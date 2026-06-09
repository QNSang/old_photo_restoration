from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.color_dataset import ColorRestorationDataset
from src.losses.color_restoration import ColorRestorationLoss, reconstruct_lab_ab
from src.models.color_unet import ColorRestorationUNet, VALID_COLOR_MODEL_MODES
from src.restoration.color_restoration import CHECKPOINT_FORMAT_VERSION, resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Pik-Fix-inspired color restoration U-Net.")
    parser.add_argument("--config", default="configs/color_restoration.yaml")
    parser.add_argument("--run-id", default="color-unet-rgb-r001-s42")
    parser.add_argument("--mode", choices=sorted(VALID_COLOR_MODEL_MODES))
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--dataset-root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-run", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(json.dumps(config))
    if args.mode:
        config["model"]["mode"] = args.mode
    if args.base_channels is not None:
        config["model"]["base_channels"] = args.base_channels
    if args.dataset_root:
        config["dataset"]["root"] = args.dataset_root
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    return config


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_kaggle_environment() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


def resolve_num_workers(value: Any) -> int:
    if value == "auto" or value is None:
        return 0 if os.name == "nt" or is_kaggle_environment() else 2
    workers = int(value)
    if workers < 0:
        raise ValueError("num_workers must be >= 0")
    return workers


def make_loader(dataset: ColorRestorationDataset, batch_size: int, num_workers: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )


def prediction_to_rgb(model: ColorRestorationUNet, inputs: torch.Tensor) -> torch.Tensor:
    prediction = model(inputs)
    return reconstruct_lab_ab(inputs, prediction) if model.mode == "lab_ab" else prediction


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = ((tensor.detach().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5).permute(1, 2, 0).numpy()
    return np.clip(image, 0, 255).astype(np.uint8)


def save_sample_grid(path: Path, degraded: torch.Tensor, predicted: torch.Tensor, clean: torch.Tensor, count: int = 4) -> None:
    rows = []
    for index in range(min(count, degraded.shape[0])):
        rows.append(np.concatenate([tensor_to_rgb_uint8(degraded[index]), tensor_to_rgb_uint8(predicted[index]), tensor_to_rgb_uint8(clean[index])], axis=1))
    if not rows:
        return
    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def average_metrics(sums: dict[str, float], batches: int) -> dict[str, float]:
    return {name: value / max(1, batches) for name, value in sums.items()}


def run_epoch(
    model: ColorRestorationUNet,
    loader: DataLoader,
    loss_fn: ColorRestorationLoss,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> tuple[dict[str, float], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None]:
    training = optimizer is not None
    model.train(training)
    sums = {"total": 0.0, "rgb_l1": 0.0, "ab_l1": 0.0, "ssim": 0.0}
    sample_batch = None
    for batch in loader:
        inputs = batch["degraded"].to(device, non_blocking=True)
        targets = batch["clean"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = prediction_to_rgb(model, inputs)
                losses = loss_fn(predictions, targets)
            if training:
                assert scaler is not None
                scaler.scale(losses["total"]).backward()
                scaler.step(optimizer)
                scaler.update()
        for name in sums:
            sums[name] += float(losses[name].detach().cpu())
        if sample_batch is None:
            sample_batch = (inputs.detach().cpu(), predictions.detach().cpu(), targets.detach().cpu())
    return average_metrics(sums, len(loader)), sample_batch


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: ColorRestorationUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    metrics: dict[str, Any],
    config: dict[str, Any],
    dataset_id: str,
    early_stopping: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "metrics": metrics,
            "model_config": model.get_config(),
            "dataset_config": config["dataset"],
            "training_config": config["training"],
            "dataset_id": dataset_id,
            "input_profile": config["dataset"].get("input_profile", ""),
            "early_stopping": early_stopping,
            "residual_scale": float(model.residual_scale.detach().cpu()),
        },
        path,
    )


def write_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def read_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_run_dir(path: Path, overwrite: bool, resume: bool) -> None:
    if path.exists() and not resume:
        if not overwrite:
            raise FileExistsError(f"Run already exists: {path}. Use --overwrite-run or --resume.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = apply_overrides(load_config(config_path), args)
    dataset_root = resolve_path(config["dataset"]["root"])
    model_config = dict(config["model"])
    training = config["training"]
    loss_config = config["loss"]
    device = resolve_torch_device(args.device)
    seed = int(training.get("seed", 42))
    set_seed(seed)

    train_dataset = ColorRestorationDataset(dataset_root, "train", int(config["dataset"]["image_size"]), augment=True)
    val_dataset = ColorRestorationDataset(dataset_root, "val", int(config["dataset"]["image_size"]), augment=False)
    num_workers = resolve_num_workers(training.get("num_workers", "auto"))
    train_loader = make_loader(train_dataset, int(training["batch_size"]), num_workers, True, device)
    val_loader = make_loader(val_dataset, int(training["batch_size"]), num_workers, False, device)
    model = ColorRestorationUNet(**model_config).to(device)
    loss_fn = ColorRestorationLoss(**loss_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["lr"]), weight_decay=float(training.get("weight_decay", 1e-4)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]))
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    if args.dry_run:
        metrics, sample = run_epoch(model, [next(iter(train_loader))], loss_fn, device, optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled)
        print(json.dumps({"device": str(device), "num_workers": num_workers, "metrics": metrics, "sample_shapes": [list(t.shape) for t in sample] if sample else []}, indent=2))
        return 0

    checkpoint_root = resolve_path(config["outputs"]["checkpoint_root"]) / args.run_id
    experiment_root = resolve_path(config["outputs"]["experiment_root"]) / args.run_id
    prepare_run_dir(checkpoint_root, args.overwrite_run, bool(args.resume))
    prepare_run_dir(experiment_root, args.overwrite_run, bool(args.resume))
    (checkpoint_root / "config_snapshot.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    start_epoch = 1
    best_val = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = read_history(experiment_root / "metrics.csv") if args.resume else []
    if args.resume:
        resume_checkpoint = torch.load(resolve_path(args.resume), map_location=device, weights_only=False)
        if resume_checkpoint.get("model_config") != model.get_config():
            raise ValueError("Resume checkpoint model_config does not match the active training config")
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if resume_checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val = float(resume_checkpoint.get("early_stopping", {}).get("best_val_loss", best_val))
        epochs_without_improvement = int(resume_checkpoint.get("early_stopping", {}).get("epochs_without_improvement", 0))

    patience = int(training.get("patience", 10))
    min_delta = float(training.get("min_delta", 1e-4))
    dataset_id = dataset_root.name
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_metrics, _ = run_epoch(model, train_loader, loss_fn, device, optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled)
        with torch.no_grad():
            val_metrics, sample = run_epoch(model, val_loader, loss_fn, device, amp_enabled=amp_enabled)
        scheduler.step()
        improved = val_metrics["total"] < best_val - min_delta
        if improved:
            best_val = val_metrics["total"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        early_stopping = {
            "patience": patience,
            "min_delta": min_delta,
            "best_val_loss": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "triggered": epochs_without_improvement >= patience > 0,
        }
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], "residual_scale": float(model.residual_scale.detach().cpu())}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        row["improved"] = int(improved)
        history.append(row)
        write_history(experiment_root / "metrics.csv", history)
        if sample:
            save_sample_grid(experiment_root / "samples" / f"epoch_{epoch:03d}.png", *sample)
        save_checkpoint(checkpoint_root / "last.pth", epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=row, config=config, dataset_id=dataset_id, early_stopping=early_stopping)
        if improved:
            save_checkpoint(checkpoint_root / "best.pth", epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, metrics=row, config=config, dataset_id=dataset_id, early_stopping=early_stopping)
        print(f"epoch={epoch} train_total={train_metrics['total']:.6f} val_total={val_metrics['total']:.6f} scale={row['residual_scale']:.4f} improved={improved}")
        if early_stopping["triggered"]:
            print(f"early_stopping: epoch={epoch} best_val_loss={best_val:.6f}")
            break

    summary = {
        "run_id": args.run_id,
        "dataset_id": dataset_id,
        "device": str(device),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "best_val_loss": best_val,
        "epochs_executed": int(history[-1]["epoch"]) if history else 0,
        "checkpoint": str(checkpoint_root / "best.pth"),
    }
    (experiment_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
