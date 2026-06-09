# Color Restoration Training

This module trains the optional color-restoration stage used after inpainting and
conservative quality restoration.

## 1. Generate paired data

```powershell
python scripts\generate_color_dataset.py `
  --clean-dir D:\datasets\clean_images `
  --output-dir data\processed\ds-color-restoration-320-v001 `
  --train-variants 3 `
  --eval-variants 1
```

The generator audits source dimensions before splitting source IDs into
train/val/test. Images smaller than `320px` on either side are skipped by
default. Inspect `source_audit.csv`, `manifest.csv`, and
`previews/comparison_grid.png` before training.

## 2. Train

Run a one-batch contract check:

```powershell
python scripts\train_restoration.py `
  --config configs\color_restoration.yaml `
  --run-id color-unet-rgb-r001-s42 `
  --dry-run
```

Run the baseline:

```powershell
python scripts\train_restoration.py `
  --config configs\color_restoration.yaml `
  --run-id color-unet-rgb-r001-s42
```

Switch to Lab-ab prediction with `--mode lab_ab`. The default config uses
early stopping with patience `10`, AMP on CUDA, and Kaggle-safe DataLoader
workers.

## 3. Evaluate and infer

```powershell
python scripts\evaluate_color_restoration.py `
  --checkpoint checkpoints\color_restoration\color-unet-rgb-r001-s42\best.pth `
  --dataset-root data\processed\ds-color-restoration-320-v001 `
  --output-dir outputs\color_eval

python scripts\infer_color_restoration.py `
  --input outputs\lama_result.png `
  --checkpoint checkpoints\color_restoration\color-unet-rgb-r001-s42\best.pth `
  --output outputs\color_restored.png
```

## 4. Use in the full pipeline

```powershell
python scripts\run_restoration_pipeline.py `
  --image data\demo_inputs\my_photos\real_0089.png `
  --output-dir outputs\r013_color_pipeline `
  --mode auto_r011_union_refined `
  --checkpoint checkpoints\segmenter\seg-unet-attn-r013-gen120-fixed118-local\best_val_iou.pth `
  --backend official_lama `
  --post-pipeline pikfix_experimental `
  --color-checkpoint checkpoints\color_restoration\color-unet-rgb-r001-s42\best.pth `
  --face-mode auto
```
