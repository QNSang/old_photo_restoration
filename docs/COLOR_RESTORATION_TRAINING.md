# Color Restoration Training

This module trains the optional color-restoration stage independently:

```text
clean portrait -> synthetic color degradation -> color model -> clean portrait
```

OpenCV quality restoration is not part of the default training path. It remains
available only as an explicit ablation through `--quality-mode opencv_conservative`.

## 1. Generate paired data

```powershell
python scripts\generate_color_dataset.py `
  --clean-dir D:\datasets\clean_images `
  --output-dir data\processed\ds-color-restoration-320-v001 `
  --train-variants 3 `
  --eval-variants 1 `
  --quality-mode off
```

The generator audits source dimensions before splitting source IDs into
train/val/test. Images smaller than `320px` on either side are skipped by
default. Inspect `source_audit.csv`, `manifest.csv`, and
`previews/comparison_grid.png` before training.

With `--quality-mode off`, each preview contains:

```text
clean target | synthetic degraded model input
```

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
  --quality-mode off `
  --color-checkpoint checkpoints\color_restoration\color-unet-rgb-r001-s42\best.pth `
  --face-mode auto
```

The full pipeline must use the same input profile used during training. A model
trained with `--quality-mode off` should therefore be run with
`--quality-mode off`.

## 5. Fine-tune for real yellow/sepia portraits

Use 1,000 FFHQ sources and the heavy real-old-photo profile:

```bash
python scripts/generate_color_dataset.py \
  --clean-dir /kaggle/input/ffhq-dataset \
  --output-dir /kaggle/working/ds-color-ffhq-heavy-1000 \
  --max-sources 1000 \
  --degradation-profile real_old_photo_heavy \
  --target-profile conservative_real_old_photo \
  --crop-size 320 \
  --train-variants 3 \
  --eval-variants 1 \
  --quality-mode off \
  --num-previews 20 \
  --overwrite
```

This profile produces four useful groups:

```text
35% strong sepia
30% warm near-grayscale
25% faded old color
10% mild or identity samples
```

The conservative target follows the remaining color information:

```text
faded old color       -> mostly colorful target
strong yellow/sepia   -> moderately saturated target
warm near-grayscale   -> low-saturation target
identity/mild         -> clean RGB target
```

This prevents the model from being forced to invent full FFHQ colors for a
nearly black-and-white input, which is a major cause of global red/pink casts.

Fine-tune the existing RGB residual checkpoint into a new run:

```bash
python scripts/train_restoration.py \
  --config configs/color_restoration.yaml \
  --dataset-root /kaggle/working/ds-color-ffhq-heavy-1000 \
  --run-id color-ffhq-rgb-heavy-ft-r001 \
  --mode rgb_residual \
  --base-channels 32 \
  --epochs 15 \
  --batch-size 8 \
  --lr 0.00002 \
  --init-checkpoint checkpoints/color_restoration/color-ffhq-rgb-r001/best.pth \
  --device cuda
```

`--init-checkpoint` loads only model weights. It starts a new optimizer,
scheduler, epoch count, and output run, which is the intended fine-tuning
behavior.

Before training, inspect the preview carefully. Each row uses:

```text
clean FFHQ source | old-photo synthetic input | conservative training target
```

The third panel should become less saturated as the second panel approaches
warm gray or black-and-white. Do not fine-tune if the generated inputs are
mostly red/pink, heavily blurred, or structurally damaged.
