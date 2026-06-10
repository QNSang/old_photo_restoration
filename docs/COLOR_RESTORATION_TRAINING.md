# Color Restoration Training

This module trains the optional color-restoration stage independently:

```text
clean portrait -> synthetic color degradation -> color model -> clean portrait
```

The target is always the original clean RGB crop. Color, saturation, tone, and
luminance are never modified to create a softer target. Only paired geometric
augmentation is allowed.

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
  --run-id color-unet-lab-ab-r001-s42 `
  --dry-run
```

Run the baseline:

```powershell
python scripts\train_restoration.py `
  --config configs\color_restoration.yaml `
  --run-id color-unet-lab-ab-r001-s42
```

Lab-ab prediction is the default training mode. The model preserves the input
Lab luminance channel and predicts the two chroma channels. RGB residual mode
remains available through `--mode rgb_residual` only for compatibility and
ablation.

The default objective is:

```text
L1 RGB + L1 Lab-ab + 0.2 SSIM + 0.1 Histogram EMD Lab-ab
```

Histogram EMD compares the global predicted and target chroma distributions.
The default config uses early stopping with patience `10`, AMP on CUDA, and
Kaggle-safe DataLoader workers.

## 3. Evaluate and infer

```powershell
python scripts\evaluate_color_restoration.py `
  --checkpoint checkpoints\color_restoration\color-unet-lab-ab-r001-s42\best.pth `
  --dataset-root data\processed\ds-color-restoration-320-v001 `
  --output-dir outputs\color_eval

python scripts\infer_color_restoration.py `
  --input outputs\lama_result.png `
  --checkpoint checkpoints\color_restoration\color-unet-lab-ab-r001-s42\best.pth `
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
  --color-checkpoint checkpoints\color_restoration\color-unet-lab-ab-r001-s42\best.pth `
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

Every generated sample uses the unmodified clean FFHQ crop as its target,
including strong sepia and warm near-grayscale inputs. Do not reuse datasets
generated with the removed `conservative_real_old_photo` target profile.

Train a fresh Lab-ab run:

```bash
python scripts/train_restoration.py \
  --config configs/color_restoration.yaml \
  --dataset-root /kaggle/working/ds-color-ffhq-heavy-1000 \
  --run-id color-ffhq-lab-ab-heavy-r001 \
  --mode lab_ab \
  --base-channels 32 \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.0002 \
  --device cuda
```

Before training, inspect the preview carefully. Each row uses:

```text
clean RGB target | old-photo synthetic input
```

Do not train if the target differs in color from the clean source, or if the
generated inputs are mostly red/pink, heavily blurred, or structurally damaged.
