# Color Restoration Training

This module trains the optional color-restoration stage independently:

```text
clean portrait -> synthetic color degradation -> color model -> clean portrait
```

The target is always the original clean RGB crop. Color, saturation, tone, and
luminance are never modified to create a softer target. Only paired geometric
augmentation is allowed.

V2 trains on the same input profile used by the experimental post-pipeline:

```text
synthetic faded-color degradation -> OpenCV conservative cleanup -> color model
```

The V2 degradation profile preserves usable chroma. Near-grayscale inputs belong
to a separate colorization problem and are not generated for this model.

## 1. Generate paired data

```powershell
python scripts\generate_color_dataset.py `
  --clean-dir D:\datasets\clean_images `
  --output-dir data\processed\ds-color-restoration-v2-320 `
  --degradation-profile faded_color_v2 `
  --train-variants 3 `
  --eval-variants 1 `
  --quality-mode opencv_conservative
```

Repeat `--clean-dir` to mix datasets without copying them into one directory:

```bash
python scripts/generate_color_dataset.py \
  --clean-dir /kaggle/input/ffhq \
  --clean-dir /kaggle/input/deepfashion-person-subset \
  --clean-dir /kaggle/input/coco-person-subset \
  --output-dir /kaggle/working/ds-color-v2-mixed \
  --max-sources 15000 \
  --degradation-profile faded_color_v2 \
  --quality-mode opencv_conservative \
  --overwrite
```

The generator audits source dimensions and mean Lab chroma before splitting
source IDs into train/val/test. V2 defaults to rejecting clean sources with mean
chroma below `4.0`. Inspect `source_audit.csv`, `manifest.csv`, and
`previews/comparison_grid.png` before training.

Each V2 preview contains:

```text
clean target | synthetic degraded | model input after OpenCV conservative
```

## 2. Train

Run a one-batch contract check:

```powershell
python scripts\train_restoration.py `
  --config configs\color_restoration_v2.yaml `
  --run-id color-unet-lab-residual-v2-r001-s42 `
  --dry-run
```

Run the baseline:

```powershell
python scripts\train_restoration.py `
  --config configs\color_restoration_v2.yaml `
  --run-id color-unet-lab-residual-v2-r001-s42
```

V2 predicts bounded Lab residuals:

```text
L output  = L input  + 15 * residual L
ab output = ab input + 40 * residual ab
```

It uses GroupNorm for stable small-batch color training. Legacy `lab_ab` and
`rgb_residual` checkpoints remain loadable.

The default objective is:

```text
0.5 L1 RGB + 0.5 L1 Lab-L + L1 Lab-ab
+ 0.2 SSIM + 0.1 Histogram EMD Lab-ab + 0.2 identity preservation
```

Histogram EMD compares the global predicted and target chroma distributions.
The default config uses early stopping with patience `10`, AMP on CUDA, and
Kaggle-safe DataLoader workers.

## 3. Evaluate and infer

```powershell
python scripts\evaluate_color_restoration.py `
  --checkpoint checkpoints\color_restoration\color-unet-lab-residual-v2-r001-s42\best.pth `
  --dataset-root data\processed\ds-color-restoration-v2-320 `
  --output-dir outputs\color_eval

python scripts\infer_color_restoration.py `
  --input outputs\lama_result.png `
  --checkpoint checkpoints\color_restoration\color-unet-lab-residual-v2-r001-s42\best.pth `
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
  --quality-mode opencv_conservative `
  --color-checkpoint checkpoints\color_restoration\color-unet-lab-residual-v2-r001-s42\best.pth `
  --face-mode auto
```

The full pipeline validates the checkpoint's training quality mode. A V2
checkpoint trained after `opencv_conservative` is rejected if runtime uses
`--quality-mode off`.

## 5. Train V2 on color-preserving old-photo degradation

Use 5,000 FFHQ sources for the first full V2 run:

```bash
python scripts/generate_color_dataset.py \
  --clean-dir /kaggle/input/ffhq-dataset \
  --output-dir /kaggle/working/ds-color-ffhq-v2-5000 \
  --max-sources 5000 \
  --degradation-profile faded_color_v2 \
  --crop-size 320 \
  --train-variants 3 \
  --eval-variants 1 \
  --quality-mode opencv_conservative \
  --num-previews 20 \
  --overwrite
```

This profile produces:

```text
5% identity
15% mild yellow/fading
45% faded old color
25% moderate sepia with retained chroma
10% hard color degradation with retained chroma
```

Every generated sample uses the unmodified clean FFHQ crop as its target. The
V2 profile deliberately excludes near-grayscale degradation. Do not reuse
datasets generated with `real_old_photo_heavy` or the removed
`conservative_real_old_photo` target profile.

All non-identity V2 samples receive a warm yellow paper cast. Faded, sepia, and
hard-color samples use progressively stronger warm casts while retaining some
random channel variation.

Train a fresh Lab-residual run:

```bash
python scripts/train_restoration.py \
  --config configs/color_restoration_v2.yaml \
  --dataset-root /kaggle/working/ds-color-ffhq-v2-5000 \
  --run-id color-ffhq-lab-residual-v2-r001 \
  --mode lab_residual \
  --base-channels 32 \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.0002 \
  --device cuda
```

Before training, inspect the preview carefully. Each row uses:

```text
clean RGB target | synthetic degraded | input after OpenCV conservative
```

Do not train if the target differs in color from the clean source, or if the
generated inputs are mostly red/pink, heavily blurred, or structurally damaged.
