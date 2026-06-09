# R013 Checkpoint

- Model name: `r013_gen120_fixed118_local`
- Purpose: segmentation / repair-mask model for old photo restoration runtime and demo flows
- Source path before copy: `outputs\r013_finetune\r013_gen120_fixed118_local\best_val_iou.pth`
- Canonical checkpoint path: `checkpoints\segmenter\seg-unet-attn-r013-gen120-fixed118-local\best_val_iou.pth`
- SHA256: `a63381ade991cb936e2262e80fa6001c3a1fe9d10b1075be0d3c7f617c0a5725`
- File size: `1585714` bytes

## Model Note

- Base: fine-tune từ historical `r011` checkpoint
- Current status: current operational candidate checkpoint
- Dataset: `r013` fixed dataset, `118` valid image-mask pairs
- Train / val / test: `83 / 18 / 17`
- Best checkpoint: `best_val_iou.pth`

## Recommended Thresholds

- `0.50`: balanced
- `0.40`: sensitive

## Fair Test Comparison

### r011 @ 0.55

- IoU: `0.252702`
- F1: `0.402465`
- Precision: `0.411155`
- Recall: `0.408300`

### r013 @ 0.50

- IoU: `0.345656`
- F1: `0.509736`
- Precision: `0.588667`
- Recall: `0.466958`

## Copied Companion Files

- `best_val_f1.pth`
- `last.pth`
- `metrics_summary.json`
- `run_metadata.json`
- `r013_candidate_evaluation_summary.md`
- `fair_r011_vs_r013_test.csv`
- `threshold_sweep_val.csv`

## Usage Examples

```powershell
python scripts\run_restoration_pipeline.py --image <image> --mode auto_r013_union_refined --backend official_lama --face-mode off
python scripts\run_restoration_pipeline.py --image <image> --mode auto_r013_union_repair_wide --backend official_lama --face-mode off
```

## Warnings

- Do not rename this checkpoint to `r011`.
- Original `r011` / `r010` checkpoints are missing locally.
- This checkpoint is `r013` candidate, not `r011`.
- Do not commit this checkpoint into Git if large-file policy forbids it.
