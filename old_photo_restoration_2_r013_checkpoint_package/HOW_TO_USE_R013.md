# How To Use R013

- Place checkpoint at: `checkpoints\segmenter\seg-unet-attn-r013-gen120-fixed118-local\best_val_iou.pth`
- Use modes:
  - `auto_r013_union_refined`
  - `auto_r013_union_repair_wide`
- Thresholds:
  - `0.50` balanced
  - `0.40` sensitive
- Do not use `r011` locally because the original `r011` checkpoint is missing.
