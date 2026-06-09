# r013 Candidate Evaluation Summary

## Checkpoints
- r011 checkpoint: `F:\deeplearning\old_photo_restoration_2\checkpoints\segmenter\seg-unet-attn-r011-repair-ft-s42\best_iou.ckpt`
- r013 checkpoint: `F:\deeplearning\old_photo_restoration_2\outputs\r013_finetune\r013_gen120_fixed118_local\best_val_iou.pth`

## Threshold Selection On Val
- r011 best IoU threshold: `0.50` (IoU `0.209298`, F1 `0.342801`)
- r011 best F1 threshold: `0.55` (IoU `0.209214`, F1 `0.342918`)
- r013 best IoU threshold: `0.50` (IoU `0.381231`, F1 `0.550253`)
- r013 best F1 threshold: `0.50` (IoU `0.381231`, F1 `0.550253`)
- r013 sensitive threshold used for demo review: `0.40` (precision `0.493744`, recall `0.628110`)

Fair comparison used thresholds selected on val only: r011 `0.55` by best F1, r013 `0.50` by best IoU/F1.

## Fair Val Comparison
| model | split | threshold | iou | f1 | precision | recall | mask_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r011 | val | 0.55 | 0.209214 | 0.342918 | 0.282854 | 0.509079 | 0.133967 |
| r013 | val | 0.50 | 0.381231 | 0.550253 | 0.519363 | 0.598631 | 0.075404 |

## Fair Test Comparison
| model | split | threshold | iou | f1 | precision | recall | mask_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r011 | test | 0.55 | 0.252702 | 0.402465 | 0.411155 | 0.408300 | 0.079972 |
| r013 | test | 0.50 | 0.345656 | 0.509736 | 0.588667 | 0.466958 | 0.063429 |

r013 vẫn tốt hơn r011 sau threshold tuning công bằng trên val: test IoU tăng từ `0.252702` lên `0.345656`, F1 tăng từ `0.402465` lên `0.509736`, precision tăng từ `0.411155` lên `0.588667`, recall tăng từ `0.408300` lên `0.466958`.

## Visual Segmentation Review
- Grid fair val: `fair_comparison_grid_val.png`.
- Grid fair test: `fair_comparison_grid_test.png`.
- Grid được chọn theo các case có chênh lệch metric, case khó/cả hai fail, mask nhỏ và mask lớn. Nhìn nhanh cho thấy r013 giảm false positive nền/texture so với r011, đặc biệt ở vùng sách, núi/cây, nền ảnh và biên ảnh.
- Hạn chế quan sát: r013 có mask gọn hơn, nên một số vết mảnh hoặc vùng nứt rất nhạt cần review full-size trước khi thay default.

## Real Demo Mask Review
- Ảnh demo ngoài r013 dataset: `data/demo_inputs/real_manual_3/demo1.jpg`, `demo2.png`, `demo3.png`.
- Output: `real_demo_eval/`.
- Contact sheet: `real_demo_eval/demo_mask_contact_sheet.png`.

| image | r011_ratio | r013_t050_ratio | r013_sensitive_ratio |
| --- | --- | --- | --- |
| demo1.jpg | 0.006664 | 0.002791 | 0.003908 |
| demo2.png | 0.014429 | 0.013333 | 0.015599 |
| demo3.png | 0.043880 | 0.042615 | 0.049695 |

Nhận xét demo: r013 `0.50` thường tạo mask gọn hơn r011 trên demo1 và tương đương hơn trên demo2/demo3; r013 sensitive `0.40` mở rộng vùng mask vừa phải. Đây là dấu hiệu tốt cho giảm false positive, nhưng cần xem full-size vì demo không có ground-truth đầy đủ, trừ manual mask demo3.

## Restoration Ablation
- Đã chạy `official_lama` với face off trên demo1 và demo3.
- Cases: r011 t0.55, r013 t0.50, r013 sensitive t0.40; demo3 có thêm oracle/manual mask.
- Output root: `restoration_ablation/`.
- Contact sheet: `restoration_ablation/restoration_ablation_contact_sheet.png`.

| image | case | mask_ratio | backend | fallback |
| --- | --- | --- | --- | --- |
| demo1 | r011_t055 | 0.006664 | official_lama | False |
| demo1 | r013_t050 | 0.002791 | official_lama | False |
| demo1 | r013_sensitive_t040 | 0.003908 | official_lama | False |
| demo3 | r011_t055 | 0.043880 | official_lama | False |
| demo3 | r013_t050 | 0.042615 | official_lama | False |
| demo3 | r013_sensitive_t040 | 0.049695 | official_lama | False |
| demo3 | oracle | 0.106163 | official_lama | False |

Nhận xét restoration: official_lama chạy được và không fallback trong các case đã chạy. r013 t0.50 thường ít can thiệp hơn r011 trên demo1; trên demo3 r013 t0.50 gần r011, còn sensitive t0.40 mở rộng thêm vùng sửa. Oracle demo3 có mask ratio lớn hơn rõ (`0.106163`), cho thấy các mask tự động vẫn chưa bao phủ toàn bộ vùng manual/oracle.

## Decision
- Giữ r013 làm candidate mạnh: có metric val/test tốt hơn r011 sau threshold tuning công bằng.
- Chưa thay default r011 ngay: cần thêm visual review full-size trên ảnh thật ngoài dataset và kiểm tra end-to-end hybrid DL+CV của UI.
- Có thể cân nhắc thêm r013 vào UI như optional/experimental mode sau khi review thủ công contact sheet và một vài ảnh full-size.
- Chưa cần train thêm ngay nếu mục tiêu là candidate Module 1; nếu muốn thay default, nên bổ sung thêm ảnh thật/oracle và đánh giá pipeline đầy đủ.

## Limitations
- Dataset r013 là generated/synthetic-assisted, chưa đại diện đầy đủ cho ảnh thật.
- Số ảnh thật ngoài dataset còn ít: 3 demo, restoration ablation trên 2 ảnh.
- Chưa train LaMa.
- Final mask trong UI có thể là hybrid DL + CV + refinement, nên metric segmentation thuần chưa đủ để quyết định thay default.
- Contact sheet chỉ để xem nhanh, không thay thế review full-size.
