# Old Photo Restoration Blueprint 2.1

Pipeline nghiên cứu phục hồi ảnh cũ gồm repair-mask segmentation, mask refinement,
inpainting và face restoration tùy chọn.

Research pipeline for old-photo restoration using repair-mask segmentation, mask
refinement, inpainting, and optional face restoration.

## Documentation

- [Bilingual Technical Report / Báo cáo kỹ thuật song ngữ](docs/TECHNICAL_REPORT_BILINGUAL.md)
- [Architecture Blueprint / Kiến trúc mục tiêu](ARCHITECTURE.md)
- [Final Pipeline Status / Trạng thái pipeline](docs/FINAL_PIPELINE_STATUS.md)
- [Final Pipeline Usage / Hướng dẫn sử dụng](docs/FINAL_PIPELINE_USAGE.md)
- [Evaluation Protocol / Quy trình đánh giá](docs/EVALUATION_PROTOCOL.md)

## Quick Start

```powershell
pip install -r requirements.txt
python app_gradio.py
```

Stable automatic segmentation baseline: `r011`. It is a functional baseline, not
a solved final restoration system.
