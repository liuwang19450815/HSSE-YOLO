# HSSE-YOLO

Code implementation of HSSE-YOLO, a lightweight multiscale object detection framework based on YOLO11n for fine-grained visual analysis of historic street scenes.

> ⚠️ **Note**: This work is currently under peer review. The code and dataset are provided for review purposes.

## Modules

HSSE-YOLO integrates three modules into YOLO11n:

| Module | File | Purpose |
|--------|------|---------|
| **ODCA** (Orthogonal–Diagonal Coordinate Attention) | `ultralytics/nn/odca.py` | Boundary & attachment-structure representation |
| **P2L-Head** (P2-enhanced Lightweight Detection Head) | `HSSE_YOLO-n.yaml` | Small-object detection via P2 high-resolution features |
| **BAW-Concat** (Adaptive Weighted Concatenation) | `ultralytics/nn/modules/block.py` | Learnable branch-level multiscale feature fusion |

## Installation

```bash
git clone https://github.com/liuwang19450815/HSSE-YOLO.git
cd HSSE-YOLO
pip install ultralytics
```

Then copy the modified files into your ultralytics installation, or run directly from this directory.

## Usage

```bash
python train_HSSE.py
```

The training script uses the HSSE dataset config (`HSSE_data.yaml`) and the full HSSE-YOLO model config (`HSSE_YOLO-n.yaml`).

## Dataset

The HSSE dataset contains 4,534 annotated street-view images with 10 object categories and 48,743 instances, collected from the historic urban area of Quanzhou. The dataset is available on Figshare.

## Modified Files

| File | Change |
|------|--------|
| `ultralytics/nn/odca.py` | New — ODCA attention module |
| `ultralytics/nn/modules/block.py` | Added `BAW_Concat` class |
| `ultralytics/nn/modules/__init__.py` | Registered ODCA and BAW_Concat |
| `ultralytics/nn/tasks.py` | Added BAW_Concat channel parsing |

## License

Built upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics). AGPL-3.0 License.
