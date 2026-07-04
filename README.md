# HSSE-YOLO: Lightweight Multiscale Object Detection for Fine-Grained Visual Analysis of Historic Street Scenes

Official implementation of the HSSE-YOLO framework. HSSE-YOLO integrates three targeted modules — **ODCA**, **P2L-Head**, and **BAW-Concat** — into YOLO11n for fine-grained detection of spontaneous spatial elements in historic street scenes.

> 📄 **Paper**: HSSE-YOLO: Lightweight Multiscale Object Detection for Fine-Grained Visual Analysis of Historic Street Scenes  
> 📦 **Dataset**: HSSE dataset (4,534 images, 10 categories, 48,743 instances) — available on Figshare  
> 🖥️ **Inference Interface**: [HSSE-prototype](https://github.com/liuwang19450815/HSSE-prototype)

## Three Core Modules

### ODCA (Orthogonal–Diagonal Coordinate Attention)
Enhances the model's representation of non-orthogonal boundaries, oblique edges, and partially occluded objects. Based on Coordinate Attention, ODCA fuses horizontal and vertical coordinate features to generate **diagonal compensation features**, supplementing cross-directional structural representation while maintaining a lightweight design. Placed in the backbone at P2/4.

📄 `ultralytics/nn/odca.py`

### P2L-Head (P2-enhanced Lightweight Detection Head)
Introduces high-resolution **P2/4 shallow features** to improve small-object detection. Instead of simply adding a fourth detection scale, P2L-Head reconfigures the final detection output to **P2/P4/P5**, retaining P3 for neck-level feature propagation but removing it as an independent output to control prediction overhead.

📄 Implemented via YAML model configurations

### BAW-Concat (BiFPN-inspired Adaptive Weighted Concatenation)
Replaces standard `Concat` with **learnable branch-level adaptive weighting**. Two learnable parameters, normalized via ReLU + softmax, adaptively adjust the contributions of features from different scales before concatenation, enabling more effective multiscale feature fusion in the neck.

📄 `ultralytics/nn/modules/block.py` → class `BAW_Concat`

## Architecture

```
HSSE-YOLO = YOLO11n + ODCA (Backbone) + P2L-Head (Detection) + BAW-Concat (Neck)
```

| Component | Location | Purpose |
|-----------|----------|---------|
| ODCA | Backbone P2/4 | Boundary & attachment-structure representation |
| P2L-Head | Detection Head | Small-object detection via P2 features |
| BAW-Concat | Neck (all fusion points) | Adaptive multiscale feature weighting |

## Ablation Study Configs

| Config File | ODCA | P2L-Head | BAW-Concat |
|-------------|:----:|:--------:|:----------:|
| `yolo11n.yaml` (baseline) | | | |
| `ODCA-n.yaml` | ✅ | | |
| `BAW_Concat-n.yaml` | | | ✅ |
| `P2L_Head-n.yaml` | | ✅ | |
| `ODCA_BAW-n.yaml` | ✅ | | ✅ |
| `ODCA_P2L_Head-n.yaml` | ✅ | ✅ | |
| `P2L_Head_BAW-n.yaml` | | ✅ | ✅ |
| **`HSSE_YOLO-n.yaml`** | ✅ | ✅ | ✅ |

## Results Summary

### HSSE Dataset (Validation Set)

| Model | mAP50 | mAP50-95 | Params | Size |
|-------|:-----:|:--------:|:------:|:----:|
| YOLO11n (baseline) | 59.5% | 43.7% | 2.5M | 5.4 MB |
| HSSE-YOLO | **64.7%** | **50.3%** | 2.8M | 8.9 MB |

### External Validation

| Dataset | Metric | YOLO11n | HSSE-YOLO |
|---------|--------|:-------:|:---------:|
| CCTSDB2021 | mAP50 | 80.5% | **84.0%** |
| KITTI | mAP50 | 89.4% | **91.4%** |

## Quick Start

### Installation

```bash
git clone https://github.com/liuwang19450815/HSSE-YOLO.git
cd HSSE-YOLO
pip install ultralytics
```

Then copy the modified files to your ultralytics installation, or run directly from this repository.

### Training

```bash
# HSSE dataset
python train_HSSE.py

# CCTSDB2021 dataset  
python train_CCTSDB.py

# KITTI dataset
python train_Kitti.py
```

Modify the training script to switch model configs:

```python
# Ablation: single module
model = YOLO('ODCA-n.yaml')
model = YOLO('BAW_Concat-n.yaml')
model = YOLO('P2L_Head-n.yaml')

# Ablation: dual module
model = YOLO('ODCA_BAW-n.yaml')
model = YOLO('ODCA_P2L_Head-n.yaml')
model = YOLO('P2L_Head_BAW-n.yaml')

# Full model
model = YOLO('HSSE_YOLO-n.yaml')

# Baseline
model = YOLO('yolo11n.yaml')
```

### Per-Size Evaluation (CCTSDB)

```bash
python eval_per_size.py --weights runs/detect/train/weights/best.pt
```

## Project Structure

```
HSSE-YOLO/
├── ultralytics/nn/
│   ├── odca.py                  # ODCA module implementation
│   ├── modules/
│   │   ├── block.py             # BAW_Concat class
│   │   └── __init__.py          # Module registration
│   └── tasks.py                 # Parsing logic for custom modules
├── HSSE_YOLO-n.yaml             # Full HSSE-YOLO config
├── ODCA-n.yaml                  # ODCA-only ablation
├── BAW_Concat-n.yaml            # BAW-Concat-only ablation
├── P2L_Head-n.yaml              # P2L-Head-only ablation
├── ODCA_BAW-n.yaml              # Dual-module ablation
├── ODCA_P2L_Head-n.yaml         # Dual-module ablation
├── P2L_Head_BAW-n.yaml          # Dual-module ablation
├── HSSE_data.yaml               # HSSE dataset config
├── CCTSDB_data.yaml             # CCTSDB2021 dataset config
├── kitti_data.yaml              # KITTI dataset config
├── train_HSSE.py                # Training script
├── train_CCTSDB.py              # Training script
├── train_Kitti.py               # Training script
├── eval_per_size.py             # Per-size evaluation
└── README.md
```

## Citation

If you use HSSE-YOLO or the HSSE dataset in your research, please cite our paper:

```bibtex
@article{liu2025hsse,
  title={HSSE-YOLO: Lightweight Multiscale Object Detection for Fine-Grained Visual Analysis of Historic Street Scenes},
  author={Liu, Wangchenxiao and Lin, Yan and Li, Keran},
  journal={},
  year={2025}
}
```

## License

This project is built upon [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and follows the AGPL-3.0 License.
