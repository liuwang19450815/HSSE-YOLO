# HSSE-YOLO: minimal reproduction package

This repository contains only the final model definitions and code required to
reproduce the final ablation study. It intentionally excludes the HSSE images,
labels, experiment outputs, checkpoints, and earlier failed design attempts.

## Included

- `modules/csodca.py`: Cross-Scale Semantic-guided Orthogonal Directional Channel Attention (CSODCA).
- `modules/csgrsf.py`: Cross-Scale Guided Residual Spatial Fusion (CSGRSF).
- `models/`: YOLO11n baseline, three single-module models, three two-module models, and the full HSSE-YOLO YAML.
- `run_all_ablation.py`: one-click, scratch-training ablation runner.
- `install_hsse_modules.py`: minimal, idempotent patch that registers the two custom modules with Ultralytics' YAML parser.
- `data_hsse.yaml`: relative-path dataset configuration template.

## Dataset layout

Place a YOLO-format HSSE dataset under `dataset/HSSE`:

```text
dataset/HSSE/
  images/train/
  images/val/
  labels/train/
  labels/val/
```

The dataset itself is excluded by `.gitignore` and is not uploaded to GitHub.

## Run

```bash
pip install -r requirements.txt
python run_all_ablation.py
```

The runner first registers the two custom modules, then trains the eight
ablations in this order: baseline, P2Head, CSODCA, CSGRSF, three two-module
variants, and Full HSSE-YOLO. All runs use scratch initialization,
`imgsz=640`, `batch=16`, `seed=0`, and a maximum of 300 epochs. Existing
completed runs are skipped; interrupted runs resume from their own `last.pt`.

## Important implementation note

This code was verified with `ultralytics==8.4.7`. The installer modifies the
installed Ultralytics parser only by adding imports and parser branches for
`CSODCA` and `CSGRSF`. The underlying Ultralytics project is distributed under
AGPL-3.0; users must comply with its license when distributing derived work.
