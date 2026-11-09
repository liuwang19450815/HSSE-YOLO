"""One-click reproduction of the final HSSE-YOLO ablation study.

Place the dataset under ``dataset/HSSE`` using the paths in data_hsse.yaml,
then run this file.  It installs the two custom modules into the installed
Ultralytics 8.4.7 package once, and trains every ablation from scratch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_hsse.yaml"
PROJECT = ROOT / "runs" / "ablation_640_seed0"

# Shared controlled training configuration.
EPOCHS = 300
IMGSZ = 640
BATCH = 16
DEVICE = 0
SEED = 0

# The execution order is also the final ablation-table order.
GROUPS = (
    ("baseline", "yolo11n-Baseline11.yaml", "baseline_640_seed0"),
    ("p2head", "yolo11n-P2Head11.yaml", "p2head_640_seed0"),
    ("csodca", "yolo11n-CSODCA11.yaml", "csodca_640_seed0"),
    ("csgrsf", "yolo11n-CSGRSF11.yaml", "csgrsf_640_seed0"),
    ("p2head_csodca", "yolo11n-P2Head-CSODCA11.yaml", "p2head_csodca_640_seed0"),
    ("p2head_csgrsf", "yolo11n-P2Head-CSGRSF11.yaml", "p2head_csgrsf_640_seed0"),
    ("csodca_csgrsf", "yolo11n-CSODCA-CSGRSF11.yaml", "csodca_csgrsf_640_seed0"),
    ("full", "yolo11n-Full-P2Head-CSODCA-CSGRSF11.yaml", "full_640_seed0"),
)


def install_custom_modules() -> None:
    subprocess.run([sys.executable, str(ROOT / "install_hsse_modules.py")], check=True)


def train_one(YOLO, label: str, yaml_name: str, run_name: str) -> None:
    run_dir = PROJECT / run_name
    complete_marker = run_dir / "TRAINING_COMPLETE"
    last_checkpoint = run_dir / "weights" / "last.pt"

    if complete_marker.is_file():
        print(f"[skip] {label}: completed")
        return
    if last_checkpoint.is_file():
        print(f"[resume] {label}: {last_checkpoint}")
        YOLO(last_checkpoint).train(resume=True)
    else:
        print(f"[start] {label}: scratch training")
        YOLO(ROOT / "models" / yaml_name).train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            seed=SEED,
            pretrained=False,
            project=PROJECT,
            name=run_name,
        )
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text("completed\n", encoding="utf-8")


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML is missing: {DATA}")
    os.chdir(ROOT)
    install_custom_modules()
    # Import only after the installer has patched the local Ultralytics parser.
    from ultralytics import YOLO

    for group in GROUPS:
        train_one(YOLO, *group)


if __name__ == "__main__":
    main()
