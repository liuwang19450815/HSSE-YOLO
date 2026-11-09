"""Install the two HSSE-YOLO custom operators into Ultralytics 8.4.7.

This is deliberately small and idempotent: it copies only the two modules and
adds two parser branches required by the model YAMLs.  It does not alter
datasets, checkpoints, training hyperparameters, or any other Ultralytics code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import ultralytics
import ultralytics.nn.tasks as tasks


ROOT = Path(__file__).resolve().parent
MODULES = ROOT / "modules"
TASKS = Path(tasks.__file__)
NN_DIR = TASKS.parent
IMPORT_MARKER = "# HSSE-YOLO custom modules\n"
PARSER_MARKER = "        # HSSE-YOLO custom parser branches\n"


def install_modules() -> None:
    version = getattr(ultralytics, "__version__", "unknown")
    if version != "8.4.7":
        raise RuntimeError(
            f"This release was verified with ultralytics==8.4.7, but found {version}. "
            "Install the pinned version from requirements.txt before running."
        )
    shutil.copy2(MODULES / "csodca.py", NN_DIR / "hsse_csodca.py")
    shutil.copy2(MODULES / "csgrsf.py", NN_DIR / "hsse_csgrsf.py")


def patch_tasks() -> None:
    text = TASKS.read_text(encoding="utf-8")
    if IMPORT_MARKER not in text:
        import_anchor = "import torch.nn as nn\n"
        imports = (
            import_anchor
            + "\n"
            + IMPORT_MARKER
            + "from ultralytics.nn.hsse_csodca import CSODCA\n"
            + "from ultralytics.nn.hsse_csgrsf import CSGRSF\n"
        )
        if import_anchor not in text:
            raise RuntimeError("Unexpected Ultralytics tasks.py: import anchor not found.")
        text = text.replace(import_anchor, imports, 1)

    if PARSER_MARKER not in text:
        parser_anchor = "        elif m is CBFuse:\n"
        branches = (
            PARSER_MARKER
            + "        elif m is CSGRSF:\n"
            + "            c2 = make_divisible(min(args[0], max_channels) * width, 8)\n"
            + "            args = [[ch[x] for x in f], c2]\n"
            + "        elif m is CSODCA:\n"
            + "            c2 = make_divisible(min(args[0], max_channels) * width, 8)\n"
            + "            args = [[ch[x] for x in f], c2, *args[1:]]\n"
        )
        if parser_anchor not in text:
            raise RuntimeError("Unexpected Ultralytics tasks.py: parser anchor not found.")
        text = text.replace(parser_anchor, branches + parser_anchor, 1)

    TASKS.write_text(text, encoding="utf-8")


def main() -> None:
    install_modules()
    patch_tasks()
    print(f"HSSE-YOLO modules installed in: {NN_DIR}")


if __name__ == "__main__":
    main()
