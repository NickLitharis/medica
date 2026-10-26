#!/usr/bin/env python3
"""
Import this at the top of any pipeline entry point to get Kaggle-aware paths.

    from config.bootstrap import paths
    paths.indexes_dir          # writable output folder
    paths.images_dir           # uploaded figure images
    paths.captions             # cleaned captions JSONL

Resolution, precedence and defaults are documented in config/paths.py.
Override with MEDICA_INPUT_DIR, MEDICA_CAPTIONS and MEDICA_WORK_DIR.
"""

import sys
from pathlib import Path

# Make `config` importable whether the script is run from the repo root or
# copied into a Kaggle notebook's working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.paths import Paths, check, resolve  # noqa: E402

paths = resolve()

__all__ = ["Paths", "check", "paths", "resolve"]
