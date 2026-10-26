#!/usr/bin/env python3
"""
Generate the Kaggle notebooks from code kept in this file.

The notebooks are committed as .ipynb so Kaggle can import them directly.
Keeping the generator in the repo makes it obvious how they were built and lets
them be regenerated after a pipeline change:

    python3 kaggle/build_notebooks.py

Each notebook starts with a cell that clones (or copies) the repository so that
`scripts/`, `config/` and the root modules are importable under /kaggle/working/medica.
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/NickLitharis/medica.git"

SETUP = f'''# --- Setup: get the repo into the notebook runtime -----------------------
# On Kaggle: add the repo as a "utility" input dataset, or set REPO_URL.
# The clone lands in /kaggle/working, which is the only writable area and
# what gets saved with the version.
import os, sys, subprocess
from pathlib import Path

REPO_URL = "{REPO_URL}"
WORK = Path("/kaggle/working")
REPO = WORK / "medica"

if not REPO.exists():
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO)], check=True)
    except Exception as e:
        print("git clone failed:", e)
        print("Upload the repo as a Kaggle dataset and set REPO to that folder instead.")
    else:
        print("Cloned to", REPO)

if REPO.exists():
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    print("Working directory:", os.getcwd())
else:
    print("Repo missing: the rest of this notebook cannot run.")
'''

PATHS_CELL = '''# --- Resolve input and output paths --------------------------------------
# Attach the datasets first (right sidebar -> Add Data):
#   * figure images  (PNG files, folder name does not matter)
#   * s2_cleaned.json (cleaned captions JSONL)
from config.paths import resolve, check
paths = resolve()
check(paths)
'''

INDEX_CELL = '''# --- Stage 1: index the images and verify the caption matches -------------
!python3 scripts/create_image_index.py
!python3 scripts/fix_and_create_verified_index.py
'''

PREP_CELL = '''# --- Stage 2: preprocess images to 512x512 -------------------------------
# Writes into /kaggle/working/medica/preprocessed/images. Kaggle gives a
# notebook about 20 GB of working disk, so 105k images at 512px will not fit.
# Run this stage with --limit for a subset, or reduce the size to 384/224.
!python3 scripts/preprocess_images.py --limit 20000
'''

FEATURE_CELL = '''# --- Stage 3: ViT features (cached as one .npy per figure) ----------------
# Optional: the attention captioner reads the images directly, so it does not
# need these files. Use this stage only for the classical encoder path or to
# cache features for a lighter notebook.
!pip install -q transformers
!python3 scripts/extract_features.py
'''

TRAIN_CELL = '''# --- Stage 4: train -----------------------------------------------------
# Calls the repository's own training script (TensorFlow/Keras 2 pipeline).
# --out is where model_best.keras and encoding.json land.
!python3 train.py --dataset-dir /kaggle/input/dataset \
    --out /kaggle/working/medica/run --epochs 30 --batch-size 128 --arch seq2seq
'''

EVAL_CELL = '''# --- Stage 5: score on the held-out split -------------------------------
# evaluate.py reads model_best.keras + encoding.json from --run and prints
# BLEU-1..4 against the published ImageCLEF 2017 baselines.
!python3 evaluate.py --dataset-dir /kaggle/input/dataset \
    --run /kaggle/working/medica/run --beam-width 1
'''

EXAMPLES_CELL = '''# --- Stage 6: a few generated captions, side by side --------------------
import json
from pathlib import Path
import tensorflow as tf
import pandas as pd

run = Path("/kaggle/working/medica/run")
enc = json.load(open(run / "encoding.json"))
print("run configuration:", {k: enc[k] for k in list(enc)[:8]})
print("artifacts:", sorted(p.name for p in run.iterdir()))
'''


def nb(cells, title):
    out = {
        "cells": [],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu" if "train" in title.lower() else None},
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for kind, src in cells:
        if kind == "md":
            out["cells"].append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
        else:
            out["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {},
                                 "outputs": [], "source": src.splitlines(True)})
    return out




NOTEBOOKS = {
    "01_pipeline.ipynb": nb([
        ("md", "# Medica: Dataset Pipeline on Kaggle\n\n"
               "Index the uploaded figures, verify caption to image matches, "
               "then preprocess. Attach both datasets before running."),
        ("code", SETUP),
        ("code", PATHS_CELL),
        ("code", INDEX_CELL),
        ("code", PREP_CELL),
        ("code", FEATURE_CELL),
    ], "Medica 01 pipeline"),
    "02_train.ipynb": nb([
        ("md", "# Medica: training and evaluation\n\n"
               "Runs the repository's own `train.py` and `evaluate.py` on the held-out "
               "split. Attach the `dataset/` folder as a Kaggle Dataset, then run. "
               "See `readme.md` for the architecture and the measured scores."),
        ("code", SETUP),
        ("code", PATHS_CELL),
        ("code", TRAIN_CELL),
        ("code", EVAL_CELL),
        ("code", EXAMPLES_CELL),
    ], "Medica 02 train"),
}



def main():
    for name, notebook in NOTEBOOKS.items():
        path = OUT / name
        path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
