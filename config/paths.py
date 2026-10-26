#!/usr/bin/env python3
"""
Central path resolution for the medica pipeline.

On Kaggle, input datasets are mounted read-only under /kaggle/input and the
only writable place is /kaggle/working. This module resolves every path the
pipeline needs from three environment variables, so no script hardcodes a
machine-specific location:

    MEDICA_INPUT_DIR   folder holding the uploaded figure images
                       default: first Kaggle dataset containing PNG files,
                       otherwise ./dataset/images
    MEDICA_CAPTIONS    cleaned captions JSONL
                       default: first Kaggle dataset file named s2_cleaned.json,
                       otherwise ./data/processed/s2_cleaned.json
    MEDICA_WORK_DIR    writable working root
                       default: /kaggle/working on Kaggle, otherwise ./outputs

Usage from any script:

    from config.paths import resolve
    paths = resolve()
    print(paths.images_dir, paths.work_dir)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Repository root: this file lives in <root>/config/paths.py
REPO_ROOT = Path(__file__).resolve().parent.parent

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".dcm"}


def on_kaggle() -> bool:
    return KAGGLE_INPUT.is_dir()


def _find_images_dir() -> Path:
    """Locate the folder that actually holds the figure images."""
    env = os.environ.get("MEDICA_INPUT_DIR")
    if env:
        return Path(env)

    if KAGGLE_INPUT.is_dir():
        # Prefer the dataset folder with the most images. Dataset roots can be
        # nested (dataset/key/hash_figure.png), so search two levels deep.
        best, best_count = None, 0
        for root in sorted(KAGGLE_INPUT.iterdir()):
            if not root.is_dir():
                continue
            for candidate in [root, *[p for p in root.iterdir() if p.is_dir()]]:
                try:
                    with os.scandir(candidate) as it:
                        count = sum(1 for e in it if e.is_file()
                                    and os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS)
                except OSError:
                    continue
                if count > best_count:
                    best, best_count = candidate, count
        if best is not None:
            return best

    return REPO_ROOT / "dataset" / "images"


def _find_captions() -> Path:
    env = os.environ.get("MEDICA_CAPTIONS")
    if env:
        return Path(env)

    if KAGGLE_INPUT.is_dir():
        for path in KAGGLE_INPUT.rglob("s2_cleaned.json"):
            return path

    return REPO_ROOT / "data" / "processed" / "s2_cleaned.json"


def _find_work_dir() -> Path:
    env = os.environ.get("MEDICA_WORK_DIR")
    if env:
        return Path(env)
    if KAGGLE_WORKING.is_dir():
        return KAGGLE_WORKING / "medica"
    return REPO_ROOT / "outputs"


@dataclass(frozen=True)
class Paths:
    images_dir: Path
    captions: Path
    work_dir: Path

    # Where each pipeline stage writes
    @property
    def indexes_dir(self) -> Path:
        return self.work_dir / "indexes"

    @property
    def preprocessed_dir(self) -> Path:
        return self.work_dir / "preprocessed"

    @property
    def features_dir(self) -> Path:
        return self.work_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.work_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.work_dir / "reports"

    @property
    def image_index(self) -> Path:
        return self.indexes_dir / "image_index.json"

    @property
    def verified_index(self) -> Path:
        return self.indexes_dir / "verified_training_index.json"

    @property
    def preprocessed_index(self) -> Path:
        return self.indexes_dir / "verified_training_index_preprocessed.json"

    def make_dirs(self) -> None:
        """Create every writable directory the pipeline uses."""
        for d in (self.indexes_dir, self.preprocessed_dir, self.features_dir,
                  self.models_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return (f"images_dir = {self.images_dir}\n"
                f"captions   = {self.captions}\n"
                f"work_dir   = {self.work_dir}")


def resolve(create: bool = True) -> Paths:
    paths = Paths(
        images_dir=_find_images_dir(),
        captions=_find_captions(),
        work_dir=_find_work_dir(),
    )
    if create:
        paths.make_dirs()
    return paths


def check(paths: Paths) -> bool:
    """Print resolved paths and report anything missing. Returns True if usable."""
    ok = True
    print("=" * 70)
    print("MEDICA PATHS")
    print("=" * 70)
    print(paths.describe())
    print()

    if not paths.images_dir.is_dir():
        print(f"[missing] image folder not found: {paths.images_dir}")
        print("          set MEDICA_INPUT_DIR, or attach the Kaggle dataset with the figures")
        ok = False
    else:
        with os.scandir(paths.images_dir) as it:
            n = sum(1 for e in it if e.is_file()
                    and os.path.splitext(e.name)[1].lower() in IMAGE_EXTENSIONS)
        print(f"[ok]      {n:,} images in {paths.images_dir}")

    if not paths.captions.is_file():
        print(f"[missing] captions file not found: {paths.captions}")
        print("          set MEDICA_CAPTIONS, or attach the Kaggle dataset with s2_cleaned.json")
        ok = False
    else:
        print(f"[ok]      captions file {paths.captions.stat().st_size / 1e6:.1f} MB")

    print()
    return ok


if __name__ == "__main__":
    sys.exit(0 if check(resolve()) else 1)
