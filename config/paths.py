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


# Files that make a folder a usable training dataset. train.py needs the first
# and third; captions are present as either the original or the repaired file.
DATASET_MARKERS = ("image_features.pkl", "captions.pkl")
REPAIRED_CAPTIONS = "captions_fixed.pkl"


def _looks_like_dataset(folder: Path) -> bool:
    return all((folder / name).is_file() for name in DATASET_MARKERS)


def find_dataset_dir() -> Path:
    """
    Locate the folder holding image_features.pkl and captions.pkl.

    Search order:
      1. MEDICA_DATASET_DIR, if set
      2. /kaggle/input, by content - every dataset folder, then one level down,
         because the slug is not known in advance and may nest one folder deep
      3. ./dataset in the repository

    Finding by content rather than by name means the dataset can be called
    anything and attached in any order.
    """
    env = os.environ.get("MEDICA_DATASET_DIR")
    if env:
        return Path(env)

    if KAGGLE_INPUT.is_dir():
        for root in sorted(KAGGLE_INPUT.iterdir()):
            if not root.is_dir():
                continue
            candidates = [root, *sorted(p for p in root.iterdir() if p.is_dir())]
            for candidate in candidates:
                if _looks_like_dataset(candidate):
                    return candidate

    return REPO_ROOT / "dataset"


def dataset_dir_with_repaired_captions(dataset_dir: Path) -> Path:
    """
    Return a directory whose captions.pkl is the repaired caption file.

    train.py always reads captions.pkl. When the dataset ships captions_fixed.pkl
    (the complete captions, see readme.md) this builds a small directory in the
    work folder that symlinks the large artefacts and copies captions_fixed.pkl
    in as captions.pkl, so training uses the better captions without touching
    the read-only input mount or copying the features file.
    """
    dataset_dir = Path(dataset_dir)
    fixed = dataset_dir / REPAIRED_CAPTIONS
    if not fixed.is_file() or not _looks_like_dataset(dataset_dir):
        return dataset_dir

    target = _find_work_dir() / "dataset_fixed"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("image_features.pkl", "max_length.pkl"):
        src, dst = dataset_dir / name, target / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    # Small file: copy it so the swap is a real rename of the name train.py reads.
    (target / "captions.pkl").write_bytes(fixed.read_bytes())
    return target


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


def report_dataset() -> bool:
    """Find the training dataset and say what is in it. Returns True if usable."""
    found = find_dataset_dir()
    print("=" * 70)
    print("TRAINING DATASET")
    print("=" * 70)
    if not _looks_like_dataset(found):
        print(f"[missing] no folder with image_features.pkl + captions.pkl")
        print(f"          searched {KAGGLE_INPUT} by content, then {found}")
        print("          attach the dataset, or set MEDICA_DATASET_DIR")
        print()
        return False

    print(f"found: {found}")
    for name in ("image_features.pkl", "captions.pkl", "max_length.pkl", REPAIRED_CAPTIONS):
        path = found / name
        if path.is_file():
            print(f"  {name:<22} {path.stat().st_size / 1e6:>8.1f} MB")
    if (found / REPAIRED_CAPTIONS).is_file():
        print()
        print(f"  {REPAIRED_CAPTIONS} is present: training will use the complete")
        print("  captions via dataset_dir_with_repaired_captions().")
    print()
    return True


if __name__ == "__main__":
    paths = resolve()
    ok = check(paths)
    ok = report_dataset() and ok
    sys.exit(0 if ok else 1)
