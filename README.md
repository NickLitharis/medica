# Medica

Medical image captioning for the thesis. The full project description, results,
architecture and data provenance are in [`readme.md`](readme.md); that file is
the authoritative document.

This `README.md` covers one thing: **how to run the repository on Kaggle**.

## What is on Kaggle

Two inputs, both attached as Kaggle Datasets:

| Dataset | Contents |
|---|---|
| Preprocessed dataset | the `dataset/` folder the pipeline expects: `image_features.pkl`, `captions.pkl`, `max_length.pkl` |
| Figure images (optional) | the PNG figures, if you want to rebuild the dataset from source |

There is no data in git. `readme.md` explains what each file contains.

## Path resolution

`config/paths.py` gives every script one place to find those inputs, so nothing
hardcodes a machine-specific path. Three environment variables control it:

| Variable | Default on Kaggle |
|---|---|
| `MEDICA_INPUT_DIR` | first Kaggle dataset folder that holds image files |
| `MEDICA_CAPTIONS` | first `s2_cleaned.json` under `/kaggle/input` |
| `MEDICA_WORK_DIR` | `/kaggle/working/medica` |

```bash
python3 -m config.paths      # show what was resolved, and what is missing
```

On a normal machine the defaults point at `./dataset/images`,
`./data/processed/s2_cleaned.json` and `./outputs`.

## Notebooks

`kaggle/` holds the notebooks to import into Kaggle. They are generated from
`kaggle/build_notebooks.py`, which is committed beside them so the code is
reviewable as plain text:

```bash
python3 kaggle/build_notebooks.py
```

| Notebook | Purpose |
|---|---|
| `kaggle/01_pipeline.ipynb` | Builds the image index, verifies caption-to-image matches, preprocesses images, extracts features |
| `kaggle/02_train.ipynb` | Training run: calls the repository's own `train.py`, then `evaluate.py` |

Each notebook clones the repository into `/kaggle/working/medica` and changes
into it, so `train.py`, `data.py` and `evaluate.py` import as they do locally.
If a notebook runs without internet access, upload the repository as a dataset
and point `REPO` in the setup cell at that folder.

## Disk budget

A Kaggle notebook has about 20 GB of writable space. Preprocessing every figure
at 512×512 PNG does not fit, so `01_pipeline.ipynb` preprocesses a subset by
default. Use `--limit` for a smaller run, or a smaller size for the full set.

## Relationship to the rest of the repository

The scripts under `scripts/` are the dataset pipeline: cleaning, analysis,
indexing, verification, preprocessing and feature extraction. They all resolve
paths through `config/paths.py` and accept command-line overrides.

The model, training loop, evaluator and tests live at the repository root
(`model.py`, `train.py`, `evaluate.py`, `tests/`) and are documented in
`readme.md`. Nothing here changes how they behave; it only supplies the paths
they need on Kaggle.
