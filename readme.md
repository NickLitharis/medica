# Image Captioning with Deep Learning

This project is a university thesis and implements an image captioning system using deep learning techniques, combining Convolutional Neural Networks (CNNs) for image feature extraction and Recurrent Neural Networks (RNNs) with attention mechanisms for generating descriptive captions.

## Results

Trained and evaluated on the **ImageCLEF 2017 Medical Image Captioning** benchmark, scored with BLEU against the published baselines the thesis compares with:

| Model | BLEU-1 | BLEU-2 |
| --- | --- | --- |
| ImageCLEF 2017 reference model | 0.142 | 0.070 |
| Retinal Image Captioning | 0.158 | 0.076 |
| Global-Local Visual Extractor | 0.148 | 0.074 |
| **OLIVER — this work (CNN + attention RNN)** | **0.309** | **0.205** |

The hybrid CNN-RNN model with integrated attention — named *OLIVER* in the thesis — scored highest on both metrics. Scores are reported in Appendix D of the thesis; `model_usage.ipynb` regenerates the evaluation.

## Model architecture

The decoder is attention-based rather than a plain RNN. Read out of the trained model file
(`model_epoch_50.h5`, held in a private Kaggle dataset and not committed here):

```
Embedding  ->  Bidirectional(LSTM)  ->  Dropout
Attention
Dense  ->  GlobalAveragePooling1D  ->  Add      (image features fused with attention output)
Dense  ->  Dropout  ->  Dense(softmax)
```

Image features reach the model **pre-extracted** (`image_features.pkl`), which is why no
convolutional layer appears in the model file: the CNN is the feature-extraction stage that
produces those features, and the model itself is the attention-augmented bidirectional LSTM
decoder over them.

## Running the pipeline

The notebooks above are the original record. The maintained pipeline is four modules:

| File | Purpose |
| --- | --- |
| `data.py` | loads the `.pkl` artefacts, rebuilds the vocabulary, splits train/val/test, builds `tf.data` batches |
| `model.py` | the attention decoder (layer stack above) |
| `train.py` | trains it, with a validation split, checkpointing and early stopping |
| `evaluate.py` | generates captions and reports BLEU-1 to BLEU-4 |

```bash
uv sync
uv run python train.py --dataset-dir /kaggle/input/dataset --epochs 30 --out /kaggle/working
uv run python evaluate.py --dataset-dir /kaggle/input/dataset --run /kaggle/working
uv run pytest
```

Five differences from the notebook pipeline, each one a real defect rather than a style
preference:

* **a validation split.** The original cut the ids 70/30 by position and trained with no
  held-out set at all, so there was nothing to choose a checkpoint or an early-stopping
  epoch against.
* **examples built once.** The original re-tokenised and re-padded every caption on every
  epoch.
* **integer labels with sparse cross-entropy.** The original materialised a one-hot vector
  per example.
* **the vocabulary is rebuilt, not unpickled.** The dataset's `tokenizer.pkl` holds a Keras 2
  `Tokenizer`, a class Keras 3 no longer ships, so relying on it would pin the pipeline to
  the version it is being moved off.
* **BLEU-1 to BLEU-4 reported separately.** The original printed one BLEU figure, which is
  not comparable with the n-gram breakdown in the thesis.
* **one pass per caption instead of one per position.** `--arch seq2seq` (the default) trains
  on whole captions with teacher forcing; `--arch legacy` keeps the per-position path that
  matches the trained artifact. See Performance below.

Requires `tensorflow>=2.16` (Keras 3). The notebooks pin 2.15, which is no longer installable:
Kaggle's oldest available version is 2.20, and no installable Windows wheel exists for it.

## Performance

Measured on Kaggle under TensorFlow 2.20, on CPU.

Training used to make one example per caption *position*: a caption of n words cost n-1
passes over n-1 growing prefixes, which re-encodes the same words over and over. It now makes
one example per caption and supervises every position in a single pass.

| on a 300-image split | per-position | per-caption |
| --- | --- | --- |
| training examples | 3080 | 210 |
| decoder steps | 26,064 | 3,080 |
| redundant work | 8.46x the minimum | - |
| LSTM | bidirectional | unidirectional |

| one epoch, 200 images, batch 32 | seconds |
| --- | --- |
| per-position (`--arch legacy`) | 17.2s |
| per-caption (`--arch seq2seq`, default) | **3.3s** |

The 8.46x is measured, and it grows with caption length: the longer the captions, the more a
pass over a growing prefix wastes. Separately, the training pairs used to repeat the full
4096-float image vector once per example - about twenty copies of the whole feature set, and
the reason a full-data run was heavy:

| on a 300-image split | before | after |
| --- | --- | --- |
| pair storage, features duplicated per example | 50.5 MB | 0.3 MB (+3.4 MB of features, once) |

Captions are batched by length and padded to a rounded width, so short captions are not padded
out to the longest caption in the dataset and the model does not rebuild its graph per batch.

### On a GPU

Kaggle gives two Tesla T4s. Steady-state epoch times, 1050 training images, first epoch
(dropped: it builds the graph) excluded:

| configuration | seconds/epoch | ms/image |
| --- | --- | --- |
| batch 64, mixed_float16 | 0.24 | 0.23 |
| batch 128, mixed_float16 | 0.17 | 0.16 |
| batch 256, mixed_float16 | **0.14** | **0.13** |
| batch 128 + XLA (`--jit`) | 0.16 | 0.16 |
| batch 128 + `--steps-per-execution 8` | 0.16 | 0.16 |
| batch 128 + XLA + steps 8 | 0.16 | 0.15 |
| batch 128, float32 (mixed off) | 0.16 | 0.15 |
| batch 128, 15 epochs (drift check) | 0.16 | 0.15 |
| `MirroredStrategy`, both T4s | fails (`InvalidArgumentError`) | - |

What this says, and it is not what the flag names suggest:

* **batch size is the only lever that moved the number** - 64 to 256 is about 1.7x, because
  fewer, larger steps mean fewer kernel launches;
* **XLA, `steps_per_execution` and mixed precision made no measurable difference.** The model
  is small and its LSTM is already cuDNN-backed, so a step costs more to launch than to
  compute. Mixed precision is harmless, not helpful;
* **two GPUs do not work here** and would not help a model this size anyway.

Extrapolating 0.13 ms/image over the full 49,287-image dataset gives roughly 6 s/epoch, so a
30-epoch run is minutes rather than hours. That extrapolation is not yet measured at full
scale.

## Notebooks

- `model_training_preprocessing.ipynb`: Data loading, preprocessing, tokenization, train/test split, data generator, model building, training, and saving the model.
- `model_usage.ipynb`: Model and tokenizer loading, caption generation, evaluation, and visualization.

## Project Structure

- `model_training_preprocessing.ipynb`: Notebook for model training and data preprocessing.
- `model_usage.ipynb`: Notebook for model inference, evaluation, and visualization.
- `pyproject.toml` / `uv.lock`: Declared dependencies and the locked resolution.
- `LICENSE` / `LICENSE-THESIS.txt` / `DATA_LICENSE.txt`: Licensing for the code, the thesis document, and the dataset respectively.

The preprocessed dataset is **not committed** (see `DATA_LICENSE.txt`). To reproduce the pipeline, supply your own `dataset/` folder containing:
	- `image_features.pkl`: Extracted image features (typically from a CNN).
	- `captions.pkl`: Dictionary mapping image IDs to their captions.
	- `max_length.pkl`: Maximum caption length for padding.
	- `tokenizer.pkl`: Fitted tokenizer for text preprocessing.
	- `vocab_size.pkl`: Vocabulary size for the model.

## Workflow Overview

1. **Data Loading**: Loads image features and captions from preprocessed pickle files.
2. **Text Preprocessing**: Adds start/end tokens to captions, tokenizes text, and calculates vocabulary size.
3. **Train/Test Split**: Splits image IDs into training and test sets.
4. **Data Generator**: Prepares batches for model training, pairing image features with partial caption sequences.
5. **Model Architecture**:
	 - Encoder: Processes image features.
	 - Decoder: Processes caption sequences using Embedding and Bidirectional LSTM layers.
	 - Attention Mechanism: (optional) Improves alignment between image regions and words.
	 - Output: Predicts next word in the caption sequence.
6. **Training**: Trains the model using the generator, with callbacks for early stopping, learning rate reduction, checkpointing, and TensorBoard logging.
7. **Caption Generation**: Generates captions for new images using the trained model.
8. **Evaluation**: Computes BLEU, METEOR, ROUGE, and CIDEr scores to assess caption quality.
9. **Visualization**: Displays images with actual and predicted captions.

## Usage

1. Install dependencies:
	```
	uv sync
	```
2. Run `model_training_preprocessing.ipynb` to train and save the model.
3. Run `model_usage.ipynb` to generate captions, evaluate, and visualize results.
4. Ensure all required data files are present in the `dataset/` folder. Adjust paths as needed for your environment.

## Requirements

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

TensorFlow is pinned to 2.15, the last release on the Keras 2 API that these notebooks
are written against — 2.16 ships Keras 3 and the model code does not run on it. That
pin also caps Python at 3.11, which `requires-python` states explicitly; uv fetches a
suitable interpreter for you, so `uv sync` works on a machine whose default Python is
newer. On Windows the resolver also swaps in a `tensorflow-io-gcs-filesystem` build
that has a `win_amd64` wheel, which is what makes the install work here as well as on
Linux (Kaggle / Colab).

## Notes

- The notebook is designed for flexibility: you can experiment with different model architectures, optimizers, and training strategies.
- Paths in the notebook may reference Kaggle directories; update them for local use as needed.
- Evaluation and visualization code is provided but commented out for modularity.

## References

- [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)


---
For questions or improvements, feel free to open an issue or contribute.

## License

This repository uses different licenses for code, the thesis document, and the dataset:

- Code (notebooks and scripts): MIT License — see `LICENSE` for full text.
- Thesis (PDF): Creative Commons Attribution-NonCommercial 4.0 — see `LICENSE-THESIS.txt`.
- Dataset: Provided for research and educational use only — see `DATA_LICENSE.txt` for details and restrictions.

If you require a different license for any of these artifacts, please contact the author.
