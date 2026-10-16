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

The two OLIVER figures are in-sample, on 100 training images — see
[Where the 0.309 comes from](#where-the-0309-comes-from) below.

The hybrid CNN-RNN model with integrated attention — named *OLIVER* in the thesis — scored highest on both metrics. Scores are reported in Appendix D of the thesis; `model_usage.ipynb` regenerates the evaluation.

### Modernised pipeline, and a like-for-like comparison

The thesis artifact (`model_epoch_50.h5`) and the modernised model, scored on the identical
seeded test split (9,857 images) with identical BLEU code:

| metric | thesis artifact | modernised |
| --- | --- | --- |
| BLEU-1 | 0.1627 | 0.1623 |
| BLEU-2 | 0.0815 | 0.0838 |
| BLEU-3 | 0.0377 | 0.0413 |
| BLEU-4 | 0.0178 | 0.0210 |

**The modernised model matches the artifact on BLEU-1 and edges ahead on BLEU-2 to BLEU-4**,
while training in minutes instead of the artifact's original cost. It also trains on a
9,026-word vocabulary against the artifact's 23,615, and the comparison shows that pruning
did not cost quality.

The artifact scored BLEU-1 0.1994 and corpus BLEU 0.0319 when evaluated the way
`model_usage.ipynb` does — over every image, including the ones it trained on — so that
figure is in-sample and inflated by construction.

**Nothing reproduces the 0.309 in the table above.** Every checkpoint saved to Kaggle was
scored on this same split with this same BLEU implementation:

| checkpoint | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | corpus BLEU |
| --- | --- | --- | --- | --- | --- |
| `cnn-rnn-att100/model_epoch_100.h5` | **0.1681** | 0.0890 | 0.0463 | 0.0265 | 0.0265 |
| `cnn-rnn-att/model_epoch_50.h5` | 0.1627 | 0.0815 | 0.0377 | 0.0178 | 0.0178 |
| `med-model/model_50.h5` | 0.1617 | 0.0756 | 0.0348 | 0.0179 | 0.0179 |
| `trained-model/model.h5` | 0.1432 | 0.0611 | 0.0294 | 0.0180 | 0.0180 |
| `model80/model_epoch_40(1).h5` | 0.1271 | 0.0625 | 0.0273 | 0.0120 | 0.0120 |
| `newmodel/model_final.h5` | 0.1104 | 0.0461 | 0.0171 | 0.0064 | 0.0064 |
| `model70/model_epoch_40.h5` | 0.1004 | 0.0450 | 0.0170 | 0.0061 | 0.0061 |

The best of the seven scores 0.1681 and the modernised model's 0.1623 sits inside this range.

Two notes on that table. `model_final.h5` is the **weakest** of the seven despite its name:
`thesis.ipynb` cell 18 loads an existing model and saves it again under that name, so the
filename says nothing about which epoch it holds. And two further artifacts (`newh5model`,
`h5model`) do not load under Keras 3 at all — their saved configs feed an LSTM initial state
positionally, which Keras 3 rejects — so they are untested here.

### Where the 0.309 comes from

`thesis.ipynb` cell 10 splits positionally —
`train = image_ids[:int(len(image_ids) * 0.7)]` — and cell 28 then evaluates
`list(mapping.keys())[:100]`: **the first 100 images, every one of them a training image.**
Reproducing that cell exactly, against 100 held-out images from the same checkpoints:

| checkpoint | set | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 |
| --- | --- | --- | --- | --- | --- |
| `cnn-rnn-att/model_epoch_50.h5` | first 100 (training) | 0.4556 | 0.3153 | 0.2468 | 0.1464 |
| `cnn-rnn-att/model_epoch_50.h5` | 100 held-out | 0.1584 | 0.0846 | 0.0587 | 0.0168 |
| `cnn-rnn-att100/model_epoch_100.h5` | first 100 (training) | 0.6300 | 0.5155 | 0.4538 | 0.3475 |
| `cnn-rnn-att100/model_epoch_100.h5` | 100 held-out | 0.1692 | 0.0864 | 0.0515 | 0.0185 |
| `newmodel/model_final.h5` | first 100 (training) | 0.3568 | 0.1938 | 0.1233 | 0.0470 |
| `newmodel/model_final.h5` | 100 held-out | 0.1056 | 0.0399 | 0.0204 | 0.0000 |

The reported 0.309 / 0.205 sits inside the training-100 column and far above every held-out
number, so it is a 100-image, in-sample measurement — the same code, the same weights, but
evaluated on data the model had trained on. The ImageCLEF rows at the top of this section are
published held-out scores, so that comparison is not like for like.

Scoring the modernised model against the raw captions (rather than the `<start>`-wrapped
ones) moved its BLEU-1 from 0.1330 to 0.1623; the earlier figure was depressed by a bug in
this repository's reference construction, not by the model.

Two Keras 2 constructs in the artifact no longer run under Keras 3 and had to be worked
around: its saved LSTM config carries a `time_major` key Keras 3 rejects, and the notebook's
`pad_sequences` left-pads, which trips Keras 3's mask assertion on the LSTM.

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

## Does the image matter?

An ablation on 3,000 held-out images: the same checkpoints scored with the real features, with
another image's features (a permutation, so the distribution is unchanged), and with the image
zeroed.

| model | real | permuted | no image |
| --- | --- | --- | --- |
| modernised, BLEU-1 | 0.1665 | 0.1154 | 0.1419 |
| modernised, BLEU-4 | 0.0234 | 0.0076 | 0.0121 |
| `model_epoch_50.h5`, BLEU-1 | 0.1598 | 0.1188 | 0.0856 |

**The image is used.** Handing a model a different image costs it 0.041–0.051 BLEU-1, well
outside seed noise, so it is reading the features rather than reciting captions.

**But the language prior carries most of the score.** The modernised model still reaches 0.1419
with no image at all — about 85% of its 0.1665 — so only a small part of the result comes from
looking at the picture. The zeroed-image row is a weaker control than the permuted one (a zero
vector is out of distribution), which is why the permuted column is the one to read.

That combination is the diagnosis for this dataset: a single pooled feature vector per image
and roughly one caption per image. The decoder has learned the shape of a radiology caption
well, and has little image detail available to be more specific with. It is the argument for
spatial features over better decoding.

## Training-side experiments

Four configs, identical settings otherwise: the full training split, 30 epochs, batch 256,
greedy decoding, scored on the same 1,500 held-out images.

| config | BLEU-1 | BLEU-2 | BLEU-4 | mean words | vocabulary |
| --- | --- | --- | --- | --- | --- |
| baseline (embed 256, dropout 0.5, `min_count` 3) | 0.1618 | 0.0857 | 0.0229 | 9.3 | 9,025 |
| wider (embed 512, dropout 0.3) | 0.1163 | 0.0641 | 0.0184 | 6.9 | 9,025 |
| label smoothing 0.1 | **0.1741** | **0.0898** | 0.0216 | 10.1 | 9,025 |
| `min_count` 1 | **0.1785** | **0.0916** | 0.0215 | 10.8 | 23,618 |

**Label smoothing and a larger vocabulary both help** — together roughly +0.02 BLEU-1 over the
baseline. `min_count 1` gives a 23,618-word vocabulary, which is within four words of the
23,615 the original Keras tokenizer carried: the original was effectively `min_count 1`, and
pruning to 9,025 was costing accuracy by turning rare clinical terms into `<unk>`.

**More capacity does not help.** Embed 512 with dropout 0.3 scored 0.1163 and early-stopped
after about 13 epochs. The model is not capacity-limited, which agrees with the image
ablation: the limit is the supervision and the image representation, not the decoder.

The baseline was re-run here at 0.1618, against 0.1652 for the same configuration in the
decoding experiment. That 0.003 spread between two runs of one configuration is the noise
floor, so only larger gaps are worth believing.

## Decoding: greedy beats beam search here

Beam search is the usual free win in captioning. On this model and this data it is a large
loss, so it is measured and written down rather than recommended. 1,500 held-out images, the
same weights:

| decoding | BLEU-1 | BLEU-2 | BLEU-4 | mean words |
| --- | --- | --- | --- | --- |
| greedy | **0.1652** | 0.0873 | 0.0235 | 9.6 |
| beam 3, no length penalty | 0.0875 | 0.0543 | 0.0177 | 5.5 |
| beam 3, length penalty 0.6 | 0.0912 | 0.0570 | 0.0188 | 5.6 |
| beam 3, length penalty 1.0 | 0.1033 | 0.0640 | 0.0210 | 6.1 |
| beam 5, no length penalty | 0.0623 | 0.0391 | 0.0119 | 4.7 |

The captions say why. Greedy produces `chest xray showing bilateral pleural effusion`; beam
search produces `chest xray`, and hands the same three or four generic templates to very
different images. Maximising a sequence's joint probability leads to the safest run of words,
and with one reference caption per image the safest run is generic. Greedy, which commits to
each position's most likely token and then continues from it, does not collapse that way.

`evaluate.py --beam-width` therefore defaults to 1. Raising the length penalty recovers part
of the loss and never reaches greedy, so the remaining headroom is in the supervision and the
image representation, not in the search.

## Data provenance

The images and captions come from **MedICaT** (`allenai/medicat`, Allen AI): figures and
captions extracted from PubMed Central open-access articles, with reference text from S2ORC.
Keys are `<pdf_hash>_<n>-Figure<k>-<m>`; MedICaT's own `image_id` also carries the figure key
and the extension, as `<pdf_hash>_Figure<k>_<n>-Figure<k>-<m>.png`. Normalising for that
matters: comparing the two key formats literally reports zero overlap when the real figure is
99.6%.

* 49,112 of the 49,287 images here are present in MedICaT.
* The captions in `captions.pkl` are MedICaT's `s2_caption`, cleaned and capped at 20 words —
  median 13 against 17 in the source, and 98.7% of the source captions are longer. The cap is
  a **filter** (`dataset_analysis.ipynb` drops rows with more than 20 words), and the damaged
  text is done by the cleaning steps below: all non-alphanumerics become spaces, digits are
  removed, single letters are removed, and `mm`/`cm`/`dm` are removed. Against
  whitespace-damaged PDF-extracted captions that turns `5-cm anastomotic stricture` into
  `anastomotic stricture` and `borders` into `border`, so the references BLEU is scored
  against have had their measurements deleted.
* The images are not in this repository. The `images/` files in the Kaggle copy are Git LFS
  pointers, and no public URL resolves the keys. MedICaT publishes the real files at
  https://ai2-s2-medicat.s3.us-west-2.amazonaws.com/2020-10-05/medicat_release.tar.gz
  (104 GB, non-commercial research use).
* A 217,060-row caption index for the full MedICaT release is kept in the
  `med-image-captioning` Kaggle dataset (`data/csv/dataset.csv`), which is enough to restore
  the untruncated captions for the images used here without downloading any images.

### Restoring the untruncated captions

Because 99.6% of the images are in MedICaT, the complete captions can be restored from the
217,060-row index already held in the `med-image-captioning` dataset — no images, no download.
Two models trained with identical settings except for their captions, each scored against both
reference sets:

| trained on | scored against | BLEU-1 | BLEU-2 | BLEU-4 |
| --- | --- | --- | --- | --- |
| truncated | truncated | **0.1675** | 0.0875 | 0.0225 |
| truncated | full | 0.1103 | 0.0568 | 0.0115 |
| full | truncated | 0.1051 | 0.0507 | 0.0095 |
| full | full | **0.1329** | 0.0690 | 0.0163 |

Read *down* a column: each column scores against one fixed reference set, so only that
comparison is valid, never across columns.

* **Against full references the full-caption model wins, 0.1329 to 0.1103.** That is +0.023,
  several times the 0.003 noise floor: training on complete captions produces a better model.
* Against the truncated references the same model scores 0.1051 to 0.1675. It is not worse —
  it writes complete captions while those references stop at 20 words, and BLEU charges it for
  every extra word.

So repairing the captions improves the model while lowering the headline number: 0.1675 was a
score against truncated references. The truncated column is the inflated one, and it inflates
every figure recorded in this readme, including the 0.309. Comparisons against published
baselines, which are scored on complete captions, were never like for like.

The corrected run lengthens captions from a median of 13 words to 15 (max 51) and raises the
sequence width from 22 to 25, which costs about 50% more per epoch (~21s to ~32s).

## How the captions were preprocessed

Taken from `dataset_cleaner.ipynb` and `dataset_analysis.ipynb` in the `med-image-captioning`
dataset. Applied in order, to lowercased captions:

1. Replace every character outside `[a-z0-9]` with a space — punctuation, hyphens and slashes
   all become word breaks.
2. Strip a leading `figure` / `fig`.
3. **Delete every digit**, so `18F-FDG`, `5-cm` and `95%` lose their numbers.
4. Rewrite `x ray` → `xray`, `ct scan` and bare `ct` → `ctscan`, `mri` → `mriscan`.
5. **Delete single letters** and the units `mm`, `cm`, `dm`.
6. Strip a leading or trailing `and` / `the`, then collapse whitespace.
7. **Keep only captions of 20 words or fewer** (`df[df['sentence_length'] <= 20]`), after an
   earlier pass that dropped anything over 100.

Steps 1, 3 and 5 meet PDF-extracted captions whose word spacing is already broken, which is why
fragments like `...the left lobe of the li` and `...obstruction caused by` appear: the text was
not cut, it was cleaned. The result is a training set whose targets contain no measurements —
the numbers and units were removed — and the model faithfully writes captions without them.

Note the padding disagreement: this notebook pads `padding='post'` while the training generator
pads `padding='pre'`. It is harmless only because the padded array was never saved (the
`caption_features.pkl` write is commented out).

### The images

The image side is **not documented in any surviving notebook**. `thesis.ipynb` loads
`image_features.pkl` as a finished artefact; the extraction step is not present. The width is
4096, the signature of VGG16's `fc2` layer — the standard recipe in the Show-Attend-Tell
image-captioning tutorial — while the commented-out attempts in `medicat.ipynb` use ResNet-50
and the separate chest X-ray notebook uses DenseNet121. The step that actually produced the
file has not been found.

## How the original thesis trained

Read from the three notebooks (`thesis.ipynb` on Kaggle, plus `model_training_preprocessing.ipynb`
and `model_usage.ipynb` here). The steps, in the order they run:

1. `pip install tensorflow==2.15.0`.
2. Load `image_features.pkl`, `captions.pkl`, `max_length.pkl` from `/kaggle/input/dataset/`.
3. Wrap every caption as `<start> ... <end>`, mutating `mapping` in place.
4. Fit a `Tokenizer()` over all captions (23,614 words); save the tokenizer and `vocab_size`.
5. Split positionally and unshuffled: `train = image_ids[:int(n * 0.7)]`, `test` the rest.
   The repo notebook uses 80/20 instead, so the two artefacts disagree.
6. A generator that re-tokenises every caption on every epoch, one example per position,
   `pad_sequences(padding='pre')`, `to_categorical` labels.
7. Build the model: four variants, three commented out. The one marked *"This is the correct
   one"* is the attention + `GlobalAveragePooling1D` model.
8. Train: **continue from `model80/model_epoch_40(1).h5`** for 20 epochs, in a manual Python
   loop calling `fit(..., epochs=1)`, batch 128, checkpoint every 10 epochs.
9. Generate greedily from a `<start>` prompt, pre-padded prefixes, stop at `<end>`, strip the
   markers from the output.
10. Evaluate with `corpus_bleu` over `list(mapping.keys())[:100]`.
11. Visualise one image from `/kaggle/input/testing-images/`.

Four things in that, read from the code rather than the write-up:

* **There is no validation set anywhere.** `EarlyStopping` and `ReduceLROnPlateau` are passed
  monitoring `val_loss`, but nothing ever supplies validation data, so `val_loss` does not
  exist and both callbacks are inert. Nothing stopped the run early and nothing reduced the
  learning rate; checkpoints were saved by epoch number, not by best score.
* **Evaluation ran over the first 100 images, which are training images** — the source of the
  0.309, as measured in [Where the 0.309 comes from](#where-the-0309-comes-from).
* **The references carried `<start>` and `<end>` while the predictions did not** (step 3 mutates
  `mapping` in place; step 9 strips the markers from the generated caption). That deflates
  every BLEU score the evaluation produced — the same defect that was fixed in the modernised
  `evaluate.py`, where it moved the score from 0.1330 to 0.1623.
* **The model in `model_epoch_50.h5` is commented out in the shipped notebook.** The attention
  cell is not executed in `thesis.ipynb`, and the repo's training notebook builds a model
  *without* attention. `model.py` is the only definition in this repository that matches the
  saved artifact, which is why it was reconstructed from the artifact's config.

The checkpoints that exist are stages of one continued run (epoch 40, plus 20 more, saved every
10), not independent experiments, so comparing them compares training stages rather than
methods.

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
