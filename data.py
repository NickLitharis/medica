"""Data plumbing for the captioning pipeline.

Replaces the notebook's inline preprocessing. Three deliberate differences from the
original:

* the vocabulary is rebuilt from the captions rather than unpickled. The dataset ships a
  tokenizer.pkl saved by a Keras 2 `Tokenizer`, a class Keras 3 no longer provides, so
  depending on it would tie the pipeline to the old stack it is being moved off.
* a real train/validation/test split, seeded and shuffled. The original split the image
  ids 70/30 by position with no validation set at all, which leaves nothing to select a
  checkpoint or an early-stopping epoch against.
* padding is applied to a fixed width here, once, instead of inside the batch generator.
  The original re-tokenised and re-padded every caption on every epoch.

TensorFlow is imported lazily, inside :func:`make_dataset`, so everything else in this
module can be tested without it.
"""
from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Iterable, Mapping, Sequence

START = "<start>"
END = "<end>"
UNK = "<unk>"
PAD_INDEX = 0  # Keras convention: index 0 is the padding slot, used with mask_zero=True

ARTIFACT_FILES = {
    "features": "image_features.pkl",
    "captions": "captions.pkl",
    "max_length": "max_length.pkl",
}


def load_artifacts(dataset_dir: str | Path) -> tuple[dict, dict, int]:
    """Load ``image_features.pkl``, ``captions.pkl`` and ``max_length.pkl``.

    ``features`` maps image id -> array of feature vectors; the pipeline uses the first,
    which is what the original notebooks did (``features[key][0]``).
    """
    root = Path(dataset_dir)
    loaded = {}
    for key, filename in ARTIFACT_FILES.items():
        path = root / filename
        if not path.exists():
            raise FileNotFoundError(f"missing dataset artefact: {path}")
        with path.open("rb") as handle:
            loaded[key] = pickle.load(handle)
    return loaded["features"], loaded["captions"], int(loaded["max_length"])


def wrap_captions(mapping: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    """Put the start and end markers around every caption, in place of the raw text."""
    return {
        image_id: [f"{START} {caption.strip()} {END}" for caption in captions]
        for image_id, captions in mapping.items()
    }


def build_vocabulary(
    captions: Iterable[str], min_count: int = 1
) -> tuple[dict[str, int], dict[int, str]]:
    """Build word <-> index maps from the captions.

    Index 0 is reserved for padding. Rare words fall back to ``<unk>`` at encode time
    rather than being dropped, so a caption never silently loses a position.
    """
    counts: dict[str, int] = {}
    for caption in captions:
        for word in caption.split():
            counts[word] = counts.get(word, 0) + 1

    words = sorted(word for word, count in counts.items() if count >= min_count)
    if START not in words:
        words.append(START)
    if END not in words:
        words.append(END)
    if UNK not in words:
        words.append(UNK)

    word_to_index = {word: index for index, word in enumerate(words, start=1)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    index_to_word[PAD_INDEX] = ""
    return word_to_index, index_to_word


def split_ids(
    image_ids: Sequence[str],
    val_size: float = 0.1,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Shuffle the ids once, then cut them into train/validation/test.

    Returns a dict with the three keys. The sets are disjoint and cover every id.
    """
    if not 0.0 <= val_size < 1.0 or not 0.0 <= test_size < 1.0:
        raise ValueError("val_size and test_size must be in [0, 1)")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must leave something for training")

    ids = list(image_ids)
    if not ids:
        raise ValueError("cannot split an empty id list")
    random.Random(seed).shuffle(ids)

    n_test = int(len(ids) * test_size)
    n_val = int(len(ids) * val_size)
    return {
        "test": ids[:n_test],
        "val": ids[n_test : n_test + n_val],
        "train": ids[n_test + n_val :],
    }


def encode(caption: str, word_to_index: Mapping[str, int]) -> list[int]:
    """Caption -> token indices, unknown words mapped to ``<unk>``."""
    return [word_to_index.get(word, word_to_index[UNK]) for word in caption.split()]


def training_pairs(
    image_ids: Iterable[str],
    mapping: Mapping[str, Sequence[str]],
    word_to_index: Mapping[str, int],
    max_length: int,
) -> tuple[list[int], list[list[int]], list[int]]:
    """Every (image row, caption prefix, next token) triple for the given images.

    The image is returned as a *row index*, not a vector. A caption of n words yields n-1
    examples, all sharing one 4096-float feature, so repeating the vector per example would
    mean ~20 copies of the whole feature set. The feature is gathered once per batch
    instead, in :func:`make_dataset`.
    """
    ids = list(image_ids)
    row_of = {image_id: row for row, image_id in enumerate(ids)}
    rows: list[int] = []
    prefixes: list[list[int]] = []
    targets: list[int] = []

    for image_id in ids:
        for caption in mapping[image_id]:
            sequence = encode(caption, word_to_index)
            for position in range(1, len(sequence)):
                prefix = sequence[:position][-max_length:]
                # left-pad, as the notebooks did: the prefix sits at the end of the row
                rows.append(row_of[image_id])
                prefixes.append([PAD_INDEX] * (max_length - len(prefix)) + prefix)
                targets.append(sequence[position])
    return rows, prefixes, targets


def feature_matrix(features: Mapping[str, Sequence], image_ids: Iterable[str]):
    """One feature row per image - not one per training example."""
    import numpy as np

    return np.asarray([features[image_id][0] for image_id in image_ids], dtype="float32")


def decode(indices: Iterable[int], index_to_word: Mapping[int, str]) -> str:
    """Token indices -> text, stopping at the end marker and dropping padding."""
    words = []
    for index in indices:
        word = index_to_word.get(int(index), UNK)
        if word == END:
            break
        if word and word != START:
            words.append(word)
    return " ".join(words).strip()


def caption_examples(
    image_ids: Iterable[str],
    mapping: Mapping[str, Sequence[str]],
    word_to_index: Mapping[str, int],
    max_length: int,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """One example per caption: the caption in, the same caption shifted by one out.

    The alternative, :func:`training_pairs`, makes one example per *position*, so teaching a
    five-word caption costs four passes over four growing prefixes and four weight updates.
    This is the same supervision in one pass. Positions that are padding are masked out of
    the loss by :func:`make_caption_dataset`.
    """
    ids = list(image_ids)
    rows: list[int] = []
    inputs: list[list[int]] = []
    targets: list[list[int]] = []

    for row, image_id in enumerate(ids):
        for caption in mapping[image_id]:
            sequence = encode(caption, word_to_index)
            if len(sequence) < 2:
                continue
            rows.append(row)
            inputs.append(sequence[:-1][:max_length])
            targets.append(sequence[1:][:max_length])
    return rows, inputs, targets


def _padded_batches(rows, inputs, targets, batch_size: int, width_step: int = 8):
    """Group captions into fixed-size batches, padding each to a rounded-up width.

    Padding every batch to a different width makes the model rebuild its graph for each new
    shape, which costs far more than the padding saves. Rounding the width up to a multiple
    of ``width_step`` keeps the number of distinct shapes small while still avoiding padding
    short captions out to the longest caption in the dataset.
    """
    import numpy as np

    order = sorted(range(len(rows)), key=lambda index: len(inputs[index]))
    batches = []
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        width = max(len(inputs[index]) for index in chosen)
        width = ((width + width_step - 1) // width_step) * width_step

        batch_rows = np.asarray([rows[index] for index in chosen], dtype="int32")
        batch_inputs = np.full((len(chosen), width), PAD_INDEX, dtype="int32")
        batch_targets = np.full((len(chosen), width), PAD_INDEX, dtype="int32")
        for position, index in enumerate(chosen):
            length = len(inputs[index])
            batch_inputs[position, :length] = inputs[index]
            batch_targets[position, :length] = targets[index]
        batches.append((batch_rows, batch_inputs, batch_targets))
    return batches


def make_caption_dataset(
    features: dict,
    mapping: Mapping[str, Sequence[str]],
    word_to_index: Mapping[str, int],
    max_length: int,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
):
    """A ``tf.data.Dataset`` of (image feature, caption) -> (next token per position, weight).

    One example per caption, batched by length so a batch is padded to its own longest
    caption. The weight is 1.0 on real positions and 0.0 on padding, which is how the loss
    ignores padding without the model needing to know where a caption ends.
    """
    import tensorflow as tf

    ids = list(mapping.keys())
    rows, inputs, targets = caption_examples(ids, mapping, word_to_index, max_length)
    if not rows:
        raise ValueError("no usable captions: every caption was shorter than two tokens")
    feature_rows = tf.constant(feature_matrix(features, ids))
    batches = _padded_batches(rows, inputs, targets, batch_size)

    dataset = tf.data.Dataset.from_generator(
        lambda: iter(batches),
        output_signature=(
            tf.TensorSpec((None,), tf.int32),
            tf.TensorSpec((None, None), tf.int32),
            tf.TensorSpec((None, None), tf.int32),
        ),
    )
    if shuffle:
        # whole batches, so the length grouping above survives
        dataset = dataset.shuffle(buffer_size=max(len(batches), 2), seed=seed,
                                  reshuffle_each_iteration=True)

    return dataset.map(
        lambda row, tokens, target: (
            (tf.gather(feature_rows, row), tokens),
            target,
            tf.cast(tf.not_equal(target, PAD_INDEX), "float32"),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


def make_dataset(
    features: dict,
    mapping: Mapping[str, Sequence[str]],
    word_to_index: Mapping[str, int],
    max_length: int,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
):
    """Build a ``tf.data.Dataset`` of (image feature, prefix) -> next token.

    Two things keep this cheap:

    * the pairs carry an image *row index*, and the feature is gathered once per batch, so
      the feature set is not copied per example;
    * only the small index/prefix/target arrays are cached, and the gather happens on the
      batched tensor, off the critical path.

    Labels stay integers: the model uses sparse categorical cross-entropy, where the
    original built a one-hot vector per example with ``to_categorical``.
    """
    import numpy as np
    import tensorflow as tf

    ids = list(mapping.keys())
    rows, prefixes, targets = training_pairs(ids, mapping, word_to_index, max_length)
    feature_rows = tf.constant(feature_matrix(features, ids))

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            (np.asarray(rows, dtype="int32"), np.asarray(prefixes, dtype="int32")),
            np.asarray(targets, dtype="int32"),
        )
    )
    # the index/prefix/target arrays are small; the feature gather is left out of the cache
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(targets), 100_000), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda inputs, label: (
            (tf.gather(feature_rows, inputs[0]), inputs[1]),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
