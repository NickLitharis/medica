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
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    """Every (image feature, caption prefix, next token) triple for the given images.

    One example per position, which is how the original generator worked; doing it once
    up front is what removes the per-epoch re-tokenisation.
    """
    features, prefixes, targets = [], [], []
    for image_id in image_ids:
        for caption in mapping[image_id]:
            sequence = encode(caption, word_to_index)
            for position in range(1, len(sequence)):
                prefix = sequence[:position][-max_length:]
                # left-pad, as the notebooks did: the prefix sits at the end of the row
                padded = [PAD_INDEX] * (max_length - len(prefix)) + prefix
                features.append(image_id)
                prefixes.append(padded)
                targets.append(sequence[position])
    return features, prefixes, targets


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

    Labels stay as integers: the model uses sparse categorical cross-entropy, where the
    original built a one-hot vector per example with ``to_categorical``.
    """
    import numpy as np
    import tensorflow as tf

    ids, prefixes, targets = training_pairs(list(mapping.keys()), mapping, word_to_index, max_length)
    feature_matrix = np.asarray([features[image_id][0] for image_id in ids], dtype="float32")
    prefix_matrix = np.asarray(prefixes, dtype="int32")
    target_vector = np.asarray(targets, dtype="int32")

    dataset = tf.data.Dataset.from_tensor_slices(
        ((feature_matrix, prefix_matrix), target_vector)
    )
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(len(target_vector), 10_000), seed=seed, reshuffle_each_iteration=True
        )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
