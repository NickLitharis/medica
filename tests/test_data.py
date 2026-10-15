"""Tests for the data plumbing.

No TensorFlow: `data.py` imports it lazily, so everything except `make_dataset` is
testable anywhere.

Run with::

    uv run pytest tests -v
"""
import pytest

from data import (
    END,
    PAD_INDEX,
    START,
    UNK,
    build_vocabulary,
    caption_examples,
    decode,
    encode,
    feature_matrix,
    split_ids,
    training_pairs,
    usable_mapping,
    wrap_captions,
)

RAW = {
    "a": ["a small image", "a small cat"],
    "b": ["a dog"],
    "c": ["a bird"],
    "d": ["a fish"],
    "e": ["a mouse"],
    "f": ["a horse"],
    "g": ["a sheep"],
    "h": ["a goat"],
    "i": ["a cow"],
    "j": ["a pig"],
}


def test_wrap_captions_marks_the_boundaries():
    wrapped = wrap_captions({"a": ["small cat"]})
    assert wrapped["a"] == [f"{START} small cat {END}"]


def test_vocabulary_reserves_zero_for_padding():
    word_to_index, index_to_word = build_vocabulary([f"{START} cat {END}"])
    assert PAD_INDEX not in word_to_index.values(), "0 must be left for padding"
    assert index_to_word[PAD_INDEX] == ""
    assert word_to_index[START] >= 1 and word_to_index["cat"] >= 1


def test_vocabulary_indices_are_a_dense_bijection():
    """Indices start at 1 and leave no gaps, which the embedding table size assumes."""
    word_to_index, index_to_word = build_vocabulary(["a cat sat on a mat", "a dog"])
    assert sorted(word_to_index.values()) == list(range(1, len(word_to_index) + 1))
    for word, index in word_to_index.items():
        assert index_to_word[index] == word


def test_vocabulary_always_has_the_special_tokens():
    word_to_index, _ = build_vocabulary(["a plain caption with no markers"])
    for token in (START, END, UNK):
        assert token in word_to_index


def test_encode_maps_unknown_words_to_unk():
    word_to_index, _ = build_vocabulary([f"{START} cat {END}"])
    encoded = encode(f"{START} cat zebra {END}", word_to_index)
    assert encoded[0] == word_to_index[START]
    assert encoded[2] == word_to_index[UNK]


def test_split_is_disjoint_exhaustive_and_deterministic():
    ids = list("abcdefghij")
    first = split_ids(ids, val_size=0.2, test_size=0.2, seed=1)
    second = split_ids(ids, val_size=0.2, test_size=0.2, seed=1)

    assert first == second, "same seed must give the same split"
    everything = first["train"] + first["val"] + first["test"]
    assert sorted(everything) == sorted(ids)
    assert len(set(everything)) == len(ids), "the three sets must not overlap"
    assert first["val"] and first["test"] and first["train"]


def test_split_uses_a_different_order_for_a_different_seed():
    ids = list("abcdefghij")
    assert split_ids(ids, seed=1) != split_ids(ids, seed=2)


def test_split_rejects_impossible_fractions():
    ids = list("abcdefghij")
    with pytest.raises(ValueError):
        split_ids(ids, val_size=0.6, test_size=0.6)
    with pytest.raises(ValueError):
        split_ids(ids, val_size=-0.1)
    with pytest.raises(ValueError):
        split_ids([])


def test_training_pairs_cover_every_position_once():
    mapping = wrap_captions({"a": ["one two three"]})
    word_to_index, _ = build_vocabulary(mapping["a"])
    rows, prefixes, targets = training_pairs(["a"], mapping, word_to_index, max_length=8)

    sequence = encode(mapping["a"][0], word_to_index)
    assert len(targets) == len(sequence) - 1, "one example per position after the first token"
    assert targets == sequence[1:]
    # the image is referenced by row, not copied in - that is what keeps memory flat
    assert rows == [0] * len(targets)


def test_training_pairs_reference_the_right_image_row():
    mapping = wrap_captions({"a": ["one two"], "b": ["one two"]})
    word_to_index, _ = build_vocabulary(["one two"])
    rows, _, _ = training_pairs(["a", "b"], mapping, word_to_index, max_length=6)
    assert set(rows) == {0, 1}
    # identical captions, so each image contributes the same number of examples
    assert rows.count(0) == rows.count(1) > 0


def test_caption_examples_make_one_example_per_caption():
    mapping = wrap_captions({"a": ["one two three", "four five"]})
    word_to_index, _ = build_vocabulary(mapping["a"])
    rows, inputs, targets = caption_examples(["a"], mapping, word_to_index, max_length=10)

    # the per-position pairing gives 3 examples here, one per position
    assert len(inputs) == 2, "one example per caption, not one per position"
    assert rows == [0, 0]
    assert len(targets[0]) == len(inputs[0])


def test_caption_examples_shift_the_target_by_one():
    mapping = wrap_captions({"a": ["one two"]})
    word_to_index, _ = build_vocabulary(["one two"])
    _, inputs, targets = caption_examples(["a"], mapping, word_to_index, max_length=10)

    sequence = encode(mapping["a"][0], word_to_index)
    assert inputs[0] == sequence[:-1]
    assert targets[0] == sequence[1:]
    assert targets[0][-1] == word_to_index[END]


def test_caption_examples_respect_max_length():
    mapping = wrap_captions({"a": ["one two three four five"]})
    word_to_index, _ = build_vocabulary(["one two three four five"])
    _, inputs, targets = caption_examples(["a"], mapping, word_to_index, max_length=2)
    assert len(inputs[0]) == 2 and len(targets[0]) == 2


def test_caption_examples_reference_the_right_image_row():
    mapping = wrap_captions({"a": ["one two"], "b": ["one two"]})
    word_to_index, _ = build_vocabulary(["one two"])
    rows, inputs, _ = caption_examples(["a", "b"], mapping, word_to_index, max_length=8)
    assert rows == [0, 1]
    assert len(inputs) == 2


def test_padded_batches_preserve_every_example_and_pad_to_a_multiple_of_eight():
    from data import _padded_batches

    rows = list(range(5))
    inputs = [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
    targets = [[len(item)] * len(item) for item in inputs]
    batches = _padded_batches(rows, inputs, targets, batch_size=2)

    seen = []
    for batch_rows, batch_inputs, batch_targets in batches:
        assert batch_inputs.shape == batch_targets.shape
        # a rounded width is what keeps the model from rebuilding its graph per batch
        assert batch_inputs.shape[1] % 8 == 0
        seen.extend(batch_rows.tolist())
        for position, row in enumerate(batch_inputs):
            real = [token for token in row if token != PAD_INDEX]
            assert real == inputs[batch_rows[position]], "real tokens first, padding after"
    assert sorted(seen) == rows, "every caption appears exactly once"


def test_padded_batches_group_similar_lengths_together():
    from data import _padded_batches

    rows = list(range(6))
    inputs = [[1], [2], [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2], [1, 2, 3]]
    targets = [[1] * len(item) for item in inputs]
    batches = _padded_batches(rows, inputs, targets, batch_size=2)

    widths = [batch_inputs.shape[1] for _, batch_inputs, _ in batches]
    # short captions must not be padded out to the longest caption in the dataset
    assert min(widths) < max(widths)


def test_usable_mapping_drops_captions_without_a_feature_row():
    from data import usable_mapping

    mapping = {"a": ["one two"], "b": ["three four"], "c": ["five six"]}
    features = {"a": [0], "c": [0]}  # b has no extracted feature
    usable = usable_mapping(mapping, features)
    assert set(usable) == {"a", "c"}
    assert usable["a"] == ["one two"], "captions are kept as they were"


def test_usable_mapping_keeps_everything_when_nothing_is_missing():
    from data import usable_mapping

    mapping = {"a": ["one two"], "b": ["three four"]}
    features = {"a": [0], "b": [0]}
    assert usable_mapping(mapping, features) == mapping


def test_feature_matrix_has_one_row_per_image_not_per_example():
    import numpy as np

    features = {"a": [np.arange(4, dtype="float32")], "b": [np.arange(4, dtype="float32") + 10]}
    matrix = feature_matrix(features, ["a", "b"])
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32
    assert matrix[1][0] == 10


def test_prefixes_are_left_padded_to_max_length():
    mapping = wrap_captions({"a": ["one two three"]})
    word_to_index, _ = build_vocabulary(mapping["a"])
    _, prefixes, _ = training_pairs(["a"], mapping, word_to_index, max_length=5)

    assert all(len(p) == 5 for p in prefixes)
    # the first prefix is just <start>, so it is all padding but its last slot
    assert prefixes[0][:4] == [PAD_INDEX] * 4
    assert prefixes[0][4] == word_to_index[START]


def test_prefix_is_capped_at_max_length():
    long_caption = " ".join(["word"] * 50)
    mapping = wrap_captions({"a": [long_caption]})
    word_to_index, _ = build_vocabulary(mapping["a"])
    _, prefixes, _ = training_pairs(["a"], mapping, word_to_index, max_length=6)
    assert all(len(p) == 6 for p in prefixes)


def test_decode_stops_at_the_end_marker_and_drops_padding():
    word_to_index, index_to_word = build_vocabulary([f"{START} cat sat {END}"])
    tokens = [PAD_INDEX, word_to_index[START], word_to_index["cat"], word_to_index[END], word_to_index["sat"]]
    assert decode(tokens, index_to_word) == "cat"


def test_decode_keeps_going_when_there_is_no_end_marker():
    word_to_index, index_to_word = build_vocabulary([f"{START} cat sat"])
    tokens = [word_to_index[START], word_to_index["cat"], word_to_index["sat"]]
    assert decode(tokens, index_to_word) == "cat sat"
