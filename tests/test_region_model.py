"""Tests for the region-conditioned decoder.

Like test_model.py these need TensorFlow, so they run on Kaggle rather than locally. They pin
the two things that would silently ruin a training run: the input contract and the fact that
the image, not the caption, decides the output.
"""
import numpy as np
import pytest

tensorflow = pytest.importorskip("tensorflow")

from model import build_region_model  # noqa: E402

REGION_COUNT, REGION_DIM = 49, 512


def test_output_has_a_distribution_for_every_caption_position():
    model = build_region_model(vocab_size=120, region_dim=REGION_DIM,
                               region_count=REGION_COUNT)
    regions = np.zeros((2, REGION_COUNT, REGION_DIM), dtype="float32")
    tokens = np.zeros((2, 7), dtype="int32")
    output = model.predict([regions, tokens], verbose=0)

    assert output.shape == (2, 7, 120), "one distribution per position"
    assert np.allclose(output.sum(axis=-1), 1.0, atol=1e-4), "positions must be distributions"


def test_inputs_are_the_regions_and_the_caption():
    model = build_region_model(vocab_size=120, region_dim=REGION_DIM,
                               region_count=REGION_COUNT)
    names = [tensor.name.split(":")[0] for t in model.inputs]
    assert "image_regions" in names[0], names
    assert "caption_tokens" in names[1], names


def test_the_image_changes_the_output():
    """If swapping the image left the output identical, the decoder is ignoring it."""
    model = build_region_model(vocab_size=120, region_dim=REGION_DIM,
                               region_count=REGION_COUNT)
    tokens = np.zeros((1, 5), dtype="int32")
    first = model.predict([np.zeros((1, REGION_COUNT, REGION_DIM), dtype="float32"), tokens],
                          verbose=0)
    second = model.predict([np.ones((1, REGION_COUNT, REGION_DIM), dtype="float32"), tokens],
                           verbose=0)
    assert not np.allclose(first, second), "the image must influence the prediction"


def test_region_count_is_respected():
    model = build_region_model(vocab_size=60, region_dim=64, region_count=16)
    output = model.predict([np.zeros((1, 16, 64), dtype="float32"),
                            np.zeros((1, 4), dtype="int32")], verbose=0)
    assert output.shape == (1, 4, 60)


def test_vocab_and_region_arguments_are_validated():
    with pytest.raises(ValueError):
        build_region_model(vocab_size=1)
    with pytest.raises(ValueError):
        build_region_model(vocab_size=10, region_count=0)
    with pytest.raises(ValueError):
        build_region_model(vocab_size=10, region_dim=0)


def test_attention_is_not_causal_over_regions():
    """The image has no future to hide, so a later position may use an earlier region."""
    model = build_region_model(vocab_size=40, region_dim=32, region_count=9, embed_dim=16)
    layer_names = [layer.name for layer in model.layers]
    assert "region_attention" in layer_names
    assert "image_regions" in [t.name.split(":")[0] for t in model.inputs]
