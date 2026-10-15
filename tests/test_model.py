"""Tests for the captioning model.

These run without the dataset: the model is built with random tensors, which is enough to
prove the architecture is constructible and that the attention layer is present — the thing
the committed notebook is missing.

Run with::

    uv run pytest
"""
import numpy as np
import pytest

from model import build_captioning_model, layer_names

VOCAB = 500
FEATURES = 64  # smaller than the real 4096, to keep the tests quick
EMBED = 16


@pytest.fixture(scope="module")
def model():
    return build_captioning_model(vocab_size=VOCAB, feature_dim=FEATURES, embed_dim=EMBED, max_length=12)


def test_has_the_attention_layer(model):
    """The regression that matters: the architecture must attention-pool, as the trained model does."""
    assert "Attention" in layer_names(model)


def test_matches_the_trained_model_layer_stack(model):
    """Compare against the layer classes the trained model's own config records.

    Reading `model.to_json()` rather than `model.layers` keeps this independent of how a
    given Keras version exposes nested layers (a Bidirectional's LSTMs are sub-layers),
    and it compares like for like: the trained model's layer list was read from its
    config in exactly the same way.
    """
    config = model.to_json().replace(" ", "").replace("\n", "")
    for expected in ("InputLayer", "Embedding", "Bidirectional", "LSTM", "Attention",
                     "GlobalAveragePooling1D", "Add", "Dense"):
        assert f'"class_name":"{expected}"' in config, f"{expected} missing from the model config"


def test_two_inputs_one_output(model):
    assert len(model.inputs) == 2
    assert len(model.outputs) == 1
    assert {i.name for i in model.inputs} == {"image_features", "caption_tokens"}


def test_output_is_a_distribution_over_the_vocabulary(model):
    assert model.output_shape == (None, VOCAB)


def test_forward_pass_produces_valid_probabilities(model):
    features = np.random.rand(4, FEATURES).astype("float32")
    tokens = np.random.randint(1, VOCAB, size=(4, 12)).astype("int32")
    predictions = model.predict([features, tokens], verbose=0)
    assert predictions.shape == (4, VOCAB)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-4)
    assert (predictions >= 0).all()


def test_a_bad_vocabulary_size_is_rejected():
    with pytest.raises(ValueError):
        build_captioning_model(vocab_size=1)


def test_variable_length_captions_are_supported():
    """The notebook passes max_length=None so the token input stays flexible."""
    flexible = build_captioning_model(vocab_size=VOCAB, feature_dim=FEATURES, embed_dim=EMBED)
    assert flexible.inputs[1].shape[1] is None
