"""The captioning model.

This reproduces the architecture of the trained model shipped with the thesis. The layer
stack below was read out of the saved weights (`model_epoch_50.h5`):

    Embedding -> Bidirectional(LSTM) -> Dropout
    Attention
    Dense -> GlobalAveragePooling1D -> Add
    Dense -> Dropout -> Dense(softmax)

`model_training_preprocessing.ipynb` builds the same inputs but **without the attention
layer**, so the model it defines cannot reproduce the trained weights or the reported BLEU
scores. This module is the corrected definition; the notebook is left as the original record.

Why the image input is a vector and not an image: features arrive pre-extracted
(`image_features.pkl`, 4096-dim, produced by a CNN in a separate step), so no convolutional
layer appears here. The CNN is the feature-extraction stage; this model is the decoder over
its output.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Add,
    Attention,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    Reshape,
)
from tensorflow.keras.models import Model

# Shapes taken from the trained model.
DEFAULT_FEATURE_DIM = 4096
DEFAULT_EMBED_DIM = 256
DEFAULT_DROPOUT = 0.5


def build_captioning_model(
    vocab_size: int,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
    max_length: int | None = None,
) -> Model:
    """Build the attention-augmented decoder.

    Two inputs, matching the trained model: pre-extracted image features, and the caption
    tokens seen so far. One output: a distribution over the vocabulary.

    The output is pooled over the sequence, which is what the trained model does — see the
    note in README.md about per-sample versus per-position prediction.
    """
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}")
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    # Image features -> the vector that the decoder output is added to.
    # Note the width: a Bidirectional LSTM emits embed_dim per direction, so the attended
    # sequence is 2 * embed_dim wide and this projection must match it for the Add below.
    # The trained model does the same thing - Dense(512) against LSTM(256) per direction.
    features_in = Input(shape=(feature_dim,), name="image_features")
    fe = Dropout(dropout, name="feature_dropout")(features_in)
    fe = Dense(2 * embed_dim, activation="relu", name="feature_projection")(fe)

    # Caption tokens -> sequence representation.
    tokens_in = Input(shape=(max_length,) if max_length else (None,), name="caption_tokens")
    tokens = Embedding(vocab_size, embed_dim, mask_zero=True, name="token_embedding")(tokens_in)
    tokens = Dropout(dropout, name="token_dropout")(tokens)
    sequence = Bidirectional(
        LSTM(embed_dim, return_sequences=True, dropout=dropout, recurrent_dropout=dropout),
        name="bidirectional_encoder",
    )(tokens)

    # Attention over the sequence, then pooled back to a single vector.
    context = Attention(name="attention")([sequence, sequence])
    context = GlobalAveragePooling1D(name="attention_pool")(context)

    # Fuse the image vector with the attended text vector (both 2 * embed_dim wide).
    merged = Add(name="fuse")([fe, context])
    merged = Dense(embed_dim, activation="relu", name="decoder_dense")(merged)
    merged = Dropout(dropout, name="decoder_dropout")(merged)

    # float32 on the output keeps a mixed_float16 run numerically stable
    outputs = Dense(vocab_size, activation="softmax", name="next_token", dtype="float32")(merged)

    return Model(inputs=[features_in, tokens_in], outputs=outputs, name="captioning_model")


def build_seq2seq_model(
    vocab_size: int,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    embed_dim: int = DEFAULT_EMBED_DIM,
    dropout: float = DEFAULT_DROPOUT,
) -> Model:
    """The teacher-forced decoder: one pass over a whole caption, a distribution per position.

    Three differences from :func:`build_captioning_model`, all of which cut training work:

    * the sequence is not pooled, so one pass supervises every position instead of one;
    * the LSTM is unidirectional. Teacher forcing feeds the whole caption at once, so a
      bidirectional decoder would let each position see the tokens after the one it is
      predicting; a causal decoder is required, and it also costs half as much. (The
      per-position model fed only the prefix, so bidirectionality was harmless there -
      merely expensive.)
    * no recurrent dropout, which is what stops XLA from compiling the step.

    The image conditions every position through cross-attention over a one-step sequence,
    rather than being added once to a pooled vector.
    """
    if vocab_size <= 1:
        raise ValueError(f"vocab_size must be greater than 1, got {vocab_size}")
    if feature_dim <= 0:
        raise ValueError(f"feature_dim must be positive, got {feature_dim}")

    features_in = Input(shape=(feature_dim,), name="image_features")
    image = Dropout(dropout, name="feature_dropout")(features_in)
    image = Dense(embed_dim, activation="relu", name="feature_projection")(image)
    # a length-1 sequence so every caption position can attend to the image
    image_seq = Reshape((1, embed_dim), name="image_as_sequence")(image)

    # Length is left dynamic so a batch is only padded to its own longest caption.
    tokens_in = Input(shape=(None,), name="caption_tokens")
    tokens = Embedding(vocab_size, embed_dim, mask_zero=True, name="token_embedding")(tokens_in)
    tokens = Dropout(dropout, name="token_dropout")(tokens)
    sequence = LSTM(
        embed_dim, return_sequences=True, dropout=dropout, name="decoder_lstm"
    )(tokens)
    sequence = Dropout(dropout, name="sequence_dropout")(sequence)

    # query = caption positions, key/value = the image, so the result is per position
    context = Attention(name="image_attention")([sequence, image_seq])
    merged = Add(name="fuse")([sequence, context])
    merged = Dense(embed_dim, activation="relu", name="decoder_dense")(merged)
    merged = Dropout(dropout, name="decoder_dropout")(merged)

    # float32 on the output keeps a mixed_float16 run numerically stable
    outputs = Dense(vocab_size, activation="softmax", name="next_token", dtype="float32")(merged)

    return Model(inputs=[features_in, tokens_in], outputs=outputs, name="captioning_seq2seq")


def layer_names(model: Model) -> list[str]:
    """Every layer class in the graph, including layers nested inside wrappers.

    A `Bidirectional` wrapper keeps its `LSTM`s as sub-layers, and Keras 3 no longer
    lists those at the top level, so walking recursively keeps this working on both
    Keras 2 (the pinned TensorFlow 2.15) and Keras 3.
    """
    names: list[str] = []

    def walk(layer) -> None:
        names.append(layer.__class__.__name__)
        for sub in getattr(layer, "layers", []) or []:
            walk(sub)

    for layer in model.layers:
        walk(layer)
    return names
