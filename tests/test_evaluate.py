"""Tests for the evaluation step.

`evaluate.py` imports TensorFlow only inside `main`, so these run without a GPU or a
TensorFlow install.
"""
from data import wrap_captions

from evaluate import reference_tokens


def test_reference_tokens_exclude_the_markers():
    """decode() strips start/end from a generated caption, so references must not have them."""
    raw = {"a": ["chest xray showing pleural effusion"]}
    references = reference_tokens(raw, ["a"])
    assert references == [[["chest", "xray", "showing", "pleural", "effusion"]]]


def test_wrapped_references_would_add_two_unmatchable_tokens():
    """The regression this guards: passing the wrapped mapping adds two tokens per reference."""
    raw = {"a": ["chest xray showing pleural effusion"]}
    wrapped = wrap_captions(raw)

    good = reference_tokens(raw, ["a"])[0][0]
    bad = reference_tokens(wrapped, ["a"])[0][0]

    assert len(bad) == len(good) + 2, "the wrapped form really does carry two extra tokens"
    assert "<start>" in bad and "<end>" in bad
    assert "<start>" not in good and "<end>" not in good


def test_reference_tokens_keep_multiple_references_per_image():
    raw = {"a": ["one two", "three four"]}
    assert reference_tokens(raw, ["a"]) == [[["one", "two"], ["three", "four"]]]
