"""Tests for the evaluation step.

`evaluate.py` imports TensorFlow only inside `main`, so these run without a GPU or a
TensorFlow install.
"""
from data import wrap_captions

from evaluate import normalised_score, reference_tokens


def test_normalised_score_removes_length_bias_at_penalty_one():
    """Same per-token log probability must score the same, whatever the length."""
    per_token = -2.0
    assert normalised_score(per_token * 5, 5, 1.0) == normalised_score(per_token * 20, 20, 1.0)


def test_normalised_score_reduces_the_length_bias_at_the_default_penalty():
    """The default penalty of 0.6 softens the short-caption bias without removing it.

    Without normalisation, equal-quality captions of 5 and 20 tokens differ by a factor of 4.
    """
    per_token = -2.0
    raw_gap = abs(per_token * 20) - abs(per_token * 5)
    penalty = 0.6
    normalised_gap = (abs(normalised_score(per_token * 20, 20, penalty))
                      - abs(normalised_score(per_token * 5, 5, penalty)))
    assert 0 < normalised_gap < raw_gap


def test_normalised_score_with_no_penalty_is_the_raw_score():
    assert normalised_score(-10.0, 5, length_penalty=0.0) == -10.0


def test_normalised_score_prefers_the_better_caption():
    # -1.0 per token over 10 tokens beats -2.0 per token over 10 tokens
    assert normalised_score(-10.0, 10) > normalised_score(-20.0, 10)


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
