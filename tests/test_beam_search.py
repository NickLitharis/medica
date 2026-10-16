"""Tests for beam search, using a fake decoder so they run without TensorFlow.

`beam_search_captions` only ever calls `model.predict`, so a small stand-in that produces a
chosen next-token distribution exercises the whole search: the beam bookkeeping, the length
normalisation, and the finished-beam handling.
"""
import numpy as np

from data import PAD_INDEX, START, decode
from evaluate import beam_search_captions, generate_captions, normalised_score

# a tiny vocabulary and a hand-built "language" the fake decoder follows:
#   <start> -> A (0.55) or B (0.45)
#   A       -> <end>                      (a dead end: greedy goes here and stops after 1 word)
#   B -> C -> D -> <end>                  (same total probability, four words)
VOCAB = [PAD_INDEX, "<start>", "<end>", "A", "B", "C", "D"]
WORD_TO_INDEX = {"<start>": 1, "<end>": 2, "A": 3, "B": 4, "C": 5, "D": 6}
INDEX_TO_WORD = {1: "<start>", 2: "<end>", 3: "A", 4: "B", 5: "C", 6: "D"}

NEXT = {
    1: {3: 0.55, 4: 0.45},   # from <start>
    3: {2: 1.0},             # from A: end
    4: {5: 1.0},             # from B: C
    5: {6: 1.0},             # from C: D
    6: {2: 1.0},             # from D: end
    2: {2: 1.0},
}


class FakeDecoder:
    """Returns a (batch, time, vocab) distribution driven by the last real token."""

    def __init__(self, next_tokens=None, vocab_size=len(VOCAB)):
        self.next_tokens = next_tokens if next_tokens is not None else NEXT
        self.vocab_size = vocab_size

    def predict(self, inputs, verbose=0, batch_size=64):
        features, prefixes = inputs
        batch, time = prefixes.shape
        output = np.zeros((batch, time, self.vocab_size), dtype="float32")
        for row in range(batch):
            real = [int(token) for token in prefixes[row] if int(token) != PAD_INDEX]
            last = real[-1] if real else WORD_TO_INDEX["<start>"]
            for token, probability in self.next_tokens.get(last, {0: 1.0}).items():
                output[row, :, token] = probability
        return output


def test_beam_search_runs_at_all():
    """Regression: the length argument is a length, not the token list."""
    captions = beam_search_captions(FakeDecoder(), np.zeros((3, 4), dtype="float32"),
                                    WORD_TO_INDEX, INDEX_TO_WORD, max_length=5, beam_width=2)
    assert len(captions) == 3


def test_greedy_takes_the_locally_best_token_and_stops_at_the_dead_end():
    captions = generate_captions(FakeDecoder(), np.zeros((1, 4), dtype="float32"),
                                 WORD_TO_INDEX, INDEX_TO_WORD, max_length=5, batch_size=1)
    assert captions == ["A"], f"greedy should walk into the dead end, got {captions}"


def test_beam_search_finds_the_better_caption_greedy_cannot():
    """A is more likely at step one but ends after one word; B C D <end> scores better.

    Same total probability, four output words instead of one, so the length penalty makes it
    the better caption - and greedy cannot find it because it commits to A at step one.
    """
    features = np.zeros((1, 4), dtype="float32")
    captions = beam_search_captions(FakeDecoder(), features, WORD_TO_INDEX, INDEX_TO_WORD,
                                    max_length=6, beam_width=2)
    assert captions == ["B C D"], f"beam search should prefer B C D, got {captions}"


def test_beam_search_with_width_one_matches_greedy():
    features = np.zeros((2, 4), dtype="float32")
    greedy = generate_captions(FakeDecoder(), features, WORD_TO_INDEX, INDEX_TO_WORD,
                               max_length=5, batch_size=2)
    beam = beam_search_captions(FakeDecoder(), features, WORD_TO_INDEX, INDEX_TO_WORD,
                                max_length=5, beam_width=1)
    assert greedy == beam


def test_beam_search_respects_max_length():
    # a decoder that never emits <end> must still terminate
    loop = {1: {6: 1.0}, 6: {6: 1.0}}
    captions = beam_search_captions(FakeDecoder(loop), np.zeros((1, 4), dtype="float32"),
                                    WORD_TO_INDEX, INDEX_TO_WORD, max_length=3, beam_width=2)
    assert len(captions[0].split()) <= 3


def test_decode_stops_at_the_end_marker():
    assert decode([1, 3, 2, 5], INDEX_TO_WORD) == "A"
