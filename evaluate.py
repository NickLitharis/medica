"""Generate captions and score them.

Replaces the notebook's evaluation cells. Two things it does differently:

* BLEU is reported per n-gram (BLEU-1 to BLEU-4). The original printed a single BLEU
  figure, which cannot be compared with the breakdown in the thesis.
* Generation is batched. The obvious loop calls the model once per token per image; this
  calls it once per token *for the whole batch*, which is the difference between a few
  hundred model calls and a few dozen on a normal test set.

Captions are produced by feeding the growing prefix back through the model one token at a
time, which is the scheme the model is trained on (predict the next token given the prefix).

Usage::

    python evaluate.py --dataset-dir /kaggle/input/dataset --run /kaggle/working
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data import (
    END,
    PAD_INDEX,
    START,
    UNK,
    decode,
    feature_matrix,
    load_artifacts,
    wrap_captions,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/kaggle/input/dataset")
    parser.add_argument("--run", default="/kaggle/working", help="dir with model_best.keras + encoding.json")
    parser.add_argument("--model", default="", help="override the model path")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N test images")
    parser.add_argument("--batch-size", type=int, default=256, help="images generated at once")
    parser.add_argument("--beam-width", type=int, default=1,
                        help="1 = greedy (the default, and what the readme's numbers use); "
                             "3-5 = beam search, which usually scores higher")
    parser.add_argument("--length-penalty", type=float, default=0.6,
                        help="beam search length normalisation exponent")
    return parser.parse_args(argv)


def generate_captions(model, features, word_to_index, index_to_word, max_length, batch_size=64):
    """Greedy-decode captions for a matrix of image features.

    ``features`` is ``(n_images, feature_dim)``. Returns a list of ``n_images`` captions.

    Images are decoded in chunks of ``batch_size``, and within a chunk one model call covers
    a whole token position for the whole chunk. Both are deliberate: asking the model for the
    full test split at once means materialising ``(n_images, max_length, vocab_size)``, which
    on the real split is 9857 x 22 x 9025 - 7.8 GB as float32, and that is an out-of-memory
    error rather than a slow run.

    Prefixes are **right**-padded: real tokens first, padding after. Padding on the left
    makes the embedding mask go false-then-true, which cuDNN's LSTM rejects outright - and
    cuDNN is what runs on a GPU. For a per-position model the distribution is then read at
    each caption's last real token rather than at the final column, which is padding.
    """
    import numpy as np

    total = len(features)
    tokens = [[] for _ in range(total)]

    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        chunk_features = features[start:stop]
        size = stop - start

        prefixes = np.full((size, max_length), PAD_INDEX, dtype="int32")
        prefixes[:, 0] = word_to_index[START]
        lengths = np.ones(size, dtype="int32")
        finished = np.zeros(size, dtype=bool)

        for _ in range(max_length):
            predictions = model.predict([chunk_features, prefixes], verbose=0,
                                        batch_size=batch_size)
            if predictions.ndim == 3:
                # per-position model: read each caption's last real token, not the padding
                predictions = predictions[np.arange(size), lengths - 1]
            next_indices = np.argmax(predictions, axis=1).astype("int32")

            for i in range(size):
                if finished[i]:
                    continue
                if index_to_word.get(int(next_indices[i])) == END:
                    finished[i] = True
                    continue
                tokens[start + i].append(int(next_indices[i]))
                if lengths[i] < max_length:
                    prefixes[i, lengths[i]] = next_indices[i]
                    lengths[i] += 1
            if finished.all():
                break

    return [decode([word_to_index[START], *row], index_to_word) for row in tokens]


def reference_tokens(raw_mapping, image_ids):
    """Reference token lists for BLEU - the original captions, with no start/end markers.

    ``decode`` strips the markers from a generated caption, so a reference that kept them
    would carry two tokens no hypothesis can ever match, which deflates every n-gram
    precision. Pass the *unwrapped* mapping here.
    """
    return [[reference.split() for reference in raw_mapping[image_id]] for image_id in image_ids]


def normalised_score(total_log_prob: float, length: int, length_penalty: float = 0.6) -> float:
    """Rank a beam by summed log probability, adjusted for how long it is.

    Without this, beam search prefers short captions: every extra token can only add a
    negative log probability. Dividing by ``length ** length_penalty`` is the usual fix.
    """
    return total_log_prob / max(length, 1) ** length_penalty


def beam_search_captions(
    model,
    features,
    word_to_index,
    index_to_word,
    max_length,
    beam_width=3,
    batch_size=64,
    length_penalty=0.6,
):
    """Beam search over the same per-position decoder.

    Greedy decoding commits to the single most likely token at every step, so a word that
    looks best now but leads nowhere cannot be taken back. Beam search keeps the best
    ``beam_width`` partial captions per image and only narrows at the end.

    The expensive part - one model call per token position - is batched over every active
    beam of ``batch_size`` images at once, so this costs about ``beam_width`` times greedy
    rather than ``beam_width`` separate passes. Prefixes are right-padded, as in
    :func:`generate_captions`, and each row is read at its own last real token.
    """
    import numpy as np

    start_id = word_to_index[START]
    end_id = word_to_index.get(END)
    total = len(features)
    captions = [""] * total

    for chunk_start in range(0, total, batch_size):
        chunk = features[chunk_start : chunk_start + batch_size]
        count = len(chunk)
        # (summed log probability, tokens, finished)
        beams = [[(0.0, [start_id], False)] for _ in range(count)]

        for _ in range(max_length):
            rows = [(image, index)
                    for image in range(count)
                    for index, (_, tokens, done) in enumerate(beams[image])
                    if not done]
            if not rows:
                break

            row_features = np.stack([chunk[image] for image, _ in rows])
            prefixes = np.full((len(rows), max_length), PAD_INDEX, dtype="int32")
            lengths = np.empty(len(rows), dtype="int32")
            for position, (image, index) in enumerate(rows):
                tokens = beams[image][index][1]
                prefixes[position, : len(tokens)] = tokens
                lengths[position] = len(tokens)

            predictions = model.predict([row_features, prefixes], verbose=0,
                                        batch_size=batch_size)
            if predictions.ndim == 3:
                predictions = predictions[np.arange(len(rows)), lengths - 1]
            log_probabilities = np.log(np.clip(predictions, 1e-9, None))

            candidates = [[] for _ in range(count)]
            for position, (image, index) in enumerate(rows):
                score = beams[image][index][0]
                tokens = beams[image][index][1]
                # the beam_width most likely next tokens for this beam
                top = np.argpartition(-log_probabilities[position], beam_width)[:beam_width]
                for token in top:
                    token = int(token)
                    extended = tokens + [token]
                    candidates[image].append((
                        score + float(log_probabilities[position][token]),
                        extended,
                        token == end_id or len(extended) >= max_length,
                    ))

            for image in range(count):
                # beams already finished stay in the running, unextended
                kept = [beam for beam in beams[image] if beam[2]]
                pool = kept + candidates[image]
                # len(tokens) - 1: <start> is the prompt, not an output token, so it must not
                # inflate the length penalty for short captions
                pool.sort(key=lambda beam: normalised_score(beam[0], len(beam[1]) - 1,
                                                            length_penalty),
                          reverse=True)
                beams[image] = pool[:beam_width]

        for image in range(count):
            best = max(beams[image],
                       key=lambda beam: normalised_score(beam[0], len(beam[1]) - 1,
                                                         length_penalty))
            captions[chunk_start + image] = decode(best[1], index_to_word)
    return captions


def bleu_scores(hypotheses, references):
    """BLEU-1..4 as a dict; references is a list of reference lists per hypothesis."""
    from nltk.translate.bleu_score import corpus_bleu

    scores = {}
    for n in (1, 2, 3, 4):
        weights = tuple(1.0 / n if i < n else 0.0 for i in range(4))
        try:
            scores[f"BLEU-{n}"] = corpus_bleu(references, hypotheses, weights=weights)
        except ZeroDivisionError:
            scores[f"BLEU-{n}"] = 0.0
    return scores


def main(argv=None) -> int:
    args = parse_args(argv)
    import tensorflow as tf

    run = Path(args.run)
    with (run / "encoding.json").open(encoding="utf-8") as handle:
        encoding = json.load(handle)
    word_to_index = encoding["word_to_index"]
    index_to_word = {int(index): word for index, word in encoding["index_to_word"].items()}
    max_length = encoding["max_length"]

    model = tf.keras.models.load_model(args.model or (run / "model_best.keras"))

    features, raw_mapping, _ = load_artifacts(args.dataset_dir)
    mapping = wrap_captions(raw_mapping)
    test_ids = encoding["test_ids"]
    if args.limit:
        test_ids = test_ids[: args.limit]
    test_ids = [image_id for image_id in test_ids if image_id in features and image_id in mapping]

    started = time.time()
    matrix = feature_matrix(features, test_ids)
    if args.beam_width > 1:
        print(f"beam search, width {args.beam_width}, length penalty {args.length_penalty}")
        captions = beam_search_captions(
            model, matrix, word_to_index, index_to_word, max_length,
            beam_width=args.beam_width, batch_size=args.batch_size,
            length_penalty=args.length_penalty,
        )
    else:
        captions = generate_captions(
            model, matrix, word_to_index, index_to_word, max_length, batch_size=args.batch_size
        )
    elapsed = time.time() - started

    hypotheses = [caption.split() for caption in captions]
    references = reference_tokens(raw_mapping, test_ids)

    print(f"scored {len(hypotheses)} captions in {elapsed:.1f}s "
          f"({len(hypotheses) / max(elapsed, 1e-9):.1f} captions/s)")
    if not hypotheses:
        print("nothing to score - check the test ids in encoding.json")
        return 1

    scores = bleu_scores(hypotheses, references)
    print("metric      score")
    for name, value in scores.items():
        print(f"{name:<10}  {value:.4f}")

    (run / "bleu.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    for image_id, caption in list(zip(test_ids, captions))[:5]:
        print(f"  {image_id[:12]}...  {caption}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
