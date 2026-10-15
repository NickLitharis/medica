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
    parser.add_argument("--batch-size", type=int, default=64, help="images generated at once")
    return parser.parse_args(argv)


def generate_captions(model, features, word_to_index, index_to_word, max_length, batch_size=64):
    """Greedy-decode captions for a matrix of image features.

    ``features`` is ``(n_images, feature_dim)``. Returns a list of ``n_images`` captions.
    One model call per token position, over the whole batch, rather than one per token per
    image.
    """
    import numpy as np

    count = len(features)
    # prefixes are left-padded, so the tokens sit at the end and padding at the front
    prefixes = np.full((count, max_length), PAD_INDEX, dtype="int32")
    prefixes[:, -1] = word_to_index[START]

    finished = np.zeros(count, dtype=bool)
    tokens = [[] for _ in range(count)]

    for _ in range(max_length):
        predictions = model.predict([features, prefixes], verbose=0, batch_size=batch_size)
        if predictions.ndim == 3:
            # a per-position model (seq2seq) speaks at every step; the last step is the next token
            predictions = predictions[:, -1, :]
        next_indices = np.argmax(predictions, axis=1).astype("int32")

        for i in range(count):
            if finished[i]:
                continue
            if index_to_word.get(int(next_indices[i])) == END:
                finished[i] = True
            else:
                tokens[i].append(int(next_indices[i]))
        if finished.all():
            break

        # shift left and append, keeping the last max_length tokens
        prefixes = np.concatenate([prefixes[:, 1:], next_indices.reshape(-1, 1)], axis=1)

    return [decode([word_to_index[START], *row], index_to_word) for row in tokens]


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

    features, mapping, _ = load_artifacts(args.dataset_dir)
    mapping = wrap_captions(mapping)
    test_ids = encoding["test_ids"]
    if args.limit:
        test_ids = test_ids[: args.limit]
    test_ids = [image_id for image_id in test_ids if image_id in features and image_id in mapping]

    started = time.time()
    matrix = feature_matrix(features, test_ids)
    captions = generate_captions(
        model, matrix, word_to_index, index_to_word, max_length, batch_size=args.batch_size
    )
    elapsed = time.time() - started

    hypotheses = [caption.split() for caption in captions]
    references = [[reference.split() for reference in mapping[image_id]] for image_id in test_ids]

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
