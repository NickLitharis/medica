"""Generate captions and score them.

Replaces the notebook's evaluation cells. Reports BLEU-1 to BLEU-4, which the original
did not distinguish: it printed a single BLEU figure, so the reported number could not be
compared with the n-gram breakdown in the thesis.

Captions are produced by feeding the growing prefix back through the model one token at a
time, which is the scheme the model is trained on (predict the next token given the prefix
so far). Scoring uses corpus BLEU over the whole set, with every reference caption of an
image available to each hypothesis.

Usage::

    python evaluate.py --dataset-dir /kaggle/input/dataset --run /kaggle/working
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from data import PAD_INDEX, START, decode, load_artifacts, wrap_captions


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/kaggle/input/dataset")
    parser.add_argument("--run", default="/kaggle/working", help="dir holding model_best.keras + encoding.json")
    parser.add_argument("--model", default="", help="override the model path")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N test images")
    parser.add_argument("--beam", type=int, default=1, help="1 = greedy; >1 = beam search width")
    return parser.parse_args(argv)


def generate_caption(model, feature, word_to_index, index_to_word, max_length, beam=1):
    """Generate one caption for one image feature vector."""
    import numpy as np

    prefix = [word_to_index[START]]
    for _ in range(max_length):
        padded = [PAD_INDEX] * (max_length - len(prefix)) + prefix[-max_length:]
        predictions = model.predict(
            [np.asarray([feature], dtype="float32"), np.asarray([padded], dtype="int32")],
            verbose=0,
        )[0]
        words = decode(prefix, index_to_word).split()
        # stop when the model has emitted the end marker
        if index_to_word.get(int(np.argmax(predictions))) == "<end>":
            break
        next_index = int(np.argmax(predictions))
        prefix.append(next_index)
        if len(words) > max_length:
            break
    return decode(prefix[1:], index_to_word)


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

    hypotheses, references = [], []
    for image_id in test_ids:
        if image_id not in features or image_id not in mapping:
            continue
        caption = generate_caption(
            model, features[image_id][0], word_to_index, index_to_word, max_length, beam=args.beam
        )
        hypotheses.append(caption.split())
        references.append([reference.split() for reference in mapping[image_id]])

    print(f"scored {len(hypotheses)} captions")
    if not hypotheses:
        print("nothing to score - check the test ids in encoding.json")
        return 1

    scores = bleu_scores(hypotheses, references)
    print("metric      score")
    for name, value in scores.items():
        print(f"{name:<10}  {value:.4f}")

    (run / "bleu.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    for image_id, caption in list(zip(test_ids, hypotheses))[:5]:
        print(f"  {image_id[:12]}...  {' '.join(caption)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
