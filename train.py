"""Train the captioning model.

Replaces the notebook's training cells. Differences that matter:

* a validation split drives checkpointing and early stopping, so the stopping point is
  chosen on data the model has not seen (the original had no validation set);
* examples are built once, not re-tokenised every epoch;
* labels stay as integers and the loss is sparse categorical cross-entropy, instead of a
  one-hot vector per example;
* the vocabulary and run configuration are saved next to the weights, so evaluation and
  generation do not have to rebuild them.

Usage::

    python train.py --dataset-dir /kaggle/input/dataset --epochs 30 --out /kaggle/working
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from data import (
    build_vocabulary,
    load_artifacts,
    make_caption_dataset,
    make_dataset,
    split_ids,
    wrap_captions,
)
from model import build_captioning_model, build_seq2seq_model


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/kaggle/input/dataset")
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--limit-images", type=int, default=0,
                        help="use only the first N images (0 = all); for smoke runs")
    parser.add_argument("--min-count", type=int, default=5,
                        help="drop words appearing fewer times than this")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="train in mixed_float16; a large speedup on GPU")
    parser.add_argument("--steps-per-execution", type=int, default=1,
                        help="batches per graph execution; raises throughput on GPU")
    parser.add_argument("--jit", action="store_true",
                        help="compile with XLA; faster where it applies, not always compatible")
    parser.add_argument("--arch", choices=["seq2seq", "legacy"], default="seq2seq",
                        help="seq2seq = one pass per caption (default, much less work); "
                             "legacy = one pass per position, matching the trained artifact")
    parser.add_argument("--num-parallel-calls", type=int, default=0,
                        help="0 = let tf.data decide (AUTOTUNE)")
    return parser.parse_args(argv)


def epoch_timer():
    """A real Keras callback that prints each epoch's wall-clock time, so tuning is measured."""
    import tensorflow as tf

    clock = {"started": 0.0}

    def on_epoch_begin(epoch, logs=None):
        clock["started"] = time.time()

    def on_epoch_end(epoch, logs=None):
        print(f"  [timing] epoch {epoch}: {time.time() - clock['started']:.1f}s", flush=True)

    return tf.keras.callbacks.LambdaCallback(
        on_epoch_begin=on_epoch_begin, on_epoch_end=on_epoch_end
    )


def main(argv=None) -> int:
    args = parse_args(argv)

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        CSVLogger,
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    features, mapping, max_length = load_artifacts(args.dataset_dir)
    mapping = wrap_captions(mapping)
    if args.limit_images:
        keep = sorted(mapping)[: args.limit_images]
        mapping = {key: mapping[key] for key in keep}

    all_captions = [caption for captions in mapping.values() for caption in captions]
    word_to_index, index_to_word = build_vocabulary(all_captions, min_count=args.min_count)
    vocab_size = len(word_to_index) + 1  # + the padding slot

    splits = split_ids(list(mapping), val_size=args.val_size, test_size=args.test_size)
    print(f"images: {len(mapping)}  vocab: {vocab_size}  max_length: {max_length}  arch: {args.arch}")
    print(f"train: {len(splits['train'])}  val: {len(splits['val'])}  test: {len(splits['test'])}")

    def dataset_for(ids, shuffle):
        subset = {key: mapping[key] for key in ids}
        builder = make_caption_dataset if args.arch == "seq2seq" else make_dataset
        return builder(features, subset, word_to_index, max_length,
                       batch_size=args.batch_size, shuffle=shuffle)

    train_ds = dataset_for(splits["train"], shuffle=True)
    val_ds = dataset_for(splits["val"], shuffle=False)

    feature_dim = int(features[splits["train"][0]][0].shape[0])
    if args.arch == "seq2seq":
        model = build_seq2seq_model(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
        )
    else:
        model = build_captioning_model(
            vocab_size=vocab_size,
            feature_dim=feature_dim,
            embed_dim=args.embed_dim,
            dropout=args.dropout,
            max_length=max_length,
        )
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("mixed precision: mixed_float16", flush=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
        steps_per_execution=args.steps_per_execution,
        jit_compile=args.jit,
    )
    model.summary()

    best = out / "model_best.keras"
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[
            ModelCheckpoint(best, monitor="val_loss", save_best_only=True, verbose=1),
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5, verbose=1),
            CSVLogger(out / "training_log.csv"),
            epoch_timer(),
        ],
    )

    # Everything evaluation needs to reproduce the encoding, saved beside the weights.
    with (out / "encoding.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "max_length": max_length,
                "arch": args.arch,
                "vocab_size": vocab_size,
                "embed_dim": args.embed_dim,
                "feature_dim": feature_dim,
                "min_count": args.min_count,
                "word_to_index": word_to_index,
                "index_to_word": index_to_word,
                "test_ids": splits["test"],
                "splits": {name: len(ids) for name, ids in splits.items()},
            },
            handle,
        )
    print(f"saved: {best} and {out / 'encoding.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
