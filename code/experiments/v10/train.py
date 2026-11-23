"""Fit matched hard-label controls and the distilled fastText candidate."""
import argparse
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from inputs import SPLITS, TEACHER, digest, save, split
from views import training_views
from predict_masked_v9 import ASSETS, MaskedBodyEncoder
from bili_ai.metrics import select_threshold
from bili_ai.model import DEFAULT_PARAMS, positive_scores, train_model
from bili_ai.preprocess import Preprocessor

COPIES = 8
TEACHER_WEIGHT = 0.70
OPTIONS = {**DEFAULT_PARAMS, "epoch": 4}
ARMS = ("raw_hard", "aug_hard", "v10")


def targets(rows, arm, teacher_path):
    labels = [int(row["label"]) for row in rows]
    if arm != "v10":
        return [x * COPIES for x in labels]
    cache = json.loads(teacher_path.read_bytes())
    soft = cache["soft_targets"]
    if (cache["teacher"] != TEACHER or cache["train_sha256"] != SPLITS["train"]
            or cache["temperature"] != 2.0 or len(soft) != len(labels)
            or not all(math.isfinite(p) and 0 <= p <= 1 for p in soft)):
        raise ValueError("Teacher cache must match the frozen teacher and ordered training split")
    # Unbiased stochastic rounding. Every view of a row receives the same soft target.
    rng = random.Random(20261123)
    return [int(COPIES * ((1 - TEACHER_WEIGHT) * y + TEACHER_WEIGHT * p) + rng.random())
            for y, p in zip(labels, soft, strict=True)]


def write_corpus(train_file, rows, encoder, counts, arm):
    rng = random.Random(42)
    lines = []
    changed = Counter()
    for row in rows:
        text = row["text"]
        kind = rng.randrange(4)
        views = training_views(text, kind) if arm != "raw_hard" else (text,) * 4
        encoded = {t: encoder.encode(t, "", "text") for t in set(views)}
        original = encoded[text]
        if original == "field_empty":
            raise ValueError("Ineligible original in frozen training split")
        for i, t in enumerate(views):
            # Preserve row weight if an edit turns a valid comment into an abstention.
            line = encoded[t] if encoded[t] != "field_empty" else original
            lines.append(line)
            changed[str(i)] += line != original
    order = list(range(len(lines) * COPIES))
    random.Random(42).shuffle(order)
    with train_file.open("w", encoding="utf-8", newline="\n") as f:
        for position in order:
            view, copy = divmod(position, COPIES)
            label = int(copy < counts[view // 4])
            f.write(f"__label__{label} {lines[view]}\n")
    return dict(changed)


def train(processed, target, arm, teacher_path):
    if target.exists():
        raise FileExistsError(f"Choose a new output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = split(processed, "train")
    val = split(processed, "validation")
    base = Preprocessor(ASSETS)
    encoder = MaskedBodyEncoder(base)
    counts = targets(rows, arm, teacher_path)
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        tmp = Path(tmp)
        bundle = tmp / "bundle"
        corpus = tmp / "train.txt"
        changed = write_corpus(corpus, rows, encoder, counts, arm)
        print(f"{arm}: fitting {len(rows)*4*COPIES} examples", flush=True)
        model = train_model(corpus, bundle / "model.bin", OPTIONS)
        features = [encoder.encode(r["text"], "", "text") for r in val]
        missing = [s == "field_empty" for s in features]
        scores = positive_scores(model, features)
        selected = select_threshold([int(r["label"]) for r in val], scores, 0.01, missing)
        meta = dict(arm=arm, mode="text", encoder="masked_v9_structure",
                    threshold=selected["threshold"], validation=selected,
                    dictionary_sha256=base.config["dictionary_sha256"],
                    preprocess_sha256=digest(ASSETS / "preprocess.json"),
                    model_sha256=digest(bundle / "model.bin"),
                    train_sha256=SPLITS["train"], validation_sha256=SPLITS["validation"],
                    params=OPTIONS, unique_training_rows=len(rows), views_per_row=4, copies_per_view=COPIES,
                    changed_encoded_views=changed, training_file_sha256=digest(corpus),
                    nonbinary_rows=sum(0 < c < COPIES for c in counts))
        if arm == "v10":
            meta.update(teacher=TEACHER, teacher_weight=TEACHER_WEIGHT, temperature=2.0,
                        teacher_cache_sha256=digest(teacher_path))
        save(bundle / "model.bin.json", meta)
        bundle.rename(target)
    print(f"{arm}: validation threshold frozen at {selected['threshold']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm", choices=ARMS)
    p.add_argument("processed", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--teacher-cache", type=Path)
    a = p.parse_args()
    if a.arm == "v10" and not a.teacher_cache:
        p.error("v10 requires --teacher-cache")
    train(a.processed, a.output, a.arm, a.teacher_cache)
