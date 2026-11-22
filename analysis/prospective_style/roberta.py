"""Body-only Chinese RoBERTa reference, fitted once on the historical weak labels."""

import argparse
import csv
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from collect import freeze, read
from transformers import BertForSequenceClassification, BertTokenizer

REVISION = "5c58d0b8ec1d9014354d691c538661bf00bfdb44"
SEED, EPOCHS, BATCH, MAX_LENGTH = 20261122, 1, 16, 128


def batches(model, tokenizer, texts, device, labels=None):
    training = labels is not None
    model.train(training)
    generator = np.random.default_rng(SEED)
    order = generator.permutation(len(texts)) if training else np.arange(len(texts))
    for offset in range(0, len(order), BATCH):
        ids = order[offset : offset + BATCH]
        tokens = tokenizer(
            [texts[i] for i in ids],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.set_grad_enabled(training):
            if training:
                yield model(**tokens, labels=torch.tensor([labels[i] for i in ids], device=device)).loss
            else:
                yield model(**tokens).logits.softmax(-1)[:, 1].detach().cpu().tolist()


def predict(model, tokenizer, texts, device):
    return [v for batch in batches(model, tokenizer, texts, device) for v in batch]


def file_sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fit(root, folder):
    if (folder / "roberta-fitted").exists() or (folder / "roberta-selection.json").exists():
        raise FileExistsError("Use a fresh private directory to refit without overwriting a frozen model")
    src = folder / "roberta-source"
    info = read(src / "source.json")
    weight = next(
        x
        for x in info["siblings"]
        if x["rfilename"] in ("pytorch_model.bin", "model.safetensors") and (src / x["rfilename"]).exists()
    )
    if info["sha"] != REVISION or file_sha(src / weight["rfilename"]) != weight["lfs"]["sha256"]:
        raise ValueError("Invalid base model provenance")
    data = root / "fasttext-classifier/data-text-specialist-20261114-v8/processed"
    partitions = {}
    for split in ("train", "validation"):
        with (data / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            partitions[split] = list(csv.DictReader(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    tokenizer = BertTokenizer(vocab=str(src / "vocab.txt"))
    config = {
        "base": "hfl/chinese-roberta-wwm-ext",
        "revision": REVISION,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch": BATCH,
        "max_tokens": MAX_LENGTH,
        "optimizer": "AdamW",
        "lr": 2e-5,
        "weight_decay": 0.01,
        "validation_fpr_cap": 0.01,
        "threshold_rule": "maximum recall; then fewer FP; then higher threshold",
        "split_sha256": {s: file_sha(data / f"{s}.csv") for s in partitions},
        "rows": {s: len(v) for s, v in partitions.items()},
    }
    freeze(folder / "roberta-protocol.json", config)
    model = BertForSequenceClassification.from_pretrained(src, local_files_only=True, num_labels=2).to(
        device
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train = partitions["train"]
    for i, loss in enumerate(
        batches(model, tokenizer, [r["text"] for r in train], device, [int(r["label"]) for r in train])
    ):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 100 == 0:
            print("Historical training batch", i, flush=True)
    val = partitions["validation"]
    scores = predict(model, tokenizer, [r["text"] for r in val], device)
    labels = np.array([int(r["label"]) for r in val])
    pos, neg = int(sum(labels)), int(sum(labels == 0))
    if min(pos, neg) == 0:
        raise ValueError("Validation requires both classes")
    arr = np.array(scores)
    candidates = []
    for threshold in [math.nextafter(1.0, math.inf), *np.unique(arr)]:
        pred = arr >= threshold
        tp, fp = int(sum(pred & (labels == 1))), int(sum(pred & (labels == 0)))
        if fp / neg <= 0.01:
            candidates.append((tp, -fp, float(threshold)))
    tp, minus_fp, threshold = max(candidates)
    target = folder / "roberta-fitted"
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    freeze(
        folder / "roberta-selection.json",
        {
            "threshold": threshold,
            "tp": tp,
            "fp": -minus_fp,
            "positives": pos,
            "negatives": neg,
            "weights_sha256": file_sha(target / "model.safetensors"),
        },
    )
    print("Validation threshold frozen", threshold, flush=True)


def score(folder):
    meta = read(folder / "roberta-selection.json")
    source = folder / "roberta-fitted"
    if file_sha(source / "model.safetensors") != meta["weights_sha256"]:
        raise ValueError("Fitted model digest differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(source, local_files_only=True).to(device)
    tokenizer = BertTokenizer.from_pretrained(source, local_files_only=True)
    rows = read(folder / "frame.json")
    values = predict(model, tokenizer, [r["text"] for r in rows], device)
    freeze(
        folder / "roberta.json",
        [
            {"id": r["rpid"], "score": s, "blocked": s >= meta["threshold"]}
            for r, s in zip(rows, values, strict=True)
        ],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["fit", "score"])
    ap.add_argument("workspace", type=Path)
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    if a.action == "fit":
        fit(a.workspace, a.folder)
    else:
        score(a.folder)
