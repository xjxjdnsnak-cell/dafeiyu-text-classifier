"""One preregistered character CNN vs the frozen mention-masked v9 baseline."""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from predict_masked_v9 import MODEL, THRESHOLD, MaskedV9Predictor, mask_mentions, is_mention_only
from bili_ai.common import sha256
from bili_ai.metrics import select_threshold
from bili_ai.preprocess import normalize

SEEDS = (17, 37, 73)
KERNELS = (1, 2, 3, 5)
BATCH = 128


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_split(data, name, manifest):
    key = name + ".csv"
    path = data / "processed" / key
    if sha256(path) != manifest["files"][key]:
        raise ValueError(f"Frozen split hash differs: {name}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def body(row, training=False):
    text = row["text"]
    if not training:
        if is_mention_only(text):
            return ""
        text = mask_mentions(text)
    return normalize(text)[:4096]


def encode(texts, vocab):
    return [torch.tensor([vocab.get(c, 1) for c in text], dtype=torch.long) for text in texts]


def batch_ids(items, device):
    ids = pad_sequence(items, batch_first=True, padding_value=0)
    if ids.shape[1] < max(KERNELS):
        ids = nn.functional.pad(ids, (0, max(KERNELS) - ids.shape[1]))
    return ids.to(device)


class CharCNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.conv = nn.ModuleList(nn.Conv1d(32, 32, width) for width in KERNELS)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(32 * len(KERNELS), 1))

    def forward(self, ids):
        lengths = ids.ne(0).sum(1)
        vectors = self.embed(ids).transpose(1, 2)
        pooled = []
        for k, conv in zip(KERNELS, self.conv):
            local = conv(vectors).relu()
            valid = (lengths - k + 1).clamp(min=1)
            pos = torch.arange(local.shape[2], device=ids.device)
            pad = pos.unsqueeze(0) >= valid.unsqueeze(1)
            pooled.append(local.masked_fill(pad.unsqueeze(1), 0).amax(2))
        return self.head(torch.cat(pooled, dim=1)).squeeze(1)


def train_one(seed, tokens, labels, vocab, device):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CharCNN(len(vocab) + 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    loss = nn.BCEWithLogitsLoss()
    for epoch in range(10):
        indices = list(range(len(labels)))
        rng.shuffle(indices)
        for start in range(0, len(indices), BATCH):
            take = indices[start:start + BATCH]
            ids = batch_ids([tokens[i] for i in take], device)
            target = labels[take].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss(model(ids), target).backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def cnn_scores(model, tokens, device):
    scores = []
    for i in range(0, len(tokens), BATCH):
        part = tokens[i:i + BATCH]
        values = model(batch_ids(part, device)).sigmoid().cpu().tolist()
        scores.extend(v if len(x) else None for x, v in zip(part, values))
    return scores


def positive(scores, threshold):
    return np.array([s is not None and s >= threshold for s in scores])


def summarize(rows, scores, flags, base, ordinary_regex, k):
    y = np.array([int(r["label"]) for r in rows], dtype=bool)
    short = np.array([1 <= len(r["text"]) <= 20 for r in rows])
    ordinary = np.array([bool(r["username"].strip()) and not ordinary_regex.search(r["username"]) for r in rows])
    values = np.array([s if s is not None else -np.inf for s in scores])
    negative = np.sort(values[~y])[::-1]
    rank_hits = values > negative[k]
    groups = {"all": y, "short": y & short, "ordinary": y & ordinary, "short_ordinary": y & ordinary & short}
    return {
        "negative_rows": int((~y).sum()), "negative_hits": int((flags & ~y).sum()),
        "negative_new": int((flags & ~base & ~y).sum()), "negative_lost": int((~flags & base & ~y).sum()),
        "rank_negative_hits": int((rank_hits & ~y).sum()),
        "positive_groups": {name: {"n": int(g.sum()), "hits": int((flags & g).sum()),
            "new": int((flags & ~base & g).sum()), "lost": int((~flags & base & g).sum()),
            "equal_cost_rank_hits": int((rank_hits & g).sum())} for name, g in groups.items()},
    }


def main(data, out):
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((data / "processed/manifest.json").read_text(encoding="utf-8-sig"))
    protocol = json.loads((data / "protocol.json").read_text(encoding="utf-8-sig"))
    ordinary = re.compile(protocol["ordinary_nickname_pattern"], re.I)
    tr = read_split(data, "train", manifest)
    va = read_split(data, "validation", manifest)
    train_text = [body(r, training=True) for r in tr]
    val_text = [body(r) for r in va]
    counts = Counter("".join(train_text))
    vocab = {char: i + 2 for i, char in enumerate(sorted(c for c, n in counts.items() if n >= 2))}
    save(out / "vocab.json", vocab)
    tr_ids, va_ids = encode(train_text, vocab), encode(val_text, vocab)
    tr_y = torch.tensor([int(r["label"]) for r in tr], dtype=torch.float)
    y_val = [int(r["label"]) for r in va]
    ft = MaskedV9Predictor()
    def ft_scores(rows):
        return [item["ai_style_score"] for item in ft.predict_many([{"text": r["text"]} for r in rows])]
    vf = ft_scores(va)
    base_val = positive(vf, THRESHOLD)
    budget = int(base_val[np.array(y_val) == 0].sum())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    frozen = {"fasttext_threshold": THRESHOLD, "fasttext_sha256": sha256(MODEL),
              "budget_val": budget, "n_negative_val": y_val.count(0), "vocab_size_including_special": len(vocab)+2,
              "split_hashes": {k:manifest["files"][k+".csv"] for k in ("train","validation","test")},
              "torch": str(torch.__version__), "device": str(device), "cnn": {}}
    validation = {"fasttext": summarize(va, vf, base_val, base_val, ordinary, budget)}
    phone = [{"text": r["text"], "fasttext": s} for r, s in zip(va[:100], vf[:100])]
    for seed in SEEDS:
        start = time.perf_counter()
        model = train_one(seed, tr_ids, tr_y, vocab, device)
        sv = cnn_scores(model, va_ids, device)
        seconds = time.perf_counter() - start
        ckpt = out / f"seed{seed}.pt"
        torch.save(model.state_dict(), ckpt)
        sel = select_threshold(y_val, [s if s is not None else 0.0 for s in sv],
                               budget / y_val.count(0), [s is None for s in sv])
        th = sel["threshold"]
        val = summarize(va, sv, positive(sv, th), base_val, ordinary, budget)
        assert val["negative_hits"] <= budget
        validation[str(seed)] = val
        frozen["cnn"][str(seed)] = {"sha256": sha256(ckpt), "threshold": th,
                                    "train_and_val_score_s": round(seconds, 3)}
        if seed == SEEDS[0]:
            for item, score in zip(phone, sv):
                item["cnn"] = score
        print("fitted", seed, "time", round(seconds,1), "val negatives", val["negative_hits"], flush=True)
    save(out / "freeze.json", frozen)
    save(out / "phone-sample.private.json", phone)
    test = read_split(data, "test", manifest)
    te_ids = encode([body(r) for r in test], vocab)
    sf = ft_scores(test)
    base_te = positive(sf, THRESHOLD)
    kt = sum(flag and int(row["label"]) == 0 for row, flag in zip(test, base_te))
    results = {"validation": validation, "test": {"fasttext": summarize(test, sf, base_te, base_te, ordinary, kt)}}
    paired = {"fasttext": sf}
    for seed in SEEDS:
        path = out / f"seed{seed}.pt"
        entry = frozen["cnn"][str(seed)]
        assert sha256(path) == entry["sha256"]
        model = CharCNN(len(vocab)+2).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        scores = cnn_scores(model.eval(), te_ids, device)
        flags = positive(scores, entry["threshold"])
        results["test"][str(seed)] = summarize(test, scores, flags, base_te, ordinary, kt)
        paired[str(seed)] = scores
    save(out / "metrics.json", results)
    with (out / "test-scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["test_row_1based", "label", *paired])
        for i, r in enumerate(test):
            w.writerow([i+1, r["label"], *(scores[i] for scores in paired.values())])
    print(json.dumps(results["test"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.data, args.output)
