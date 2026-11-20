"""Compare a single rule-flip exclusion arm with its frozen masked-v9 parent."""

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from audit import csvrows, normalize
from predict_mention_trained import MentionTrainedPredictor, simple_invitation
from predict_masked_v9 import MaskedBodyEncoder
from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import select_threshold
from bili_ai.model import positive_scores, train_model
from bili_ai.preprocess import Preprocessor

PARENT_SHA = "4b4c8fdb38aa86b6bc8dce88080accb975664e1b253e3ea6d5f290f00338ecc9"


def key(row):
    return row["uid"], normalize(row["text"])


def summarize(scored):
    metrics = {}
    for name, subset in (("all", scored), ("ordinary", [r for r in scored if r["ordinary"] == "True"]),
                         ("short", [r for r in scored if int(r["length"]) <= 20])):
        metrics[name] = {}
        for y in (0, 1):
            group = [r for r in subset if int(r["label"]) == y]
            pairs = Counter((int(r["v9_mask_trained_hit"]), r["rule_flip_hit"]) for r in group)
            metrics[name][str(y)] = dict(rows=len(group), parent=pairs[1, 1] + pairs[1, 0],
                candidate=pairs[0, 1] + pairs[1, 1], added=pairs[0, 1], lost=pairs[1, 0])
    return metrics


def train(workspace, audit_dir, out):
    out.mkdir(parents=True, exist_ok=False)
    previous = workspace / "out/mention-training-results-20261120"
    parent_meta = read_json(previous / "candidate/candidate.json")
    if parent_meta["model_sha256"] != PARENT_SHA or sha256(previous / "train-masked.txt") != parent_meta["training_input_sha256"]:
        raise ValueError("上一版对照工件与冻结记录不一致")
    previous_predictor = MentionTrainedPredictor(previous / "candidate")
    removal_path = audit_dir / "set-aside.json"
    drop = {key(r) for r in read_json(removal_path)}
    ledger = [r for r in csvrows(previous / "training-ledger.csv") if r["reason"] == "included"]
    random.Random(parent_meta["params"]["seed"]).shuffle(ledger)
    pre = Preprocessor()
    encoder = MaskedBodyEncoder(pre)
    old_lines = (previous / "train-masked.txt").read_text(encoding="utf-8").splitlines()
    removed = set()
    train_file = out / "train-masked.txt"
    with train_file.open("w", encoding="utf-8", newline="\n") as f:
        for r, old in zip(ledger, old_lines, strict=True):
            expected = "__label__" + r["label"] + " " + encoder.encode(r["text"], "", "text")
            if expected != old:
                raise ValueError("复原的排序或编码与实际上一版训练行不符")
            if r["group"] == "original" and key(r) in drop:
                if r["label"] != "1" or key(r) in removed:
                    raise ValueError("隔离行必须是唯一的原正类")
                removed.add(key(r))
            else:
                f.write(old + "\n")
    if removed != drop:
        raise ValueError("待隔离身份与实际移除样本有差异")
    processed = workspace / "fasttext-classifier/data-text-specialist-20261114-v8/processed"
    val_path = processed / "validation.csv"
    if sha256(val_path) != parent_meta["split_sha256"]["validation"]:
        raise ValueError("验证划分已更改")
    model = train_model(train_file, out / "candidate/model.bin", parent_meta["params"])
    val = csvrows(val_path)
    val_scores = positive_scores(model, [encoder.encode(r["text"], "", "text") for r in val])
    selection = select_threshold([int(r["label"]) for r in val], val_scores,
        parent_meta["validation_fpr_cap"], abstain=[simple_invitation(r["text"]) for r in val])
    meta = {k: parent_meta[k] for k in ("encoder", "pass_rule", "params", "dictionary_sha256",
            "preprocess_sha256", "threshold_source", "validation_fpr_cap", "selection_rule",
            "split_sha256", "reviewed_labels_sha256")}
    meta.update(model_sha256=sha256(out / "candidate/model.bin"), threshold=selection["threshold"],
        training_input_sha256=sha256(train_file), training_rows=len(old_lines)-len(removed),
        parent_model_sha256=PARENT_SHA, set_aside_rows=len(removed), set_aside_sha256=sha256(removal_path))
    write_json(out / "candidate/candidate.json", meta)
    new = MentionTrainedPredictor(out / "candidate")
    metrics = {}
    for split in ("validation", "test", "development", "held_out"):
        scored = csvrows(previous / (split + "-scores.csv"))
        rows = [{"text": r["text"]} for r in scored]
        old_preds = previous_predictor.predict_many(rows)
        new_preds = new.predict_many(rows)
        if split == "validation" and [r["ai_style_score"] for r in new_preds] != val_scores:
            raise ValueError("推理入口与训练校准评分不一致")
        for r, before, after in zip(scored, old_preds, new_preds, strict=True):
            if (before["ai_style_score"] != float(r["v9_mask_trained_score"])
                    or int(before["blocked"]) != int(r["v9_mask_trained_hit"])):
                raise ValueError("冻结模型未复现旧的逐条结果")
            r.update(rule_flip_score=after["ai_style_score"], rule_flip_hit=int(after["blocked"]),
                     rule_flip_reason=after["reason"])
        metrics[split] = summarize(scored)
        with (out / (split + "-scores.csv")).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scored[0])
            w.writeheader()
            w.writerows(scored)
    write_json(out / "metrics.json", metrics)
    print(json.dumps({k: m["all"] for k,m in metrics.items()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for flag in ("workspace", "audit-dir", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    args = p.parse_args()
    train(args.workspace.resolve(), args.audit_dir.resolve(), args.output.resolve())
