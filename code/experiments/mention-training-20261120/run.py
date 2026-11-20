"""Train one masked v9 candidate, freeze on historical validation, then compare."""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
from predict_masked_v9 import MODEL, MODEL_SHA256, THRESHOLD, MaskedBodyEncoder, MaskedV9Predictor
from predict_mention_trained import MentionTrainedPredictor, simple_invitation
from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import select_threshold
from bili_ai.model import DEFAULT_PARAMS, Predictor, positive_scores, train_model
from bili_ai.preprocess import ASSETS, Preprocessor, normalize
from structure_features import encode_structure_augmented
from text_policy import is_mention_only

ARMS = ("v8", "v9", "v9_mask_only", "v9_mask_trained")


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def reviews(directory):
    source = {r["rpid"]: r for r in csv_rows(directory / "new-video-short-comments.csv")}
    groups = {"development": [], "held_out": []}
    seen = set()
    for r in csv_rows(directory / "71条短评-人工判别.csv"):
        reply = r["评论ID"]
        s = source[reply]
        if reply in seen or (s["text"], s["split"]) != (r["正文"], r["原分组"]):
            raise ValueError("复核表与原采集行不一致")
        seen.add(reply)
        groups[s["split"]].append(dict(uid=s["uid"], username=s["nickname"], bvid=s["bvid"],
                                       text=s["text"], label={"AI": 1, "真人": 0}[r["你的判断"]],
                                       row_id=r["序号"], rpid=reply))
    if seen != set(source) or [len(v) for v in groups.values()] != [50, 21]:
        raise ValueError("短评复核分组或数量发生了变化")
    return groups


def check_groups(lineage, development, held):
    old_uids, old_videos = set(), set()
    with lineage.open(encoding="utf-8-sig") as f:
        for line in f:
            r = json.loads(line)
            old_uids.add(str(r["uid"]))
            old_uids.update(map(str, r["associated_uids"]))
            old_videos.update(r["bvids"])
    for field, old in (("uid", old_uids), ("bvid", old_videos)):
        dev_values = {r[field] for r in development}
        held_values = {r[field] for r in held}
        if dev_values & held_values or old & (dev_values | held_values):
            raise ValueError("复核样本分组发生重合: " + field)


def prepare(train, development, protected, encoder):
    lines, known, ledger = [], {}, []
    # Keep the original v9 multiplicities. De-duplicate the added reviews by
    # the complete masked encoding, including its sentence-structure tokens.
    for group, rows in (("original", train), ("review", development)):
        for row in rows:
            line = encoder.encode(row["text"], "", "text")
            label = int(row["label"])
            reason = "included"
            if line in protected:
                reason = "protected_input"
            elif line == "field_empty":
                reason = "empty_input"
            elif group == "review" and line in known:
                if known[line] != {label}:
                    raise ValueError("新增复核标签在遮蔽后与训练标签冲突")
                reason = "duplicate_input"
            if reason == "included":
                lines.append(f"__label__{label} {line}")
                known.setdefault(line, set()).add(label)
            ledger.append(dict(group=group, row_id=row.get("row_id", ""), uid=row["uid"],
                               text=row["text"], label=label, reason=reason))
    return lines, ledger


class RawStructureEncoder:
    def __init__(self, pre):
        self.pre = pre

    def encode(self, text, username, mode):
        if not normalize(text) or is_mention_only(text):
            return "field_empty"
        return encode_structure_augmented(text, self.pre)


def evaluate(models, rows, ordinary, included_ids):
    # All four inference paths receive only the body.
    inputs = [{"text": r["text"]} for r in rows]
    scored = [dict(row_id=r.get("row_id", str(i)), rpid=r.get("rpid", ""), text=r["text"],
                   label=int(r["label"]), included_review=r.get("row_id", "") in included_ids,
                   ordinary=not ordinary.search(r["username"]), length=len(r["text"]))
              for i, r in enumerate(rows, 1)]
    for name, model in models.items():
        for row, result in zip(scored, model.predict_many(inputs), strict=True):
            row[name + "_score"] = result["ai_style_score"]
            row[name + "_hit"] = int(result["blocked"])
            if name == "v9_mask_trained":
                row["candidate_reason"] = result["reason"]
                row["candidate_model_hit"] = result["model_label"]
    subsets = {"all": scored, "ordinary": [r for r in scored if r["ordinary"]],
               "short": [r for r in scored if r["length"] <= 20]}
    metrics = {}
    for group, part in subsets.items():
        group_results = {}
        for label in (0, 1):
            subset = [r for r in part if r["label"] == label]
            group_results[str(label)] = {"rows": len(subset), "hits": {
                name: sum(r[name + "_hit"] for r in subset) for name in ARMS}}
        metrics[group] = group_results
    return scored, metrics


def run(workspace, out):
    data = workspace / "fasttext-classifier/data-text-specialist-20261114-v8"
    review = workspace / "out/ordinary-short-evidence-20261120"
    processed = data / "processed"
    manifest = read_json(processed / "manifest.json")
    if out.exists() and any(out.iterdir()):
        raise ValueError("请使用空输出目录")
    out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for s in ("train", "validation", "test"):
        file = processed / (s + ".csv")
        if sha256(file) != manifest["files"][file.name]:
            raise ValueError("原冻结划分已变动: " + s)
        splits[s] = csv_rows(file)
    v8_path = REPO / "model/text/model.bin"
    v8 = Predictor(v8_path, assets=ASSETS)
    if sha256(MODEL) != MODEL_SHA256 or v8.metadata["params"] != DEFAULT_PARAMS:
        raise ValueError("基线工件或训练参数发生变化")
    pre = Preprocessor()
    masked = MaskedBodyEncoder(pre)
    reviewed = reviews(review)
    check_groups(data / "sample-lineage.jsonl", reviewed["development"], reviewed["held_out"])
    # Use eval inputs only to exclude train overlap. Eval scores and label
    # readouts are computed only after the candidate threshold has been saved.
    protected = {masked.encode(r["text"], "", "text") for pool in
                 (splits["validation"], splits["test"], reviewed["held_out"]) for r in pool}
    lines, ledger = prepare(splits["train"], reviewed["development"], protected, masked)
    random.Random(DEFAULT_PARAMS["seed"]).shuffle(lines)
    train_file = out / "train-masked.txt"
    with train_file.open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(line + "\n" for line in lines)
    save_csv(out / "training-ledger.csv", ledger)
    folder = out / "candidate"
    model_path = folder / "model.bin"
    model = train_model(train_file, model_path, DEFAULT_PARAMS)
    val = splits["validation"]
    feats = [masked.encode(r["text"], "", "text") for r in val]
    if "field_empty" in feats:
        raise ValueError("冻结验证集意外包含无正文样本")
    scores = positive_scores(model, feats)
    exemption = [simple_invitation(r["text"]) for r in val]
    sel = select_threshold([int(r["label"]) for r in val], scores, 0.01, abstain=exemption)
    counts = Counter((r["group"], r["reason"], r["label"]) for r in ledger)
    meta = dict(model_sha256=sha256(model_path), threshold=sel["threshold"],
                threshold_source="historical validation only, 1% negative-trigger cap",
                validation_fpr_cap=0.01, selection_rule=sel["selection_rule"],
                encoder="MaskedBodyEncoder(txt+sfx)", pass_rule="simple_invitation",
                dictionary_sha256=pre.config["dictionary_sha256"],
                preprocess_sha256=sha256(pre.assets / "preprocess.json"),
                training_input_sha256=sha256(train_file), params=DEFAULT_PARAMS, training_rows=len(lines),
                training_composition=[dict(group=g, reason=r, label=y, count=n)
                                      for (g, r, y), n in sorted(counts.items())],
                split_sha256={s: manifest["files"][s + ".csv"] for s in splits},
                reviewed_labels_sha256=sha256(review / "71条短评-人工判别.csv"))
    write_json(folder / "candidate.json", meta)
    new = MentionTrainedPredictor(folder)
    v9 = Predictor(MODEL, mode="text", threshold=THRESHOLD, assets=ASSETS)
    v9.pre = RawStructureEncoder(v9.pre)
    models = dict(zip(ARMS, (v8, v9, MaskedV9Predictor(), new), strict=True))
    ordinary = re.compile(read_json(data / "protocol.json")["ordinary_nickname_pattern"], re.I)
    include = {r["row_id"] for r in ledger if r["group"] == "review" and r["reason"] == "included"}
    report = {}
    for split, pool in (("validation", val), ("test", splits["test"]), *reviewed.items()):
        result, report[split] = evaluate(models, pool, ordinary, include)
        if split == "validation":
            assert [r["v9_mask_trained_score"] for r in result] == scores
            assert [r["v9_mask_trained_hit"] for r in result] == [
                int(s >= meta["threshold"] and not ex) for s, ex in zip(scores, exemption, strict=True)]
        save_csv(out / (split + "-scores.csv"), result)
    write_json(out / "metrics.json", report)
    print(json.dumps({"training_rows": len(lines), "threshold": meta["threshold"],
                      "counts": {k: v["all"] for k, v in report.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace.resolve(), args.output.resolve())
