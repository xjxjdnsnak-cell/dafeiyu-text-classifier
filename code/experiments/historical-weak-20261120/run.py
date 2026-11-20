"""Two body-only data variants on the frozen v8 partitions. See the accompanying report."""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EXPERIMENTS = REPO / "code/experiments"
sys.path[:0] = [str(REPO / "code/src"), str(EXPERIMENTS / "text-specialist-20261113"),
                str(EXPERIMENTS / "sentence-structure-ablation-20261114")]

from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.preprocess import Preprocessor, normalize
from structure_features import encode_structure_augmented
from text_policy import is_mention_only

CUTOFF = 1790784000  # Oct 1, 2026 00:00 UTC+8, exclusive upper bound
SEED = 42
ENCODERS = {"v8": "txt", "v8.1": "txt", "v9": "txt+sfx", "v9.1": "txt+sfx"}
BASELINES = {"v8": "model/text/model.bin", "v9": "models/sentence-structure/B/model.bin"}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return str(row["uid"]), normalize(row["text"])


class Evidence:
    def __init__(self, decisions, partition):
        self.records = {}
        self.human_uids = set()
        for r in read_jsonl(decisions):
            sources = r.get("sources", [])
            if any(s.get("manual_account_label") == "human" for s in sources):
                self.human_uids.add(str(r["uid"]))
            manual = (r.get("manual_account_review") or {}).get("判定") == "人类"
            manual |= (r.get("legacy_comment_review") or {}).get("label") == "human"
            manual |= any(s.get("legacy_comment_judgment") == "human"
                          or s.get("legacy_comment_label") == "human" for s in sources)
            score = r.get("rule_score", r.get("rule_score_max"))
            self.records[key(r)] = (manual, score, [str(s["rpid"]) for s in sources if s.get("rpid")])
        self.times = {}
        self.times_by_reply = {}
        for r in read_jsonl(partition):
            if r.get("ctime_unix") is None:
                continue
            timestamp = int(r["ctime_unix"])
            self.times.setdefault(key(r), timestamp)
            self.times_by_reply.setdefault(str(r["rpid"]), timestamp)

    def group(self, row):
        if int(row["label"]) == 1:
            return "positive"
        identity = key(row)
        manual, score, replies = self.records.get(identity, (False, None, []))
        if manual or identity[0] in self.human_uids:
            return "human_evidence"
        timestamp = self.times.get(identity)
        if timestamp is None:
            timestamp = next((self.times_by_reply[i] for i in replies if i in self.times_by_reply), None)
        if timestamp is not None and timestamp < CUTOFF:
            return "historical_existing"
        if score is not None and int(score) < 30:
            return "weak_low_rule"
        return "unresolved"


def historical_rows(corpus, lineage, benchmark):
    bench = read_rows(benchmark)
    uids = {r["uid"] for r in bench}
    videos = {r["bvid"] for r in bench}
    texts = {normalize(r["text"]) for r in bench}
    aliases = set()
    for r in read_jsonl(lineage):
        texts.add(normalize(r["text"]))
        if r["split"] in ("validation", "test"):
            uids.add(str(r["uid"]))
            aliases.update(map(str, r["associated_uids"]))
            videos.update(r["bvids"])
    eligible, excluded = [], Counter()
    for r in read_rows(corpus):
        if r["source"] != "new_collection":
            continue
        uid, text = key(r)
        if uid in uids:
            excluded["uid"] += 1
        elif text in texts:
            excluded["normalized_text"] += 1
        elif r["bvid"] in videos:
            excluded["video"] += 1
        elif r["training_pool_eligible"] != "True":
            excluded["corpus_ineligible"] += 1
        elif uid in aliases:
            excluded["associated_uid"] += 1
        else:
            if int(r["ctime_unix"]) >= CUTOFF or r["proposed_label"] != "0":
                raise ValueError("unexpected historical candidate")
            eligible.append({**r, "label": "0"})
    if len({key(r) for r in eligible}) != len(eligible):
        raise ValueError("duplicate historical training identity")
    return eligible, dict(excluded)


def shared_training(train, evidence, extra, out):
    rng = np.random.RandomState(SEED)
    ledger = []
    selected = []
    counts = Counter()
    kept = Counter()
    for r in train + extra:
        group = "historical_new" if r.get("source") == "new_collection" else evidence.group(r)
        keep = rng.rand() < 0.5 if group == "weak_low_rule" else group != "unresolved"
        counts[group] += 1
        ledger.append({"uid": r["uid"], "text": r["text"], "label": r["label"],
                       "group": group, "kept": int(keep)})
        if keep:
            kept[group] += 1
            selected.append(r)
    # One draw per weak row; both feature encoders consume the very same shuffled list.
    random.Random(SEED).shuffle(selected)
    write_rows(out / "training-ledger.csv", ledger)
    return selected, {"available": dict(counts), "retained": dict(kept),
                      "total_retained": len(selected), "seed": SEED, "weak_keep_probability": 0.5}


def encode(text, encoder, pre):
    if encoder == "txt+sfx":
        return encode_structure_augmented(text, pre)
    if encoder == "txt":
        return pre.encode(text, "", "text")
    raise ValueError(encoder)


def eligible_features(rows, encoder, pre):
    lines = [encode(r["text"], encoder, pre) for r in rows]
    if any(line == "field_empty" or is_mention_only(r["text"]) for line, r in zip(lines, rows)):
        raise ValueError("frozen partition contains an ineligible body")
    return lines


def baseline_records():
    v8 = read_json(REPO / "model/text/model.bin.json")
    v9 = read_json(REPO / "reports/sentence-structure/audit-threshold-record.json")
    records = {"v8": {"threshold": v8["threshold"], "model_sha256": v8["model_sha256"]},
               "v9": {"threshold": v9["threshold_fpr_1pct"],
                      "model_sha256": v9["evaluation_conditions"]["model_sha256"]}}
    for name, meta in records.items():
        if sha256(REPO / BASELINES[name]) != meta["model_sha256"]:
            raise ValueError("frozen baseline model mismatch: " + name)
    return records


def model_path(out, name):
    return REPO / BASELINES[name] if name in BASELINES else out / name / "model.bin"


def summarize(scored, ordinary_indices):
    subsets = {"positive": [], "negative": [], "ordinary_positive": [], "short_positive": [],
               "historical_existing": [], "human_evidence": []}
    for i, r in enumerate(scored):
        if int(r["label"]):
            subsets["positive"].append(i)
            if i in ordinary_indices:
                subsets["ordinary_positive"].append(i)
            if len(r["text"]) <= 20:
                subsets["short_positive"].append(i)
        else:
            subsets["negative"].append(i)
            if r["group"] in ("historical_existing", "human_evidence"):
                subsets[r["group"]].append(i)
    results = {}
    for group, indices in subsets.items():
        results[group] = {"rows": len(indices)}
        for name in ENCODERS:
            hits = {i for i in indices if scored[i][name + "_hit"]}
            entry = {"hits": len(hits)}
            if name.endswith(".1"):
                old = {i for i in indices if scored[i][name[:2] + "_hit"]}
                entry.update(added=len(hits - old), lost=len(old - hits))
            results[group][name] = entry
    return results


def run(workspace, out):
    data = workspace / "fasttext-classifier/data-text-specialist-20261114-v8"
    processed = data / "processed"
    corpus = workspace / "out/pre-fish-human-corpus-20261114"
    decisions = workspace / "fasttext-classifier/data-rule-ge30-20261114-r2/label-decisions.jsonl"
    benchmark = workspace / "out/prospective-benchmark-20261114/benchmark/comments.csv"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("select a new or empty output directory")
    manifest = read_json(processed / "manifest.json")

    def partition(name):
        path = processed / (name + ".csv")
        if sha256(path) != manifest["files"][path.name]:
            raise ValueError("frozen partition mismatch: " + name)
        return read_rows(path)

    pre = Preprocessor()
    if pre.config["dictionary_sha256"] != manifest["dictionary_sha256"] or (
            sha256(pre.assets / "preprocess.json") != manifest["preprocess_sha256"]):
        raise ValueError("frozen preprocessing mismatch")
    pattern = re.compile(read_json(data / "protocol.json")["ordinary_nickname_pattern"], re.I)
    records = baseline_records()
    evidence = Evidence(decisions, corpus / "existing-partition.jsonl")
    extra, excluded = historical_rows(corpus / "historical-human-candidates.csv",
                                      data / "sample-lineage.jsonl", benchmark)
    training, composition = shared_training(partition("train"), evidence, extra, out)
    composition["new_exclusions"] = excluded
    composition["new_videos"] = len({r["bvid"] for r in extra})
    val = partition("validation")
    val_features = {}
    for name in ("v8.1", "v9.1"):
        encoder = ENCODERS[name]
        folder = out / name
        folder.mkdir()
        lines = eligible_features(training, encoder, pre)
        trainfile = folder / "train.txt"
        with trainfile.open("w", encoding="utf-8", newline="\n") as f:
            for row, line in zip(training, lines, strict=True):
                f.write(f"__label__{row['label']} {line}\n")
        path = model_path(out, name)
        model = train_model(trainfile, path, DEFAULT_PARAMS)
        features = eligible_features(val, encoder, pre)
        val_features[encoder] = features
        scores = positive_scores(model, features)
        selected = select_threshold([int(r["label"]) for r in val], scores, 0.01)
        records[name] = {"threshold": selected["threshold"], "encoder": encoder,
                         "model_sha256": sha256(path), "training_input_sha256": sha256(trainfile),
                         "validation_selection": selected}
        print(name, len(training), "threshold", selected["threshold"], flush=True)
    freeze = {"models": records, "training": composition, "params": DEFAULT_PARAMS,
              "cutoff_exclusive": "2026-10-01T00:00:00+08:00", "validation_negative_budget": 0.01,
              "split_sha256": {s: manifest["files"][s + ".csv"] for s in ("train", "validation", "test")},
              "ledger_sha256": sha256(out / "training-ledger.csv")}
    write_json(out / "frozen-before-test.json", freeze)
    frozen = read_json(out / "frozen-before-test.json")["models"]
    summary = {}
    for split, rows in (("validation", val), ("test", partition("test"))):
        features = val_features if split == "validation" else {
            enc: eligible_features(rows, enc, pre) for enc in val_features}
        scored = [{**r, "group": evidence.group(r)} for r in rows]
        for name, encoder in ENCODERS.items():
            path = model_path(out, name)
            if sha256(path) != frozen[name]["model_sha256"]:
                raise ValueError("model changed after freeze")
            scores = positive_scores(load_model(path), features[encoder])
            for row, score in zip(scored, scores, strict=True):
                row[name + "_score"] = score
                row[name + "_hit"] = int(score >= frozen[name]["threshold"])
        ordinary = {i for i, r in enumerate(rows) if not pattern.search(r["username"])}
        summary[split] = summarize(scored, ordinary)
        write_rows(out / (split + "-scores.csv"), scored)
    write_json(out / "results.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def predict(out, name, text):
    meta = read_json(out / "frozen-before-test.json")["models"][name]
    path = model_path(out, name)
    if sha256(path) != meta["model_sha256"]:
        raise ValueError("candidate model hash mismatch")
    score = None
    if normalize(text) and not is_mention_only(text):
        pre = Preprocessor()
        score = positive_scores(load_model(path), [encode(text, meta["encoder"], pre)])[0]
    return {"score": score, "threshold": meta["threshold"],
            "hit": score is not None and score >= meta["threshold"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="command", required=True)
    fit = cmds.add_parser("train")
    fit.add_argument("--workspace", type=Path, required=True)
    single = cmds.add_parser("predict")
    single.add_argument("--model", choices=["v8.1", "v9.1"], required=True)
    single.add_argument("--text", required=True)
    for sub in (fit, single):
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        run(args.workspace.resolve(), args.output.resolve())
    else:
        print(json.dumps(predict(args.output.resolve(), args.model, args.text), ensure_ascii=False))
