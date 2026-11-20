"""Re-score the source observations for the historical >=30 labeling rule."""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from predict_masked_v9 import mask_mentions
from bili_ai.common import sha256, write_json
from bili_ai.preprocess import normalize

RULE_SHA = "6dde9149a5f27359211ea40cb54de9b09a442be98c44efff0e694f9221436f4d"
VERSION = "liyupi/dafeiyu-cleaner@32866038316f81d77557f20a213b8c3723bf9ce0"


def jsonl(path):
    with path.open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def csvrows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def location(s):
    # The original builder resolves successive observations to the last one.
    # The r2 expansion instead records its export location in provenance[0].
    obs = (s.get("score_observations") or [])[-1:] or (s.get("provenance") or [])[:1]
    if not obs:
        return None
    parts = obs[0]["file"].replace("\\", "/").split("/")
    return "/".join(parts[parts.index("out"):]), int(obs[0]["line"])


def replay(workspace, sources, detector):
    refs = {location(s) for r in sources for s in r["sources"]} - {None}
    raws, hashes = {}, {}
    for name in sorted({name for name, _ in refs}):
        path = workspace / name
        hashes[name] = sha256(path)
        for i, raw in enumerate(jsonl(path), 1):
            if (name, i) in refs:
                raws[name, i] = raw
    ids = list(raws)
    prepared = []
    for ref in ids:
        raw = raws[ref]
        item = {k: raw[k] for k in ("account", "reasons", "text", "root", "up_mid")}
        prepared.append({**item, "masked": mask_mentions(raw["text"])})
    proc = subprocess.run(["node", str(HERE / "replay.cjs"), str(detector)],
                          input=json.dumps(prepared, ensure_ascii=False).encode("utf-8"),
                          stdout=subprocess.PIPE, check=True)
    for ref, result in zip(ids, json.loads(proc.stdout), strict=True):
        raw = raws[ref]
        raw["replay_ok"] = (raw["rule_version"] == VERSION and raw["history_boost"] == 0
                            and result["original"] == raw["score"]
                            and result["originalText"] == raw["text_score"]
                            and result["originalAccount"] == raw["account_feature_score"]
                            and result["reasonsMatch"])
        raw["mask_score"] = result["masked"]
    return raws, hashes


def audit_source(r, raws):
    if not r["sources"]:
        return None, None, "no_source"
    orig, masked = [], []
    for s in r["sources"]:
        raw = raws.get(location(s))
        if (raw is None or not raw["replay_ok"] or raw["score"] != s["rule_score"]
                or str(raw["mid"]) != str(s["uid"]) or str(raw["rpid"]) != str(s["rpid"])
                or normalize(raw.get("training_text") or raw["text"]) != normalize(s["text"])):
            return None, None, "not_reproduced"
        orig.append(raw["score"])
        masked.append(raw["mask_score"])
    # The original labeling rule is "any source reaches 30" for this identity.
    before, after = max(orig), max(masked)
    if before >= 30 > after:
        return before, after, "falls_below_30"
    if before < 30 <= after:
        return before, after, "rises_to_30"
    return before, after, "same_side"


def audit(workspace, out):
    out.mkdir(parents=True, exist_ok=False)
    detector = workspace / "tools/dafeiyu/upstream-detector.js"
    if sha256(detector) != RULE_SHA:
        raise ValueError("旧规则的 SHA256 不匹配")
    base = workspace / "fasttext-classifier/data-text-specialist-20261114-v8"
    source = workspace / "fasttext-classifier/data-rule-ge30-20261114-r2/label-decisions.jsonl"
    decisions = jsonl(source)
    raws, hashes = replay(workspace, decisions, detector)
    human_ids = {r["样本ID"] for r in csvrows(base / "user-corrections-20261114.csv")}
    human_ids.update(r["sample_id"] for r in jsonl(base / "user-reviewed-ai.jsonl"))
    checked_uids = {r["UID"] for r in csvrows(base / "legacy_uid_decisions.csv")}
    checked_comments = {(str(r["uid"]), normalize(r["text"]))
                        for r in jsonl(base / "legacy_comment_judgments.jsonl")}
    counts, records = Counter(), {}
    for r in decisions:
        before, after, flag = audit_source(r, raws)
        protected = (r["sample_id"] in human_ids or str(r["uid"]) in checked_uids
                     or (str(r["uid"]), normalize(r["text"])) in checked_comments
                     or any(s.get("manual_account_label") in ("human", "ai")
                            or s.get("legacy_comment_judgment") in ("human", "ai")
                            for s in r["sources"]))
        records[r["sample_id"]] = dict(sample_id=r["sample_id"], text=r["text"],
            label=r["label"], label_source=r["label_source"], before=before, after=after,
            flag=flag, manual_evidence=protected)
        counts[r["label"], r["label_source"], flag] += 1
    flagged_rules = [m for m in records.values() if m["flag"] == "falls_below_30"
                     and m["label_source"] == "user_rule_score_ge_30"]
    for name, exported in (("source-rule-audit.csv", records.values()),
                           ("flagged-rule-positives.csv", flagged_rules)):
        with (out / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=next(iter(records.values())))
            w.writeheader()
            w.writerows(exported)
    train = [r for r in jsonl(base / "sample-lineage.jsonl") if r["split"] == "train"]
    removed, training_stats = [], Counter()
    for r in train:
        if int(r["label"]) != 1:
            continue
        members = [records[sid] for sid in r["source_metadata_ids"]]
        if not any(m["flag"] == "falls_below_30" for m in members):
            continue
        if (r["user_reviewed_ai"] or r["user_corrected_ai"]
                or any(m["manual_evidence"] for m in members)):
            reason = "keep_human_review"
        elif not all(m["flag"] == "falls_below_30" and m["label"] == "1"
                     and m["label_source"] == "user_rule_score_ge_30" for m in members):
            reason = "keep_inherited_or_other_evidence"
        else:
            reason = "set_aside_in_candidate"
            removed.append({"uid": r["uid"], "text": r["text"], "sample_id": r["sample_id"]})
        training_stats[reason] += 1
    write_json(out / "set-aside.json", removed)
    summary = dict(source_rows=len(records), source_hash=sha256(source), rule_hash=RULE_SHA,
                   exports=hashes, observations=len(raws),
                   observations_with_exact_replay=sum(r["replay_ok"] for r in raws.values()),
                   groups=[dict(label=y, source=s, flag=f, count=n)
                           for (y,s,f),n in sorted(counts.items())],
                   positive_training_rows_with_a_flip=dict(training_stats))
    write_json(out / "audit-summary.json", summary)
    print(summary["observations"], summary["observations_with_exact_replay"], training_stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit(args.workspace.resolve(), args.output.resolve())
