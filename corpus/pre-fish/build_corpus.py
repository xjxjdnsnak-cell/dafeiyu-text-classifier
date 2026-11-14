"""Build the pre-outbreak historical human-negative candidate corpus.

Inputs:
- Phase-0 existing-partition.jsonl (existing data classified by ctime into P/N/U/quarantine)
- The new collection export (out/pre-fish-human-corpus-20261114/collection/export/all_scored.jsonl)

Outputs (all new files; nothing existing is modified):
- historical-human-candidates.csv : N-pool rows (existing corpus + new collection),
  each with proposed_label=0, label_source=historical_timestamp, authorship_verified=false,
  ctime traceability, and training_pool_eligible flags.
- quarantine.csv : every excluded row with a reason (time missing/invalid, rule conflict,
  gray policy, mention-only, duplicates, eval overlaps).
- existing-partition-stats.json / collection-report.md / manifest.json.

Training-eligibility rule (user-specified): a candidate may enter a FUTURE training
pool only if it does not overlap the old validation set, the old test set, or the
frozen prospective benchmark (uid / normalized body / bvid). Rows overlapping the old
TRAINING set are eligible but flagged (counted separately, never presented as new
increment).
"""

import csv
import datetime
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
sys.path.insert(0, str(CLASSIFIER / "experiments/text-training-20261114"))

from bili_ai.common import sha256, write_json
from bili_ai.preprocess import normalize
from text_policy import compile_rules, human_exclusion_matches, is_mention_only

HERE = Path(__file__).resolve().parent
BENCH = ROOT / "out/prospective-benchmark-20261114/benchmark"
COLLECTION = HERE / "collection/export/all_scored.jsonl"
CANDIDATE_FIELDS = ("candidate_id", "source", "uid", "rpid", "bvid", "text", "length_group",
                    "rule_score", "ctime_unix", "ctime_beijing", "topic_group",
                    "proposed_label", "label_source", "authorship_verified",
                    "training_pool_eligible", "overlap_flags")
BEIJING = datetime.timezone(datetime.timedelta(hours=8))


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def length_group(text):
    if len(text) <= 20:
        return "1-20"
    if len(text) <= 60:
        return "21-60"
    return "61+"


def beijing_iso(seconds):
    return datetime.datetime.fromtimestamp(int(seconds), tz=datetime.timezone.utc).astimezone(BEIJING).isoformat(timespec="seconds")


def main():
    rules = compile_rules(json.loads((CLASSIFIER / "experiments/text-training-20261114/human_exclusion_policy.json").read_text(encoding="utf-8")))
    partition = load_jsonl(HERE / "existing-partition.jsonl")
    discovery = json.loads((HERE / "video-discovery.json").read_text(encoding="utf-8"))
    topic_by_bvid = {p["bvid"]: p["topic"] for p in discovery["picks"]}

    # v8 partitions + frozen benchmark exclusion sets
    lineage_split, lineage_bvids = {}, {}
    for row in load_jsonl(CLASSIFIER / "data-text-specialist-20261114-v8/sample-lineage.jsonl"):
        identity = (str(row["uid"]), normalize(row["text"]))
        lineage_split[identity] = row.get("split", "")
        lineage_bvids[identity] = set(row.get("bvids", []) or [])
    bench_uids, bench_texts, bench_bvids = set(), set(), set()
    with (BENCH / "comments.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            bench_uids.add(str(row["uid"]))
            bench_texts.add(normalize(row["text"]))
            bench_bvids.add(row["bvid"])

    def overlap_flags(row, bvids):
        identity = (str(row["uid"]), normalize(row["text"]))
        flags = []
        split = lineage_split.get(identity, "")
        if split == "train":
            flags.append("v8_train")
        elif split == "validation":
            flags.append("v8_validation")
        elif split == "test":
            flags.append("v8_test")
        if str(row["uid"]) in bench_uids:
            flags.append("benchmark_uid")
        if normalize(row["text"]) in bench_texts:
            flags.append("benchmark_text")
        if bvids & bench_bvids:
            flags.append("benchmark_bvid")
        return split, flags

    candidates, quarantine = [], []
    candidate_ids = set()

    def candidate_id(row, source):
        return ("EX-" if source == "existing_corpus" else "NC-") + \
            hashlib.sha256((row["uid"] + "|" + row["text"] + "|" + str(row.get("rpid", ""))).encode("utf-8")).hexdigest()[:20]

    # --- existing N pool (phase-0 classified) ---
    existing_n = [r for r in partition if r["partition"] == "N"]
    existing_seen_identity = set()
    for row in existing_n:
        identity = (str(row["uid"]), normalize(row["text"]))
        if identity in existing_seen_identity:
            quarantine.append({"source": "existing_corpus", "uid": row["uid"], "rpid": row["rpid"],
                               "bvid": row["bvid"], "text": row["text"], "reason": "duplicate_uid_body_existing",
                               "rule_score": row.get("rule_score"), "ctime_beijing": row.get("ctime_beijing")})
            continue
        existing_seen_identity.add(identity)
        split, flags = overlap_flags(row, lineage_bvids.get(identity, set()))
        flags.append("existing_corpus_row")
        eligible = not any(f in ("v8_validation", "v8_test", "benchmark_uid", "benchmark_text", "benchmark_bvid") for f in flags)
        text = row["text"]
        candidates.append({
            "candidate_id": candidate_id(row, "existing_corpus"), "source": "existing_corpus",
            "uid": row["uid"], "rpid": row["rpid"], "bvid": row["bvid"], "text": text,
            "length_group": length_group(text), "rule_score": row.get("rule_score"),
            "ctime_unix": row.get("ctime_unix"), "ctime_beijing": row.get("ctime_beijing"),
            "topic_group": "existing_corpus", "proposed_label": 0,
            "label_source": "historical_timestamp", "authorship_verified": False,
            "training_pool_eligible": eligible, "overlap_flags": "|".join(flags) or "none",
        })

    # --- new collection rows ---
    new_rows = load_jsonl(COLLECTION)
    seen_rpid = set()
    new_seen_identity = set()
    existing_identities = {(str(r["uid"]), normalize(r["text"])) for r in existing_n}
    for row in new_rows:
        uid, rpid = str(row.get("mid", "")), str(row.get("rpid", ""))
        text = normalize(row.get("training_text") or row.get("text", ""))
        score = int(row["score"]) if str(row.get("score", "")).lstrip("-").isdigit() else None
        bvid = row.get("bvid", "")
        ctime_raw = row.get("ctime")
        base = {"source": "new_collection", "uid": uid, "rpid": rpid, "bvid": bvid, "text": text,
                "rule_score": score, "ctime_raw": ctime_raw}
        try:
            seconds = int(ctime_raw) if ctime_raw not in (None, "", "0", 0) else None
        except (TypeError, ValueError):
            seconds = None
        if seconds is None or seconds <= 0 or seconds > 4_102_444_800:
            quarantine.append({**base, "reason": "ctime_missing_or_invalid"})
            continue
        beijing = beijing_iso(seconds)
        historical = seconds < int(datetime.datetime(2026, 10, 1, 0, 0, 0, tzinfo=BEIJING).timestamp())
        if (uid, rpid) in seen_rpid:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "duplicate_rpid"})
            continue
        seen_rpid.add((uid, rpid))
        if is_mention_only(text) or not text:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "mention_only_or_empty"})
            continue
        if not historical:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "post_cutoff"})
            continue
        if score is not None and score >= 30:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "rule_conflict_score_ge30"})
            continue
        if human_exclusion_matches(text, rules=rules):
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "gray_policy_v3"})
            continue
        identity = (uid, text)
        if identity in new_seen_identity:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "duplicate_uid_body_new"})
            continue
        new_seen_identity.add(identity)
        if identity in existing_identities:
            quarantine.append({**base, "ctime_beijing": beijing, "reason": "duplicate_uid_body_existing"})
            continue
        split, flags = overlap_flags({**row, "uid": uid, "text": text}, lineage_bvids.get(identity, set()) | {bvid})
        eligible = not any(f in ("v8_validation", "v8_test", "benchmark_uid", "benchmark_text", "benchmark_bvid") for f in flags)
        topic = topic_by_bvid.get(bvid, "unknown")
        cid = "NC-" + hashlib.sha256((uid + "|" + text + "|" + rpid).encode("utf-8")).hexdigest()[:20]
        candidates.append({
            "candidate_id": cid, "source": "new_collection",
            "uid": uid, "rpid": rpid, "bvid": bvid, "text": text,
            "length_group": length_group(text), "rule_score": score,
            "ctime_unix": seconds, "ctime_beijing": beijing,
            "topic_group": topic, "proposed_label": 0,
            "label_source": "historical_timestamp", "authorship_verified": False,
            "training_pool_eligible": eligible, "overlap_flags": "|".join(flags) or "none",
        })

    # candidate_id collision check
    ids = [c["candidate_id"] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate_id 冲突")

    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "historical-human-candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)
    with (HERE / "quarantine.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "uid", "rpid", "bvid", "text", "reason",
                                                    "rule_score", "ctime_raw", "ctime_beijing"))
        writer.writeheader()
        for row in quarantine:
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})

    new_candidates = [c for c in candidates if c["source"] == "new_collection"]
    stats = {
        "cutoff_beijing": "2026-10-01T00:00:00+08:00",
        "new_collection_raw_rows": len(new_rows),
        "new_candidates": len(new_candidates),
        "new_candidates_by_topic": dict(Counter(c["topic_group"] for c in new_candidates)),
        "new_candidates_by_month": dict(sorted(Counter(c["ctime_beijing"][:7] for c in new_candidates).items())),
        "new_candidates_by_video": {vid: sum(1 for c in new_candidates if c["bvid"] == vid)
                                    for vid in sorted({c["bvid"] for c in new_candidates})},
        "new_candidates_length": dict(Counter(c["length_group"] for c in new_candidates)),
        "new_candidates_score_bands": {"0-9": sum(1 for c in new_candidates if (c["rule_score"] or 0) <= 9),
                                       "10-19": sum(1 for c in new_candidates if c["rule_score"] is not None and 10 <= c["rule_score"] <= 19),
                                       "20-29": sum(1 for c in new_candidates if c["rule_score"] is not None and 20 <= c["rule_score"] <= 29)},
        "existing_n_pool_rows": len(existing_n),
        "existing_duplicates": sum(1 for q in quarantine if q["reason"] == "duplicate_uid_body_existing"),
        "quarantine_by_reason": dict(Counter(q["reason"] for q in quarantine)),
        "training_pool_eligible": {"new_collection": sum(1 for c in new_candidates if c["training_pool_eligible"]),
                                   "existing_corpus": sum(1 for c in candidates if c["source"] == "existing_corpus" and c["training_pool_eligible"])},
        "overlap_flag_counts": dict(Counter(f for c in candidates for f in c["overlap_flags"].split("|") if f != "none")),
        "near_duplicate_note": "完全重复已按 uid+规范化正文去重；近重复（>=0.90 相似度）仅在报告中给出规模估计，不做排除",
    }
    write_json(HERE / "build-stats.json", stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
