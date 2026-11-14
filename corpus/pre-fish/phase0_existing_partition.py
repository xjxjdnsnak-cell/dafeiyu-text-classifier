"""Phase 0: partition ALL existing collected data by comment ctime into
P (high-confidence AI positives) / N (historical human candidates) /
U (source-uncertain, post-cutoff or unlabeled) / quarantine.

Time rule (user-specified): a comment belongs to the historical pool only if its
own ctime (Unix seconds) converted to Beijing time (UTC+8) is strictly before
2026-10-01 00:00:00 CST. Missing/invalid ctime -> quarantine. Original timestamps,
converted times, bvid and rpid are preserved for traceability. Nothing in the
existing datasets is modified.

Reads: the six raw collector exports (comment-level rows with ctime/score/bvid)
and data-rule-ge30-20261114-r2/label-decisions.jsonl (corpus labels + evidence).
Writes: existing-partition.jsonl (one record per raw row) and stats JSON under
out/pre-fish-human-corpus-20261114/.
"""

import csv
import datetime
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
sys.path.insert(0, str(CLASSIFIER / "experiments/text-training-20261114"))

from bili_ai.common import sha256, write_json
from bili_ai.preprocess import normalize
from text_policy import compile_rules, human_exclusion_matches, is_mention_only

HERE = Path(__file__).resolve().parent
OUT = HERE
CUTOFF_BEIJING = datetime.datetime(2026, 10, 1, 0, 0, 0, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
CUTOFF_UNIX = CUTOFF_BEIJING.timestamp()
EXPORTS = [
    ROOT / "out/comment-dataset-20261112/export/all_scored.jsonl",
    ROOT / "out/collector-expand-20261112/export/all_scored.jsonl",
    ROOT / "out/collector-newtopics-20261112/export/all_scored.jsonl",
    ROOT / "out/text-specialist-20261113/collection/export/all_scored.jsonl",
    ROOT / "out/collector-expand-topics-20261114/export/all_scored.jsonl",
    ROOT / "out/dsh-video-20261114/export/all_scored.jsonl",
]
BEIJING = datetime.timezone(datetime.timedelta(hours=8))


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def beijing_from_unix(value):
    return datetime.datetime.fromtimestamp(int(value), tz=datetime.timezone.utc).astimezone(BEIJING)


def parse_ctime(raw):
    """Returns (unix_seconds_int, beijing_iso) or (None, None) if missing/invalid."""
    value = raw.get("ctime")
    if value in (None, "", "0", 0):
        return None, None
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return None, None
    if seconds <= 0 or seconds > 4_102_444_800:  # 2100-01-01 sanity bound
        return None, None
    beijing = beijing_from_unix(seconds)
    return seconds, beijing.isoformat(timespec="seconds")


def main():
    rules = compile_rules(json.loads((CLASSIFIER / "experiments/text-training-20261114/human_exclusion_policy.json").read_text(encoding="utf-8")))
    decisions = {}
    for meta in load_jsonl(CLASSIFIER / "data-rule-ge30-20261114-r2/label-decisions.jsonl"):
        for source in meta.get("sources", []):
            if source.get("rpid"):
                decisions[(str(meta["uid"]), str(source["rpid"]))] = meta

    partition = []
    stats = Counter()
    seen_rpid = set()
    seen_identity = set()
    for path in EXPORTS:
        for row in load_jsonl(path):
            uid, rpid = str(row.get("mid", "")), str(row.get("rpid", ""))
            text = normalize(row.get("training_text") or row.get("text", ""))
            score = int(row["score"]) if str(row.get("score", "")).lstrip("-").isdigit() else None
            seconds, beijing_iso = parse_ctime(row)
            meta = decisions.get((uid, rpid))
            label = meta["label"] if meta else ""
            record = {"uid": uid, "rpid": rpid, "bvid": row.get("bvid", ""), "text": text,
                      "rule_score": score, "ctime_raw": row.get("ctime"),
                      "ctime_beijing": beijing_iso, "in_corpus": meta is not None,
                      "corpus_label": label, "source_file": path.name}
            if (uid, rpid) in seen_rpid:
                record.update(partition="quarantine", reason="duplicate_rpid")
                stats["quarantine/duplicate_rpid"] += 1
                partition.append(record)
                continue
            seen_rpid.add((uid, rpid))
            if seconds is None:
                record.update(partition="quarantine", reason="ctime_missing_or_invalid")
                stats["quarantine/ctime_missing_or_invalid"] += 1
                partition.append(record)
                continue
            record["ctime_unix"] = seconds
            historical = seconds < CUTOFF_UNIX
            if is_mention_only(text) or not text:
                record.update(partition="quarantine", reason="mention_only_or_empty")
                stats["quarantine/mention_only_or_empty"] += 1
            elif meta is not None and label == "1":
                record.update(partition="P", reason="corpus_rule_positive_or_confirmed_ai")
                stats["P"] += 1
            elif meta is not None and label == "0" and score is not None and score >= 30:
                record.update(partition="quarantine", reason="rule_conflict_label0_score_ge30")
                stats["quarantine/rule_conflict"] += 1
            elif not historical:
                record.update(partition="U", reason="post_cutoff" if meta is not None else "unlabeled_post_cutoff")
                stats["U/post_cutoff"] += 1
            elif human_exclusion_matches(text, rules=rules):
                record.update(partition="quarantine", reason="gray_policy_v3")
                stats["quarantine/gray_policy_v3"] += 1
            else:
                identity = (uid, text)
                if identity in seen_identity:
                    record.update(partition="quarantine", reason="duplicate_uid_body")
                    stats["quarantine/duplicate_uid_body"] += 1
                else:
                    seen_identity.add(identity)
                    if meta is not None and label == "0":
                        record.update(partition="N", reason="existing_corpus_human_pre_cutoff")
                        stats["N/existing_corpus"] += 1
                    else:
                        record.update(partition="N", reason="raw_unlabeled_pre_cutoff")
                        stats["N/raw_unlabeled"] += 1
            partition.append(record)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "existing-partition.jsonl").open("w", encoding="utf-8") as handle:
        for record in partition:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    n_rows = [r for r in partition if r["partition"] == "N"]
    year_month = Counter(r["ctime_beijing"][:7] for r in n_rows)
    summary = {
        "cutoff_beijing": CUTOFF_BEIJING.isoformat(),
        "cutoff_unix": int(CUTOFF_UNIX),
        "raw_rows_total": len(partition),
        "stats": {k: v for k, v in sorted(stats.items())},
        "n_pool_rows": len(n_rows),
        "n_pool_by_year_month": dict(sorted(year_month.items())),
        "n_pool_by_video_top": dict(Counter(r["bvid"] for r in n_rows).most_common(30)),
        "n_pool_length": {"1-20": sum(1 for r in n_rows if len(r["text"]) <= 20),
                          "21-60": sum(1 for r in n_rows if 21 <= len(r["text"]) <= 60),
                          "61+": sum(1 for r in n_rows if len(r["text"]) >= 61)},
        "n_pool_score_bands": {"0-9": sum(1 for r in n_rows if (r["rule_score"] or 0) <= 9),
                               "10-19": sum(1 for r in n_rows if r["rule_score"] is not None and 10 <= r["rule_score"] <= 19),
                               "20-29": sum(1 for r in n_rows if r["rule_score"] is not None and 20 <= r["rule_score"] <= 29),
                               "unknown": sum(1 for r in n_rows if r["rule_score"] is None)},
    }
    write_json(OUT / "existing-partition-stats.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
