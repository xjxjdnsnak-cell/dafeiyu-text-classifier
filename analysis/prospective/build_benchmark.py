"""Build and FREEZE the prospective benchmark dataset.

Policy (mechanical, no user review, loop closed):
- label = 1 if raw export score >= 30 else 0 (same threshold family as the corpus);
- mention-only rows excluded; gray policy v3 hits among label-0 rows excluded;
- HARD disjointness against every row of the labeled corpus (data-rule-ge30-20261114-r2):
  a benchmark row is dropped if its uid, normalized text, or bvid appears anywhere in r2
  (r2 covers train/validation/test of v5..v8 and all earlier corpora);
- rows deduplicated by (uid, normalized text) inside the benchmark.

The output directory is written once and then never modified; comments.csv sha256 is
recorded in the manifest and verified by the evaluation script.
"""

import datetime
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
ROOT = CLASSIFIER.parent
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
sys.path.insert(0, str(CLASSIFIER / "experiments/text-training-20261114"))

from bili_ai.common import sha256, write_csv, write_json
from bili_ai.preprocess import normalize
from text_policy import compile_rules, human_exclusion_matches, is_mention_only

HERE = Path(__file__).resolve().parent
OUT = HERE / "benchmark"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
POLICY_V3 = CLASSIFIER / "experiments/text-training-20261114/human_exclusion_policy.json"
FIELDS = ("uid", "username", "text", "label", "score", "bvid", "pool")
EXPORTS = {"ai": HERE / "ai/export/all_scored.jsonl", "topics": HERE / "topics/export/all_scored.jsonl"}


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def main():
    if OUT.exists():
        raise FileExistsError("基准集只构建一次并冻结；如需重建请换目录")
    rules = compile_rules(json.loads(POLICY_V3.read_text(encoding="utf-8")))
    policy_version = json.loads(POLICY_V3.read_text(encoding="utf-8"))["version"]

    corpus_uids, corpus_texts, corpus_videos = set(), set(), set()
    corpus_rows = 0
    for row in load_jsonl(SOURCE / "label-decisions.jsonl"):
        corpus_rows += 1
        corpus_uids.add(str(row["uid"]))
        corpus_texts.add(normalize(row["text"]))
        for source in row.get("sources", []):
            if source.get("bvid"):
                corpus_videos.add(source["bvid"])

    raw_rows, dropped = [], Counter()
    seen = set()
    for pool, path in EXPORTS.items():
        for row in load_jsonl(path):
            uid, text = str(row["mid"]), normalize(row.get("training_text") or row["text"])
            bvid = row.get("bvid", "")
            if is_mention_only(text):
                dropped["mention_only"] += 1
                continue
            identity = (uid, text)
            if identity in seen:
                dropped["duplicate_in_batch"] += 1
                continue
            seen.add(identity)
            score = int(row["score"])
            label = "1" if score >= 30 else "0"
            if label == "0" and human_exclusion_matches(text, rules=rules):
                dropped["gray_v3"] += 1
                continue
            if uid in corpus_uids:
                dropped["uid_in_training_corpus"] += 1
                continue
            if text in corpus_texts:
                dropped["text_in_training_corpus"] += 1
                continue
            if bvid in corpus_videos:
                dropped["video_in_training_corpus"] += 1
                continue
            raw_rows.append({"uid": uid, "username": (row.get("account") or {}).get("uname", ""),
                             "text": text, "label": label, "score": score, "bvid": bvid, "pool": pool})

    rows_by_key = {}
    for row in raw_rows:
        rows_by_key.setdefault((row["uid"], row["text"]), row)
    rows = sorted(rows_by_key.values(), key=lambda r: (r["pool"], r["bvid"], r["uid"], r["text"]))
    conflicts = 0
    groups = defaultdict(set)
    for row in rows:
        groups[(row["uid"], row["text"])].add(row["label"])
    conflicts = sum(1 for labels in groups.values() if len(labels) > 1)
    if conflicts:
        raise ValueError(f"同一 uid+正文出现冲突标签 {conflicts} 组，需显式处理")
    if not rows:
        raise ValueError("基准集为空")
    ai_rows = [r for r in rows if r["label"] == "1"]
    if len(ai_rows) < 50:
        raise ValueError(f"AI 行仅 {len(ai_rows)} 条，不足以评估普通昵称召回；请扩充 AI 池")

    OUT.mkdir(parents=True)
    write_csv(OUT / "comments.csv", ({k: r[k] for k in FIELDS} for r in rows), FIELDS)
    manifest = {
        "frozen": True,
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "comments_sha256": sha256(OUT / "comments.csv"),
        "policy": "label = score>=30; mention-only excluded; gray policy v3 excluded (label 0 only); NO user review, NO correction loop",
        "gray_policy_version": policy_version,
        "disjointness": {
            "rule": "uid, normalized text, or bvid may not appear anywhere in data-rule-ge30-20261114-r2 (covers train/validation/test of v5..v8 and all earlier corpora)",
            "corpus_rows_checked": corpus_rows,
            "corpus_uids": len(corpus_uids), "corpus_texts": len(corpus_texts), "corpus_videos": len(corpus_videos),
            "dropped": dict(dropped),
        },
        "rows": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "videos": len({r["bvid"] for r in rows}),
        "uids": len({r["uid"] for r in rows}),
        "per_video": {r["bvid"]: {"label": dict(Counter(x["label"] for x in rows if x["bvid"] == r["bvid"]))}
                      for r in rows},
        "ai_pool_videos": sorted({r["bvid"] for r in rows if r["pool"] == "ai"}),
        "exports_sha256": {str(p): sha256(p) for p in EXPORTS.values()},
    }
    write_json(OUT / "manifest.json", manifest)
    hints = re.compile(json.load(open(CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json", encoding="utf-8"))["ordinary_nickname_pattern"], re.I)
    ordinary = sum(1 for r in ai_rows if not hints.search(r.get("username", "") or ""))
    print(json.dumps({"rows": len(rows), "labels": manifest["labels"], "videos": manifest["videos"],
                      "uids": manifest["uids"], "dropped": dict(dropped), "ai_rows": len(ai_rows),
                      "ordinary_ai_rows": ordinary, "short_ai_rows": sum(1 for r in ai_rows if len(r["text"]) <= 20),
                      "short_human_rows": sum(1 for r in rows if r["label"] == "0" and len(r["text"]) <= 20)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
