"""Fold the 2026-11-14 expansion into the labeled rule-ge30 source (r2).

Semantics follow merge_population (score30_dataset.py) with one change the user
explicitly requested: new non-qualifying rows are KEPT as human (the topics batch
exists to widen human coverage). The dsh video contributes only its >=30 rows —
its human rows never load because the AI-only export is the declared source.
Existing rows keep their labels unless a new observation scores >= 30 (rule wins,
as in the original score30 build).
"""

import json
import sys
from collections import Counter
from pathlib import Path

CLASSIFIER = Path(__file__).resolve().parents[2]
ROOT = CLASSIFIER.parent
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

from bili_ai.common import sha256, write_json
from bili_ai.preprocess import normalize
from score30_dataset import read_jsonl, review_markdown, review_summary, sample_ai, stable_id
from text_policy import EMOTES_PATH, POLICY_PATH, human_exclusion_matches, is_mention_only
from text_specialist_dataset import load_legacy_judgments, load_records, read_uid_reviews

SOURCE = CLASSIFIER / "data-rule-ge30-20261114"
OUT = CLASSIFIER / "data-rule-ge30-20261114-r2"
REVIEW_OUT = ROOT / "out/rule-ge30-review-20261114-r2"
NEW_EXPORTS = (
    ROOT / "out/collector-expand-topics-20261114/export/all_scored.jsonl",
    ROOT / "out/dsh-video-20261114/export/ai-only-ge30.jsonl",
)
MANUAL_PATH = Path("E:/human-review-uid-decisions (2).csv")
FIELDS = ("uid", "username", "text", "label")
THRESHOLD = 30


def observation_from(row):
    provenance = row["provenance"][0]
    return {"file": provenance["file"], "line": provenance["line"],
            "score": int(row["rule_score"]), "reasons": row["rule_reasons"],
            "collected_at": provenance.get("collected_at"),
            "export_label": row.get("source_weak_comment_label"), "text": row["text"]}


def merge_source(existing, new_rows):
    """Returns (records, excluded, moved, merged_count). Identities are (uid, normalized text)."""
    records = [dict(row) for row in existing]
    index = {}
    for row in records:
        identity = (str(row["uid"]), normalize(row["text"]))
        if identity in index:
            raise ValueError("源数据内部存在重复 uid+正文")
        index[identity] = row
    excluded, moved, merged = [], [], 0
    for row in new_rows:
        identity = (str(row["uid"]), normalize(row["text"]))
        text = row["text"]
        target = index.get(identity)
        if target is not None:
            merged += 1
            observation = observation_from(row)
            target.setdefault("sources", []).append(row)
            target.setdefault("score_observations", []).append(observation)
            if int(target["label"]) == 0 and observation["score"] >= THRESHOLD:
                target["label"] = "1"
                target["previous_label"] = "0"
                target["label_source"] = "user_rule_score_ge_30"
                target["matches_new_rule"] = True
                moved.append(target)
            continue
        if is_mention_only(text):
            excluded.append({"sample_id": stable_id(identity), "uid": identity[0], "text": text,
                             "reason": "mention_only", "sources": [row]})
            continue
        score = int(row["rule_score"])
        if score >= THRESHOLD:
            label, label_source = "1", "user_rule_score_ge_30"
        elif human_exclusion_matches(text):
            excluded.append({"sample_id": stable_id(identity), "uid": identity[0], "text": text,
                             "reason": "human_expression", "sources": [row]})
            continue
        else:
            label, label_source = "0", "weak_rule_human_new_collection"
        record = {**row, "sample_id": stable_id(identity), "label": label,
                  "previous_label": None, "label_source": label_source,
                  "matches_new_rule": score >= THRESHOLD, "newly_added": True,
                  "sources": [row], "score_observations": [observation_from(row)],
                  "authorship_label": "uncertain"}
        records.append(record)
        index[identity] = record
        if label == "1":
            moved.append(record)
    return records, excluded, moved, merged


def main():
    if OUT.exists() or REVIEW_OUT.exists():
        raise FileExistsError("r2 源与复核目录必须使用新目录，不覆盖任何已有版本")
    known_uids = read_uid_reviews(MANUAL_PATH)
    legacy = load_legacy_judgments(ROOT)
    hints = json.load(open(CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json", encoding="utf-8"))
    import re
    hints = re.compile(hints["ordinary_nickname_pattern"], re.I)
    existing = read_jsonl(SOURCE / "label-decisions.jsonl")
    sources = [(str(path.relative_to(ROOT)).replace("\\", "/"), False) for path in NEW_EXPORTS]
    new_rows = load_records(ROOT, sources, known_uids, legacy, hints)
    records, excluded, moved, merged = merge_source(existing, new_rows)
    identities = {(str(r["uid"]), normalize(r["text"])) for r in records}
    if len(identities) != len(records):
        raise ValueError("合并后出现重复 uid+正文")
    if any(int(r["label"]) == 0 and human_exclusion_matches(r["text"]) for r in records):
        raise ValueError("v1 名单内的人类灰区仍留在人类集")
    ai = [r for r in records if r["label"] == "1"]
    added = [r for r in moved if r.get("newly_added")]
    sample = sample_ai(records, seed=20261114)
    ai_ids = {r["sample_id"] for r in ai}
    assert {r["sample_id"] for r in sample} <= ai_ids
    assert len(sample) == (len(ai) + 99) // 100

    OUT.mkdir(parents=True)
    REVIEW_OUT.mkdir(parents=True)
    from bili_ai.common import write_csv
    write_csv(OUT / "comments.csv", ({k: r[k] for k in FIELDS} for r in records), FIELDS)
    from text_specialist_dataset import save_jsonl, save_csv
    save_jsonl(OUT / "ai.jsonl", ai)
    save_jsonl(OUT / "label-decisions.jsonl", records)
    save_jsonl(OUT / "moved-to-ai.jsonl", moved)
    save_jsonl(OUT / "excluded.jsonl", excluded)
    for path in (POLICY_PATH, EMOTES_PATH, SOURCE / "legacy_uid_decisions.csv",
                 SOURCE / "legacy_comment_judgments.jsonl"):
        out_copy = OUT / path.name
        if not out_copy.exists():
            out_copy.write_bytes(path.read_bytes())
    inherited = SOURCE / "inherited-splits.csv"
    out_copy = OUT / "inherited-splits.csv"
    if not out_copy.exists():
        out_copy.write_bytes(inherited.read_bytes())

    review_rows = [{"序号": i, "样本ID": r["sample_id"], "正文": r["text"], "判定": "", "备注": ""}
                   for i, r in enumerate(sample, 1)]
    save_csv(REVIEW_OUT / "抽样复核.csv", review_rows, ("序号", "样本ID", "正文", "判定", "备注"))
    (REVIEW_OUT / "抽样正文.md").write_text(review_markdown(review_rows), encoding="utf-8")
    save_jsonl(REVIEW_OUT / "sample.jsonl", sample)
    manifest = {"population": str(OUT / "ai.jsonl"), "population_sha256": sha256(OUT / "ai.jsonl"),
                "population_size": len(ai), "unit": "uid_and_normalized_body", "seed": 20261114,
                "fraction_requested": 0.01, "size_rounding": "ceil", "sample_size": len(sample),
                "actual_fraction": len(sample) / len(ai),
                "sample_ids_in_review_order": [r["sample_id"] for r in sample],
                "sample_sha256": sha256(REVIEW_OUT / "sample.jsonl"),
                "note": "r2 扩充后的新总体抽样；仅供用户抽查，不阻塞训练"}
    write_json(REVIEW_OUT / "sampling.json", manifest)
    write_json(REVIEW_OUT / "pending-review-summary.json",
               review_summary(review_rows, set(manifest["sample_ids_in_review_order"])))

    protected = [SOURCE / "comments.csv", SOURCE / "label-decisions.jsonl", SOURCE / "ai.jsonl",
                 POLICY_PATH, EMOTES_PATH, Path(__file__), *NEW_EXPORTS]
    report = {"policy": "user_authorized_rule_score_ge_30; new human rows kept (2026-11-14 expansion)",
              "threshold": THRESHOLD, "data_output": str(OUT), "review_output": str(REVIEW_OUT),
              "source_sha256": {str(p): sha256(p) for p in protected},
              "new_exports": [str(p) for p in NEW_EXPORTS],
              "existing_rows": len(existing), "new_export_rows": len(new_rows),
              "merged_identities": merged,
              "new_rows_added": sum(1 for r in records if r.get("newly_added")),
              "promoted_existing": len(moved) - sum(1 for r in moved if r.get("newly_added")),
              "labels_before": dict(Counter(r["label"] for r in existing)),
              "labels_after": dict(Counter(r["label"] for r in records)),
              "moved_to_ai": len(moved),
              "excluded_by_kind": dict(Counter(r["reason"] for r in excluded)),
              "ai_population": len(ai), "sample": manifest,
              "dsh_video_ai_only": "out/dsh-video-20261114/export/ai-only-ge30.jsonl (user: 只记录ai评论)"}
    write_json(OUT / "build_report.json", report)
    write_json(REVIEW_OUT / "build_report.json", report)
    print(json.dumps({k: report[k] for k in ("existing_rows", "new_export_rows", "labels_before",
                                             "labels_after", "moved_to_ai", "excluded_by_kind",
                                             "ai_population")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
