"""Reconcile the new text exclusions, protected sources and output partitions."""

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from bili_ai.common import read_csv, read_json, sha256, write_json
from bili_ai.data import FIELDS, read_splits
from text_policy import human_exclusion_matches, is_mention_only
from text_specialist_dataset import partition_audit


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    classifier = Path(__file__).resolve().parents[2]
    root = classifier.parent
    source = classifier / "data-augment-70-15-15-clean-uid"
    filtered = classifier / "data-augment-70-15-15-text-gray-20261114-v2"
    bundle = root / "out/text-specialist-20261113/bundle-v6"
    report = read_json(filtered / "build_report.json")
    original = read_csv(source / "comments.csv", FIELDS)
    kept = read_csv(filtered / "comments.csv", FIELDS)
    removed = jsonl(filtered / "excluded_rows.jsonl")
    counter = lambda rows: Counter(tuple(row[field] for field in FIELDS) for row in rows)
    assert counter(original) == counter(kept) + counter(removed)
    assert not any(is_mention_only(row["text"]) for row in kept)
    assert not any(row["label"] == "0" and human_exclusion_matches(row["text"]) for row in kept)
    assert all(row["label"] == "0" for row in removed if row["exclusion_kind"] == "human_expression")
    original_map = read_splits(source / "splits.csv", original)
    new_map = read_splits(filtered / "splits.csv", kept)
    assert all(original_map[uid] == split for uid, split in new_map.items())
    assert all(sha256(source / name) == digest for name, digest in report["source_sha256"].items())
    corpus, candidates = jsonl(bundle / "corpus.jsonl"), jsonl(bundle / "candidates.jsonl")
    audit = read_json(bundle / "audit.json")
    assert len(corpus) == 21628 and len(candidates) == audit["candidates"]
    assert all(not is_mention_only(row["text"]) for row in candidates)
    assert all(row["text_score"] is None for row in corpus if not row["is_old_source"])
    assert all(row["authorship_label"] == "uncertain" for row in corpus)
    assert all(row["weak_comment_label"] != "human" for row in corpus if not row["human_label_eligible"])
    assert all(row["weak_comment_label"] != "ai" and row["text_score"] is None
               for row in corpus if is_mention_only(row["text"]))
    assert not any(any(counts.values()) for counts in partition_audit(corpus).values())
    assert (bundle / "legacy_uid_decisions.csv").read_bytes() == Path("E:/human-review-uid-decisions (2).csv").read_bytes()
    assert len(jsonl(bundle / "legacy_comment_judgments.jsonl")) == 138
    for rel, digest in audit["source_hashes"].items():
        path = Path(rel)
        assert sha256(path if path.is_absolute() else root / path) == digest
    assert sha256(Path(__file__).with_name("text_specialist_dataset.py")) == audit["builder_sha256"]
    by_id = {row["sample_id"]: row for row in corpus}
    pairs = read_csv(bundle / "matched-controls.csv", ("positive_candidate_id", "control_id"))
    for pair in pairs:
        positive, control = by_id[pair["positive_candidate_id"]], by_id[pair["control_id"]]
        assert control["human_label_eligible"] and control["text_model_eligible"]
        assert not human_exclusion_matches(control["text"]) and not is_mention_only(positive["text"])
        assert positive["bvid"] == control["bvid"] and positive["uid"] != control["uid"]
        assert positive["length_group"] == control["length_group"]
    html = (bundle / "review.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
    checks = Path(__file__).parent / "checks"
    checks.mkdir(exist_ok=True)
    generated = checks / "review-v6-generated.js"
    generated.write_text(script, encoding="utf-8")
    result = subprocess.run(["node", "--check", str(generated)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    verification = {"passed": True, "filtered_rows": len(kept), "removed_rows": len(removed),
        "removed_by_kind": report["excluded_by_kind"], "labels_changed": 0, "uid_assignments_changed": 0,
        "bundle_candidates": len(candidates), "matched_controls_checked": len(pairs),
        "all_original_sources_and_manual_history_preserved": True,
        "no_mention_only_in_active_review_or_ai_samples": True,
        "no_user_expression_in_human_controls_or_retained_label_0": True,
        "javascript_syntax_valid": True, "browser_visual_check": "not performed; previously blocked by browser URL policy",
        "model_training_run": False}
    write_json(bundle / "verification.json", verification)
    write_json(filtered / "verification.json", verification)
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
