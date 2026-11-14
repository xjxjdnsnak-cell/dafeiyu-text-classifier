"""Merge semantics for the r2 source: keep new humans, rule >=30 wins, dedup by identity."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text-specialist-20261113"))

from build_source_r2 import merge_source, observation_from


def corpus_row(uid, rpid, text, score, username="昵称", bvid="BVx", line=1, label="human"):
    return {"sample_id": "TXT-" + rpid, "uid": uid, "rpid": rpid, "bvid": bvid,
            "video_title": "视频", "url": "https://b23.tv/" + rpid, "username": username,
            "text_raw": text, "text": text, "length": len(text), "length_group": "1-20",
            "nickname_group": "ordinary", "body_pattern_tags": [], "template_text": text,
            "template_id": "tpl-" + rpid, "is_old_source": False,
            "provenance": [{"file": "out/x/export/all_scored.jsonl", "line": line,
                            "raw_ref": "raw\\" + rpid, "collected_at": "2026-11-14"}],
            "text_score": None, "manual_account_review": None, "legacy_comment_review": None,
            "rule_reasons": [], "candidate_tags": [], "rule_score": score,
            "source_weak_comment_label": label, "weak_comment_label": label,
            "text_model_eligible": True, "text_exclusion_reason": "",
            "human_label_eligible": True, "human_exclusion_matches": [],
            "authorship_label": "uncertain"}


def existing_row(uid, text, label):
    return {"sample_id": "DS30-" + uid, "uid": uid, "text": text, "username": "旧昵称",
            "label": label, "label_source": "preserved_existing_label",
            "sources": [{"sample_id": "TXT-old", "uid": uid, "bvid": "BVold"}],
            "rule_score": 5, "score_observations": []}


def test_new_qualifying_row_becomes_ai_and_human_row_is_kept():
    rows = [corpus_row("1", "r1", "这条超过三十分的内容", 45),
            corpus_row("2", "r2", "普通的人类发言", 8)]
    records, excluded, moved, merged = merge_source([existing_row("9", "既有内容", "0")], rows)
    by_uid = {r["uid"]: r for r in records}
    assert by_uid["1"]["label"] == "1" and by_uid["1"]["label_source"] == "user_rule_score_ge_30"
    assert by_uid["2"]["label"] == "0" and by_uid["2"]["newly_added"]
    assert not excluded and len(moved) == 1 and merged == 0


def test_mention_only_and_v1_gray_rows_are_excluded():
    # 源层沿用 v1 名单；v2/v3 新短语（如 这句最戳我）在源里保留为人类，
    # 由训练构建的灰区流程排除。
    rows = [corpus_row("1", "r1", "@某人", 60),
            corpus_row("2", "r2", "这话我记下了", 5),
            corpus_row("3", "r3", "这句最戳我", 5)]
    records, excluded, moved, _ = merge_source([], rows)
    assert {r["uid"] for r in records} == {"3"}
    assert records[0]["label"] == "0"
    assert {r["reason"] for r in excluded} == {"mention_only", "human_expression"}


def test_same_identity_merges_and_rule_promotes_human():
    rows = [corpus_row("7", "rNew", "既有正文但出现新视频", 62, bvid="BVnew")]
    records, excluded, moved, merged = merge_source([existing_row("7", "既有正文但出现新视频", "0")], rows)
    assert merged == 1 and len(records) == 1 and not excluded
    assert records[0]["label"] == "1" and len(records[0]["sources"]) == 2
    assert len(moved) == 1 and moved[0]["previous_label"] == "0"


def test_existing_ai_row_keeps_label_on_merge():
    rows = [corpus_row("7", "rNew", "既是AI的旧正文", 40, bvid="BVnew")]
    base = existing_row("7", "既是AI的旧正文", "1")
    base["previous_label"] = None
    records, _, moved, merged = merge_source([base], rows)
    assert merged == 1 and records[0]["label"] == "1" and records[0]["previous_label"] is None
    assert moved == []


def test_observation_fields_record_score_and_source():
    row = corpus_row("3", "r3", "内容", 31)
    observation = observation_from(row)
    assert observation["score"] == 31 and observation["file"].endswith("all_scored.jsonl")
