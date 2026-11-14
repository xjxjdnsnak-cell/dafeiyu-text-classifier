"""Verify the user's new label policy and the population used for sampling."""

import copy
import math

import pytest

try:
    import score30_dataset as module
except ModuleNotFoundError:
    module = None


def impl():
    assert module is not None, "score30_dataset is not implemented"
    return module


def source(uid, text, score, identifier=None, **extra):
    return {"uid": uid, "text": text, "username": "普通昵称", "rule_score": score,
            "sample_id": identifier or uid, "rpid": identifier or uid,
            "manual_account_label": "unreviewed", "pool_role": "development", **extra}


def base(uid, text, label):
    return {"uid": uid, "text": text, "username": "原昵称", "label": label}


def test_30_is_inclusive_and_29_is_not_promoted():
    records, excluded = impl().merge_population([], [source("a", "够三十分", 30), source("b", "二十九分", 29)])
    assert [(row["uid"], row["label"]) for row in records] == [("a", "1")]
    assert excluded == []


def test_mention_only_is_excluded_even_with_score_100_or_previous_ai_label():
    records, excluded = impl().merge_population([base("old", "@甲 [doge]", "1")],
        [source("new", "回复 @乙：", 100)])
    assert records == []
    assert len(excluded) == 2


def test_new_user_rule_overrides_effective_human_label_and_preserves_history():
    history = {"判定": "人类", "备注": "原人工结论"}
    originals = [base("a", "真实正文", "0")]
    raw = [source("a", "真实正文", 30, manual_account_label="human", manual_account_review=history)]
    snapshot = copy.deepcopy((originals, raw))
    records, _ = impl().merge_population(originals, raw)
    row = records[0]
    assert row["label"] == "1"
    assert row["previous_label"] == "0"
    assert row["base_row"] == originals[0]
    assert row["sources"][0]["manual_account_review"] == history
    assert row["label_source"] == "user_rule_score_ge_30"
    assert (originals, raw) == snapshot


def test_uid_text_dedup_preserves_all_comment_ids_and_old_ai_outside_threshold():
    records, _ = impl().merge_population([base("old", "保留以前的 AI", "1")],
        [source("same", "Ａ同文", 30, "r1"), source("same", "a同文", 65, "r2")])
    assert len(records) == 2
    row = next(row for row in records if row["uid"] == "same")
    assert {r["rpid"] for r in row["sources"]} == {"r1", "r2"}
    assert row["rule_score_max"] == 65
    assert next(row for row in records if row["uid"] == "old")["label"] == "1"


def test_below_threshold_human_expression_stays_out_of_human_and_is_not_promoted():
    records, excluded = impl().merge_population([base("a", "这话我记下了", "0")],
                                               [source("a", "这话我记下了", 29)])
    assert records == []
    assert excluded[0]["reason"] == "human_expression"


def test_expression_is_allowed_in_ai_when_score_qualifies():
    records, _ = impl().merge_population([], [source("a", "这话我记下了", 30)])
    assert records[0]["label"] == "1"


def test_sampling_is_exactly_ceil_one_percent_of_all_ai_without_replacement():
    rows = [{"sample_id": f"s{i:04}", "label": "1"} for i in range(6025)]
    rows.append({"sample_id": "human", "label": "0"})
    selected = impl().sample_ai(rows, seed=20261114)
    assert len(selected) == math.ceil(6025 / 100) == 61
    assert len({r["sample_id"] for r in selected}) == 61
    assert all(r["label"] == "1" for r in selected)
    assert selected == impl().sample_ai(list(reversed(rows)), seed=20261114)
    assert selected != impl().sample_ai(rows, seed=20261115)


def test_sampling_rejects_duplicate_population_ids():
    with pytest.raises(ValueError, match="duplicate"):
        impl().sample_ai([{"sample_id": "same", "label": "1"}] * 2)


def test_expanded_export_score_supersedes_earlier_score_and_keeps_both(tmp_path):
    files = []
    for name, score in [("first.jsonl", 14), ("expanded.jsonl", 34)]:
        path = tmp_path / name
        import json
        path.write_text(json.dumps({"mid": "a", "rpid": "r", "text": "正文", "score": score,
                                    "collected_at": "2026-11-12T09:05:23+00:00", "label": "review"}), encoding="utf-8")
        files.append(path)
    original = source("a", "正文", 14, "r")
    refreshed = impl().refresh_rule_scores([original], files)[0]
    assert refreshed["rule_score"] == 34
    assert refreshed["snapshot_rule_score"] == 14
    assert [r["score"] for r in refreshed["score_observations"]] == [14, 34]
    assert original["rule_score"] == 14


def test_review_counts_pending_and_uncertain_without_claiming_they_are_ai():
    rows = [{"样本ID": "1", "判定": "AI"}, {"样本ID": "2", "判定": "真人"},
            {"样本ID": "3", "判定": "不确定"}, {"样本ID": "4", "判定": ""}]
    result = impl().review_summary(rows, {"1", "2", "3", "4"})
    assert result["reviewed_false_discovery_fraction"] == 0.5
    assert result["confirmed_human_fraction_of_sample"] == 0.25
    assert result["unresolved"] == 2
    assert result["sample_error_bounds_with_unresolved"] == [0.25, 0.75]
    assert result["review_complete"] is False


def test_unfilled_review_does_not_report_zero_error():
    result = impl().review_summary([{"样本ID": "1", "判定": ""}], {"1"})
    assert result["reviewed_false_discovery_fraction"] is None


@pytest.mark.parametrize("rows", [
    [{"样本ID": "other", "判定": "AI"}],
    [{"样本ID": "1", "判定": "AI"}, {"样本ID": "1", "判定": "真人"}],
    [],
    [{"样本ID": "1", "判定": "猜测"}],
])
def test_invalid_or_incomplete_review_file_cannot_change_sampling_denominator(rows):
    with pytest.raises(ValueError):
        impl().review_summary(rows, {"1"})
