"""User labels override effective labels; pattern hits only remove negatives."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "text-specialist-20261113"))
from text_policy import compile_rules, human_exclusion_matches

try:
    import apply_user_review as module
except ModuleNotFoundError:
    module = None


@pytest.mark.parametrize("text", ["这句最戳我", "不是命令，而是知道该装哪一个", "不是命令,是知道该装哪一个", "带『引号』的正文"])
def test_extended_human_exclusions_cover_the_requested_phrases(text):
    rules = compile_rules(json.loads((HERE / "human_exclusion_policy.json").read_text(encoding="utf-8")))
    assert human_exclusion_matches(text, rules=rules)


def test_policy_v3_corner_bracket_does_not_match_everything():
    rules = compile_rules(json.loads((HERE / "human_exclusion_policy.json").read_text(encoding="utf-8")))
    assert human_exclusion_matches("正常讨论内容，没有特殊符号", rules=rules) == []
    assert human_exclusion_matches("说完就跑了", rules=rules) == []


def test_phrase_in_mentioned_nickname_does_not_exclude_a_human_body():
    rules = compile_rules(json.loads((HERE / "human_exclusion_policy.json").read_text(encoding="utf-8")))
    assert not human_exclusion_matches("@这句最戳我 装一下", rules=rules)


def test_explicit_ai_overrides_gray_while_other_pattern_humans_only_become_gray():
    assert module is not None, "apply_user_review is not implemented"
    rows = [{"sample_id": "explicit", "uid": "1", "text": "这句最戳我", "label": "0"},
            {"sample_id": "gray", "uid": "2", "text": "不是命令，是知道如何做", "label": "0"},
            {"sample_id": "old_ai", "uid": "3", "text": "这句最戳我", "label": "1"},
            {"sample_id": "human", "uid": "4", "text": "正常讨论内容", "label": "0"}]
    kept, gray = module.apply_corrections(rows, {"explicit": 5})
    assert {r["sample_id"]: r["label"] for r in kept} == {"explicit": "1", "old_ai": "1", "human": "0"}
    assert gray[0]["sample_id"] == "gray" and gray[0]["label"] == "0"
    assert next(r for r in kept if r["sample_id"] == "explicit")["previous_effective_label"] == "0"
    assert rows[0]["label"] == "0"


def test_review_id_must_exist():
    assert module is not None, "apply_user_review is not implemented"
    with pytest.raises(ValueError):
        module.apply_corrections([], {"unknown": 5})


def test_all_review_batches_merge_without_collision():
    assert module is not None, "apply_user_review is not implemented"
    corrections = module.correction_map()
    assert len(corrections) == 30
    batches = {detail["batch"] for detail in corrections.values()}
    assert batches == {1, 2, 3, 4, 5, 6}
    reviewed = module.reviewed_sample_ids()
    assert len(reviewed) == 97
    assert set(corrections) <= reviewed
    assert len(module.review_csvs()) == 7
