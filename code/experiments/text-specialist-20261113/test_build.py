"""Guard source/style separation, manual judgments and prospective split isolation."""

import csv
import json
from pathlib import Path

import pytest

try:
    import text_specialist_dataset as builder
except ModuleNotFoundError:
    builder = None


def impl():
    assert builder is not None, "Text-specialist dataset builder is not implemented"
    return builder


def test_manual_human_account_is_not_overwritten_by_weak_ai_score():
    fields = impl().evidence_fields(99, "ai", "human", "")
    assert fields["manual_account_label"] == "human"
    assert fields["weak_comment_label"] == "ai"
    assert fields["body_style_label"] == "unreviewed"
    assert fields["authorship_label"] == "uncertain"


def test_body_style_annotation_never_becomes_authorship_evidence():
    module = impl()
    rows = [{"sample_id": "s1", "body_style_label": "unreviewed", "authorship_label": "uncertain", "manual_account_label": "human"}]
    result = module.apply_body_reviews(rows, [{"sample_id": "s1", "body_style_label": "ai_like", "style_notes": "formal phrasing"}])
    assert result[0]["body_style_label"] == "ai_like"
    assert result[0]["authorship_label"] == "uncertain"
    assert result[0]["manual_account_label"] == "human"
    assert rows[0]["body_style_label"] == "unreviewed"


def test_unknown_body_review_id_is_rejected():
    with pytest.raises(ValueError):
        impl().apply_body_reviews([], [{"sample_id": "missing", "body_style_label": "ai_like"}])


def test_uid_review_does_not_coerce_uncertain_into_human(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("UID,判定\n123,不确定\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        impl().read_uid_reviews(path)


def test_conflicting_uid_decisions_require_resolution(tmp_path):
    path = tmp_path / "decisions.csv"
    path.write_text("UID,判定\n123,AI\n123,人类\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        impl().read_uid_reviews(path)


def test_template_normalization_catches_changed_mentions_and_numbers():
    module = impl()
    assert module.template_text("@first 这里有12只猫。") == module.template_text("@second 这里有30只猫！")


def record(uid, bvid, text, old=False):
    return {"sample_id": f"{uid}-{bvid}", "uid": uid, "bvid": bvid,
            "text": text, "is_old_source": old, "template_text": impl().template_text(text)}


def test_old_uid_is_quarantined_without_training_on_test_video():
    module = impl()
    rows = [record("old", "old_video", "早先讨论的正文", True), record("old", "new_test", "新的正文")]
    result = module.assign_roles(rows, {"new_test": "test"})
    assert result[0]["pool_role"] == "development"
    assert result[1]["pool_role"] == "quarantine"
    assert "uid_seen_in_old_data" in result[1]["quarantine_reasons"]


def test_uid_crossing_new_video_partitions_is_quarantined_on_both_sides():
    result = impl().assign_roles([record("same", "v1", "甲"), record("same", "v2", "乙")],
                                {"v1": "validation", "v2": "test"})
    assert [r["pool_role"] for r in result] == ["quarantine", "quarantine"]


def test_same_template_across_partitions_is_quarantined():
    result = impl().assign_roles([record("a", "v1", "@first 这里有12只猫。"), record("b", "v2", "@second 这里有30只猫！")],
                                {"v1": "validation", "v2": "test"})
    assert [r["pool_role"] for r in result] == ["quarantine", "quarantine"]


def test_new_video_new_uid_and_distinct_template_can_enter_holdout():
    result = impl().assign_roles([record("a", "v1", "这是一段全新的内容")], {"v1": "test"})
    assert result[0]["pool_role"] == "test"


def test_uid_known_to_model_but_absent_from_raw_is_still_excluded():
    result = impl().assign_roles([record("trained", "new", "另一段正文")],
                                {"new": "test"}, known_uids={"trained"})
    assert result[0]["pool_role"] == "quarantine"


def test_template_known_to_model_but_absent_from_raw_is_still_excluded():
    row = record("new", "new", "这段正文此前参与了模型训练")
    result = impl().assign_roles([row], {"new": "test"}, known_templates={row["template_text"]})
    assert result[0]["pool_role"] == "quarantine"


def test_controls_match_video_and_length_without_fabrication():
    module = impl()
    positive = {"sample_id": "p", "uid": "a", "bvid": "v1", "text": "abcdefghij"}
    controls = [{"sample_id": "wrong", "uid": "b", "bvid": "v2", "text": "abcdefghij"},
                {"sample_id": "right", "uid": "c", "bvid": "v1", "text": "abcdefghijkl"}]
    pairs = module.match_controls([positive], controls)
    assert pairs[0]["control_id"] == "right"
    assert pairs[0]["length_difference"] == 2


def test_controls_do_not_cross_the_predeclared_length_groups():
    positive = {"sample_id": "p", "uid": "a", "bvid": "v", "text": "x" * 20}
    controls = [{"sample_id": "aaa", "uid": "b", "bvid": "v", "text": "x" * 21},
                {"sample_id": "zzz", "uid": "c", "bvid": "v", "text": "x" * 19}]
    pairs = impl().match_controls([positive], controls)
    assert len(pairs) == 1
    assert pairs[0]["control_id"] == "zzz"


def test_preview_annotations_cannot_be_imported_as_real_review():
    with pytest.raises(ValueError):
        impl().apply_body_reviews(
            [{"sample_id": "s", "body_style_label": "unreviewed", "authorship_label": "uncertain"}],
            [{"sample_id": "s", "body_style_label": "ai_like", "style_reviewer": "qa_preview"}],
        )


def test_csv_display_protects_formula_text_and_long_uids():
    module = impl()
    assert module.csv_cell("=1+1") == "'=1+1"
    assert module.csv_cell("3747576996170599") == "'3747576996170599"
    assert module.csv_cell("普通文本") == "普通文本"


def test_readme_accepts_relative_output_directory(tmp_path, monkeypatch):
    module = impl()
    monkeypatch.chdir(tmp_path)
    output = Path("relative-bundle")
    output.mkdir()
    report = {"fresh_comments": 1, "raw_unique_comments": 1, "candidates": 1,
              "manual_uid_decisions_preserved": 0, "legacy_comment_decisions_preserved": 0,
              "matched_control_pairs": 0}
    module.write_readme(output, report)
    assert output.resolve().as_posix() in (output / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("text", ["@昵称", "　＠某用户　", "@甲 @乙", "@昵称。", "@昵称 [doge]", "回复 @昵称："])
def test_mention_without_body_is_detected(text):
    assert impl().is_mention_only(text)


@pytest.mark.parametrize("text", ["@昵称 这句有正文", "看看 @昵称", "@昵称 123", "正常正文", "@昵称 https://example.com"])
def test_real_body_after_or_before_mention_is_retained(text):
    assert not impl().is_mention_only(text)


@pytest.mark.parametrize("text", ["@用户 [这段是我写的，不是表情]", "@用户[真实的正文内容]"])
def test_unknown_bracketed_prose_is_not_discarded_as_an_emote(text):
    assert not impl().is_mention_only(text)


def test_high_rule_score_does_not_make_mention_only_an_ai_sample():
    row = record("old", "old_video", "@某用户", old=True)
    row.update(impl().evidence_fields(99, "ai", "ai", "ai"))
    result = impl().assign_roles([row], {})[0]
    assert result["pool_role"] == "excluded_text"
    assert result["weak_comment_label"] == "review"
    assert result["source_weak_comment_label"] == "ai"
    assert result["text_exclusion_reason"] == "mention_only"
    assert result["manual_account_label"] == result["legacy_comment_judgment"] == "ai"
    assert result["authorship_label"] == "uncertain"


def test_old_review_cannot_label_mention_only_as_ai_style():
    with pytest.raises(ValueError):
        impl().apply_body_reviews(
            [{"sample_id": "s", "text": "@昵称", "body_style_label": "unreviewed", "authorship_label": "uncertain"}],
            [{"sample_id": "s", "body_style_label": "ai_like"}],
        )


def test_mention_only_is_not_used_as_matched_control():
    positives = [{"sample_id": "p", "uid": "a", "bvid": "v", "text": "真实正文"}]
    controls = [{"sample_id": "c", "uid": "b", "bvid": "v", "text": "@昵称"}]
    assert impl().match_controls(positives, controls) == []


def test_every_user_expression_is_covered_by_the_gray_policy():
    policy = json.loads(Path(__file__).with_name("human_exclusion_policy.json").read_text(encoding="utf-8"))
    for phrase in dict.fromkeys(policy["source_phrases"]):
        example = phrase.replace("xxx", "这个").replace("xx", "这个")
        assert phrase in impl().human_exclusion_matches(example), phrase


@pytest.mark.parametrize("text", ["我记下了。", "睡吧, 我守着", "这件事比刚才沉", "我没接你那句玩笑", "这话是她告诉你的，还是你自己想明白的？"])
def test_user_expression_variants_are_gray_not_human(text):
    assert impl().human_exclusion_matches(text)


@pytest.mark.parametrize("text", ["今天去了公园", "旅行，今天到了北京", "执行，这一步完成了", "@我记住了 你好"])
def test_unrelated_body_and_mentioned_names_do_not_match_gray_policy(text):
    assert impl().human_exclusion_matches(text) == []


def test_gray_human_is_not_promoted_to_ai_and_manual_history_survives():
    row = {"text": "我记下了", **impl().evidence_fields(6, "human", "human", "human")}
    result = impl().apply_text_policy(row)
    assert result["weak_comment_label"] == "review"
    assert result["source_weak_comment_label"] == "human"
    assert result["human_label_eligible"] is False
    assert result["text_model_eligible"] is True
    assert result["manual_account_label"] == result["legacy_comment_judgment"] == "human"
    assert result["authorship_label"] == "uncertain"


def test_existing_weak_ai_is_kept_when_matching_a_human_gray_expression():
    row = {"text": "这话我记下了", **impl().evidence_fields(90, "ai", "", "")}
    result = impl().apply_text_policy(row)
    assert result["weak_comment_label"] == "ai"
    assert result["human_label_eligible"] is False


def test_gray_expression_is_not_used_as_normal_control():
    positives = [{"sample_id": "p", "uid": "a", "bvid": "v", "text": "正常正文"}]
    controls = [{"sample_id": "c", "uid": "b", "bvid": "v", "text": "我记下了"}]
    assert impl().match_controls(positives, controls) == []


def test_labeled_dataset_drops_gray_humans_and_all_mentions_but_keeps_ai():
    rows = [
        {"uid": "1", "username": "a", "text": "@昵称", "label": "1"},
        {"uid": "2", "username": "b", "text": "我记下了", "label": "0"},
        {"uid": "3", "username": "c", "text": "我记下了", "label": "1"},
        {"uid": "4", "username": "d", "text": "今天天气很好", "label": "0"},
    ]
    kept, excluded = impl().filter_labeled_rows(rows)
    assert [r["uid"] for r in kept] == ["3", "4"]
    assert [r["label"] for r in kept] == ["1", "0"]
    assert [r["exclusion_kind"] for r in excluded] == ["mention_only", "human_expression"]
    assert [r["label"] for r in rows] == ["1", "0", "1", "0"]
