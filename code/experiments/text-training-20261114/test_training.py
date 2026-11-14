"""Small software fixtures only; these examples are never model training data."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "text-specialist-20261113"))
try:
    import train_text as module
except ModuleNotFoundError:
    module = None


def impl():
    assert module is not None, "train_text is not implemented"
    return module


def row(identifier, uid, text, video, label=1, ordinary=True):
    return {"sample_id": identifier, "uid": uid, "text": text, "username": "昵称",
            "label": label, "videos": [video] if video else [], "ordinary": ordinary}


def test_template_preserves_substantive_brackets_and_uses_model_visible_prefix():
    m = impl()
    assert m.template_key("@某人 [这段是实质正文]") == "这段是实质正文"
    assert m.template_key("@某人 [doge] 还有正文123。") == "还有正文"
    prefix = "甲" * 4096
    assert m.template_key(prefix + "后文一") == m.template_key(prefix + "后文二")


def test_account_and_template_overlap_is_removed_from_holdouts():
    m = impl()
    rows = [row("train", "1", "训练中的正文", "a"),
            row("uid_overlap", "1", "另一个正文", "b"),
            row("template_overlap", "2", "训练中的正文！", "c"),
            row("valid", "3", "完全不同的保留内容", "b"),
            row("test", "4", "测试集合独有的表述", "c")]
    result = m.partition_rows(rows, {"a": "train", "b": "validation", "c": "test"})
    roles = {r["sample_id"]: r["split"] for r in result}
    assert roles == {"train": "train", "uid_overlap": "quarantine", "template_overlap": "quarantine",
                     "valid": "validation", "test": "test"}


def test_unknown_video_is_train_only_and_cross_video_row_is_quarantined():
    result = impl().partition_rows([row("unknown", "1", "没有视频出处", None),
                                   {**row("cross", "2", "跨视频的同一正文", "a"), "videos": ["a", "b"]}],
                                  {"a": "train", "b": "test"})
    assert [r["split"] for r in result] == ["train", "quarantine"]


def test_original_phone_oid_and_collector_bvid_use_same_video_identity():
    metadata = {"uid": "1", "text": "原始正文", "sources": [{"bvid": "BVexample"}]}
    phone = {("1", "原始正文"): [{"mid": "1", "text": "原始正文", "oid": 123,
                                    "replyType": 1, "rpid": "9"}]}
    videos, trace = impl().recover_video_groups(metadata, {"BVexample": "123"}, phone, {}, {})
    assert videos == ["video:123"]
    assert trace[0]["rpid"] == "9"


def test_prior_import_comment_id_can_recover_a_changed_training_body():
    metadata = {"uid": "1", "text": "训练正文", "sources": []}
    provenance = {("1", "训练正文"): [{"rpid": "9", "source_archive": "original.zip"}]}
    videos, trace = impl().recover_video_groups(metadata, {}, {}, provenance, {("1", "9"): {"aid": 123}})
    assert videos == ["video:123"] and trace[0]["match"] == "prior_import_uid_body_then_uid_rpid"


def test_global_body_dedup_keeps_both_original_accounts_and_videos():
    members = [{"uid": "1", "text": "同一正文", "sample_id": "one", "sources": [{"bvid": "a", "sample_id": "ra"}]},
               {"uid": "2", "text": "同一正文", "sample_id": "two", "sources": [{"bvid": "b", "sample_id": "rb"}]}]
    merged = impl().merge_body_lineage(members, ({"a": "123", "b": "456"}, {}, {}, {}))
    assert merged["associated_uids"] == ["1", "2"]
    assert merged["videos"] == ["video:123", "video:456"]
    assert merged["source_metadata_ids"] == ["one", "two"]
    representative = {**row("one", "1", "同一正文", "a"), **merged}
    partitioned = impl().partition_rows([representative], {"video:123": "train", "video:456": "validation"})
    assert partitioned[0]["split"] == "quarantine"


def test_removed_duplicate_account_still_guards_uid_isolation():
    rows = [{**row("a", "1", "合并了账号来源的正文", "a"), "associated_uids": ["1", "2"]},
            row("b", "2", "同账号在验证里的另一条正文", "b")]
    result = impl().partition_rows(rows, {"a": "train", "b": "validation"})
    assert result[1]["split"] == "quarantine" and "uid_cross_split" in result[1]["quarantine_reasons"]


def test_near_duplicates_do_not_remain_across_splits():
    text = "这是一条专门用来检查近重复隔离的软件测试正文以及后续内容"
    result = impl().partition_rows([row("a", "1", text + "甲", "a"),
                                   row("b", "2", text + "乙", "b")], {"a": "train", "b": "test"})
    assert result[0]["split"] == "train" and result[1]["split"] == "quarantine"


def test_video_split_is_deterministic_and_assigns_each_video_once():
    rows = [row(str(i), str(i), f"独有正文{i}", f"video{i}", label=i % 2) for i in range(15)]
    a = impl().assign_video_splits(rows)
    assert a == impl().assign_video_splits(list(reversed(rows)))
    assert set(a) == {r["videos"][0] for r in rows}
    assert set(a.values()) == {"train", "validation", "test"}


def test_ordinary_positive_weight_only_repeats_actual_training_rows():
    rows = [row("a", "1", "a", "a"), row("b", "2", "b", "a", ordinary=False),
            row("c", "3", "c", "a", label=0)]
    lines = ["__label__1 txt_a", "__label__1 txt_b", "__label__0 txt_c"]
    weighted, ids = impl().weighted_training_lines(rows, lines, weight=3)
    assert weighted.count(lines[0]) == 3 and weighted.count(lines[1]) == weighted.count(lines[2]) == 1
    assert sorted(ids) == ["a", "a", "a", "b", "c"]
    assert set(weighted) == set(lines)


def test_validation_rank_prioritizes_ordinary_recall_before_overall():
    m = impl()
    assert m.validation_rank(0.8, {"recall": 0.5, "precision": 0.9}, 1000) > m.validation_rank(
        0.2, {"recall": 0.99, "precision": 1.0}, 500)


def test_user_review_requires_exact_draw_and_keeps_authorship_uncertain():
    m = impl()
    rows = [{"sample_id": "s1", "label": "1", "authorship_label": "uncertain"}]
    result = m.user_reviews(rows, ["s1"])
    assert result[0]["review_label"] == "AI" and result[0]["reviewer"] == "user"
    assert result[0]["authorship_label"] == "uncertain"
    with pytest.raises(ValueError):
        m.user_reviews(rows, ["unknown"])


def test_mention_only_inference_does_not_load_or_call_a_model():
    result = impl().predict_text(Path("must-not-be-opened.ftz"), "回复 @某人：[doge]")
    assert result["label"] is None and result["reason"] == "insufficient_body"
