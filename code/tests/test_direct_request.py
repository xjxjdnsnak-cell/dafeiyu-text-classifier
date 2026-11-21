import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict_masked_v9 import MaskedV9Predictor, is_direct_request


@pytest.mark.parametrize(("text", "expected"), [
    ("回复 ＠示例：能帮我装这个吗？", True),
    ("@示例 那就自己装", True),
    ("@示例 怎么看？", True),
    ("@示例 你看一下这个插件，我试过了。", False),
    ("@示例 你看一下@另一个人", False),
    ("@示例 这是我自己装好的", False),
    ("你能帮我装吗？", False),
    ("请看 @示例 能不能装", False),
    ("@示例 那就自己装" + "一" * 20, False),
    ("@示例", False),
    (None, False),
])
def test_frozen_request_boundaries(text, expected):
    assert is_direct_request(text) is expected


def test_policy_only_changes_protected_predictions():
    parent = MaskedV9Predictor()
    guarded = MaskedV9Predictor(protect_direct_requests=True)
    rows = [{"text": "@示例 那就自己装", "uid": "123"}, {"text": "@示例"},
            {"text": "@示例 插件的安装过程值得记录。"}, {"text": ""}]
    raw = parent.predict_many(rows)
    result = guarded.predict_many(rows)
    assert raw[0]["reason"] == "model"
    assert result[0] == dict(raw[0], blocked=False, label=None, model_label=None,
                             ai_style_score=None, reason="direct_request")
    assert result[1:] == raw[1:]
    assert guarded.predict_one(**rows[0]) == result[0]
    assert guarded.predict_many([]) == []
