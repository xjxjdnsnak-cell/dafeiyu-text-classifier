import json
import subprocess
import sys

import pytest

from inputs import SPLITS, TEACHER
from train import COPIES, targets
from views import CODE, broad, teacher_pair, training_views
from predict_v10 import V10Predictor


def test_training_edits_preserve_mentions_and_body_tokens():
    body = "回复 @甲-乙 :「插件」好用！[点赞] https://a.example/x"
    original, punctuation, toggled, both = training_views(body, 0)
    assert original == body
    assert "@甲-乙" in punctuation and "「" not in punctuation
    assert "[点赞]" in punctuation and "https://a.example/x" in punctuation
    assert "@" not in toggled and "@" not in both
    assert "插件" in both
    assert "@甲-乙" in broad(body)
    assert training_views("想看教程", 2) == (
        "想看教程", "「想看教程」", "@小林 想看教程", "@小林 「想看教程」")
    assert teacher_pair("！") == ("！", "！")


def test_distillation_uses_teacher_signal_and_checks_alignment(tmp_path):
    rows = [{"label": "0"}, {"label": "1"}]
    cache = {"soft_targets": [0.99, 0.01], "temperature": 2.0,
             "train_sha256": SPLITS["train"], "teacher": TEACHER}
    path = tmp_path / "teacher.json"
    path.write_text(json.dumps(cache))
    a, b = targets(rows, "v10", path)
    assert 0 < b < a < COPIES
    assert targets(rows, "aug_hard", None) == [0, COPIES]
    cache["train_sha256"] = "other split"
    path.write_text(json.dumps(cache))
    with pytest.raises(ValueError):
        targets(rows, "v10", path)


def test_native_batch_matches_single_and_skips_mentions(tmp_path):
    rows = [{"text": "回复 @小林 : 可以离线安装吗？"}, {"text": "回复 @大肥鱼 : 可以离线安装吗？"},
            {"text": "回复 @小林 :"}, {"text": " \n "}, {"text": "今天学会了。"}]
    model = V10Predictor()
    expected = [model.predict_one(text=r["text"]) for r in rows]
    assert expected[0]["ai_style_score"] == expected[1]["ai_style_score"]
    assert expected[0]["ai_style_score"] is not None
    assert all(expected[i]["ai_style_score"] is None and not expected[i]["blocked"] for i in (2, 3))
    path = tmp_path / "samples.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    output = tmp_path / "scores.jsonl"
    subprocess.run([sys.executable, str(CODE / "predict_v10.py"), "batch",
                    "--input", str(path), "--output", str(output)], check=True)
    assert [json.loads(s) for s in output.read_text(encoding="utf-8").splitlines()] == expected
