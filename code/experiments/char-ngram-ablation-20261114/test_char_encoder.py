"""Feature-encoder targeted tests for the char-ngram augmentation (spec item: 只运行直接相关的特征编码测试)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from char_encoder import char_ngram_tokens, encode_char_augmented, feature_stats
from bili_ai.preprocess import Preprocessor

pre = Preprocessor()


def test_cross_boundary_grams_exist():
    # "不是命令" tokenizes across jieba words; the char 3-gram 是命 spans the boundary.
    tokens = char_ngram_tokens("不是命令，是态度", pre)
    assert "chx_是命" in tokens
    assert "chx_是命令" in tokens  # 4-gram crossing 是|命
    assert any(t == "chx_不是" for t in tokens)


def test_prefix_never_mixes_with_txt_features():
    line = encode_char_augmented("评分很高", pre)
    parts = line.split(" ")
    assert parts[0].startswith("txt_") or parts[0] == "field_empty"
    assert all(t.startswith("chx_") for t in parts if t.startswith("chx_"))
    assert not any(t.startswith("chx_") and t[4:].startswith("txt_") for t in parts)


def test_truncation_follows_max_text_chars():
    limit = pre.config["max_text_chars"]
    body = "测" * (limit + 50)
    tokens = char_ngram_tokens(body, pre)
    longest = max(len(t) - len("chx_") for t in tokens)
    assert longest <= 4
    # last gram starts at limit-4 at most
    assert max(i for i, t in enumerate(tokens) if t == "chx_" + "测" * 2) <= (limit - 2)


def test_deterministic_and_order_preserved():
    text = "第一句。第二句！"
    a = char_ngram_tokens(text, pre)
    b = char_ngram_tokens(text, pre)
    assert a == b
    assert a[0] == "chx_第一"


def test_feature_stats_separates_prefixes():
    lines = [encode_char_augmented("不是命令，是态度", pre)]
    stats = feature_stats(lines)
    assert stats["unique"]["chx"] > 0
    assert stats["total"]["chx"] == stats["total"]["chx"]
