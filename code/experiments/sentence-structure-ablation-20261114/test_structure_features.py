"""Targeted tests for the sentence-structure feature extractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from structure_features import encode_structure_augmented, split_sentences, structure_tokens
from bili_ai.preprocess import Preprocessor

pre = Preprocessor()


def test_patterns_fire():
    tokens = structure_tokens("不是命令，而是态度。", pre)
    assert "sfx_不是而是_1" in tokens
    tokens = structure_tokens("与其说这是工具，不如说这是玩具。", pre)
    assert "sfx_与其说不如说_1" in tokens
    tokens = structure_tokens("不仅没有帮助，还浪费了时间。", pre)
    assert "sfx_不仅还_1" in tokens


def test_non_matching_text_does_not_fire_patterns():
    tokens = structure_tokens("今天天气很好。", pre)
    assert not any(t.startswith("sfx_不是而是") or t.startswith("sfx_与其说") or t.startswith("sfx_不仅还") for t in tokens)


def test_sentence_count_and_length_buckets():
    tokens = structure_tokens("一句话。", pre)
    assert "sfx_句子数_1" in tokens
    tokens = structure_tokens("第一句。第二句！第三句？", pre)
    assert "sfx_句子数_2-3" in tokens
    tokens = structure_tokens("很短。" + "长" * 40 + "。", pre)
    assert any(t.startswith("sfx_句长变化_高") for t in tokens)


def test_line_keeps_txt_sequence_first_and_sfx_last():
    line = encode_structure_augmented("第一句。第二句！", pre)
    parts = line.split(" ")
    assert parts[0].startswith("__label__") is False
    txt_positions = [i for i, t in enumerate(parts) if t.startswith("txt_")]
    sfx_positions = [i for i, t in enumerate(parts) if t.startswith("sfx_")]
    assert sfx_positions and max(txt_positions) < min(sfx_positions) - 0
    assert parts[0].startswith("txt_") or parts[0] == "field_empty"


def test_empty_body_returns_base_only():
    line = encode_structure_augmented("   ", pre)
    assert line == "field_empty"
