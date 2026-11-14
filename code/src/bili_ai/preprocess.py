"""Frozen jieba DAG segmentation; the JavaScript port uses the same dictionary."""

import logging
import re
from pathlib import Path

import jieba

from .common import read_json, sha256

ASSETS = Path(__file__).parent / "assets"
MODES = ("text", "nickname", "combined")
# Explicit code points keep Unicode database upgrades out of normalization.
SPACE_POINTS = frozenset(range(33)) | {127, 160, 5760, *range(8192, 8208), 8232, 8233, 8239, 8287, 12288, 65279}


def normalize(value):
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("评论和昵称必须是字符串")
    chars = []
    for ch in value:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            cp -= 0xFEE0
        if cp in SPACE_POINTS:
            cp = 32
        if 65 <= cp <= 90:
            cp += 32
        if 0xD800 <= cp <= 0xDFFF:
            cp = 0xFFFD
        chars.append(chr(cp))
    return re.sub(" +", " ", "".join(chars)).strip(" ")


class Preprocessor:
    def __init__(self, assets=None):
        self.assets = Path(assets) if assets else ASSETS
        self.config = read_json(self.assets / "preprocess.json")
        dictionary = self.assets / "jieba.dict.txt"
        if sha256(dictionary) != self.config["dictionary_sha256"]:
            raise ValueError("jieba 词典校验失败；必须使用训练时的词典")
        jieba.setLogLevel(logging.WARNING)
        self.tokenizer = jieba.Tokenizer(str(dictionary))
        self.tokenizer.initialize()

    def encode(self, text="", username="", mode="combined"):
        if mode not in MODES:
            raise ValueError(f"未知特征模式: {mode}")
        tokens = []
        if mode in ("text", "combined"):
            body = normalize(text)[:self.config["max_text_chars"]]
            tokens.extend("txt_" + t for t in self.tokenizer.cut(body, HMM=False) if t and t != " ")
        if mode in ("nickname", "combined"):
            nick = normalize(username).replace(" ", "")[:self.config["max_nickname_chars"]]
            chars = [format(ord(c), "x") for c in nick]
            for n in range(1, self.config["nickname_ngram_max"] + 1):
                prefix = "nick_c_" if n == 1 else f"nick_g{n}_"
                tokens.extend(prefix + "_".join(chars[i:i+n]) for i in range(len(chars)-n+1))
            if chars:
                for n in range(1, min(3, len(chars)) + 1):
                    tokens.extend((f"nick_start{n}_" + "_".join(chars[:n]), f"nick_end{n}_" + "_".join(chars[-n:])))
                tokens.append(f"nick_len_{min(len(chars) // 4, 16)}")
                for match in re.finditer(r"(.)\1+", nick):
                    tokens.append(f"nick_repeat_{min(len(match[0]), 8)}")
                shape = []
                for c in nick:
                    kind = "d" if "0" <= c <= "9" else "a" if "a" <= c <= "z" else "h" if "\u4e00" <= c <= "\u9fd5" else "o"
                    if not shape or shape[-1] != kind:
                        shape.append(kind)
                tokens.append("nick_shape_" + "".join(shape))
        return " ".join(tokens) or "field_empty"
