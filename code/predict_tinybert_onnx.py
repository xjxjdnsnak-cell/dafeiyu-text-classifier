"""Offline predictor for the frozen TinyBERT per-channel QInt8 ONNX text classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from transformers import BertTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "tinybert-perchannel-int8"
EMOTES_PATH = ROOT / "policy" / "known_bilibili_emotes.json"
PREPROCESS_PATH = ROOT / "code" / "src" / "bili_ai" / "assets" / "preprocess.json"

REPLY_PREFIX_RE = re.compile(r"^回复[\s\u0085\u001c-\u001f]*(?=@)")
MENTION_ONLY_RE = re.compile(r"@[^\s\u0085\u001c-\u001f:：,，。!！?？\[\]【】]+")
BRACKET_EMOTE_RE = re.compile(r"\[[^\]]*\]")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_space(text: str, space_points: set[int]) -> str:
    out: list[str] = []
    in_space = False
    for ch in text:
        if ord(ch) in space_points or ch.isspace():
            if out and not in_space:
                out.append(" ")
                in_space = True
        else:
            out.append(ch)
            in_space = False
    if out and out[-1] == " ":
        out.pop()
    return "".join(out)


def _has_letter_or_digit(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


class TinyBertOnnxPredictor:
    """Frozen TinyBERT per-channel QInt8 text-only classifier."""

    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR, threads: int = 2) -> None:
        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "model.json").read_text(encoding="utf-8"))
        model_path = self.model_dir / self.meta["artifact_file"]
        tok_path = self.model_dir / self.meta["tokenizer_file"]

        if _sha256(model_path) != self.meta["model_sha256"]:
            raise ValueError(f"Model SHA-256 mismatch: {model_path}")
        if _sha256(tok_path) != self.meta["tokenizer_sha256"]:
            raise ValueError(f"Tokenizer SHA-256 mismatch: {tok_path}")

        pre_cfg = json.loads(PREPROCESS_PATH.read_text(encoding="utf-8"))
        self._space_points = set(pre_cfg["space_points"])
        emotes_raw = json.loads(EMOTES_PATH.read_text(encoding="utf-8"))["tokens"]
        self._emotes = {_normalize_space(e, self._space_points) for e in emotes_raw}

        self._mention_re = re.compile(self.meta["preprocessing"]["mention_pattern"])
        self._mention_repl = self.meta["preprocessing"]["mention_replacement"]
        self._max_length = int(self.meta["preprocessing"]["max_length"])
        self.threshold = float(self.meta["threshold"])

        self.tokenizer = BertTokenizer.from_pretrained(str(self.model_dir), local_files_only=True)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            model_path.read_bytes(),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def is_mention_only(self, text: str) -> bool:
        body = _normalize_space(text, self._space_points)
        if not MENTION_ONLY_RE.search(body):
            return False
        stripped = MENTION_ONLY_RE.sub("", REPLY_PREFIX_RE.sub("", body))
        remaining = BRACKET_EMOTE_RE.sub(lambda m: "" if m.group(0) in self._emotes else m.group(0), stripped)
        return not _has_letter_or_digit(remaining)

    def mask_mentions(self, text: str) -> str:
        return self._mention_re.sub(self._mention_repl, text)

    def predict_one(self, text: str) -> dict:
        if self.is_mention_only(text):
            return {
                "eligible": False,
                "ai_style_score": None,
                "threshold": self.threshold,
                "predicted_label": 0,
            }
        masked = self.mask_mentions(text)
        if not masked.strip():
            return {
                "eligible": False,
                "ai_style_score": None,
                "threshold": self.threshold,
                "predicted_label": 0,
            }
        enc = self.tokenizer(masked, truncation=True, max_length=self._max_length)
        feed = {
            k: np.array([enc[k]], dtype=np.int64)
            for k in ("input_ids", "attention_mask", "token_type_ids")
        }
        logits = self.session.run(["logits"], feed)[0][0]
        l0, l1 = float(logits[0]), float(logits[1])
        m = max(l0, l1)
        e0, e1 = math.exp(l0 - m), math.exp(l1 - m)
        prob = min(1.0, max(0.0, e1 / (e0 + e1)))
        return {
            "eligible": True,
            "ai_style_score": prob,
            "threshold": self.threshold,
            "predicted_label": int(prob >= self.threshold),
        }

    def predict_many(self, texts: Iterable[str]) -> list[dict]:
        return [self.predict_one(t) for t in texts]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score comments with frozen TinyBERT per-channel INT8 ONNX model.")
    parser.add_argument("texts", nargs="*", help="Comment texts to score (reads lines from stdin if omitted).")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)

    predictor = TinyBertOnnxPredictor(args.model_dir, threads=args.threads)
    inputs = args.texts if args.texts else [line.rstrip("\r\n") for line in sys.stdin if line.strip()]
    for text, res in zip(inputs, predictor.predict_many(inputs), strict=True):
        print(json.dumps({"text": text, **res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
