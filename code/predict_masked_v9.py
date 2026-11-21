"""Body-only candidate using the frozen v9 weights and the all-mentions mask."""

import argparse
import json
import re
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path[:0] = [
    str(CODE / "src"),
    str(CODE / "experiments" / "sentence-structure-ablation-20261114"),
    str(CODE / "experiments" / "text-specialist-20261113"),
]

from bili_ai.cli import _batch
from bili_ai.common import sha256
from bili_ai.model import Predictor
from bili_ai.preprocess import ASSETS
from structure_features import encode_structure_augmented
from text_policy import is_mention_only
from direct_request import is_direct_request

MODEL = ROOT / "models" / "sentence-structure" / "B" / "model.bin"
MODEL_SHA256 = "0d2ea96fdd6d92880fc8d58411f07b1300e629bb896f5df8de7fda2bce5f40bc"
THRESHOLD = 0.9387965748548248
# Same pattern, placeholder and pre-normalization order as make_ablation_input.py.
MENTION = re.compile(r"@([^\s@:：，,。！!?？]{1,24})")


def mask_mentions(text):
    return MENTION.sub("@某人", text)


class MaskedBodyEncoder:
    def __init__(self, base):
        self.base = base

    def encode(self, text, username, mode):
        if mode != "text":
            raise ValueError("候选仅支持正文模式")
        if text is None or is_mention_only(text):
            return "field_empty"
        return encode_structure_augmented(mask_mentions(text), self.base)


class MaskedV9Predictor(Predictor):
    """Single and batch inference share the same masking and scoring path."""

    def __init__(self, *, protect_direct_requests=False):
        if sha256(MODEL) != MODEL_SHA256:
            raise ValueError("v9 模型 SHA256 与冻结记录不符")
        super().__init__(MODEL, mode="text", threshold=THRESHOLD, assets=ASSETS)
        self.pre = MaskedBodyEncoder(self.pre)
        self.protect_direct_requests = protect_direct_requests

    def predict_many(self, rows):
        if not self.protect_direct_requests:
            return super().predict_many(rows)
        # Keep name boundaries until after the policy decision. A protected row
        # uses the parent's unscored path, preserving row order and UID handling.
        protected = [is_direct_request(row.get("text")) for row in rows]
        model_rows = [dict(row, text=None) if protect else row
                      for row, protect in zip(rows, protected, strict=True)]
        results = super().predict_many(model_rows)
        for result, protect in zip(results, protected, strict=True):
            if protect and result["reason"] == "insufficient_input":
                result["reason"] = "direct_request"
        return results


def main():
    parser = argparse.ArgumentParser(description="v9 正文候选：评分前遮蔽正文中的 @昵称")
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("predict")
    single.add_argument("--text", required=True)
    batch = commands.add_parser("batch")
    batch.add_argument("--input", type=Path, required=True, help="包含 text 字段的 CSV 或 JSONL")
    batch.add_argument("--output", type=Path, required=True, help="逐行预测 JSONL")
    for command in (single, batch):
        command.add_argument("--protect-direct-requests", action="store_true",
                             help="启用候选策略，放行符合短句模板的点名请求")
    args = parser.parse_args()
    predictor = MaskedV9Predictor(protect_direct_requests=args.protect_direct_requests)
    if args.command == "predict":
        result = predictor.predict_one(text=args.text)
    else:
        result = _batch(predictor, args.input, args.output, batch_size=64)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
