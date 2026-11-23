"""Offline, body-only fastText v10 candidate."""
import argparse
import json
from pathlib import Path

from predict_masked_v9 import ROOT, MaskedBodyEncoder
from bili_ai.cli import _batch
from bili_ai.model import Predictor
from bili_ai.preprocess import ASSETS

MODEL = ROOT / "models" / "v10" / "model.bin"


class V10Predictor(Predictor):
    def __init__(self, path=MODEL):
        super().__init__(path, mode="text", assets=ASSETS)
        if self.metadata.get("encoder") != "masked_v9_structure":
            raise ValueError("Unexpected encoder for this candidate")
        self.pre = MaskedBodyEncoder(self.pre)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    single = commands.add_parser("predict")
    single.add_argument("--text", required=True)
    batch = commands.add_parser("batch")
    batch.add_argument("--input", type=Path, required=True)
    batch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = V10Predictor()
    if args.command == "predict":
        result = model.predict_one(text=args.text)
    else:
        result = _batch(model, args.input, args.output, batch_size=64)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
