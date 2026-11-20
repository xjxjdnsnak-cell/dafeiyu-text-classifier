"""Offline inference for the separately trained masked-text candidate."""

import argparse
import json
import re
from pathlib import Path

from predict_masked_v9 import MaskedBodyEncoder, mask_mentions
from bili_ai.common import read_json, sha256
from bili_ai.model import Predictor
from bili_ai.preprocess import ASSETS, normalize

# Only the entire invitation qualifies. Additional words still go to the model.
INVITATION = re.compile(r"(?:(?:回复\s*)?@某人[\s:,]*)*快来学学[。!?.]*")


def simple_invitation(text):
    return INVITATION.fullmatch(normalize(mask_mentions(text or ""))) is not None


class MentionTrainedPredictor(Predictor):
    def __init__(self, folder):
        folder = Path(folder)
        meta = read_json(folder / "candidate.json")
        if sha256(folder / "model.bin") != meta["model_sha256"]:
            raise ValueError("候选权重与冻结记录不符")
        super().__init__(folder / "model.bin", mode="text", threshold=meta["threshold"], assets=ASSETS)
        if (self.pre.config["dictionary_sha256"] != meta["dictionary_sha256"]
                or sha256(self.pre.assets / "preprocess.json") != meta["preprocess_sha256"]):
            raise ValueError("候选词典或预处理配置与冻结记录不符")
        self.pre = MaskedBodyEncoder(self.pre)

    def predict_many(self, rows):
        rows = list(rows)
        results = super().predict_many(rows)
        for row, result in zip(rows, results, strict=True):
            if simple_invitation(row.get("text")):
                result.update(label=0, blocked=False, reason="simple_invitation")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(MentionTrainedPredictor(args.model_dir).predict_one(text=args.text), ensure_ascii=False))
