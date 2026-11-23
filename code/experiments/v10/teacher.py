"""Score only the original training partition with the frozen Chinese RoBERTa."""
import argparse
from pathlib import Path

import torch
import transformers
from transformers import BertForSequenceClassification, BertTokenizer

from inputs import SPLITS, TEACHER, check, save, split
from views import teacher_pair

TEMPERATURE = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for name, expected in TEACHER.items():
        check(args.teacher / name, expected)
    rows = split(args.processed, "train")
    pairs = [teacher_pair(r["text"]) for r in rows]
    texts = sorted({t for pair in pairs for t in pair}, key=lambda t: (len(t), t))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BertForSequenceClassification.from_pretrained(args.teacher, local_files_only=True).to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(args.teacher, local_files_only=True)
    scores = {}
    with torch.inference_mode():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            logits = model(**tokens.to(device)).logits
            soft = (logits / TEMPERATURE).softmax(-1)[:, 1].cpu().tolist()
            scores.update(zip(batch, soft, strict=True))
            if i % 3200 == 0:
                print(f"Teacher {i}/{len(texts)} on {device}", flush=True)
    save(args.output, {"train_sha256": SPLITS["train"], "teacher": TEACHER,
                       "temperature": TEMPERATURE, "max_tokens": 128, "device": device,
                       "transformers": transformers.__version__, "torch": str(torch.__version__),
                       "unique_inputs": len(texts), "rows": len(rows),
                       "soft_targets": [(scores[a] + scores[b]) / 2 for a, b in pairs]})


if __name__ == "__main__":
    main()
