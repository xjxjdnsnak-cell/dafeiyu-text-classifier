"""Learn addressee-directed questions and requests from body-only intent annotations.

fit INPUT.csv MODEL_DIR OUTPUT_DIR expects text and intent (0 or 1).
score INPUT.csv OUTPUT_DIR SCORED.csv reads text and preserves other input columns.
MODEL_DIR holds the original Chinese TinyBERT config, vocabulary and safetensors.
"""
import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizer
from safetensors.torch import load_file, save_file


MENTION = re.compile(r"^(?:回复\s*)?@[^\s@,，:：。?？!！]{1,32}(?:\s+|[,，:：])")


def addressed_body(text):
    normalized = unicodedata.normalize("NFKC", str(text)).strip()
    match = MENTION.match(normalized)
    if not match:
        return ""
    body = normalized[match.end():].lstrip(" ,:，：").strip()
    if not 1 <= len(body) <= 64 or "@" in body:
        return ""
    return body


class IntentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config, add_pooling_layer=False)
        self.head = nn.Linear(config.hidden_size, 2)

    def forward(self, tokens):
        states = self.bert(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (states * mask).sum(1) / mask.sum(1)
        return self.head(pooled)


def tokens_for(tokenizer, texts, device):
    return tokenizer(texts, padding=True, truncation=True, max_length=96,
                     return_tensors="pt").to(device)


def fit(rows, pretrained, output, device):
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    config = BertConfig.from_json_file(str(pretrained / "config.json"))
    tokenizer = BertTokenizer(vocab=str(pretrained / "vocab.txt"))
    model = IntentModel(config)
    source = load_file(str(pretrained / "model.safetensors"))
    model.bert.load_state_dict({key: source["bert." + key] for key in model.bert.state_dict()})
    model.to(device)
    bodies = [addressed_body(text) for text in rows.text]
    labels = rows.intent.to_numpy(dtype=int)
    if not all(bodies) or set(labels) != {0, 1}:
        raise ValueError("Fit requires eligible addressed examples of both intent classes.")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    model.train()
    for _ in range(6):
        for batch in np.array_split(rng.permutation(len(rows)), int(np.ceil(len(rows) / 16))):
            encoded = tokens_for(tokenizer, [bodies[i] for i in batch], device)
            loss = nn.functional.cross_entropy(model(encoded), torch.tensor(labels[batch], device=device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output)
    config.to_json_file(str(output / "config.json"), use_diff=False)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(output / "intent.safetensors"))


def score(rows, checkpoint, device):
    tokenizer = BertTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = IntentModel(BertConfig.from_json_file(str(checkpoint / "config.json"))).to(device)
    model.load_state_dict(load_file(str(checkpoint / "intent.safetensors")))
    model.eval()
    bodies = [addressed_body(text) for text in rows.text]
    indices = [i for i, body in enumerate(bodies) if body]
    values = np.full(len(rows), np.nan)
    with torch.inference_mode():
        for start in range(0, len(indices), 32):
            batch = indices[start:start + 32]
            tokens = tokens_for(tokenizer, [bodies[i] for i in batch], device)
            values[batch] = model(tokens).softmax(dim=-1)[:, 1].cpu().numpy()
    # Fixed before looking at evaluation results. Outside the routing scope, abstain.
    return rows.assign(intent_score=values, predicted_request=values >= 0.80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "score"))
    parser.add_argument("input", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input, keep_default_na=False, dtype={"text": str})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.action == "fit":
        fit(data, args.model, args.output, device)
    else:
        score(data, args.model, device).to_csv(args.output, index=False, encoding="utf-8-sig")
