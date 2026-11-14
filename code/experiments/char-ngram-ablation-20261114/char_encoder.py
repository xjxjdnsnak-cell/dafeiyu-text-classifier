"""Character N-gram feature encoder for the augmentation ablation.

Extracts explicit continuous character 2/3/4-grams over the normalized full
comment body (crossing jieba token boundaries), with the dedicated prefix
"chx_" so they cannot collide with the existing "txt_" word features.

The original "txt_" token sequence is preserved VERBATIM and FIRST: fastText
builds wordNgrams over consecutive tokens in the line, so appending chx tokens
at the end keeps every original txt_ bigram intact. The only new bigrams are at
the txt_/chx_ boundary and within the chx run (hashed into the same bucket
space; reported as a feature-count change).

Truncation follows the frozen preprocess rule (max_text_chars on the
normalized body before tokenization).
"""

import json
from pathlib import Path

from bili_ai.preprocess import Preprocessor, normalize

CHAR_NGRAM_SIZES = (2, 3, 4)
CHAR_PREFIX = "chx_"


def char_ngram_tokens(text, pre: Preprocessor, sizes=CHAR_NGRAM_SIZES, max_chars=None):
    """Continuous char n-grams over the normalized body (crossing token boundaries)."""
    limit = max_chars or pre.config["max_text_chars"]
    body = normalize(text)[:limit]
    tokens = []
    for n in sizes:
        for i in range(len(body) - n + 1):
            tokens.append(CHAR_PREFIX + body[i:i + n])
    return tokens


def encode_char_augmented(text, pre: Preprocessor, sizes=CHAR_NGRAM_SIZES, max_chars=None):
    """Original txt_ feature line + appended cross-boundary char n-grams."""
    base = pre.encode(text, "", "text")
    if base == "field_empty":
        return base
    chx = char_ngram_tokens(text, pre, sizes=sizes, max_chars=max_chars)
    return base + " " + " ".join(chx) if chx else base


def feature_stats(lines):
    """Per-prefix token totals and corpus-level unique feature counts."""
    unique = {"txt": set(), "chx": set(), "other": set()}
    total = {"txt": 0, "chx": 0, "other": 0}
    per_line = []
    for line in lines:
        tokens = [t for t in line.split(" ") if t and not t.startswith("__label__")]
        counts = {"txt": 0, "chx": 0, "other": 0}
        for token in tokens:
            key = "txt" if token.startswith("txt_") else "chx" if token.startswith("chx_") else "other"
            counts[key] += 1
            unique[key].add(token)
            total[key] += 1
        per_line.append(counts)
    avg = {k: sum(c[k] for c in per_line) / max(1, len(per_line)) for k in ("txt", "chx", "other")}
    return {"unique": {k: len(v) for k, v in unique.items()},
            "total": total,
            "avg_per_line": {k: round(v, 2) for k, v in avg.items()}}


def save_encoder_config(path: Path, extra=None):
    config = {"char_ngram_sizes": list(CHAR_NGRAM_SIZES), "char_prefix": CHAR_PREFIX,
              "placement": "appended after the original txt_ token sequence (preserves all original word bigrams)",
              "extraction_source": "normalized full body after max_text_chars truncation (crosses jieba boundaries)",
              "dedup": "none (occurrence counts preserved, same as txt_ tokens)"}
    if extra:
        config.update(extra)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config
