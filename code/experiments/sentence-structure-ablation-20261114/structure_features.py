"""Sentence-structure feature extractor for the structure-feature ablation.

A dozen fixed, discretized features derived from the normalized body (text only;
no nickname/UID). Prefix "sfx_", appended after the original txt_ token sequence
(preserves all original txt_ bigrams, same placement discipline as the chx round).

Features:
1. sfx_不是而是_1        不是……而是…… (literal 而是 within 40 chars)
2. sfx_与其说不如说_1     与其说……不如说……
3. sfx_不仅还_1          不仅……还/也/而且……
4. sfx_句长变化_低/中/高   max-min sentence length spread
5. sfx_连接词比例_0/低/高   connective occurrences per sentence
6. sfx_重复句式_0/1/多     sentences sharing the same 4-char opener
7. sfx_句子数_1/2-3/4-6/7+  sentence count
8. sfx_平均句长_短/中/长    mean sentence length

Note (from the gray policy): corpus rows containing 不是…而是…/与其说…不如说…/
不仅…还… patterns were EXCLUDED from label-0 (human) training data by the gray
policy — these features can therefore fire mostly on AI rows. Firing rates are
reported per feature so this asymmetry is visible, never hidden.
"""

import re
from collections import Counter

from bili_ai.preprocess import normalize

SFX_PREFIX = "sfx_"
MAX_BODY_CHARS = 4096
PATTERNS = (
    ("sfx_不是而是", r"不是[^。！？!?;\n]{0,40}而是"),
    ("sfx_与其说不如说", r"与其说[^。！？!?;\n]{0,40}不如说"),
    ("sfx_不仅还", r"不仅[^。！？!?;\n]{0,40}(?:还|也|而且)"),
)
CONNECTIVES = ("但是", "不过", "然而", "所以", "因此", "而且", "并且", "虽然",
               "可是", "其实", "于是", "总之", "不仅", "与其说", "换句话说", "也就是说")
SENT_SPLIT = re.compile(r"[。！？!?;；\n]+")


def split_sentences(body):
    return [s for s in SENT_SPLIT.split(body) if s.strip()]


def structure_tokens(text, pre):
    body = normalize(text)[:pre.config["max_text_chars"]]
    if not body:
        return []
    sentences = split_sentences(body)
    n_sent = len(sentences) or 1
    lens = [len(s) for s in sentences]
    tokens = []
    for name, pattern in PATTERNS:
        if re.search(pattern, body):
            tokens.append(name + "_1")
    spread = (max(lens) - min(lens)) if lens else 0
    tokens.append("sfx_句长变化_" + ("低" if spread <= 4 else "中" if spread <= 15 else "高"))
    conn_ratio = sum(body.count(c) for c in CONNECTIVES) / n_sent
    tokens.append("sfx_连接词比例_" + ("0" if conn_ratio == 0 else "低" if conn_ratio < 0.5 else "高"))
    openers = Counter(s[:4] for s in sentences if len(s) >= 4)
    dup = (max(openers.values()) - 1) if openers else 0
    tokens.append("sfx_重复句式_" + ("0" if dup == 0 else "1" if dup == 1 else "多"))
    tokens.append("sfx_句子数_" + ("1" if n_sent <= 1 else "2-3" if n_sent <= 3 else "4-6" if n_sent <= 6 else "7+"))
    avg_len = sum(lens) / n_sent
    tokens.append("sfx_平均句长_" + ("短" if avg_len <= 10 else "中" if avg_len <= 25 else "长"))
    return tokens


def encode_structure_augmented(text, pre):
    base = pre.encode(text, "", "text")
    sfx = structure_tokens(text, pre)
    return base + " " + " ".join(sfx) if sfx else base


def feature_stats(lines):
    unique_sfx, unique_txt = set(), set()
    total_sfx = total_txt = 0
    per_line = []
    for line in lines:
        tokens = [t for t in line.split(" ") if t and not t.startswith("__label__")]
        sfx = [t for t in tokens if t.startswith("sfx_")]
        txt = [t for t in tokens if t.startswith("txt_")]
        unique_sfx.update(sfx)
        unique_txt.update(txt)
        total_sfx += len(sfx)
        total_txt += len(txt)
        per_line.append({"sfx": len(sfx), "txt": len(txt)})
    return {"unique_sfx": len(unique_sfx), "unique_txt": len(unique_txt),
            "total_sfx": total_sfx, "total_txt": total_txt,
            "avg_sfx_per_line": round(total_sfx / max(1, len(per_line)), 2),
            "avg_txt_per_line": round(total_txt / max(1, len(per_line)), 2)}


def firing_rate_report(rows, pre):
    """Firing rate of each pattern/bucket feature over a labeled row list (dicts with text/label)."""
    names = [name for name, _ in PATTERNS]
    fires = {name: Counter() for name in names}
    buckets = defaultdict(Counter)
    for row in rows:
        body = normalize(row["text"])
        tokens = set(structure_tokens(row["text"], pre))
        label = str(row.get("label"))
        for name, pattern in PATTERNS:
            if re.search(pattern, body):
                fires[name][label] += 1
        for token in tokens:
            buckets[token][label] += 1
    return {"pattern_fires": {name: dict(cnt) for name, cnt in fires.items()},
            "bucket_fires": {token: dict(cnt) for token, cnt in sorted(buckets.items())}}
