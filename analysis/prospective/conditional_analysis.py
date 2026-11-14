"""Conditional-matching analysis: does the high trigger rate of the three
tutorial/promo videos vs the fish-disaster recap video survive after controlling
for mechanical rule score and comment length?

Strictly analysis-only: frozen benchmark (hash-verified), frozen model binaries,
frozen thresholds, frozen prediction scores (recomputed deterministically and
cross-checked against dsh-disagreement-analysis.json within 1e-6).
No training, no tuning, no recollection, no labeling; existing artifacts untouched.

Method:
- Strata: rule-score band (0-9/10-19/20-29) x length (1-20/21-60/61+).
- Trigger: model score >= that model's frozen 1% deployment threshold.
- Direct standardization: within each rule-score band, each video's trigger rate is
  re-weighted to the POOLED length distribution of all four videos, using only
  strata with common support (primary: every compared video has >=5 rows; secondary:
  >=1 row). Empty strata are never extrapolated.
- Exact Oaxaca decomposition of the raw gap (video - BV126nt6GE4W) within each band:
    raw gap = composition term + conditional term, where
    composition = sum_s (w_s^X - w_s^R) * r_pooled,s      (length mix difference)
    conditional = sum_s w_s^X (r_X,s - r_pooled,s) - sum_s w_s^R (r_R,s - r_pooled,s)
  (telescoping identity; holds exactly when missing strata contribute zero).
- Uncertainty: Wilson intervals per cell; approximate intervals for standardized
  rates via stratum-variance propagation. Low-support strata are flagged and never
  used for deterministic claims.
"""

import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

from bili_ai.common import read_csv, sha256, write_json
from bili_ai.model import load_model, positive_scores
from bili_ai.preprocess import Preprocessor, normalize

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
ROUNDS = {"v5": "models-text-specialist-20261114-v5", "v6": "models-text-specialist-20261114-v6",
          "v7": "models-text-specialist-20261114-v7", "v8": "models-text-specialist-20261114-v8"}
VIDEOS = ("BV1GJnZ6MEFw", "BV11qJ96LELG", "BV1REx46NEc4", "BV126nt6GE4W")
REFERENCE = "BV126nt6GE4W"
BANDS = ("0-9", "10-19", "20-29")
LENGTHS = ("1-20", "21-60", "61+")
MIN_ROBUST = 5
Z = 1.959963984540054


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def band_of(score):
    value = max(0, int(score))
    for low, high in ((0, 9), (10, 19), (20, 29)):
        if low <= value <= high:
            return f"{low}-{high}"
    return None


def length_group(text):
    if len(text) <= 20:
        return "1-20"
    if len(text) <= 60:
        return "21-60"
    return "61+"


def wilson(successes, total):
    if total == 0:
        return None
    p = successes / total
    denom = 1 + Z * Z / total
    center = (p + Z * Z / (2 * total)) / denom
    margin = Z * math.sqrt(p * (1 - p) / total + Z * Z / (4 * total * total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def cell_rate(hits, total):
    rate = hits / total if total else None
    return {"triggered": hits, "rows": total, "rate": rate, "wilson": wilson(hits, total)}


def standardized_rate(idx_by_stratum, weights, triggered):
    rate = 0.0
    variance = 0.0
    rows = 0
    for stratum, weight in weights.items():
        idx = idx_by_stratum.get(stratum, [])
        if not idx:
            continue
        hits = sum(1 for i in idx if triggered[i])
        r = hits / len(idx)
        rate += weight * r
        variance += weight * weight * r * (1 - r) / len(idx)
        rows += len(idx)
    half = Z * math.sqrt(variance) if variance else 0.0
    return {"rate": rate, "approx_interval": [max(0.0, rate - half), min(1.0, rate + half)], "rows": rows}


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool"))
    if sha256(BENCH / "comments.csv") != manifest["comments_sha256"]:
        raise ValueError("基准集哈希与冻结清单不一致")
    previous = json.loads((HERE / "dsh-disagreement-analysis.json").read_text(encoding="utf-8"))

    dsh = [r for r in rows if r["pool"] == "ai" and r["label"] == "0"]
    if len(dsh) != 2450:
        raise ValueError(f"dsh 规则负类行数异常: {len(dsh)}")
    saved = {item["sample_id"]: item for item in previous["per_row"]}
    for row in dsh:
        sample_id = "DSB-" + hashlib.sha256((row["uid"] + "|" + row["text"]).encode("utf-8")).hexdigest()[:24]
        if sample_id not in saved:
            raise ValueError(f"逐行预测记录缺少样本: {sample_id}")

    pre = Preprocessor()
    features = [pre.encode(r["text"], "", "text") for r in dsh]
    thresholds, scores_by_model = {}, {}
    for tag, models_dir in ROUNDS.items():
        base = CLASSIFIER / models_dir
        metadata = json.loads((base / "text/model.bin.json").read_text(encoding="utf-8"))
        model = load_model(base / "text/model.bin")
        scores = []
        for start in range(0, len(features), 4000):
            scores.extend(positive_scores(model, features[start:start + 4000]))
        del model
        max_diff = max(abs(scores[i] - saved[sample_id][tag])
                       for i, sample_id, r in [(i, "DSB-" + hashlib.sha256((r["uid"] + "|" + r["text"]).encode("utf-8")).hexdigest()[:24], r)
                                               for i, r in enumerate(dsh)])
        if max_diff > 1e-6:
            raise ValueError(f"{tag}: 重算分数与已保存预测不一致 (max diff {max_diff})")
        scores_by_model[tag] = scores
        thresholds[tag] = metadata["threshold"]

    bands = [band_of(r["score"]) for r in dsh]
    lengths = [length_group(r["text"]) for r in dsh]
    videos = [r["bvid"] for r in dsh]
    if any(v not in VIDEOS for v in videos):
        raise ValueError("出现未知视频")
    triggered = {tag: [scores_by_model[tag][i] >= thresholds[tag] for i in range(len(dsh))] for tag in ROUNDS}

    strata_counts, strata_rates = {}, {}
    for band in BANDS:
        for length in LENGTHS:
            key = f"{band}|{length}"
            strata_counts[key] = {}
            strata_rates[key] = {}
            for video in VIDEOS:
                idx = [i for i in range(len(dsh)) if bands[i] == band and lengths[i] == length and videos[i] == video]
                hits = sum(1 for i in idx if any(triggered[tag][i] for tag in [])) if False else None
                per_model = {tag: cell_rate(sum(1 for i in idx if triggered[tag][i]), len(idx)) for tag in ROUNDS}
                if idx:
                    strata_counts[key][video] = len(idx)
                    strata_rates[key][video] = per_model
                else:
                    strata_counts[key][video] = 0

    report = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "frozen_benchmark_sha256": sha256(BENCH / "comments.csv"),
              "dsh_negative_rows": len(dsh),
              "prediction_check": "recomputed full-precision scores match recorded per-row predictions within 1e-6",
              "thresholds": thresholds,
              "strata_counts": strata_counts,
              "strata_rates": strata_rates,
              "standardized": {},
              "decomposition": {}}

    for band in BANDS:
        band_rows = sum(1 for b in bands if b == band)
        pooled_counts = {length: sum(1 for i in range(len(dsh)) if bands[i] == band and lengths[i] == length) for length in LENGTHS}
        pooled_total = sum(pooled_counts.values())
        pooled_weights = {length: pooled_counts[length] / pooled_total for length in LENGTHS if pooled_total}
        pooled_rate = {}
        for length in LENGTHS:
            idx = [i for i in range(len(dsh)) if bands[i] == band and lengths[i] == length]
            pooled_rate[length] = {}
            for tag in ROUNDS:
                hits = sum(1 for i in idx if triggered[tag][i])
                pooled_rate[length][tag] = hits / len(idx) if idx else None
        band_std = {"pooled_length_weights": pooled_weights, "pooled_trigger_rates": pooled_rate}
        for support_name, minimum in (("robust_min5", MIN_ROBUST), ("min1", 1)):
            supported = [length for length in LENGTHS
                         if all(sum(1 for i in range(len(dsh)) if bands[i] == band and lengths[i] == length and videos[i] == video) >= minimum
                                for video in VIDEOS)]
            weights = {length: pooled_weights[length] for length in supported}
            weight_sum = sum(weights.values())
            weights = {length: w / weight_sum for length, w in weights.items()} if weight_sum else {}
            band_entry = {"supported_strata": supported, "weights": weights}
            for video in VIDEOS:
                entry = {}
                for tag in ROUNDS:
                    raw_idx = [i for i in range(len(dsh)) if bands[i] == band and videos[i] == video]
                    raw_hits = sum(1 for i in raw_idx if triggered[tag][i])
                    raw = cell_rate(raw_hits, len(raw_idx))
                    idx_by_stratum = {length: [i for i in range(len(dsh)) if bands[i] == band and lengths[i] == length and videos[i] == video]
                                      for length in supported}
                    covered = all(idx_by_stratum[length] for length in supported)
                    if covered:
                        std = standardized_rate(idx_by_stratum, weights, triggered[tag])
                        entry[tag] = {"raw": raw, "standardized": std}
                    else:
                        entry[tag] = {"raw": raw, "standardized": None,
                                      "note": "存在无该视频样本的分层，不做标准化外推"}
                band_entry[video] = entry
            band_std[support_name] = band_entry
        gaps = {}
        for tag in ROUNDS:
            gap_entry = {}
            for video in VIDEOS[:3]:
                raw_idx_video = [i for i in range(len(dsh)) if bands[i] == band and videos[i] == video]
                raw_idx_ref = [i for i in range(len(dsh)) if bands[i] == band and videos[i] == REFERENCE]
                raw_rate_video = sum(1 for i in raw_idx_video if triggered[tag][i]) / len(raw_idx_video)
                raw_rate_ref = sum(1 for i in raw_idx_ref if triggered[tag][i]) / len(raw_idx_ref)
                raw_gap = raw_rate_video - raw_rate_ref
                total_video, total_ref = len(raw_idx_video), len(raw_idx_ref)
                composition, conditional = 0.0, 0.0
                for length in LENGTHS:
                    n_x = sum(1 for i in raw_idx_video if lengths[i] == length)
                    n_r = sum(1 for i in raw_idx_ref if lengths[i] == length)
                    w_x = n_x / total_video if total_video else 0.0
                    w_r = n_r / total_ref if total_ref else 0.0
                    hits_pooled = sum(1 for i in range(len(dsh)) if bands[i] == band and lengths[i] == length and triggered[tag][i])
                    r_pooled = hits_pooled / pooled_counts[length] if pooled_counts[length] else None
                    if r_pooled is not None:
                        composition += (w_x - w_r) * r_pooled
                    if n_x:
                        conditional += w_x * (sum(1 for i in raw_idx_video if lengths[i] == length and triggered[tag][i]) / n_x - r_pooled)
                    if n_r:
                        conditional -= w_r * (sum(1 for i in raw_idx_ref if lengths[i] == length and triggered[tag][i]) / n_r - r_pooled)
                gap_entry[video] = {"raw_gap": raw_gap, "composition_term": composition,
                                    "conditional_term": conditional,
                                    "identity_residual": raw_gap - (composition + conditional)}
            gaps[tag] = gap_entry
        band_std["decomposition_vs_reference"] = gaps
        report["standardized"][band] = band_std

    write_json(HERE / "conditional-analysis.json", report)
    print(json.dumps({"written": str(HERE / "conditional-analysis.json"),
                      "strata": len(report["strata_counts"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
