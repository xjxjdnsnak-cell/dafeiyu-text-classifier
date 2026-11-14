"""dsh-community disagreement analysis: models v5-v8 vs mechanical rule labels.

No training, no tuning, no benchmark modification, no human review.
Uses the FROZEN benchmark (hash-verified) and the four frozen model binaries;
per-row scores are recomputed deterministically (identical inputs -> identical scores).

Questions answered (per user protocol):
- Trigger rate per mechanical-rule-score band (0-9 / 10-19 / 20-29), per model.
- Trigger rate crossed with comment length (<=20 / 21-60 / >=61).
- Trigger rate crossed with video source (is a few videos dominating?).
- Model consistency: rows triggered by all / some / none of v5-v8.
- Drift: rows newly triggered by v8 vs stably triggered since v5.
"""

import datetime
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

import hashlib

from bili_ai.common import read_csv, sha256, write_json
from bili_ai.metrics import binary_metrics
from bili_ai.model import load_model, positive_scores
from bili_ai.preprocess import Preprocessor, normalize

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
ROUNDS = {"v5": "models-text-specialist-20261114-v5", "v6": "models-text-specialist-20261114-v6",
          "v7": "models-text-specialist-20261114-v7", "v8": "models-text-specialist-20261114-v8"}
BANDS = ((0, 9), (10, 19), (20, 29))


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def band_of(score):
    value = max(0, int(score))
    for low, high in BANDS:
        if low <= value <= high:
            return f"{low}-{high}"
    return None


def length_group(text):
    if len(text) <= 20:
        return "<=20"
    if len(text) <= 60:
        return "21-60"
    return ">=61"


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool"))
    digest = sha256(BENCH / "comments.csv")
    if digest != manifest["comments_sha256"]:
        raise ValueError("基准集哈希与冻结清单不一致")

    corpus_uids, corpus_texts, corpus_videos = set(), set(), set()
    for row in load_jsonl(SOURCE / "label-decisions.jsonl"):
        corpus_uids.add(str(row["uid"]))
        corpus_texts.add(normalize(row["text"]))
        for source in row.get("sources", []):
            if source.get("bvid"):
                corpus_videos.add(source["bvid"])
    for row in rows:
        if str(row["uid"]) in corpus_uids or normalize(row["text"]) in corpus_texts or row["bvid"] in corpus_videos:
            raise ValueError("基准集与训练语料出现交集（应为零）")

    dsh_negatives = [r for r in rows if r["pool"] == "ai" and r["label"] == "0"]
    if len(dsh_negatives) != 2450:
        raise ValueError(f"dsh 规则负类行数与冻结评估时不一致: {len(dsh_negatives)} != 2450")

    pre = Preprocessor()
    features = [pre.encode(r["text"], "", "text") for r in dsh_negatives]
    scores_by_model = {}
    for tag, models_dir in ROUNDS.items():
        base = CLASSIFIER / models_dir
        metadata = json.loads((base / "text/model.bin.json").read_text(encoding="utf-8"))
        frozen = json.loads((base / "selection-frozen-before-test.json").read_text(encoding="utf-8"))
        model = load_model(base / "text/model.bin")
        scores = []
        for start in range(0, len(features), 4000):
            scores.extend(positive_scores(model, features[start:start + 4000]))
        del model
        scores_by_model[tag] = {"scores": scores,
                                "thr1": metadata["threshold"],
                                "thr2": frozen["models"]["selected_original"]["working_points"]["0.02"]["threshold"]}

    bands = [band_of(r["score"]) for r in dsh_negatives]
    if None in bands:
        raise ValueError("存在超出 0-29 的规则负类分数")

    analysis = {
        "frozen_benchmark_sha256": digest,
        "dsh_negative_rows": len(dsh_negatives),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "trigger_definition": "model score >= 该模型当轮验证集冻结的1%部署阈值；未调参",
        "rule_score_bands": {},
        "length_within_band": {},
        "video_within_band": {},
        "model_consistency": {},
        "v5_to_v8_drift": {},
        "per_row": [],
    }

    for low, high in BANDS:
        name = f"{low}-{high}"
        idx = [i for i, r in enumerate(dsh_negatives) if band_of(r["score"]) == name]
        band = {"rows": len(idx), "triggered": {}, "mean_model_score": {}, "length_breakdown": {}, "video_breakdown": {}}
        for tag in ROUNDS:
            thr = scores_by_model[tag]["thr1"]
            scores = scores_by_model[tag]["scores"]
            trig = [i for i in idx if scores[i] >= thr]
            band["triggered"][tag] = {"count": len(trig), "rate": len(trig) / len(idx)}
            band["mean_model_score"][tag] = sum(scores[i] for i in idx) / len(idx)
        for length in ("<=20", "21-60", ">=61"):
            sub = [i for i in idx if length_group(dsh_negatives[i]["text"]) == length]
            if not sub:
                continue
            band["length_breakdown"][length] = {"rows": len(sub)}
            for tag in ROUNDS:
                thr = scores_by_model[tag]["thr1"]
                scores = scores_by_model[tag]["scores"]
                band["length_breakdown"][length][tag] = sum(1 for i in sub if scores[i] >= thr) / len(sub)
        videos = Counter(dsh_negatives[i]["bvid"] for i in idx)
        for bvid, count in videos.most_common():
            entry = {"rows": count}
            for tag in ROUNDS:
                thr = scores_by_model[tag]["thr1"]
                scores = scores_by_model[tag]["scores"]
                sub = [i for i in idx if dsh_negatives[i]["bvid"] == bvid]
                entry[tag] = sum(1 for i in sub if scores[i] >= thr) / len(sub)
            band["video_breakdown"][bvid] = entry
        analysis["rule_score_bands"][name] = band

    # model consistency at deployed thresholds
    trig_sets = {}
    for tag in ROUNDS:
        thr = scores_by_model[tag]["thr1"]
        scores = scores_by_model[tag]["scores"]
        trig_sets[tag] = {i for i in range(len(dsh_negatives)) if scores[i] >= thr}
    analysis["model_consistency"] = {
        "triggered_by_all_4": len(set.intersection(*(trig_sets[t] for t in ROUNDS))),
        "triggered_by_none": len(set.intersection(*(set(range(len(dsh_negatives))) - trig_sets[t] for t in ROUNDS))),
        "triggered_counts_histogram": dict(Counter(len({t for t in ROUNDS if i in trig_sets[t]}) for i in range(len(dsh_negatives)))),
    }
    analysis["v5_to_v8_drift"] = {
        "triggered_v5_and_v8": len(trig_sets["v5"] & trig_sets["v8"]),
        "newly_triggered_by_v8_only": len(trig_sets["v8"] - trig_sets["v5"]),
        "dropped_by_v8": len(trig_sets["v5"] - trig_sets["v8"]),
        "band_of_newly_triggered": dict(Counter(band_of(dsh_negatives[i]["score"]) for i in trig_sets["v8"] - trig_sets["v5"])),
        "band_of_stable": dict(Counter(band_of(dsh_negatives[i]["score"]) for i in trig_sets["v5"] & trig_sets["v8"])),
    }
    # per-row audit artifact (no text / no uid)
    for i, row in enumerate(dsh_negatives):
        analysis["per_row"].append({
            "sample_id": "DSB-" + hashlib.sha256((row["uid"] + "|" + row["text"]).encode("utf-8")).hexdigest()[:24],
            "rule_score": int(row["score"]), "band": band_of(row["score"]),
            "length_group": length_group(row["text"]), "bvid": row["bvid"],
            **{tag: round(scores_by_model[tag]["scores"][i], 6) for tag in ROUNDS},
        })
    write_json(HERE / "dsh-disagreement-analysis.json", analysis)

    # markdown summary
    lines = ["# dsh 社区规则负类 × v5–v8 分歧分析", "",
             f"行数 {len(dsh_negatives)}（冻结基准集，哈希已验证；模型与阈值全部冻结，未调参、未训练、未人工审核）", "",
             "触发 = 模型分 ≥ 该模型当轮验证集冻结的 1% 部署阈值。", "",
             "## 按机械规则分数分带", "",
             "| 规则分 | 行数 | v5 触发 | v6 触发 | v7 触发 | v8 触发 | v8 平均模型分 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for low, high in BANDS:
        name = f"{low}-{high}"
        band = analysis["rule_score_bands"][name]
        lines.append(f"| {name} | {band['rows']} | {band['triggered']['v5']['rate']:.1%} | {band['triggered']['v6']['rate']:.1%} | "
                     f"{band['triggered']['v7']['rate']:.1%} | {band['triggered']['v8']['rate']:.1%} | {band['mean_model_score']['v8']:.3f} |")
    lines += ["", "## 分带 × 长度（v8 触发率）", "", "| 规则分 | ≤20字 | 21–60字 | ≥61字 |", "|---|---:|---:|---:|"]
    for low, high in BANDS:
        name = f"{low}-{high}"
        breakdown = analysis["rule_score_bands"][name]["length_breakdown"]
        cells = []
        for length in ("<=20", "21-60", ">=61"):
            if length in breakdown:
                cells.append(f"{breakdown[length]['v8']:.1%}（{breakdown[length]['rows']}行）")
            else:
                cells.append("—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## 分带 × 视频（v8 触发率；检查是否少数视频主导）", "",
              "| 规则分 | 视频 | 行数 | v5 | v6 | v7 | v8 |", "|---|---|---:|---:|---:|---:|---:|"]
    for low, high in BANDS:
        name = f"{low}-{high}"
        for bvid, entry in analysis["rule_score_bands"][name]["video_breakdown"].items():
            lines.append(f"| {name} | {bvid} | {entry['rows']} | {entry['v5']:.1%} | {entry['v6']:.1%} | {entry['v7']:.1%} | {entry['v8']:.1%} |")
    mc = analysis["model_consistency"]
    lines += ["", "## 模型一致性", "",
              f"- 四个模型**全部**触发：{mc['triggered_by_all_4']} 条；全部不触发：{mc['triggered_by_none']} 条。",
              f"- 触发模型数直方图：{json.dumps(mc['triggered_counts_histogram'], ensure_ascii=False)}", "",
              "## v5 → v8 漂移", "",
              f"- v5 与 v8 都触发：{analysis['v5_to_v8_drift']['triggered_v5_and_v8']} 条（稳定触发）。",
              f"- v5 不触发、v8 触发（复核循环期间新增）：{analysis['v5_to_v8_drift']['newly_triggered_by_v8_only']} 条，分带 {json.dumps(analysis['v5_to_v8_drift']['band_of_newly_triggered'], ensure_ascii=False)}。",
              f"- v5 触发、v8 不触发：{analysis['v5_to_v8_drift']['dropped_by_v8']} 条。"]
    (HERE / "dsh-disagreement-analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: analysis[k] for k in ("dsh_negative_rows", "rule_score_bands", "model_consistency", "v5_to_v8_drift")},
                     ensure_ascii=False, indent=2)[:3000])


if __name__ == "__main__":
    main()
