"""Single ablation: does treating every rule-score<30 row as human cap recall?

Arm A: the existing v8 baseline model (weight 1 for every training row; reused,
never retrained — identity asserted via its recorded training-input hash).
Arm B: same train split, same fastText params, but UNCONFIRMED human rows
(no user human evidence in manual UID review or legacy judgments) contribute an
EXPECTED weight of 0.5 via seeded Bernoulli keep; confirmed-human rows keep
weight 1; AI rows keep weight 1. fastText has no per-example weights, so the
expected-weight semantics is realized by Bernoulli row keep (documented).

Working points for BOTH arms are selected on the v8 VALIDATION set with the
frozen protocol rule (maximize recall under 1% empirical FPR cap; 2% secondary).
The historical reference benchmark is NOT used for selection or evaluation.
Diagnostic only: neither arm is a release candidate; final verdicts need a fresh
independent test set.
"""

import datetime
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2].parent
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
sys.path.insert(0, str(CLASSIFIER / "experiments/text-training-20261114"))

from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize

sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
from score30_dataset import read_jsonl

DATA = CLASSIFIER / "data-text-specialist-20261114-v8"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
MODEL_A_DIR = CLASSIFIER / "models-text-specialist-20261114-v8"
MODELS_B = CLASSIFIER / "models-neg-ablation-20261114/B"
REPORTS = CLASSIFIER / "reports-neg-ablation-20261114"
NEG_KEEP_PROB = 0.5
SEED = 42
EXPANSION_LIST_SOURCE = ROOT / "out/collector-expand-topics-20261114/videos.txt"
EXPANSION_VIDEOS = {line.strip() for line in EXPANSION_LIST_SOURCE.read_text(encoding="utf-8-sig").splitlines() if line.strip()} | {"BV1i24P6rEmt"}


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def confirmed_human(meta):
    """确认人类 = 737-UID 人工表的判定或历史人工意见；证据在顶层或 sources[] 内嵌行。"""
    manual = meta.get("manual_account_review") or {}
    if manual.get("判定") == "人类":
        return True
    if (meta.get("legacy_comment_review") or {}).get("label") == "human":
        return True
    for source in meta.get("sources", []):
        if source.get("manual_account_label") == "human" or source.get("legacy_comment_label") == "human":
            return True
    return False


def rule_score_of(meta):
    """v1-derived rows carry rule_score_max; r2-new rows carry top-level rule_score."""
    value = meta.get("rule_score", meta.get("rule_score_max"))
    return int(value) if value is not None else None


def evaluate_arm(y_val, scores_val, threshold, val_rows, hints, expansion_set, rule_lookup):
    metrics = binary_metrics(y_val, scores_val, threshold)
    idx_ordinary_ai = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints.search(r["username"] or "")]
    ordinary = binary_metrics([y_val[i] for i in idx_ordinary_ai], [scores_val[i] for i in idx_ordinary_ai], threshold)
    low_rows = [i for i, r in enumerate(val_rows) if (rule_lookup[(str(r["uid"]), normalize(r["text"]))]["rule_score"] or 0) < 30]  # None 视为低分
    low_human = [i for i in low_rows if y_val[i] == 0]
    low_ai = [i for i in low_rows if y_val[i] == 1]
    expansion_human = [i for i, r in enumerate(val_rows)
                       if y_val[i] == 0 and set(rule_lookup[(str(r["uid"]), normalize(r["text"]))]["bvids"]) & expansion_set]
    groups = {}
    for low_bound, high_bound, name in ((1, 20, "1-20"), (21, 60, "21-60"), (61, 10**9, "61+")):
        idx = [i for i, r in enumerate(val_rows) if y_val[i] == 0 and low_bound <= len(r["text"]) <= high_bound]
        groups[name] = binary_metrics([y_val[i] for i in idx], [scores_val[i] for i in idx], threshold) if idx else None
    confirmed_rows = [i for i in low_human if rule_lookup[(str(val_rows[i]["uid"]), normalize(val_rows[i]["text"]))]["confirmed_human"]]
    unconfirmed_rows = [i for i in low_human if not rule_lookup[(str(val_rows[i]["uid"]), normalize(val_rows[i]["text"]))]["confirmed_human"]]
    return {"threshold": threshold, "all": metrics,
            "ordinary_ai_recall": ordinary,
            "low_score_rows": {"rows": len(low_rows),
                               "trigger_rate_overall": (sum(1 for i in low_rows if scores_val[i] >= threshold) / len(low_rows)) if low_rows else None,
                               "human_fpr": binary_metrics([y_val[i] for i in low_human], [scores_val[i] for i in low_human], threshold) if low_human else None,
                               "hard_ai_recall": binary_metrics([y_val[i] for i in low_ai], [scores_val[i] for i in low_ai], threshold) if low_ai else None,
                               "human_fpr_confirmed": binary_metrics([y_val[i] for i in confirmed_rows], [scores_val[i] for i in confirmed_rows], threshold) if confirmed_rows else None,
                               "human_fpr_unconfirmed": binary_metrics([y_val[i] for i in unconfirmed_rows], [scores_val[i] for i in unconfirmed_rows], threshold) if unconfirmed_rows else None,
                               "confirmed_human_rows": len(confirmed_rows), "unconfirmed_human_rows": len(unconfirmed_rows)},
            "expansion_video_human_fpr": binary_metrics([y_val[i] for i in expansion_human], [scores_val[i] for i in expansion_human], threshold) if expansion_human else None,
            "human_fpr_by_length": groups}


def main():
    processed, data_manifest = verify_prepared(DATA)
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    meta_a = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))
    train_weight1 = processed / "weighted-text/train-weight1.txt"
    if sha256(train_weight1) != meta_a["training_input_sha256"]:
        raise ValueError("A 臂训练输入哈希与 v8 元数据不一致")
    selection_a = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))
    if selection_a["selected_trial"] != "baseline" or selection_a["recommended_format"] != "bin":
        raise ValueError("A 臂必须对应 v8 baseline bin 工件")

    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    hints = re.compile(read_json(CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"], re.I)

    rng = np.random.RandomState(SEED)
    keep_flags, sampling = [], {}
    confirmed = unconfirmed = 0
    for row in train_rows:
        meta = decisions.get((str(row["uid"]), normalize(row["text"])))
        if meta is None:
            raise ValueError(f"训练行缺少来源决策记录: {row['uid']}")
        if int(row["label"]) == 1:
            keep_flags.append(True)
            continue
        score = rule_score_of(meta)
        if score is not None and score >= 30:
            raise ValueError(f"训练集存在规则分>=30的人类行: {row['uid']}")
        if confirmed_human(meta):
            confirmed += 1
            keep_flags.append(True)
        else:
            unconfirmed += 1
            keep_flags.append(bool(rng.rand() < NEG_KEEP_PROB))
    ai_train = sum(1 for r in train_rows if int(r["label"]) == 1)
    sampling = {"train_rows": len(train_rows), "ai_rows": ai_train, "human_rows": len(train_rows) - ai_train,
                "confirmed_human_kept": confirmed, "unconfirmed_total": unconfirmed,
                "unconfirmed_kept": sum(1 for i, row in enumerate(train_rows)
                                        if int(row["label"]) == 0 and not confirmed_human(decisions[(str(row["uid"]), normalize(row["text"]))]) and keep_flags[i]),
                "keep_prob": NEG_KEEP_PROB, "seed": SEED}

    MODELS_B.mkdir(parents=True, exist_ok=True)
    raw_lines = [line for line in (processed / "text/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) != len(train_rows):
        raise ValueError("train.txt 行数与 train.csv 不一致")
    kept_lines = [line for row, line, keep in zip(train_rows, raw_lines, keep_flags, strict=True) if keep]
    kept_labels = [int(row["label"]) for row, keep in zip(train_rows, keep_flags) if keep]
    if len(kept_lines) != len(kept_labels):
        raise ValueError("B 训练文件行列不一致")
    shuffled = list(zip(kept_labels, kept_lines))
    random.Random(SEED).shuffle(shuffled)
    b_train = MODELS_B / "train.txt"
    b_train.write_text("\n".join(f"__label__{label} {line}" for label, line in shuffled) + "\n", encoding="utf-8")

    b_bin = MODELS_B / "model.bin"
    b_model = train_model(b_train, b_bin, DEFAULT_PARAMS)
    b_scores_val = positive_scores(b_model, val_features)
    sel_b_1 = select_threshold(y_val, b_scores_val, 0.01)
    sel_b_2 = select_threshold(y_val, b_scores_val, 0.02)
    b_meta = {"schema": 1, "mode": "text", "format": "original",
              "model_sha256": sha256(b_bin), "model_bytes": b_bin.stat().st_size,
              "params": DEFAULT_PARAMS, "threshold": sel_b_1["threshold"],
              "threshold_source": "v8 validation selection (frozen protocol rule)",
              "training_input": str(b_train), "training_input_sha256": sha256(b_train),
              "training_rows": len(shuffled),
              "arm": "B_negative_downweight_0p5_expected",
              "parent_arm_a": str(MODEL_A_DIR / "text/model.bin"),
              "parent_arm_a_sha256": meta_a["model_sha256"],
              "sampling": sampling,
              "note": "负类降权0.5以种子Bernoulli(0.5)行保留实现（期望权重）；确认人类行与AI行权重1；fastText无逐行权重",
              "dictionary_sha256": data_manifest["dictionary_sha256"],
              "preprocess_sha256": data_manifest["preprocess_sha256"]}
    write_json(str(b_bin) + ".json", b_meta)
    del b_model

    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    del model_a

    rule_lookup = {identity: {"rule_score": rule_score_of(meta), "confirmed_human": confirmed_human(meta),
                              "bvids": sorted({s.get("bvid") for s in meta.get("sources", []) if s.get("bvid")})}
                   for identity, meta in decisions.items()}
    thr_a_1 = meta_a["threshold"]
    thr_a_2 = selection_a["models"]["selected_original"]["working_points"]["0.02"]["threshold"]
    results = {
        "A_v8_baseline": {"model_sha256": meta_a["model_sha256"],
                          "thresholds": {"fpr_1pct": thr_a_1, "fpr_2pct": thr_a_2},
                          "threshold_source": "v8 validation selection (frozen artifact)",
                          "1pct": evaluate_arm(y_val, scores_a, thr_a_1, val_rows, hints, EXPANSION_VIDEOS, rule_lookup),
                          "2pct": evaluate_arm(y_val, scores_a, thr_a_2, val_rows, hints, EXPANSION_VIDEOS, rule_lookup)},
        "B_neg_weight_0p5": {"model_sha256": sha256(b_bin),
                             "thresholds": {"fpr_1pct": sel_b_1["threshold"], "fpr_2pct": sel_b_2["threshold"]},
                             "threshold_source": "selected on v8 validation with the frozen protocol rule",
                             "1pct": evaluate_arm(y_val, b_scores_val, sel_b_1["threshold"], val_rows, hints, EXPANSION_VIDEOS, rule_lookup),
                             "2pct": evaluate_arm(y_val, b_scores_val, sel_b_2["threshold"], val_rows, hints, EXPANSION_VIDEOS, rule_lookup)},
    }
    result = {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "design": {"arm_a": "existing v8 baseline (weight 1 everywhere; reused artifact, not retrained)",
                         "arm_b": "same train split/params; unconfirmed human rows expected weight 0.5 (Bernoulli keep, seed 42); confirmed-human and AI rows weight 1",
                         "working_points": "v8 validation set, frozen protocol rule (1% cap, 2% secondary)",
                         "benchmark_use": "none (historical reference only)",
                         "status": "diagnostic ablation; not a release candidate; final verdicts need a fresh independent test set"},
              "sampling": sampling, "arms": results}
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "ablation.json", result)
    summary = {}
    for arm in results:
        one = results[arm]["1pct"]
        summary[arm] = {"recall": one["all"]["recall"], "fpr": one["all"]["fpr"],
                        "ordinary_recall": one["ordinary_ai_recall"]["recall"],
                        "low_score_trigger": one["low_score_rows"]["trigger_rate_overall"],
                        "hard_ai_recall": (one["low_score_rows"]["hard_ai_recall"] or {}).get("recall"),
                        "human_fpr_unconfirmed": (one["low_score_rows"]["human_fpr_unconfirmed"] or {}).get("fpr"),
                        "human_fpr_confirmed": (one["low_score_rows"]["human_fpr_confirmed"] or {}).get("fpr"),
                        "expansion_fpr": (one["expansion_video_human_fpr"] or {}).get("fpr")}
    print(json.dumps({"sampling": sampling, "summary_1pct": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
