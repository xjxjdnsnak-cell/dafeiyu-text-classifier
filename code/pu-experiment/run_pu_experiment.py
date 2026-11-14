"""Small-scale Bagging PU experiment (diagnostic; v8 untouched).

Data (from the v8 training split only, joined to data-rule-ge30-20261114-r2 for
evidence):
- P: all rule-positive rows (score >= 30) plus historically user-confirmed AI rows.
- U: rule-score < 30 rows WITHOUT independent human evidence. May contain humans
  and AI; used only as a temporary contrast class per bag, never labeled human.
- H: rule-score < 30 rows WITH independent human evidence (account-level: UID in
  the 737-UID manual review with 判定=人类; comment-level: uid+text in the legacy
  human judgments). H is excluded from base-model training and tracked separately.

Method: 5 fastText bags (seeds 20261101..20261105, fixed and different); each bag
trains on ALL of P plus an independently drawn U subset of ceil(|U|/5) rows
(without replacement, per-bag seed). Ensemble = mean of the five bag scores.
Thresholds for the ensemble are selected on the v8 VALIDATION set with the frozen
protocol rule (1% empirical FPR cap; 2% secondary). Arm A = the frozen v8
artifact with its own thresholds. The historical prospective benchmark is NOT
used anywhere. Diagnostic only; neither arm is a release candidate.
"""

import csv
import datetime
import json
import math
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))
sys.path.insert(0, str(CLASSIFIER / "experiments/text-training-20261114"))

from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize
from score30_dataset import read_jsonl

HERE = Path(__file__).resolve().parent
DATA = CLASSIFIER / "data-text-specialist-20261114-v8"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
MODEL_A_DIR = CLASSIFIER / "models-text-specialist-20261114-v8"
BAG_SEEDS = (20261101, 20261102, 20261103, 20261104, 20261105)
MODELS = HERE / "models"
REPORTS = HERE / "reports"


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def rule_score_of(meta):
    value = meta.get("rule_score", meta.get("rule_score_max"))
    return int(value) if value is not None else None


def has_human_evidence(meta, manual_human_uids, legacy_human_pairs):
    if str(meta.get("uid")) in manual_human_uids:
        return "account"
    if (str(meta.get("uid")), normalize(meta["text"])) in legacy_human_pairs:
        return "comment"
    for source in meta.get("sources", []):
        if source.get("manual_account_label") == "human":
            return "account"
        if source.get("legacy_comment_label") == "human":
            return "comment"
    return None


def build_partitions(train_rows, decisions, manual_human_uids, legacy_human_pairs):
    P, U, H = [], [], []
    unknown_score = 0
    for row in train_rows:
        identity = (str(row["uid"]), normalize(row["text"]))
        meta = decisions.get(identity)
        if meta is None:
            raise ValueError(f"训练行缺少来源决策记录: {row['uid']}")
        if int(row["label"]) == 1:
            P.append(row)
            continue
        score = rule_score_of(meta)
        evidence = has_human_evidence(meta, manual_human_uids, legacy_human_pairs)
        if evidence:
            H.append({**row, "evidence": evidence, "rule_score": score})
        else:
            if score is None:
                unknown_score += 1
            U.append({**row, "rule_score": score if score is not None else 0})
    if unknown_score:
        print(json.dumps({"note": "部分无真人证据行无规则分数，按 0 分计入 U", "rows": unknown_score}, ensure_ascii=False))
    ids = [(r["uid"], normalize(r["text"])) for r in P + U + H]
    if len(ids) != len(set(ids)) or set(ids) != {(r["uid"], normalize(r["text"])) for r in train_rows}:
        raise ValueError("P/U/H 划分与训练集不一致")
    if any(int(r["label"]) != 0 for r in H) or any(int(r["label"]) != 0 for r in U):
        raise ValueError("H/U 混入了规则正类")
    return P, U, H


def metrics_for(y_val, score_source, threshold, val_rows, hints, idx_h_account, idx_h_comment, idx_u_neg, idx_ordinary_ai):
    entry = {"threshold": threshold, "all": binary_metrics(y_val, score_source, threshold)}
    ordinary = binary_metrics([y_val[i] for i in idx_ordinary_ai], [score_source[i] for i in idx_ordinary_ai], threshold)
    entry["ordinary_ai_recall"] = {"recall": ordinary["recall"], "confusion_matrix": ordinary["confusion_matrix"],
                                   "rows": len(idx_ordinary_ai)}
    lengths = {}
    for low, high, name in ((1, 20, "1-20"), (21, 60, "21-60"), (61, 10**9, "61+")):
        ai_idx = [i for i in range(len(val_rows)) if y_val[i] == 1 and low <= len(val_rows[i]["text"]) <= high]
        human_idx = [i for i in range(len(val_rows)) if y_val[i] == 0 and low <= len(val_rows[i]["text"]) <= high]
        ai_m = binary_metrics([y_val[i] for i in ai_idx], [score_source[i] for i in ai_idx], threshold) if ai_idx else None
        hu_m = binary_metrics([y_val[i] for i in human_idx], [score_source[i] for i in human_idx], threshold) if human_idx else None
        lengths[name] = {"ai_recall": ai_m["recall"] if ai_m else None, "ai_rows": len(ai_idx),
                         "human_trigger": hu_m["fpr"] if hu_m else None, "human_rows": len(human_idx)}
    entry["length_groups"] = lengths
    entry["rule_negative_trigger"] = {
        "u_rows": binary_metrics([y_val[i] for i in idx_u_neg], [score_source[i] for i in idx_u_neg], threshold) if idx_u_neg else None,
        "h_account": binary_metrics([y_val[i] for i in idx_h_account], [score_source[i] for i in idx_h_account], threshold) if idx_h_account else None,
        "h_comment": binary_metrics([y_val[i] for i in idx_h_comment], [score_source[i] for i in idx_h_comment], threshold) if idx_h_comment else None,
        "h_account_rows": len(idx_h_account), "h_comment_rows": len(idx_h_comment), "u_rows_count": len(idx_u_neg),
    }
    return entry


def main():
    processed, data_manifest = verify_prepared(DATA)
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    manual_human_uids = set()
    manual_path = Path("E:/human-review-uid-decisions (2).csv")
    with manual_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["判定"] == "人类":
                manual_human_uids.add(str(row["UID"]).strip())
    legacy_human_pairs = set()
    for row in load_jsonl(DATA / "legacy_comment_judgments.jsonl"):
        if row.get("label") == "human":
            legacy_human_pairs.add((str(row["uid"]), normalize(row["text"])))

    P, U, H = build_partitions(train_rows, decisions, manual_human_uids, legacy_human_pairs)
    if len(P) < 100 or len(U) < 100 or len(H) < 50:
        raise ValueError(f"P/U/H 规模不足: P={len(P)} U={len(U)} H={len(H)}")

    raw_lines = [line for line in (processed / "text/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_lines) != len(train_rows):
        raise ValueError("train.txt 与 train.csv 行数不一致")
    line_by_identity = {}
    for row, line in zip(train_rows, raw_lines, strict=True):
        line_by_identity[(str(row["uid"]), normalize(row["text"]))] = line

    p_lines = [line_by_identity[(str(r["uid"]), normalize(r["text"]))] for r in P]
    u_lines = [line_by_identity[(str(r["uid"]), normalize(r["text"]))] for r in U]
    subset_size = math.ceil(len(U) / len(BAG_SEEDS))

    bag_meta, bag_paths = [], []
    for index, seed in enumerate(BAG_SEEDS, 1):
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(u_lines))[:subset_size]
        bag_u = [u_lines[i] for i in order]
        lines = p_lines + bag_u
        random.Random(seed).shuffle(lines)
        bag_dir = MODELS / f"bag_{index}"
        bag_dir.mkdir(parents=True, exist_ok=True)
        train_file = bag_dir / "train.txt"
        train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        bin_path = bag_dir / "model.bin"
        train_model(train_file, bin_path, DEFAULT_PARAMS)
        meta = {"schema": 1, "mode": "text", "format": "original", "model_sha256": sha256(bin_path),
                "params": DEFAULT_PARAMS, "bag_seed": seed, "bag_index": index,
                "rows": {"P": len(p_lines), "U_subset": len(bag_u), "total": len(lines)},
                "training_input": str(train_file), "training_input_sha256": sha256(train_file),
                "parent_data": str(DATA), "role": "PU bag base model; diagnostic only"}
        write_json(str(bin_path) + ".json", meta)
        bag_meta.append(meta)
        bag_paths.append(bin_path)
        print(json.dumps({"bag": index, "seed": seed, "P": len(p_lines), "U": len(bag_u)}, ensure_ascii=False), flush=True)

    pre = Preprocessor()
    val_features = [pre.encode(r["text"], "", "text") for r in val_rows]
    y_val = [int(r["label"]) for r in val_rows]
    hints = re.compile(read_json(CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"], re.I)
    idx_ordinary_ai = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints.search(r["username"] or "")]
    val_h_evidence = {}
    for i, row in enumerate(val_rows):
        meta = decisions.get((str(row["uid"]), normalize(row["text"])))
        val_h_evidence[i] = has_human_evidence(meta, manual_human_uids, legacy_human_pairs) if meta else None
    idx_h_account = [i for i in range(len(val_rows)) if y_val[i] == 0 and val_h_evidence[i] == "account"]
    idx_h_comment = [i for i in range(len(val_rows)) if y_val[i] == 0 and val_h_evidence[i] == "comment"]
    idx_u_neg = [i for i in range(len(val_rows)) if y_val[i] == 0 and val_h_evidence[i] is None]

    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    del model_a
    bag_val_scores = []
    for path in bag_paths:
        model = load_model(path)
        bag_val_scores.append(positive_scores(model, val_features))
        del model
    ensemble_scores = [sum(col) / len(col) for col in zip(*bag_val_scores, strict=True)]

    thr_a_1 = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))["threshold"]
    thr_a_2 = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))["models"]["selected_original"]["working_points"]["0.02"]["threshold"]
    sel_b_1 = select_threshold(y_val, ensemble_scores, 0.01)
    sel_b_2 = select_threshold(y_val, ensemble_scores, 0.02)

    results = {
        "A_v8_frozen": {"1pct": metrics_for(y_val, scores_a, thr_a_1, val_rows, hints, idx_h_account, idx_h_comment, idx_u_neg, idx_ordinary_ai),
                        "2pct": metrics_for(y_val, scores_a, thr_a_2, val_rows, hints, idx_h_account, idx_h_comment, idx_u_neg, idx_ordinary_ai),
                        "thresholds": {"fpr_1pct": thr_a_1, "fpr_2pct": thr_a_2},
                        "threshold_source": "v8 validation selection (frozen artifact)"},
        "B_pu_bagging": {"1pct": metrics_for(y_val, ensemble_scores, sel_b_1["threshold"], val_rows, hints, idx_h_account, idx_h_comment, idx_u_neg, idx_ordinary_ai),
                         "2pct": metrics_for(y_val, ensemble_scores, sel_b_2["threshold"], val_rows, hints, idx_h_account, idx_h_comment, idx_u_neg, idx_ordinary_ai),
                         "thresholds": {"fpr_1pct": sel_b_1["threshold"], "fpr_2pct": sel_b_2["threshold"]},
                         "threshold_source": "selected on v8 validation with the frozen protocol rule"},
    }

    perf, sizes = {}, {}
    start = time.perf_counter()
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    perf["a_load_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    bag_loaded = [load_model(path) for path in bag_paths]
    perf["b_load_ms_5_models"] = (time.perf_counter() - start) * 1000
    warm_feats = val_features[:200]
    for m in [model_a] + bag_loaded:
        positive_scores(m, warm_feats[:5])
    start = time.perf_counter()
    for _ in range(3):
        for line in warm_feats:
            positive_scores(model_a, [line])
    perf["a_score_200_comments_ms"] = (time.perf_counter() - start) * 1000 / 3
    start = time.perf_counter()
    for _ in range(3):
        for line in warm_feats:
            for m in bag_loaded:
                positive_scores(m, [line])
    perf["b_score_200_comments_ms_5_models"] = (time.perf_counter() - start) * 1000 / 3
    encode_start = time.perf_counter()
    for row in val_rows[:200]:
        pre.encode(row["text"], "", "text")
    perf["encode_200_comments_ms"] = (time.perf_counter() - encode_start) * 1000
    sizes = {"a_model_bin_bytes": (MODEL_A_DIR / "text/model.bin").stat().st_size,
             "b_ensemble_5x_bin_bytes": sum(p.stat().st_size for p in bag_paths),
             "dictionary_bytes": (CLASSIFIER / "src/bili_ai/assets/jieba.dict.txt").stat().st_size}

    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "design": {"P": len(P), "U": len(U), "H_train": len(H), "bags": len(BAG_SEEDS),
                   "u_subset_per_bag": subset_size, "seeds": list(BAG_SEEDS),
                   "aggregate": "mean of 5 bag scores",
                   "h_excluded_from_training": True,
                   "working_points": "v8 validation set, frozen protocol rule (1% cap, 2% secondary)",
                   "benchmark_use": "none (historical reference only)",
                   "status": "diagnostic; not a release candidate; v8 untouched"},
        "partitions": {"P": len(P), "U": len(U), "H": len(H),
                       "H_evidence": dict(Counter(r["evidence"] for r in H)),
                       "U_rule_score_bands": {f"{low}-{high}": sum(1 for r in U if low <= r["rule_score"] <= high)
                                              for low, high in ((0, 9), (10, 19), (20, 29))}},
        "bag_meta": bag_meta,
        "results": results,
        "perf": perf, "sizes": sizes,
        "limits": ["标签为规则分+人工裁决的弱标签；召回率指与机械规则标签的一致性，不是真实 AI 来源识别率",
                   "验证集真人证据样本规模有限，小样本触发率仅作参考",
                   "集成模型体积约为单模型 5 倍、加载与推理约 5 倍——移动端部署成本显著上升",
                   "单次实验、单划分；结果不用于调参或发布决策"]
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS / "pu-experiment.json", result)
    print(json.dumps({"A_1pct": {"recall": results["A_v8_frozen"]["1pct"]["all"]["recall"],
                                 "fpr": results["A_v8_frozen"]["1pct"]["all"]["fpr"],
                                 "ordinary": results["A_v8_frozen"]["1pct"]["ordinary_ai_recall"]["recall"]},
                      "B_1pct": {"recall": results["B_pu_bagging"]["1pct"]["all"]["recall"],
                                 "fpr": results["B_pu_bagging"]["1pct"]["all"]["fpr"],
                                 "ordinary": results["B_pu_bagging"]["1pct"]["ordinary_ai_recall"]["recall"]},
                      "partitions": result["partitions"], "perf": perf, "sizes": sizes},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
