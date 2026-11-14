"""Positive-class metrics and validation-only threshold selection."""

import math


def validate(y, scores):
    if len(y) != len(scores) or not y:
        raise ValueError("标签与评分必须等长且非空")
    if any(v not in (0, 1) for v in y):
        raise ValueError("标签必须是 0 或 1")
    if any(not math.isfinite(float(s)) or not 0 <= s <= 1 for s in scores):
        raise ValueError("评分必须为有限的 [0,1] 数值")


def binary_metrics(y, scores, threshold, abstain=None):
    validate(y, scores)
    abstain = [False] * len(y) if abstain is None else list(abstain)
    if len(abstain) != len(y) or any(not isinstance(v, bool) for v in abstain):
        raise ValueError("abstain 必须是与标签等长的布尔列表")
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError("threshold 必须在 [0, nextafter(1,+inf)]")
    tn = fp = fn = tp = 0
    for label, score, missing in zip(y, scores, abstain, strict=True):
        flagged = not missing and score >= threshold
        if label:
            tp += flagged
            fn += not flagged
        else:
            fp += flagged
            tn += not flagged
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    nneg = tn + fp
    fpr = fp / nneg if nneg else None
    # Wilson interval shows how little a small holdout can establish about FPR.
    interval = None
    if nneg:
        z = 1.959963984540054
        center = (fpr + z*z/(2*nneg)) / (1+z*z/nneg)
        margin = z*math.sqrt(fpr*(1-fpr)/nneg+z*z/(4*nneg*nneg))/(1+z*z/nneg)
        interval = [max(0.0, center-margin), min(1.0, center+margin)]
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0, "fpr": fpr, "fpr_wilson_95": interval, "confusion_matrix": [[tn, fp], [fn, tp]], "support": {"normal": nneg, "ai_style": tp+fn}, "predicted_positive": tp+fp, "abstentions": sum(abstain), "coverage": 1-sum(abstain)/len(y)}


def select_threshold(y, scores, max_fpr=0.01, abstain=None):
    validate(y, scores)
    if set(y) != {0, 1} or not 0 <= max_fpr <= 1:
        raise ValueError("阈值选择需要验证集两类及有效 max_fpr")
    candidates = {0.0, math.nextafter(1.0, math.inf)}
    for score in scores:
        candidates.add(float(score))
        candidates.add(math.nextafter(float(score), math.inf))
    feasible = []
    for threshold in sorted(candidates):
        m = binary_metrics(y, scores, threshold, abstain)
        if m["fpr"] <= max_fpr:
            feasible.append((m["recall"], m["precision"], m["f1"], threshold, m))
    recall, precision, f1, threshold, metrics = max(feasible)
    return {"threshold": threshold, "max_validation_fpr": max_fpr, "selection_rule": "maximize recall under empirical FPR cap, then precision/F1, then larger threshold; score >= threshold", "metrics": metrics, "degenerate_no_detections": metrics["predicted_positive"] == 0, "caution": "验证集经验 FPR 上限不是总体误判率保证；小样本必须结合区间与人工复核。"}
