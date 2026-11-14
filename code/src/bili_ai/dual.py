"""Independent text and nickname fastText channels fused with OR.

The two channels never share features: the text model is trained on enriched text
(including ordinary-nickname AI comments), while the nickname model only sees the
original nickname corpus, so swapped identities are not treated as AI evidence.
"""

import math
from pathlib import Path

import numpy as np

from .common import read_json, sha256
from .data import validate_uid
from .metrics import binary_metrics
from .model import load_blacklist, load_model, positive_scores
from .preprocess import Preprocessor, normalize


def _threshold(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= value <= math.nextafter(1.0, math.inf):
        raise ValueError(f"无效 {name}: {value!r}；必须在 [0, nextafter(1,+inf)]")
    return float(value)


def _candidates(values):
    values = np.asarray(values, dtype=float)
    return np.unique(np.concatenate([values, np.nextafter(values, 2.0), [0.0, math.nextafter(1.0, math.inf)]]))


def select_dual_thresholds(y, text_scores, nickname_scores, extra=(), max_fpr=0.02):
    """Pick both channel thresholds on labelled validation plus extra positives.

    The constraint is the *fused* FPR (text OR nickname), not either channel alone.
    ``extra`` holds (text_score, nickname_score) pairs of held-out ordinary-nickname
    AI comments; they contribute recall but no negatives.
    """
    y = np.asarray([int(value) for value in y])
    text_scores = np.asarray(text_scores, dtype=float)
    nickname_scores = np.asarray(nickname_scores, dtype=float)
    if not len(y) == len(text_scores) == len(nickname_scores) or not len(y):
        raise ValueError("标签与两个通道的评分必须等长且非空")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("阈值选择需要验证集同时包含 0/1 两类")
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr 必须在 [0,1]")
    for values in (text_scores, nickname_scores):
        if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError("评分必须是有限的 [0,1] 数值")
    extra = [(float(t), float(n)) for t, n in extra]
    extra_text = np.asarray([t for t, _ in extra], dtype=float)
    extra_nick = np.asarray([n for _, n in extra], dtype=float)
    for values in (extra_text, extra_nick):
        if values.size and (not np.isfinite(values).all() or values.min() < 0 or values.max() > 1):
            raise ValueError("额外的普通昵称正类评分也必须是有限的 [0,1] 数值")
    negatives = y == 0
    positives = y == 1
    nneg = int(negatives.sum())
    npos = int(positives.sum()) + len(extra)
    candidates_text = _candidates(np.concatenate([text_scores, extra_text]) if len(extra) else text_scores)
    candidates_nick = _candidates(np.concatenate([nickname_scores, extra_nick]) if len(extra) else nickname_scores)

    best = None
    fallback = None
    for text_threshold in candidates_text:
        text_flag = text_scores >= text_threshold
        extra_text_flag = extra_text >= text_threshold
        for nickname_threshold in candidates_nick:
            fused = text_flag | (nickname_scores >= nickname_threshold)
            false_positives = int(fused[negatives].sum())
            fpr = false_positives / nneg
            if fpr > max_fpr:
                if fallback is None or (fpr, -false_positives) < (fallback[0], -fallback[1]):
                    fallback = (fpr, false_positives, text_threshold, nickname_threshold)
                continue
            extra_flag = extra_text_flag | (extra_nick >= nickname_threshold) if len(extra) else np.zeros(0, bool)
            recall = (int(fused[positives].sum()) + int(extra_flag.sum())) / npos
            key = (recall, -fpr, float(text_threshold), float(nickname_threshold))
            if best is None or key > best[0]:
                best = (key, float(text_threshold), float(nickname_threshold), fpr, recall)

    if best is None:
        fpr, false_positives, text_threshold, nickname_threshold = fallback
        return {
            "fusion": "or",
            "text_threshold": text_threshold,
            "nickname_threshold": nickname_threshold,
            "max_fpr": max_fpr,
            "feasible": False,
            "metrics": None,
            "extra": None,
            "selection_rule": "no candidate satisfied the fused FPR cap; returned the minimum-FPR point",
            "caution": f"验证集 {nneg} 条负样本无法满足 FPR <= {max_fpr:.2%}；该点最小 FPR 为 {fpr:.2%}（{false_positives}/{nneg}），不可作为发布阈值。",
        }

    _, text_threshold, nickname_threshold, fpr, recall = best
    fused = (text_scores >= text_threshold) | (nickname_scores >= nickname_threshold)
    extra_flag = (extra_text >= text_threshold) | (extra_nick >= nickname_threshold) if len(extra) else np.zeros(0, bool)
    return {
        "fusion": "or",
        "text_threshold": text_threshold,
        "nickname_threshold": nickname_threshold,
        "max_fpr": max_fpr,
        "feasible": True,
        "metrics": binary_metrics(y.tolist(), fused.astype(float).tolist(), 0.5),
        "extra": {"rows": len(extra), "recall": float(extra_flag.mean()) if len(extra) else None},
        "fused_validation_recall": recall,
        "selection_rule": "maximize fused recall over labelled + extra positives under fused empirical FPR cap; score >= threshold in either channel",
        "caution": f"验证集仅 {nneg} 条负样本；融合 FPR 的分辨率是 1/{nneg}={1/nneg:.2%}，1-2% 的差异不可分辨。",
    }


class DualPredictor:
    """Load one text model and one nickname model; decide with text OR nickname."""

    def __init__(self, text_model, nickname_model, text_threshold=None, nickname_threshold=None, blacklist=None, assets=None):
        self.text_path = Path(text_model)
        self.nickname_path = Path(nickname_model)
        self.text_metadata = read_json(str(self.text_path) + ".json")
        self.nickname_metadata = read_json(str(self.nickname_path) + ".json")
        if self.text_metadata.get("mode") != "text":
            raise ValueError(f"正文模型元数据 mode 必须是 text: {self.text_path}")
        if self.nickname_metadata.get("mode") != "nickname":
            raise ValueError(f"昵称模型元数据 mode 必须是 nickname: {self.nickname_path}")
        if self.text_metadata["dictionary_sha256"] != self.nickname_metadata["dictionary_sha256"]:
            raise ValueError("两个通道使用了不同词典")
        bundle_assets = self.text_path.parent / "assets"
        self.pre = Preprocessor(assets or (bundle_assets if bundle_assets.is_dir() else None))
        preprocess_sha = sha256(self.pre.assets / "preprocess.json")
        for path, metadata, name in ((self.text_path, self.text_metadata, "正文"), (self.nickname_path, self.nickname_metadata, "昵称")):
            if metadata["model_sha256"] != sha256(path):
                raise ValueError(f"{name}模型校验失败: {path}")
            if metadata["dictionary_sha256"] != self.pre.config["dictionary_sha256"]:
                raise ValueError(f"{name}模型与分词词典不匹配")
            if metadata["preprocess_sha256"] != preprocess_sha:
                raise ValueError(f"{name}模型预处理规则不一致")
        self.text_threshold = _threshold(text_threshold if text_threshold is not None else self.text_metadata["threshold"], "text_threshold")
        self.nickname_threshold = _threshold(nickname_threshold if nickname_threshold is not None else self.nickname_metadata["threshold"], "nickname_threshold")
        self.text_model = load_model(self.text_path)
        self.nickname_model = load_model(self.nickname_path)
        self.blacklist = load_blacklist(blacklist)

    def predict_many(self, rows):
        text_lines = [self.pre.encode(row.get("text", ""), "", "text") for row in rows]
        nickname_lines = [self.pre.encode("", row.get("username", ""), "nickname") for row in rows]
        text_valid = [i for i, line in enumerate(text_lines) if line != "field_empty"]
        nickname_valid = [i for i, line in enumerate(nickname_lines) if line != "field_empty"]
        text_scores = dict(zip(text_valid, positive_scores(self.text_model, [text_lines[i] for i in text_valid]), strict=True))
        nickname_scores = dict(zip(nickname_valid, positive_scores(self.nickname_model, [nickname_lines[i] for i in nickname_valid]), strict=True))
        results = []
        for i, row in enumerate(rows):
            uid = row.get("uid", "")
            uid = validate_uid(uid) if uid else ""
            blacklisted = uid in self.blacklist
            text_score = text_scores.get(i)
            nickname_score = nickname_scores.get(i)
            text_flag = text_score is not None and text_score >= self.text_threshold
            nickname_flag = nickname_score is not None and nickname_score >= self.nickname_threshold
            model_label = int(text_flag or nickname_flag) if text_score is not None or nickname_score is not None else None
            trigger = "both" if text_flag and nickname_flag else "text" if text_flag else "nickname" if nickname_flag else None
            reason = "uid_blacklist" if blacklisted else "insufficient_input" if model_label is None else "fusion"
            results.append({
                "uid": uid,
                "text_score": text_score,
                "nickname_score": nickname_score,
                "text_flag": text_flag,
                "nickname_flag": nickname_flag,
                "trigger": trigger,
                "model_label": model_label,
                "label": 1 if blacklisted else model_label,
                "blocked": blacklisted or model_label == 1,
                "reason": reason,
                "threshold_text": self.text_threshold,
                "threshold_nickname": self.nickname_threshold,
                "fusion": "or",
                "partial_input": bool(normalize(row.get("text", ""))) != bool(normalize(row.get("username", ""))),
                "score_is_calibrated_probability": False,
            })
        return results

    def predict_one(self, text="", username="", uid=""):
        return self.predict_many([{"text": text, "username": username, "uid": uid}])[0]
