"""fastText operations and a separate local UID policy layer."""

import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import fasttext

from .common import read_json, sha256
from .data import validate_uid
from .preprocess import Preprocessor, normalize

DEFAULT_PARAMS = {"dim": 32, "bucket": 20000, "epoch": 30, "lr": 0.5, "wordNgrams": 2, "minn": 2, "maxn": 4, "minCount": 1, "loss": "softmax", "thread": 1, "seed": 42, "verbose": 0}


@contextmanager
def native_path(path, write=False):
    path = Path(path).resolve()
    if os.name != "nt" or str(path).isascii():
        yield str(path)
        return
    # fastText 0.9.2 uses narrow Windows file APIs; Python handles the Unicode copy.
    with tempfile.TemporaryDirectory(prefix="bili-ai-") as tmp:
        if not str(tmp).isascii():
            raise ValueError("Windows fastText 需要 ASCII 临时目录，请将 TEMP/TMP 配置为 ASCII 路径")
        staged = Path(tmp) / "artifact"
        if not write:
            shutil.copyfile(path, staged)
        yield str(staged)
        if write:
            shutil.copyfile(staged, path)


def load_model(path):
    with native_path(path) as staged:
        return fasttext.load_model(staged)


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with native_path(path, write=True) as staged:
        model.save_model(staged)


def train_model(train_file, output, params=None):
    options = {**DEFAULT_PARAMS, **(params or {})}
    if options["dim"] < 2 or options["bucket"] < 256 or options["thread"] != 1 or options["seed"] != 42:
        raise ValueError("dim >= 2，bucket >= 256；可复现训练固定 thread=1 / seed=42")
    if options["epoch"] < 1 or not math.isfinite(options["lr"]) or options["lr"] <= 0 or options["wordNgrams"] < 1 or options["minn"] < 0 or options["maxn"] < options["minn"]:
        raise ValueError("无效训练参数：epoch/wordNgrams >=1，lr>0，0<=minn<=maxn")
    with native_path(train_file) as staged:
        model = fasttext.train_supervised(input=staged, **options)
    if set(model.labels) != {"__label__0", "__label__1"}:
        raise ValueError("训练集必须同时且仅包含 __label__0 / __label__1")
    save_model(model, output)
    return model


def positive_scores(model, lines):
    if not lines:
        return []
    labels, probabilities = model.predict(list(lines), k=2)
    result = []
    for names, probs in zip(labels, probabilities):
        by_label = dict(zip(names, (float(p) for p in probs)))
        if set(by_label) != {"__label__0", "__label__1"}:
            raise ValueError("模型不是本项目的二分类标签格式")
        # fastText adds 1e-5 to probabilities; normalize both classes identically in JS.
        score = by_label["__label__1"] / sum(by_label.values())
        if not math.isfinite(score):
            raise ValueError("模型输出了无效评分")
        result.append(max(0.0, min(1.0, score)))
    return result


def quantize_model(raw_path, train_file, output, cutoff=20000, dsub=2):
    if Path(raw_path).resolve() == Path(output).resolve():
        raise ValueError("量化输出不得覆盖原始模型")
    if cutoff and cutoff < 256:
        raise ValueError("product quantization 至少需要 256 行，cutoff 必须为 0 或 >=256")
    model = load_model(raw_path)
    if model.is_quantized():
        raise ValueError("输入已经量化，请使用原始 .bin")
    if dsub < 1 or dsub > model.get_dimension():
        raise ValueError("dsub 必须在 [1, dim]")
    # qout is disabled: a binary output matrix has fewer than 256 rows.
    with native_path(train_file) as staged:
        model.quantize(input=staged, cutoff=cutoff, qnorm=True, qout=False, dsub=dsub, retrain=False, thread=1, verbose=0)
    save_model(model, output)
    return model


def load_blacklist(path):
    if path is None:
        return set()
    path = Path(path)
    if not path.exists():
        return set()
    return {validate_uid(line.split("#", 1)[0].strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.split("#", 1)[0].strip()}


def update_blacklist(path, action, uid=None):
    path = Path(path)
    values = load_blacklist(path)
    if action in ("add", "remove"):
        uid = validate_uid(uid)
        values.add(uid) if action == "add" else values.discard(uid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Local UID blacklist; independent of model scores\n" + "".join(u + "\n" for u in sorted(values)), encoding="utf-8")
    elif action != "list":
        raise ValueError("黑名单操作必须为 add/remove/list")
    return sorted(values)


class Predictor:
    def __init__(self, model_path, mode=None, threshold=None, blacklist=None, assets=None):
        path = Path(model_path)
        metadata_path = Path(str(path) + ".json")
        self.metadata = read_json(metadata_path) if metadata_path.exists() else {}
        if mode and self.metadata and mode != self.metadata["mode"]:
            raise ValueError("特征模式与模型元数据不一致；请加载对应的 A/B/C 模型")
        self.mode = mode or self.metadata.get("mode")
        self.threshold = threshold if threshold is not None else self.metadata.get("threshold")
        if self.mode is None or self.threshold is None:
            raise ValueError("模型缺少元数据，请显式指定 mode 和 threshold")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= math.nextafter(1.0, math.inf):
            raise ValueError("无效 threshold")
        bundle_assets = path.parent / "assets"
        self.pre = Preprocessor(assets or (bundle_assets if bundle_assets.is_dir() else None))
        if self.metadata and self.metadata["dictionary_sha256"] != self.pre.config["dictionary_sha256"]:
            raise ValueError("模型与分词词典不匹配")
        if self.metadata and (self.metadata["model_sha256"] != sha256(path) or self.metadata["preprocess_sha256"] != sha256(self.pre.assets / "preprocess.json")):
            raise ValueError("模型/预处理规则校验失败；请使用同一次训练导出的文件")
        self.model = load_model(path)
        self.blacklist = load_blacklist(blacklist)

    def predict_many(self, rows):
        features = [self.pre.encode(r.get("text", ""), r.get("username", ""), self.mode) for r in rows]
        valid = [i for i, line in enumerate(features) if line != "field_empty"]
        scores = dict(zip(valid, positive_scores(self.model, [features[i] for i in valid])))
        results = []
        for i, row in enumerate(rows):
            uid = row.get("uid", "")
            uid = validate_uid(uid) if uid else ""
            blacklisted = uid in self.blacklist
            score = scores.get(i)
            model_label = int(score >= self.threshold) if score is not None else None
            reason = "uid_blacklist" if blacklisted else "insufficient_input" if score is None else "model"
            results.append({"uid": uid, "ai_style_score": score, "model_label": model_label, "label": 1 if blacklisted else model_label, "blocked": blacklisted or model_label == 1, "reason": reason, "threshold": self.threshold, "mode": self.mode, "partial_input": self.mode == "combined" and (not normalize(row.get("text", "")) or not normalize(row.get("username", ""))), "score_is_calibrated_probability": False})
        return results

    def predict_one(self, text="", username="", uid=""):
        return self.predict_many([{"text": text, "username": username, "uid": uid}])[0]
