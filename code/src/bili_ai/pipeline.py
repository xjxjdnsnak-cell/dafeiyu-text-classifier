"""Validation selects parameters/thresholds. Test is read only by evaluate()."""

import importlib.metadata
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from .common import read_csv, read_json, write_csv, write_json, sha256
from .data import FIELDS
from .metrics import binary_metrics, select_threshold
from .model import DEFAULT_PARAMS, train_model, quantize_model, positive_scores, load_model
from .preprocess import Preprocessor, MODES, ASSETS


def verify_prepared(data_dir):
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    manifest = read_json(processed / "manifest.json")
    for path, digest in [(data_dir / "comments.csv", manifest["comments_sha256"]), (data_dir / "splits.csv", manifest["splits_sha256"]), (ASSETS / "preprocess.json", manifest["preprocess_sha256"]), (ASSETS / "jieba.dict.txt", manifest["dictionary_sha256"])]:
        if sha256(path) != digest:
            raise ValueError(f"数据/预处理校验失败，文件已变化: {path}；重新 prepare 和 train")
    for relative, digest in manifest["files"].items():
        if sha256(processed / relative) != digest:
            raise ValueError(f"预处理文件校验失败，文件已变化: {relative}")
    return processed, manifest


def _examples(processed, mode, split):
    rows = read_csv(processed / f"{split}.csv", FIELDS)
    lines = (processed / mode / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    labels, features = [], []
    for row, line in zip(rows, lines, strict=True):
        label, feature = line.split(" ", 1)
        if label != "__label__" + row["label"]:
            raise ValueError("CSV 与 fastText 标签不匹配")
        labels.append(int(row["label"]))
        features.append(feature)
    return rows, labels, features


def _metadata(mode, model, path, manifest, params, selected):
    return {"schema": 1, "mode": mode, "format": "quantized" if model.is_quantized() else "original", "model_sha256": sha256(path), "model_bytes": path.stat().st_size, "params": params, "threshold": selected["threshold"], "threshold_source": "validation", "selection_splits": ["train", "validation"], "validation": selected, "dictionary_sha256": manifest["dictionary_sha256"], "preprocess_sha256": manifest["preprocess_sha256"], "dataset": {k: manifest[k] for k in ("comments_sha256", "splits_sha256", "files")}, "versions": {p: importlib.metadata.version(p) for p in ("fasttext-wheel", "jieba", "numpy")}, "warning": "AI 风格评分不是已校准的真实 AI 概率；UID 不进入特征。"}


def train(data_dir, models_dir, modes=MODES, params=None, tune=False, max_fpr=0.01):
    processed, manifest = verify_prepared(data_dir)
    output = {}
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"无效 mode: {mode}")
        folder = Path(models_dir) / mode
        folder.mkdir(parents=True, exist_ok=True)
        _, y, features = _examples(processed, mode, "validation")
        base = {**DEFAULT_PARAMS, **(params or {})}
        candidates = [base]
        if tune:
            candidates += [{**base, "dim": 16, "epoch": 50}, {**base, "epoch": 50, "lr": 0.8}]
        best = None
        trials = []
        for i, options in enumerate(candidates):
            candidate = folder / ".candidate.bin"
            start = time.perf_counter()
            model = train_model(processed / mode / "train.txt", candidate, options)
            selected = select_threshold(y, positive_scores(model, features), max_fpr, abstain=[f == "field_empty" for f in features])
            metrics = selected["metrics"]
            trials.append({"params": options, "validation": selected, "training_seconds": time.perf_counter()-start, "model_bytes": candidate.stat().st_size})
            rank = (metrics["recall"], metrics["precision"], metrics["f1"], -candidate.stat().st_size)
            if best is None or rank > best:
                best = rank
                candidate.replace(folder / "model.bin")
                metadata = _metadata(mode, model, folder / "model.bin", manifest, options, selected)
            candidate.unlink(missing_ok=True)
            del model
        metadata["trials"] = trials
        metadata["tuning"] = "bounded validation grid" if tune else "fixed low-resource configuration"
        write_json(folder / "model.bin.json", metadata)
        output[mode] = metadata
    return output


def _check_metadata(path, manifest):
    metadata = read_json(str(path) + ".json")
    if metadata["model_sha256"] != sha256(path):
        raise ValueError(f"模型校验失败: {path}")
    if metadata["dataset"] != {k: manifest[k] for k in ("comments_sha256", "splits_sha256", "files")}:
        raise ValueError("模型训练所用数据/划分与当前 prepared 数据不同")
    if metadata["preprocess_sha256"] != manifest["preprocess_sha256"]:
        raise ValueError("模型预处理规则不一致")
    return metadata


def check_model_pair(original_path, quantized_path, manifest):
    original = _check_metadata(Path(original_path), manifest)
    quantized = _check_metadata(Path(quantized_path), manifest)
    if quantized["original_model_sha256"] != original["model_sha256"]:
        raise ValueError("量化模型来源已过期：不是由当前原始模型生成，请重新 quantize")
    return original, quantized


def quantize(data_dir, models_dir, modes=MODES, cutoff=20000, dsub=2, max_fpr=0.01):
    processed, manifest = verify_prepared(data_dir)
    output = {}
    for mode in modes:
        folder = Path(models_dir) / mode
        original = _check_metadata(folder / "model.bin", manifest)
        _, y, features = _examples(processed, mode, "validation")
        path = folder / "model.ftz"
        start = time.perf_counter()
        model = quantize_model(folder / "model.bin", processed / mode / "train.txt", path, cutoff, dsub)
        scores = positive_scores(model, features)
        selected = select_threshold(y, scores, max_fpr, abstain=[f == "field_empty" for f in features])
        metadata = _metadata(mode, model, path, manifest, original["params"], selected)
        metadata["quantization"] = {"cutoff": cutoff, "dsub": dsub, "qnorm": True, "qout": False, "retrain": False, "seconds": time.perf_counter()-start}
        metadata["original_model_sha256"] = original["model_sha256"]
        metadata["original_threshold"] = original["threshold"]
        metadata["validation_at_original_threshold"] = binary_metrics(y, scores, original["threshold"], abstain=[f == "field_empty" for f in features])
        write_json(str(path) + ".json", metadata)
        output[mode] = metadata
    return output


def _summary_ms(values):
    return {"mean_ms": statistics.mean(values), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)), "measurements": len(values)}


def benchmark(model, pre, rows, mode, features, repeats=3, batch_size=64):
    if repeats < 1:
        raise ValueError("repeats 必须 >=1")
    # Warm all paths; timings exclude disk I/O, model load and dictionary initialization.
    for row in rows[:5]:
        positive_scores(model, [pre.encode(row["text"], row["username"], mode)])
    preprocess_times, model_times, total_times, batch_model, batch_total = [], [], [], [], []
    for _ in range(repeats):
        for row, feature in zip(rows, features, strict=True):
            start = time.perf_counter_ns()
            line = pre.encode(row["text"], row["username"], mode)
            mid = time.perf_counter_ns()
            positive_scores(model, [line])
            end = time.perf_counter_ns()
            preprocess_times.append((mid-start)/1e6)
            total_times.append((end-start)/1e6)
            start = time.perf_counter_ns()
            positive_scores(model, [feature])
            model_times.append((time.perf_counter_ns()-start)/1e6)
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            start = time.perf_counter_ns()
            lines = [pre.encode(r["text"], r["username"], mode) for r in batch]
            positive_scores(model, lines)
            elapsed = (time.perf_counter_ns()-start)/1e6
            batch_total.append({"count": len(batch), "ms": elapsed})
            start = time.perf_counter_ns()
            positive_scores(model, features[i:i+batch_size])
            batch_model.append({"count": len(batch), "ms": (time.perf_counter_ns()-start)/1e6})
    def batches(samples):
        total_ms = sum(s["ms"] for s in samples)
        count = sum(s["count"] for s in samples)
        return {"batch_size": batch_size, "batches": len(samples), "total_comments": count, "mean_batch_ms": statistics.mean(s["ms"] for s in samples), "per_comment_ms": total_ms/count, "comments_per_second": count/total_ms*1000}
    return {"preprocessing_single": _summary_ms(preprocess_times), "model_only_single": _summary_ms(model_times), "end_to_end_single": _summary_ms(total_times), "model_only_batch": batches(batch_model), "end_to_end_batch": batches(batch_total), "scope": "warm desktop CPU; excludes model/dictionary initialization, file I/O and UID policy; not Android latency"}


def evaluate(data_dir, models_dir, reports_dir, modes=MODES, repeats=3):
    processed, manifest = verify_prepared(data_dir)
    reports_dir = Path(reports_dir)
    pre_start = time.perf_counter()
    pre = Preprocessor()
    pre_init_ms = (time.perf_counter()-pre_start)*1000
    results, flat, prediction_rows = [], [], []
    for mode in modes:
        rows, y, features = _examples(processed, mode, "test")
        abstain = [f == "field_empty" for f in features]
        pair = {}
        folder = Path(models_dir) / mode
        check_model_pair(folder / "model.bin", folder / "model.ftz", manifest)
        for suffix, fmt in (("bin", "original"), ("ftz", "quantized")):
            path = Path(models_dir) / mode / f"model.{suffix}"
            metadata = _check_metadata(path, manifest)
            start = time.perf_counter()
            model = load_model(path)
            load_ms = (time.perf_counter()-start)*1000
            scores = positive_scores(model, features)
            metrics = binary_metrics(y, scores, metadata["threshold"], abstain)
            result = {"mode": mode, "format": fmt, "threshold": metadata["threshold"], "threshold_source": "validation", "test_rows": len(y), "model_bytes": path.stat().st_size, "model_sha256": sha256(path), "model_load_ms": load_ms, "metrics": metrics, "latency": benchmark(model, pre, rows, mode, features, repeats)}
            if fmt == "quantized":
                original = pair["original"]
                result["original_threshold"] = original["threshold"]
                result["at_original_threshold"] = binary_metrics(y, scores, original["threshold"], abstain)
                result["size_reduction_percent"] = (1-result["model_bytes"]/original["model_bytes"])*100
                result["compression_ratio"] = original["model_bytes"]/result["model_bytes"]
                result["fpr_delta_retuned_threshold"] = metrics["fpr"]-original["metrics"]["fpr"]
                result["fpr_delta_at_original_threshold"] = result["at_original_threshold"]["fpr"]-original["metrics"]["fpr"]
            pair[fmt] = result
            results.append(result)
            flat.append({"mode": mode, "format": fmt, "threshold": result["threshold"], **{k: metrics[k] for k in ("precision", "recall", "f1", "fpr")}, "model_bytes": result["model_bytes"], "single_e2e_mean_ms": result["latency"]["end_to_end_single"]["mean_ms"], "batch_e2e_per_comment_ms": result["latency"]["end_to_end_batch"]["per_comment_ms"]})
            prediction_rows.extend({"mode": mode, "format": fmt, "uid": row["uid"], "label": truth, "ai_style_score": None if missing else score, "predicted_label": None if missing else int(score >= metadata["threshold"]), "abstained": missing, "blocked": not missing and score >= metadata["threshold"]} for row, truth, score, missing in zip(rows, y, scores, abstain, strict=True))
            del model
    import_report = Path(data_dir) / "import_report.json"
    report = {"schema": 1, "evaluation_scope": "same held-out UID test set; no threshold/parameter tuning on test; blacklist excluded", "data_warning": "Imported dataset labels include offline rule weak labels and user-confirmed positives. These are label-agreement metrics, not proof of AI authorship or real-world FPR.", "source_import": read_json(import_report) if import_report.exists() else None, "prepared_manifest": manifest, "environment": {"python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(), "fasttext_threads": 1, "timing_repeats": repeats}, "dictionary_bytes": (pre.assets / "jieba.dict.txt").stat().st_size, "dictionary_initialization_ms": pre_init_ms, "results": results}
    alignment_path = Path(data_dir) / "plan1_alignment.json"
    if alignment_path.exists():
        alignment = read_json(alignment_path)
        if alignment["shared_comments_csv_sha256"] != manifest["comments_sha256"] or alignment["shared_splits_csv_sha256"] != manifest["splits_sha256"]:
            raise ValueError("方案一对齐记录已过期，请核对共享 CSV 与 UID 划分")
        report["plan1_alignment"] = alignment
    write_json(reports_dir / "evaluation.json", report)
    write_csv(reports_dir / "comparison.csv", flat, list(flat[0]))
    write_csv(reports_dir / "test_predictions.csv", prediction_rows, ("mode", "format", "uid", "label", "ai_style_score", "predicted_label", "abstained", "blocked"))
    _write_markdown(reports_dir / "evaluation.md", report)
    return report


def _write_markdown(path, report):
    manifest = report["prepared_manifest"]
    lines = ["# fastText 同测试集评估", "", "**限制：来源含规则弱标签与用户确认正类。下列指标衡量标签一致性，不能当作真实 AI 识别准确率；评论风格不能证明作者身份。**", "", "参数与阈值只在 train/validation 上确定；测试只用于最终评估；UID 黑名单不参与。", "", f"清洗：`{manifest['cleaning']}`", "", f"划分：`{manifest['splits']}`", "", "|特征|模型|Precision|Recall|F1|FPR|阈值|模型 KiB|单条端到端 ms|批量每条 ms|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if "plan1_alignment" in report:
        lines[4:4] = ["已直接复用方案一的规范 comments.csv / splits.csv；三集合逐条记录一致，见 data/plan1_alignment.json。", ""]
    for r in report["results"]:
        m = r["metrics"]
        lines.append(f"|{r['mode']}|{r['format']}|{m['precision']:.4f}|{m['recall']:.4f}|{m['f1']:.4f}|{m['fpr']:.4%}|{r['threshold']:.9g}|{r['model_bytes']/1024:.1f}|{r['latency']['end_to_end_single']['mean_ms']:.4f}|{r['latency']['end_to_end_batch']['per_comment_ms']:.4f}|")
    for r in report["results"]:
        lines.extend(["", f"## {r['mode']} / {r['format']}", "", f"混淆矩阵（行=真实 0/1，列=预测 0/1）：`{r['metrics']['confusion_matrix']}`；FPR Wilson 95% 区间：`{r['metrics']['fpr_wilson_95']}`。", f"模型加载 {r['model_load_ms']:.2f} ms。"])
        if r["format"] == "quantized":
            lines.append(f"量化缩小 {r['size_reduction_percent']:.2f}%（{r['compression_ratio']:.2f} 倍）。原始阈值下 FPR={r['at_original_threshold']['fpr']:.4%}，FPR 变化={r['fpr_delta_at_original_threshold']:+.4%}；量化独立验证阈值下 FPR 变化={r['fpr_delta_retuned_threshold']:+.4%}。")
    lines.extend(["", f"冻结词典另外占 {report['dictionary_bytes']/1024/1024:.2f} MiB，初始化 {report['dictionary_initialization_ms']:.2f} ms；不能只拿 ftz 大小代表整个部署体积。", "", "延迟为本机桌面 CPU 预热测量，含预处理和评分，排除启动、文件 I/O、黑名单。JSON 另列模型内核、P50/P95、批量吞吐和环境；不代表 Android 真机。", "", "测试结果不用于继续调参；需要扩充人工负类、跨主题/时间/视频的独立测试与上线复核。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
