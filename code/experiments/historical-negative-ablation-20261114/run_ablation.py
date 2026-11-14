"""Historical negative-sample augmentation ablation (A/B/C), diagnostic only.

Arm A: the frozen v8 artifact (reused; thresholds untouched).
Arm B: v8 train + eligible NEW historical negatives (pre-outbreak comments from
       out/pre-fish-human-corpus-20261114, label_source=historical_timestamp_proxy).
Arm C: v8 train + the SAME NUMBER of negative rows resampled WITH replacement from
       the old training negatives (no new data) — isolates whether B's effect comes
       from new language information or merely more negative rows.

Isolation (hard requirements): new rows may not overlap the v8 validation set, the
v8 test set, or the frozen prospective benchmark by UID, normalized body, or video;
old test data never flows back to training; duplicates of existing training
identities are excluded. Every exclusion is counted with a reason.

Thresholds: A keeps its frozen artifact thresholds; B/C select on the v8 validation
set with the frozen protocol rule (1% cap; 2% secondary). After B/C thresholds are
frozen, ONE diagnostic evaluation on the demoted prospective benchmark is allowed
(historical-reference performance, not acceptance). No tuning from it.
"""

import datetime
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-training-20261114"))

from bili_ai.common import read_csv, read_json, sha256, write_json
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize

HERE = Path(__file__).resolve().parent
DATA = CLASSIFIER if False else ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
SOURCE = ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2"
MODEL_A_DIR = ROOT / "fasttext-classifier/models-text-specialist-20261114-v8"
CANDIDATES = ROOT / "out/pre-fish-human-corpus-20261114/historical-human-candidates.csv"
BENCH = ROOT / "out/prospective-benchmark-20261114/benchmark"
MODELS = HERE / "models"
SEED = 42


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def rule_score_of(meta):
    value = meta.get("rule_score", meta.get("rule_score_max"))
    return int(value) if value is not None else None


def main():
    processed, data_manifest = verify_prepared(ROOT / "fasttext-classifier/data-text-specialist-20261114-v8")
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    # ---- A identity ----
    meta_a = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))
    train_weight1 = processed / "weighted-text/train-weight1.txt"
    if sha256(train_weight1) != meta_a["training_input_sha256"]:
        raise ValueError("A 臂训练输入哈希与 v8 元数据不一致")

    # ---- protected sets (v8 partitions + frozen benchmark) ----
    lineage_split, lineage_bvids = {}, {}
    for row in load_jsonl(DATA / "sample-lineage.jsonl"):
        identity = (str(row["uid"]), normalize(row["text"]))
        lineage_split[identity] = row.get("split", "")
        lineage_bvids[identity] = set(row.get("bvids", []) or [])
    protected = {"uid": set(), "text": set(), "video": set(), "identity": set()}
    per_split_uids = {s: set() for s in ("train", "validation", "test")}
    for identity, split in lineage_split.items():
        uid, text = identity
        bvids = lineage_bvids.get(identity, set())
        protected["uid"].add(uid)
        protected["text"].add(text)
        protected["identity"].add(identity)
        protected["video"] |= bvids
        if split in per_split_uids:
            per_split_uids[split].add(uid)
    bench_uids, bench_texts, bench_bvids = set(), set(), set()
    for row in read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool")):
        bench_uids.add(str(row["uid"]))
        bench_texts.add(normalize(row["text"]))
        bench_bvids.add(row["bvid"])
    protected["uid"] |= bench_uids
    protected["text"] |= bench_texts
    protected["video"] |= bench_bvids
    protected_uids = per_split_uids["validation"] | per_split_uids["test"] | bench_uids
    protected_videos = set()
    for identity, split in lineage_split.items():
        if split in ("validation", "test"):
            protected_videos |= lineage_bvids.get(identity, set())
    protected_videos |= bench_bvids

    # ---- candidates -> eligible new negatives (full isolation) ----
    candidates = read_csv(CANDIDATES, ("candidate_id", "source", "uid", "rpid", "bvid", "text", "length_group",
                                       "rule_score", "ctime_unix", "ctime_beijing", "topic_group",
                                       "proposed_label", "label_source", "authorship_verified",
                                       "training_pool_eligible", "overlap_flags"))
    new_rows = [r for r in candidates if r["source"] == "new_collection"]
    exclusions = Counter()
    eligible = []
    train_identities = {(str(r["uid"]), normalize(r["text"])) for r in train_rows}
    for row in new_rows:
        uid, text, bvid = str(row["uid"]), normalize(row["text"]), row["bvid"]
        identity = (uid, text)
        if uid in protected_uids:
            exclusions["uid_in_validation_or_test_or_benchmark"] += 1
            continue
        if text in protected["text"]:
            exclusions["body_in_validation_or_test_or_benchmark"] += 1
            continue
        if bvid in protected_videos:
            exclusions["video_in_validation_or_test_or_benchmark"] += 1
            continue
        if identity in train_identities:
            exclusions["identity_already_in_training"] += 1
            continue
        if row["training_pool_eligible"] != "True":
            exclusions["flagged_ineligible_in_corpus_build"] += 1
            continue
        eligible.append(row)
    added_rows = len(eligible)
    if added_rows == 0:
        raise ValueError("没有通过隔离的新增负样本")

    # ---- B training file ----
    pre = Preprocessor()
    raw_train_lines = [line for line in (DATA / "processed/text/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_train_lines) != len(train_rows):
        raise ValueError("train.txt 行数不一致")
    b_extra = [f"__label__0 {pre.encode(normalize(r['text']), '', 'text')}" for r in eligible]
    b_all = raw_train_lines + b_extra
    random.Random(SEED).shuffle(b_all)
    MODELS.mkdir(parents=True, exist_ok=True)
    b_train = MODELS / "B/train.txt"
    b_train.parent.mkdir(parents=True, exist_ok=True)
    b_train.write_text("\n".join(b_all) + "\n", encoding="utf-8")

    # ---- C training file: resample OLD negatives with replacement, same added count ----
    human_idx = [i for i, r in enumerate(train_rows) if int(r["label"]) == 0]
    rng = np.random.RandomState(SEED)
    c_extra_idx = rng.choice(human_idx, size=added_rows, replace=True)
    c_all = raw_train_lines + [raw_train_lines[i] for i in c_extra_idx]
    random.Random(SEED).shuffle(c_all)
    c_train = MODELS / "C/train.txt"
    c_train.parent.mkdir(parents=True, exist_ok=True)
    c_train.write_text("\n".join(c_all) + "\n", encoding="utf-8")

    # ---- train B and C ----
    b_bin = MODELS / "B/model.bin"
    c_bin = MODELS / "C/model.bin"
    b_model = train_model(b_train, b_bin, DEFAULT_PARAMS)
    del b_model
    c_model = train_model(c_train, c_bin, DEFAULT_PARAMS)
    del c_model

    # ---- validation scoring + thresholds ----
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    del model_a
    model_b = load_model(b_bin)
    scores_b = positive_scores(model_b, val_features)
    del model_b
    model_c = load_model(c_bin)
    scores_c = positive_scores(model_c, val_features)
    del model_c
    sel_b_1 = select_threshold(y_val, scores_b, 0.01)
    sel_b_2 = select_threshold(y_val, scores_b, 0.02)
    sel_c_1 = select_threshold(y_val, scores_c, 0.01)
    sel_c_2 = select_threshold(y_val, scores_c, 0.02)
    thr_a_1 = meta_a["threshold"]
    thr_a_2 = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))["models"]["selected_original"]["working_points"]["0.02"]["threshold"]

    hints = __import__("re").compile(read_json(ROOT / "fasttext-classifier/reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"], __import__("re").I)
    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    idx_ordinary_ai = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints.search(r["username"] or "")]
    expansion_set = set()
    for name in ("videos.txt",):
        path = ROOT / "out/collector-expand-topics-20261114" / name
        expansion_set |= {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    expansion_set |= {"BV1i24P6rEmt"}

    def evaluate(y, scores, threshold):
        entry = {"threshold": threshold, "all": binary_metrics(y, scores, threshold)}
        ordinary = binary_metrics([y[i] for i in idx_ordinary_ai], [scores[i] for i in idx_ordinary_ai], threshold)
        entry["ordinary_ai_recall"] = {"recall": ordinary["recall"], "confusion_matrix": ordinary["confusion_matrix"],
                                       "rows": len(idx_ordinary_ai)}
        lengths = {}
        for low, high, name in ((1, 20, "1-20"), (21, 60, "21-60"), (61, 10**9, "61+")):
            ai_idx = [i for i, r in enumerate(val_rows) if y[i] == 1 and low <= len(r["text"]) <= high]
            m = binary_metrics([y[i] for i in ai_idx], [scores[i] for i in ai_idx], threshold) if ai_idx else None
            lengths[name] = {"recall": m["recall"] if m else None, "ai_rows": len(ai_idx)}
        entry["length_ai_recall"] = lengths
        human_idx = [i for i, r in enumerate(val_rows) if y[i] == 0]
        confirmed = [i for i in human_idx
                     if ((decisions.get((str(val_rows[i]["uid"]), normalize(val_rows[i]["text"]))) or {}).get("sources") or [])
                     and any(s.get("manual_account_label") == "human" for s in (decisions[(str(val_rows[i]["uid"]), normalize(val_rows[i]["text"]))]["sources"]))]
        entry["human_trigger"] = {"all": binary_metrics([y[i] for i in human_idx], [scores[i] for i in human_idx], threshold)["fpr"],
                                  "account_evidence_rows": len(confirmed),
                                  "account_evidence_fpr": binary_metrics([y[i] for i in confirmed], [scores[i] for i in confirmed], threshold)["fpr"] if confirmed else None}
        per_video = {}
        for i in human_idx:
            identity = (str(val_rows[i]["uid"]), normalize(val_rows[i]["text"]))
            for bvid in lineage_bvids.get(identity, set()):
                per_video.setdefault(bvid, [0, 0])
                per_video[bvid][1] += 1
                per_video[bvid][0] += int(scores[i] >= threshold)
        entry["human_trigger_per_video"] = {k: {"triggered": v[0], "rows": v[1], "rate": v[0] / v[1]} for k, v in per_video.items()}
        expansion_human = [i for i in human_idx
                           if lineage_bvids.get((str(val_rows[i]["uid"]), normalize(val_rows[i]["text"])), set()) & expansion_set]
        entry["expansion_video_human_fpr"] = binary_metrics([y[i] for i in expansion_human], [scores[i] for i in expansion_human], threshold)["fpr"] if expansion_human else None
        return entry

    arms = {
        "A_v8_frozen": {"thresholds": {"fpr_1pct": thr_a_1, "fpr_2pct": thr_a_2},
                        "threshold_source": "v8 validation selection (frozen artifact)",
                        "1pct": evaluate(y_val, scores_a, thr_a_1), "2pct": evaluate(y_val, scores_a, thr_a_2)},
        "B_historical_negatives": {"model_sha256": sha256(b_bin), "thresholds": {"fpr_1pct": sel_b_1["threshold"], "fpr_2pct": sel_b_2["threshold"]},
                                   "threshold_source": "selected on v8 validation (frozen protocol rule)",
                                   "added_negative_rows": added_rows,
                                   "1pct": evaluate(y_val, scores_b, sel_b_1["threshold"]),
                                   "2pct": evaluate(y_val, scores_b, sel_b_2["threshold"])},
        "C_negative_resample_control": {"model_sha256": sha256(c_bin), "thresholds": {"fpr_1pct": sel_c_1["threshold"], "fpr_2pct": sel_c_2["threshold"]},
                                        "threshold_source": "selected on v8 validation (frozen protocol rule)",
                                        "added_negative_rows": added_rows,
                                        "1pct": evaluate(y_val, scores_c, sel_c_1["threshold"]),
                                        "2pct": evaluate(y_val, scores_c, sel_c_2["threshold"])},
    }

    perf = {"a_load_ms": None, "b_load_ms": None, "c_load_ms": None,
            "a_score_200_ms": None, "b_score_200_ms": None, "c_score_200_ms": None}
    start = time.perf_counter()
    ma = load_model(MODEL_A_DIR / "text/model.bin")
    perf["a_load_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    mb = load_model(b_bin)
    perf["b_load_ms"] = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    mc = load_model(c_bin)
    perf["c_load_ms"] = (time.perf_counter() - start) * 1000
    warm = val_features[:200]
    for m in (ma, mb, mc):
        positive_scores(m, warm[:5])
    for key, m in (("a_score_200_ms", ma), ("b_score_200_ms", mb), ("c_score_200_ms", mc)):
        start = time.perf_counter()
        for _ in range(3):
            for line in warm:
                positive_scores(m, [line])
        perf[key] = (time.perf_counter() - start) * 1000 / 3
    sizes = {"a_model_bytes": (MODEL_A_DIR / "text/model.bin").stat().st_size,
             "b_model_bytes": b_bin.stat().st_size, "c_model_bytes": c_bin.stat().st_size}

    # ---- freeze everything, then ONE diagnostic pass on the demoted benchmark ----
    bench_rows = read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool"))
    y_bench = [int(r["label"]) for r in bench_rows]
    bench_features = [pre.encode(r["text"], "", "text") for r in bench_rows]
    scores_bench = {}
    for name, model in (("A_v8_frozen", ma), ("B_historical_negatives", mb), ("C_negative_resample_control", mc)):
        scores_bench[name] = positive_scores(model, bench_features)
    del ma, mb, mc
    dsh = [i for i, r in enumerate(bench_rows) if r["pool"] == "ai" and y_bench[i] == 0]
    topic = [i for i, r in enumerate(bench_rows) if r["pool"] == "topics" and y_bench[i] == 0]
    ordinary_bench = [i for i, r in enumerate(bench_rows) if y_bench[i] == 1 and not hints.search(r["username"] or "")]
    short_bench = [i for i, r in enumerate(bench_rows) if y_bench[i] == 1 and len(r["text"]) <= 20]
    bench_diag = {}
    for name, threshold in (("A_v8_frozen", thr_a_1), ("B_historical_negatives", sel_b_1["threshold"]),
                            ("C_negative_resample_control", sel_c_1["threshold"])):
        scores = scores_bench[name]
        bench_diag[name] = {
            "dsh_community_rule_negative_trigger": binary_metrics([y_bench[i] for i in dsh], [scores[i] for i in dsh], threshold)["fpr"],
            "general_topic_rule_negative_trigger": binary_metrics([y_bench[i] for i in topic], [scores[i] for i in topic], threshold)["fpr"],
            "ordinary_ai_recall": binary_metrics([y_bench[i] for i in ordinary_bench], [scores[i] for i in ordinary_bench], threshold)["recall"],
            "short_ai_recall": binary_metrics([y_bench[i] for i in short_bench], [scores[i] for i in short_bench], threshold)["recall"],
        }
    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "design": {"arm_a": "frozen v8 artifact", "arm_b": "v8 train + eligible historical negatives (label_source=historical_timestamp_proxy)",
                   "arm_c": "v8 train + old negatives resampled with replacement, same added count",
                   "params": DEFAULT_PARAMS, "shuffle_seed": SEED, "resample_seed": SEED,
                   "working_points": "v8 validation, frozen protocol rule; benchmark used once, after freezing, as historical reference only"},
        "isolation_audit": {"protected_sizes": {k: len(v) for k, v in protected.items()},
                            "per_split_train_uids": len(per_split_uids["train"]),
                            "candidates_new_rows": len(new_rows), "eligible_after_isolation": added_rows,
                            "exclusions": dict(exclusions),
                            "new_video_overlap_with_protected": len({r["bvid"] for r in eligible} & protected["video"])},
        "added_negative_month_profile": dict(sorted(Counter(datetime.datetime.fromtimestamp(int(r["ctime_unix"]), datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m") for r in eligible).items())),
        "added_negative_length": dict(Counter(r["length_group"] for r in eligible)),
        "results": arms,
        "perf": perf, "sizes": sizes,
        "historical_reference_diagnostic": {"benchmark_rows": len(bench_rows), "dsh_negative_rows": len(dsh),
                                            "topic_negative_rows": len(topic), "note": "demoted historical-reference performance; not acceptance",
                                            "arms": bench_diag},
        "limits": ["召回率为与机械规则标签的一致性，不是真实 AI 来源识别率",
                   "历史负样本标签是时间代理（authorship_verified=false）",
                   "验证集普通昵称 AI 仅 112 行，±1 行即 ±0.9pp",
                   "单次实验；未做超参数搜索；结果不触发发布或后续训练"]
    }
    write_json(HERE / "ablation.json", result)

    md = ["# 历史负样本增强消融：A（v8 冻结）/ B（+历史负类）/ C（旧负类重采样对照）", "",
          f"生成：{result['generated_at']}。三臂输入均仅正文；B 新增 {added_rows} 行 eligible 历史负类（label_source=historical_timestamp_proxy），"
          f"C 在旧负类内有放回重采样同数量行。阈值：A 用冻结工件值，B/C 在 v8 验证集按冻结协议选择；全部冻结后对历史参考基准做了一次诊断（非验收）。", "",
          "## 隔离审计", "",
          f"- 候选新行 {len(new_rows)}；通过完整隔离（UID/正文/视频 不与 v8 验证、测试、基准重叠，且不与既有训练身份重复）{added_rows} 行。",
          f"- 排除明细：{json.dumps(dict(exclusions), ensure_ascii=False)}",
          f"- 新增行涉及视频与受保护视频交集：{result['isolation_audit']['new_video_overlap_with_protected']}。", "",
          "## 验证集结果（1% 工作点）", "",
          "| 臂 | 阈值 | 整体召回 | 精确率 | F1 | 误拦率 | 普通昵称召回 | 硬AI召回(改判行) | 扩充视频误拦 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_historical_negatives", "B"), ("C_negative_resample_control", "C")):
        one = arms[name]["1pct"]
        all_m = one["all"]
        md.append(f"| {label} | {one['threshold']:.6g} | {all_m['recall']:.2%} | {all_m['precision']:.4f} | {all_m['f1']:.4f} | "
                  f"{all_m['fpr']:.2%} | {one['ordinary_ai_recall']['recall']:.2%} | "
                  f"{(one['low_score_rows']['hard_ai_recall'] or {}).get('recall') if one.get('low_score_rows') else '—'} | "
                  f"{(one.get('expansion_video_human_fpr') or 0):.2%} |")
    md += ["", "## 验证集结果（2% 工作点）", "",
           "| 臂 | 阈值 | 整体召回 | 普通昵称召回 | 误拦率 |", "|---|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_historical_negatives", "B"), ("C_negative_resample_control", "C")):
        two = arms[name]["2pct"]
        md.append(f"| {label} | {two['threshold']:.6g} | {two['all']['recall']:.2%} | {two['ordinary_ai_recall']['recall']:.2%} | {two['all']['fpr']:.2%} |")
    md += ["", "## 长度分组 AI 召回（1% 工作点）", "",
           "| 臂 | 1–20 字 | 21–60 字 | ≥61 字 |", "|---|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_historical_negatives", "B"), ("C_negative_resample_control", "C")):
        lengths = arms[name]["1pct"]["length_ai_recall"]
        md.append(f"| {label} | {lengths['1-20']['recall']:.2%} | {lengths['21-60']['recall']:.2%} | {lengths['61+']['recall']:.2%} |")
    md += ["", "## 历史参考基准诊断（冻结后单次；历史参考表现，非验收）", "",
           "| 臂 | dsh社区规则负类触发 | 普通话题规则负类触发 | 普通昵称AI召回 | 短文本AI召回 |", "|---|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_historical_negatives", "B"), ("C_negative_resample_control", "C")):
        d = bench_diag[name]
        md.append(f"| {label} | {d['dsh_community_rule_negative_trigger']:.2%} | {d['general_topic_rule_negative_trigger']:.2%} | "
                  f"{d['ordinary_ai_recall']:.2%} | {d['short_ai_recall']:.2%} |")
    md += ["", "## 体积与推理", "",
           "| 臂 | 模型体积 | 加载 | 逐评论推理（200条均值） |", "|---|---:|---:|---:|"]
    for name in ("A_v8_frozen", "B_historical_negatives", "C_negative_resample_control"):
        md.append(f"| {name} | {sizes[name + '_bytes' if False else {'A_v8_frozen': 'a_model_bytes', 'B_historical_negatives': 'b_model_bytes', 'C_negative_resample_control': 'c_model_bytes'}[name]]/1024:.0f} KiB | "
                  f"{perf[{'A_v8_frozen': 'a_load_ms', 'B_historical_negatives': 'b_load_ms', 'C_negative_resample_control': 'c_load_ms'}[name]]:.1f} ms | "
                  f"{perf[{'A_v8_frozen': 'a_score_200_ms', 'B_historical_negatives': 'b_score_200_ms', 'C_negative_resample_control': 'c_score_200_ms'}[name]]:.2f} ms |")
    (HERE / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"added_rows": added_rows, "exclusions": dict(exclusions),
                      "A_1pct": {"recall": arms["A_v8_frozen"]["1pct"]["all"]["recall"], "ordinary": arms["A_v8_frozen"]["1pct"]["ordinary_ai_recall"]["recall"]},
                      "B_1pct": {"recall": arms["B_historical_negatives"]["1pct"]["all"]["recall"], "ordinary": arms["B_historical_negatives"]["1pct"]["ordinary_ai_recall"]["recall"]},
                      "C_1pct": {"recall": arms["C_negative_resample_control"]["1pct"]["all"]["recall"], "ordinary": arms["C_negative_resample_control"]["1pct"]["ordinary_ai_recall"]["recall"]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
