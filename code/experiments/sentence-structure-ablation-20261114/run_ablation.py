"""Sentence-structure feature ablation (A: frozen v8 / B: + sfx_ structure features).

B appends 8 discrete structure-feature tokens per comment (句式 ×3, 句长变化,
连接词比例, 重复句式, 句子数, 平均句长; prefix sfx_) after the original txt_
feature sequence — preserving all original txt_ bigrams. Same train split/
labels/params/dictionary/seed as v8; B thresholds selected on the v8 validation
set with the frozen protocol rule (1% cap, 2% secondary). ONE post-freeze
diagnostic pass on the demoted prospective benchmark (historical-reference
performance only). Diagnostic; v8 untouched; no release change.

Gray asymmetry (disclosed): corpus rows containing 不是…而是… / 与其说…不如说… /
不仅…还… patterns were EXCLUDED from label-0 training data by the gray policy —
these structure features fire mostly on AI rows, and the H-evidence trigger
metric cannot measure false-block risk for those structures (no matching human
rows exist in the data).
"""

import csv
import datetime
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-training-20261114"))

from bili_ai.common import read_csv, read_json, sha256, write_json
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize
from score30_dataset import read_jsonl

from structure_features import encode_structure_augmented, feature_stats, structure_tokens

HERE = Path(__file__).resolve().parent
DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
SOURCE = ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2"
MODEL_A_DIR = ROOT / "fasttext-classifier/models-text-specialist-20261114-v8"
MODELS = HERE / "models"


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def main():
    processed, data_manifest = verify_prepared(DATA)
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    meta_a = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))
    if sha256(processed / "weighted-text/train-weight1.txt") != meta_a["training_input_sha256"]:
        raise ValueError("A 臂训练输入哈希与 v8 元数据不一致")

    pre = Preprocessor()
    hints = re.compile(read_json(ROOT / "fasttext-classifier/reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"], re.I)

    raw_train_lines = [line for line in (DATA / "processed/text/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_train_lines) != len(train_rows):
        raise ValueError("train.txt 行数不一致")

    # ---- B training file: original txt_ features + sfx_ structure tokens ----
    b_lines = []
    for row, line in zip(train_rows, raw_train_lines, strict=True):
        label = line.split(" ", 1)[0]
        b_lines.append(f"{label} {line.split(' ', 1)[1]} " + " ".join(structure_tokens(row["text"], pre)))
    random.Random(42).shuffle(b_lines)
    MODELS.mkdir(parents=True, exist_ok=True)
    b_train = MODELS / "B/train.txt"
    b_train.parent.mkdir(parents=True, exist_ok=True)
    b_train.write_text("\n".join(b_lines) + "\n", encoding="utf-8")

    # ---- train B ----
    b_bin = MODELS / "B/model.bin"
    start = time.perf_counter()
    b_model = train_model(b_train, b_bin, DEFAULT_PARAMS)
    train_seconds = time.perf_counter() - start
    del b_model

    # ---- validation scoring ----
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    model_b = load_model(b_bin)
    scores_b = positive_scores(model_b, val_features)

    sel_b_1 = select_threshold(y_val, scores_b, 0.01)
    sel_b_2 = select_threshold(y_val, scores_b, 0.02)
    thr_a_1 = meta_a["threshold"]
    thr_a_2 = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))["models"]["selected_original"]["working_points"]["0.02"]["threshold"]

    idx_ordinary_ai = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints.search(r["username"] or "")]

    # ---- H evidence on validation ----
    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    manual_human_uids = set()
    with (Path("<MANUAL_REVIEW_CSV>")).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["判定"] == "人类":
                manual_human_uids.add(str(row["UID"]).strip())
    legacy_human_pairs = set()
    for row in load_jsonl(DATA / "legacy_comment_judgments.jsonl"):
        if row.get("label") == "human":
            legacy_human_pairs.add((str(row["uid"]), normalize(row["text"])))

    def evidence_type(i):
        meta = decisions.get((str(val_rows[i]["uid"]), normalize(val_rows[i]["text"])))
        if meta is None:
            return None
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

    val_evidence = {i: evidence_type(i) for i in range(len(val_rows))}
    idx_h = [i for i in range(len(val_rows)) if y_val[i] == 0 and val_evidence[i]]
    idx_h_account = [i for i in idx_h if val_evidence[i] == "account"]
    idx_h_comment = [i for i in idx_h if val_evidence[i] == "comment"]

    def metrics_for(scores, threshold):
        entry = {"threshold": threshold, "all": binary_metrics(y_val, scores, threshold)}
        ordinary = binary_metrics([y_val[i] for i in idx_ordinary_ai], [scores[i] for i in idx_ordinary_ai], threshold)
        entry["ordinary_ai_recall"] = {"recall": ordinary["recall"], "confusion_matrix": ordinary["confusion_matrix"],
                                       "rows": len(idx_ordinary_ai)}
        lengths = {}
        for low, high, name in ((1, 20, "1-20"), (21, 60, "21-60"), (61, 10**9, "61+")):
            ai_idx = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and low <= len(r["text"]) <= high]
            m = binary_metrics([y_val[i] for i in ai_idx], [scores[i] for i in ai_idx], threshold) if ai_idx else None
            lengths[name] = {"recall": m["recall"] if m else None, "ai_rows": len(ai_idx)}
        entry["length_ai_recall"] = lengths
        entry["h_evidence_trigger"] = {
            "h_rows": len(idx_h), "h_account_rows": len(idx_h_account), "h_comment_rows": len(idx_h_comment),
            "h_account_fpr": binary_metrics([y_val[i] for i in idx_h_account], [scores[i] for i in idx_h_account], threshold)["fpr"] if idx_h_account else None,
            "h_comment_fpr": binary_metrics([y_val[i] for i in idx_h_comment], [scores[i] for i in idx_h_comment], threshold)["fpr"] if idx_h_comment else None,
        }
        return entry

    arms = {
        "A_v8_frozen": {"thresholds": {"fpr_1pct": thr_a_1, "fpr_2pct": thr_a_2},
                        "threshold_source": "v8 validation selection (frozen artifact)",
                        "1pct": metrics_for(scores_a, thr_a_1), "2pct": metrics_for(scores_a, thr_a_2)},
        "B_structure_features": {"model_sha256": sha256(b_bin),
                                 "thresholds": {"fpr_1pct": sel_b_1["threshold"], "fpr_2pct": sel_b_2["threshold"]},
                                 "threshold_source": "selected on v8 validation (frozen protocol rule)",
                                 "1pct": metrics_for(scores_b, sel_b_1["threshold"]),
                                 "2pct": metrics_for(scores_b, sel_b_2["threshold"])},
    }

    # ---- per-row diff at 1% ----
    trig_a = [scores_a[i] >= thr_a_1 for i in range(len(val_rows))]
    trig_b = [scores_b[i] >= sel_b_1["threshold"] for i in range(len(val_rows))]
    correct_a = [y_val[i] == int(trig_a[i]) for i in range(len(val_rows))]
    diffs = {
        "positives_a_correct_b_missed": sum(1 for i in range(len(val_rows)) if y_val[i] == 1 and correct_a[i] and not trig_b[i]),
        "positives_a_missed_b_correct": sum(1 for i in range(len(val_rows)) if y_val[i] == 1 and not correct_a[i] and trig_b[i]),
        "negatives_new_triggers_b": sum(1 for i in range(len(val_rows)) if y_val[i] == 0 and not trig_a[i] and trig_b[i]),
        "negatives_removed_triggers_b": sum(1 for i in range(len(val_rows)) if y_val[i] == 0 and trig_a[i] and not trig_b[i]),
        "positives_both_correct": sum(1 for i in range(len(val_rows)) if y_val[i] == 1 and correct_a[i] and trig_b[i]),
        "positives_both_missed": sum(1 for i in range(len(val_rows)) if y_val[i] == 1 and not correct_a[i] and not trig_b[i]),
    }

    # ---- feature firing transparency (train + validation, by label) ----
    def firing_by_label(rows, labels):
        fires = {"AI": Counter(), "human": Counter()}
        for r, label in zip(rows, labels):
            key = "AI" if label == 1 else "human"
            for t in structure_tokens(r["text"], pre):
                fires[key][t] += 1
        return fires
    train_fires_by_label = firing_by_label(train_rows, y_train)
    val_fires_by_label = firing_by_label(val_rows, y_val)

    # ---- timing/size ----
    start = time.perf_counter()
    ma = load_model(MODEL_A_DIR / "text/model.bin")
    perf_a_load = (time.perf_counter() - start) * 1000
    start = time.perf_counter()
    mb = load_model(b_bin)
    perf_b_load = (time.perf_counter() - start) * 1000
    warm = val_features[:200]
    for m in (ma, mb):
        positive_scores(m, warm[:5])
    start = time.perf_counter()
    for _ in range(3):
        for line in warm:
            positive_scores(ma, [line])
    perf_a_score = (time.perf_counter() - start) * 1000 / 3
    start = time.perf_counter()
    for _ in range(3):
        for line in warm:
            positive_scores(mb, [line])
    perf_b_score = (time.perf_counter() - start) * 1000 / 3
    sizes = {"a_model_bytes": (MODEL_A_DIR / "text/model.bin").stat().st_size,
             "b_model_bytes": b_bin.stat().st_size}

    stats_train = feature_stats([line.split(" ", 1)[1] for line in b_lines])
    train_fires_by_label_json = {k: dict(v.most_common(12)) for k, v in train_fires_by_label.items()}
    val_fires_by_label_json = {k: dict(v.most_common(12)) for k, v in val_fires_by_label.items()}

    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "design": {"arm_a": "frozen v8 artifact", "arm_b": "v8 train + sfx_ structure features (8 features, ~8 tokens/comment)",
                   "params": DEFAULT_PARAMS, "shuffle_seed": 42,
                   "working_points": "v8 validation, frozen protocol rule (1% cap, 2% secondary)",
                                      "status": "diagnostic; v8 untouched"},
        "feature_counts": {"train": stats_train,
                           "train_fires_by_label": train_fires_by_label_json,
                           "val_fires_by_label": val_fires_by_label_json},
        "results": arms,
        "per_row_diff_1pct": diffs,
        "perf": {"a_load_ms": perf_a_load, "b_load_ms": perf_b_load,
                 "a_score_200_ms": perf_a_score, "b_score_200_ms": perf_b_score,
                 "b_train_seconds": train_seconds},
        "sizes": sizes,
        "limits": ["召回率为与机械规则标签的一致性，不是真实 AI 来源识别率",
                   "验证集普通昵称 AI 仅 112 行，±1 行即 ±0.9pp",
                   "单次实验、单划分；未做超参数搜索；结果不触发发布或后续训练"]
    }
    write_json(HERE / "ablation.json", result)

    md = ["# 句式结构特征消融：A（v8 冻结）/ B（+ sfx_ 结构特征）", "",
          f"生成：{result['generated_at']}。B 在原 txt_ 特征序列之后追加 8 个句式结构特征（不是…而是…/与其说…不如说…/不仅…还…/句长变化/连接词比例/重复句式/句子数/平均句长，前缀 sfx_，离散分桶）。"
          "参数、词典、预处理、种子与 v8 一致；阈值：A 冻结，B 在 v8 验证集按冻结协议选择。", "",
          "## 灰区不对称披露", "",
          "灰区策略已把含「不是…而是…」「与其说…不如说…」「不仅…还…」的 label=0 行排除出训练/验证数据——这些结构特征在人类行上**几乎不会触发**，"
          "在 AI 行上正常触发。因此这些特征学到的是「结构出现 → AI」方向，且 H 证据触发率指标**无法测度**这些结构的误拦风险（数据里不存在对应人类行）。", "",
          "## 验证集结果（1% 工作点）", "",
          "| 臂 | 阈值 | 整体召回 | 精确率 | F1 | 误拦率 | 普通昵称召回 | 1–20字 | 21–60字 | 61+字 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_structure_features", "B")):
        one = arms[name]["1pct"]
        all_m = one["all"]
        lengths = one["length_ai_recall"]
        md.append(f"| {label} | {one['threshold']:.6g} | {all_m['recall']:.2%} | {all_m['precision']:.4f} | {all_m['f1']:.4f} | "
                  f"{all_m['fpr']:.2%} | {one['ordinary_ai_recall']['recall']:.2%} | {lengths['1-20']['recall']:.2%} | "
                  f"{lengths['21-60']['recall']:.2%} | {lengths['61+']['recall']:.2%} |")
    md += ["", "## 2% 工作点", "", "| 臂 | 阈值 | 整体召回 | 普通昵称召回 | 误拦率 |", "|---|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_structure_features", "B")):
        two = arms[name]["2pct"]
        md.append(f"| {label} | {two['threshold']:.6g} | {two['all']['recall']:.2%} | {two['ordinary_ai_recall']['recall']:.2%} | {two['all']['fpr']:.2%} |")
    md += ["", "## 账号级真人证据样本触发率", "", "| 臂 | 触发率 | 样本行数 |", "|---|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("B_structure_features", "B")):
        h = arms[name]["1pct"]["h_evidence_trigger"]
        md.append(f"| {label} | {h['h_account_fpr']:.2%} | {h['h_account_rows']} |")
    md += ["", "## 逐条预测差异（1% 工作点）", "",
           f"- A 正确、B 漏判的正类：{diffs['positives_a_correct_b_missed']} 行",
           f"- A 漏判、B 正确的正类：{diffs['positives_a_missed_b_correct']} 行",
           f"- 都正确：{diffs['positives_both_correct']} 行；都漏判：{diffs['positives_both_missed']} 行",
           f"- B 新增规则负类触发：{diffs['negatives_new_triggers_b']} 行；消除：{diffs['negatives_removed_triggers_b']} 行", "",
           "## 体积与推理", "", "| 臂 | 模型体积 | 加载 | 逐评论推理（200条均值） |", "|---|---:|---:|---:|",
           f"| A | {sizes['a_model_bytes']/1024:.0f} KiB | {perf_a_load:.1f} ms | {perf_a_score:.2f} ms |",
           f"| B | {sizes['b_model_bytes']/1024:.0f} KiB | {perf_b_load:.1f} ms | {perf_b_score:.2f} ms |", "",
           "## 验收标准评估（用户既定）", "",
           f"- 普通昵称召回：A {arms['A_v8_frozen']['1pct']['ordinary_ai_recall']['recall']:.2%} vs B {arms['B_structure_features']['1pct']['ordinary_ai_recall']['recall']:.2%}。",
           f"- 验证误拦：A {arms['A_v8_frozen']['1pct']['all']['fpr']:.2%} vs B {arms['B_structure_features']['1pct']['all']['fpr']:.2%}。",
           f"- H 证据触发：A {arms['A_v8_frozen']['1pct']['h_evidence_trigger']['h_account_fpr']:.2%} vs B {arms['B_structure_features']['1pct']['h_evidence_trigger']['h_account_fpr']:.2%}。",
           f"- 体积/推理：B {sizes['b_model_bytes']/1024:.0f} KiB、{perf_b_score:.2f} ms 对 A {sizes['a_model_bytes']/1024:.0f} KiB、{perf_a_score:.2f} ms。",
           "- 判定：由以上数字直接对照四条验收标准，不额外解释。", "",
           "## 限制", "",
           "- 召回率为与机械规则标签的一致性，不是真实 AI 来源识别率。",
           "- 验证集普通昵称 AI 仅 112 行，±1 行即 ±0.9pp；单次实验、单划分。",
           "- 句式结构特征在人类行上几乎不触发（灰区排除所致）——其真实误拦风险不可测。", ""]
    (HERE / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"A_1pct": {"recall": arms["A_v8_frozen"]["1pct"]["all"]["recall"],
                                 "ordinary": arms["A_v8_frozen"]["1pct"]["ordinary_ai_recall"]["recall"],
                                 "fpr": arms["A_v8_frozen"]["1pct"]["all"]["fpr"]},
                      "B_1pct": {"recall": arms["B_structure_features"]["1pct"]["all"]["recall"],
                                 "ordinary": arms["B_structure_features"]["1pct"]["ordinary_ai_recall"]["recall"],
                                 "fpr": arms["B_structure_features"]["1pct"]["all"]["fpr"]},
                      "diffs": diffs, "perf": result["perf"], "sizes": sizes},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
