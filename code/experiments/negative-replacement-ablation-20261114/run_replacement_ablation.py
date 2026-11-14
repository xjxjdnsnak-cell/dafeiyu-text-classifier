"""Arm D: equal-replacement negative ablation (diagnostic; v8 untouched).

Composition (exactly v8 train size):
- P: all rule-positive/confirmed-AI training rows (weight 1)          4,431
- H: human-evidence rows (account-level or comment-level), kept       1,279
- old U remaining: 7,251 unconfirmed negatives minus 5,173 removed      2,078
  (stratified removal: rule-score band x length, fixed seed, proportional)
- historical negatives added: 5,173 eligible pre-outbreak rows
Total = 12,961 = v8 train size.

Isolation: historical rows may not overlap the v8 validation set, the v8 test
set, or the frozen prospective benchmark by UID / normalized body / video, nor
duplicate any remaining training identity. Every exclusion counted with reason.

Thresholds: D selects on the v8 validation set (frozen protocol rule; 1% cap,
2% secondary). A stays frozen. After D's thresholds freeze, ONE diagnostic pass
on the demoted prospective benchmark (historical-reference performance only).
Stop criterion (user's): if D does not improve ordinary-nickname recall or
worsens H trigger rate, the historical-negative route is closed.
"""

import datetime
import json
import random
import sys
from collections import Counter, defaultdict
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
from score30_dataset import read_jsonl

HERE = Path(__file__).resolve().parent
DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
SOURCE = ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2"
MODEL_A_DIR = ROOT / "fasttext-classifier/models-text-specialist-20261114-v8"
CANDIDATES = ROOT / "out/pre-fish-human-corpus-20261114/historical-human-candidates.csv"
BENCH = ROOT / "out/prospective-benchmark-20261114/benchmark"
MODELS = HERE / "models"
REMOVE_COUNT = 5173
SEED = 42


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


def band_of(score):
    value = max(0, int(score)) if score is not None else 0
    for low, high in ((0, 9), (10, 19), (20, 29)):
        if low <= value <= high:
            return f"{low}-{high}"
    return "0-9"


def length_group(text):
    if len(text) <= 20:
        return "1-20"
    if len(text) <= 60:
        return "21-60"
    return "61+"


def main():
    processed, data_manifest = verify_prepared(DATA)
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    meta_a = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))
    if sha256(processed / "weighted-text/train-weight1.txt") != meta_a["training_input_sha256"]:
        raise ValueError("A 臂训练输入哈希与 v8 元数据不一致")

    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    manual_human_uids = set()
    import csv
    with (Path("<MANUAL_REVIEW_CSV>")).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["判定"] == "人类":
                manual_human_uids.add(str(row["UID"]).strip())
    legacy_human_pairs = set()
    for row in load_jsonl(DATA / "legacy_comment_judgments.jsonl"):
        if row.get("label") == "human":
            legacy_human_pairs.add((str(row["uid"]), normalize(row["text"])))

    # ---- partitions ----
    P, H, U = [], [], []
    for row in train_rows:
        meta = decisions.get((str(row["uid"]), normalize(row["text"])))
        if meta is None:
            raise ValueError(f"训练行缺少决策记录: {row['uid']}")
        score = rule_score_of(meta)
        if int(row["label"]) == 1:
            P.append(row)
        elif (evidence := has_human_evidence(meta, manual_human_uids, legacy_human_pairs)):
            H.append({**row, "evidence": evidence, "rule_score": score})
        else:
            U.append({**row, "rule_score": score if score is not None else 0})
    if len(P) + len(H) + len(U) != len(train_rows) or len(H) != 1279:
        raise ValueError(f"划分异常: P={len(P)} H={len(H)} U={len(U)}")

    # ---- stratified removal from U: rule band x length, proportional, seed 42 ----
    strata = defaultdict(list)
    for i, row in enumerate(U):
        strata[(band_of(row["rule_score"]), length_group(row["text"]))].append(i)
    remove_count = min(REMOVE_COUNT, len(U))
    raw_share = remove_count / len(U)
    remove_targets = {}
    assigned = 0
    strata_sorted = sorted(strata.items(), key=lambda kv: -len(kv[1]))
    remain = remove_count
    for name, indices in strata_sorted:
        take = min(len(indices), int(round(len(indices) * raw_share)))
        take = min(take, remain)
        remove_targets[name] = take
        assigned += take
        remain -= take
    for name in sorted(strata, key=lambda n: -len(strata[n])):
        if remain <= 0:
            break
        take = min(len(strata[name]) - remove_targets.get(name, 0), remain)
        remove_targets[name] = remove_targets.get(name, 0) + take
        remain -= take
    removed, remaining = set(), set()
    rng = np.random.RandomState(SEED)
    for name, indices in strata.items():
        order = rng.permutation(len(indices))
        take = remove_targets.get(name, 0)
        removed.update(indices[j] for j in order[:take])
        remaining.update(indices[j] for j in order[take:])
    if len(removed) != remove_count or removed & remaining:
        raise ValueError(f"分层移除异常: removed={len(removed)}")

    # ---- eligible historical negatives (full isolation) ----
    bench_uids, bench_texts, bench_bvids = set(), set(), set()
    for row in read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool")):
        bench_uids.add(str(row["uid"]))
        bench_texts.add(normalize(row["text"]))
        bench_bvids.add(row["bvid"])
    val_test_uids = set()
    val_test_videos = set()
    lineage_split, lineage_bvids = {}, {}
    for row in load_jsonl(DATA / "sample-lineage.jsonl"):
        identity = (str(row["uid"]), normalize(row["text"]))
        lineage_split[identity] = row.get("split", "")
        lineage_bvids[identity] = set(row.get("bvids", []) or [])
    for identity, split in lineage_split.items():
        if split in ("validation", "test"):
            val_test_uids.add(identity[0])
            val_test_videos |= lineage_bvids.get(identity, set())
    val_test_texts = {normalize(r["text"]) for r in val_rows}
    candidates = read_csv(CANDIDATES, ("candidate_id", "source", "uid", "rpid", "bvid", "text", "length_group",
                                       "rule_score", "ctime_unix", "ctime_beijing", "topic_group",
                                       "proposed_label", "label_source", "authorship_verified",
                                       "training_pool_eligible", "overlap_flags"))
    exclusions = Counter()
    train_ai_bodies = {normalize(r["text"]) for r in train_rows if int(r["label"]) == 1}
    eligible = []
    eligible_identities = set()
    remaining_identities = {(str(U[i]["uid"]), normalize(U[i]["text"])) for i in remaining}
    protected_identities = {(str(r["uid"]), normalize(r["text"])) for r in P} | \
                           {(str(r["uid"]), normalize(r["text"])) for r in H}
    train_identities = {(str(r["uid"]), normalize(r["text"])) for r in train_rows}
    new_rows = [r for r in candidates if r["source"] == "new_collection"]
    for row in new_rows:
        uid, text, bvid = str(row["uid"]), normalize(row["text"]), row["bvid"]
        identity = (uid, text)
        if uid in val_test_uids or uid in bench_uids:
            exclusions["uid_in_validation_or_test_or_benchmark"] += 1
            continue
        if text in val_test_texts or text in bench_texts:
            exclusions["body_in_validation_or_test_or_benchmark"] += 1
            continue
        if bvid in val_test_videos or bvid in bench_bvids:
            exclusions["video_in_validation_or_test_or_benchmark"] += 1
            continue
        if identity in train_identities:
            exclusions["identity_already_in_training"] += 1
            continue
        if identity in protected_identities:
            exclusions["identity_conflicts_with_P_or_H"] += 1
            continue
        if text in train_ai_bodies:
            exclusions["body_conflicts_with_train_positive"] += 1
            continue
        if identity in remaining_identities:
            exclusions["identity_duplicates_remaining_U"] += 1
            continue
        if identity in eligible_identities:
            exclusions["duplicate_candidate"] += 1
            continue
        if row["training_pool_eligible"] != "True":
            exclusions["flagged_ineligible_in_corpus_build"] += 1
            continue
        eligible.append(row)
        eligible_identities.add(identity)
    if len(eligible) != REMOVE_COUNT:
        print(json.dumps({"note": "可用历史负类数量与目标不同，按实际数量等比调整移除量",
                          "eligible": len(eligible), "target": REMOVE_COUNT}, ensure_ascii=False))
    added_rows = len(eligible)

    # if eligible count differs from removal count, re-balance the removal to match
    if added_rows != REMOVE_COUNT:
        delta = added_rows - REMOVE_COUNT
        total_u = len(U)
        new_share = added_rows / total_u
        remove_targets2 = {}
        assigned = 0
        remain = added_rows
        for name in sorted(strata, key=lambda n: -len(strata[n])):
            take = min(len(strata[name]) - (remove_targets.get(name, 0) - remove_targets.get(name, 0)), int(round(len(strata[name]) * new_share)))
            take = min(take, len(strata[name]), remain)
            remove_targets2[name] = take
            assigned += take
            remain -= take
        removed2, remaining2 = set(), set()
        for name, indices in strata.items():
            order = rng.permutation(len(indices))
            take = remove_targets2.get(name, 0)
            removed2.update(indices[j] for j in order[:take])
            remaining2.update(indices[j] for j in order[take:])
        if len(removed2) != added_rows or removed2 & remaining2:
            raise ValueError("重平衡移除异常")
        removed, remaining = removed2, remaining2

    # ---- D training file ----
    pre = Preprocessor()
    raw_train_lines = [line for line in (DATA / "processed/text/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(raw_train_lines) != len(train_rows):
        raise ValueError("train.txt 行数不一致")
    d_lines = []
    for i, row in enumerate(train_rows):
        if i in removed:
            continue
        d_lines.append(raw_train_lines[i])
    for row in eligible:
        d_lines.append(f"__label__0 {pre.encode(normalize(row['text']), '', 'text')}")
    if len(d_lines) != len(train_rows):
        raise ValueError(f"D 训练行数 {len(d_lines)} != 原始 {len(train_rows)}")
    random.Random(SEED).shuffle(d_lines)
    d_train = MODELS / "D/train.txt"
    d_train.parent.mkdir(parents=True, exist_ok=True)
    d_train.write_text("\n".join(d_lines) + "\n", encoding="utf-8")

    d_bin = MODELS / "D/model.bin"
    train_model(d_train, d_bin, DEFAULT_PARAMS)
    d_meta = {"schema": 1, "mode": "text", "format": "original", "model_sha256": sha256(d_bin),
              "params": DEFAULT_PARAMS, "arm": "D_equal_replacement",
              "training_input": str(d_train), "training_input_sha256": sha256(d_train),
              "training_rows": len(d_lines),
              "composition": {"P": len(P), "H": len(H), "old_U_remaining": len(remaining),
                              "historical_negatives": added_rows},
              "note": "等量替换消融：旧 U 分层移除 5,173，历史负类等量补入；总量与 v8 训练集一致",
              "threshold_source": "selected on v8 validation (frozen protocol rule)"}
    d_meta_path = MODELS / "D/model.bin.json"
    write_json(d_meta_path, d_meta)

    # ---- evaluation on v8 validation ----
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    del model_a
    model_d = load_model(d_bin)
    scores_d = positive_scores(model_d, val_features)
    del model_d
    sel_d_1 = select_threshold(y_val, scores_d, 0.01)
    sel_d_2 = select_threshold(y_val, scores_d, 0.02)

    idx_ordinary_ai = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints_for(r["username"])]
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
        h_account = [i for i in human_idx if (str(val_rows[i]["uid"])) in manual_human_uids]
        entry["human_trigger"] = {"all_fpr": binary_metrics([y[i] for i in human_idx], [scores[i] for i in human_idx], threshold)["fpr"],
                                  "h_account_rows": len(h_account),
                                  "h_account_fpr": binary_metrics([y[i] for i in h_account], [scores[i] for i in h_account], threshold)["fpr"] if h_account else None}
        expansion_human = [i for i in human_idx
                           if lineage_bvids.get((str(val_rows[i]["uid"]), normalize(val_rows[i]["text"])), set()) & expansion_set]
        entry["expansion_video_human_fpr"] = binary_metrics([y[i] for i in expansion_human], [scores[i] for i in expansion_human], threshold)["fpr"] if expansion_human else None
        return entry

    thr_a_1 = meta_a["threshold"]
    thr_a_2 = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))["models"]["selected_original"]["working_points"]["0.02"]["threshold"]
    arms = {
        "A_v8_frozen": {"thresholds": {"fpr_1pct": thr_a_1, "fpr_2pct": thr_a_2},
                        "1pct": evaluate(y_val, scores_a, thr_a_1), "2pct": evaluate(y_val, scores_a, thr_a_2)},
        "D_equal_replacement": {"model_sha256": sha256(d_bin),
                                "thresholds": {"fpr_1pct": sel_d_1["threshold"], "fpr_2pct": sel_d_2["threshold"]},
                                "threshold_source": "selected on v8 validation (frozen protocol rule)",
                                "1pct": evaluate(y_val, scores_d, sel_d_1["threshold"]),
                                "2pct": evaluate(y_val, scores_d, sel_d_2["threshold"])},
    }

    # ---- ONE diagnostic pass on the demoted benchmark (after freeze) ----
    bench_rows = read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool"))
    y_bench = [int(r["label"]) for r in bench_rows]
    bench_features = [pre.encode(r["text"], "", "text") for r in bench_rows]
    model_d_b = load_model(d_bin)
    scores_bench_d = positive_scores(model_d_b, bench_features)
    del model_d_b
    model_a_b = load_model(MODEL_A_DIR / "text/model.bin")
    scores_bench_a = positive_scores(model_a_b, bench_features)
    del model_a_b
    dsh = [i for i, r in enumerate(bench_rows) if r["pool"] == "ai" and y_bench[i] == 0]
    topic = [i for i, r in enumerate(bench_rows) if r["pool"] == "topics" and y_bench[i] == 0]
    ordinary_bench = [i for i, r in enumerate(bench_rows) if y_bench[i] == 1 and not hints_for(r["username"])]
    short_bench = [i for i, r in enumerate(bench_rows) if y_bench[i] == 1 and len(r["text"]) <= 20]

    def bench_metrics(scores, threshold):
        return {"dsh_community_trigger": binary_metrics([y_bench[i] for i in dsh], [scores[i] for i in dsh], threshold)["fpr"],
                "general_topic_trigger": binary_metrics([y_bench[i] for i in topic], [scores[i] for i in topic], threshold)["fpr"],
                "ordinary_ai_recall": binary_metrics([y_bench[i] for i in ordinary_bench], [scores[i] for i in ordinary_bench], threshold)["recall"],
                "short_ai_recall": binary_metrics([y_bench[i] for i in short_bench], [scores[i] for i in short_bench], threshold)["recall"]}

    bench_diag = {"A_v8_frozen": bench_metrics(scores_bench_a, thr_a_1),
                  "D_equal_replacement": bench_metrics(scores_bench_d, sel_d_1["threshold"])}

    result = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "design": {"arm_a": "frozen v8 artifact", "arm_d": "equal replacement: 5,173 old U removed (stratified rule band x length), 5,173 historical negatives added",
                   "params": DEFAULT_PARAMS, "shuffle_seed": SEED, "removal_seed": SEED,
                   "composition": {"P": len(P), "H": len(H), "old_U_remaining": len(remaining), "historical_negatives": added_rows,
                                   "total": len(d_lines)},
                   "working_points": "v8 validation, frozen protocol rule; benchmark diagnostic once after freeze",
                   "status": "diagnostic; v8 untouched; stop criterion applies"},
        "removal": {"targets": {f"{k[0]}|{k[1]}": v for k, v in sorted(remove_targets.items())},
                    "by_band": dict(Counter(band_of(U[i]["rule_score"]) for i in removed)),
                    "by_length": dict(Counter(length_group(U[i]["text"]) for i in removed))},
        "added": {"month_profile": dict(sorted(Counter(datetime.datetime.fromtimestamp(int(r["ctime_unix"]), datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m") for r in eligible).items())),
                  "length": dict(Counter(r["length_group"] for r in eligible)),
                  "topic": dict(Counter(r["topic_group"] for r in eligible))},
        "isolation_audit": {"candidates": len(candidates), "new_rows": len(new_rows), "eligible": added_rows,
                            "train_ai_body_conflict_check": True,
                            "exclusions": dict(exclusions),
                            "protected_uids_val_test_bench": len(val_test_uids | bench_uids),
                            "protected_videos_val_test_bench": len(val_test_videos | bench_bvids)},
        "results": arms,
        "benchmark_diagnostic": bench_diag,
        "limits": ["召回率为与机械规则标签的一致性，不是真实 AI 来源识别率",
                   "历史负类为时间代理标签（authorship_verified=false）",
                   "验证集普通昵称 AI 仅 112 行；±1 行即 ±0.9pp",
                   "基准为历史参考表现，非验收；未根据其结果调参"]
    }
    write_json(HERE / "ablation.json", result)

    md = ["# 等量替换消融：D（旧 U 分层移除 + 历史负类等量补入）", "",
          f"生成：{result['generated_at']}。训练总量与 v8 完全一致（12,961 行）：P 4,431 + H 1,279 + 旧 U 剩余 {len(remaining)} + 历史负类 {added_rows}。"
          "移除按规则分×长度分层等比、种子 42；历史负类来源为鱼灾前候选语料（label_source=historical_timestamp_proxy）。"
          "阈值：D 在 v8 验证集按冻结协议选择；A 冻结不变；基准在全部冻结后单次诊断。", "",
          "## 隔离审计", "",
          f"- 候选新行 {len(new_rows)}，通过完整隔离 {added_rows} 行；排除：{json.dumps(dict(exclusions), ensure_ascii=False)}",
          f"- 移除分层（规则分带×长度）：{json.dumps({f'{k[0]}|{k[1]}': v for k, v in sorted(remove_targets.items())}, ensure_ascii=False)}",
          f"- 移除行分布——规则分带：{dict(Counter(band_of(U[i]['rule_score']) for i in removed))}；长度：{dict(Counter(length_group(U[i]['text']) for i in removed))}",
          f"- 历史负类月份分布：{json.dumps(result['added']['month_profile'], ensure_ascii=False)}", "",
          "## 验证集结果", "",
          "| 臂 | 工作点 | 阈值 | 整体召回 | 精确率 | F1 | 误拦率 | 普通昵称召回 | 61+字召回 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("D_equal_replacement", "D")):
        for cap in ("1pct", "2pct"):
            one = arms[name][cap]
            all_m = one["all"]
            md.append(f"| {label} | {cap} | {one['threshold']:.6g} | {all_m['recall']:.2%} | {all_m['precision']:.4f} | {all_m['f1']:.4f} | "
                      f"{all_m['fpr']:.2%} | {one['ordinary_ai_recall']['recall']:.2%} | {one['length_ai_recall']['61+']['recall']:.2%} |")
    md += ["", "长度分组 AI 召回（1% 工作点）：", ""]
    for name, label in (("A_v8_frozen", "A"), ("D_equal_replacement", "D")):
        lengths = arms[name]["1pct"]["length_ai_recall"]
        md.append(f"- {label}: 1–20 字 {lengths['1-20']['recall']:.2%}；21–60 字 {lengths['21-60']['recall']:.2%}；≥61 字 {lengths['61+']['recall']:.2%}")
    md += ["", "## 历史参考基准诊断（冻结后单次；历史参考表现，非验收）", "",
           "| 臂 | dsh社区规则负类触发 | 普通话题规则负类触发 | 普通昵称AI召回 | 短文本AI召回 |", "|---|---:|---:|---:|---:|"]
    for name, label in (("A_v8_frozen", "A"), ("D_equal_replacement", "D")):
        d = bench_diag[name]
        md.append(f"| {label} | {d['dsh_community_trigger']:.2%} | {d['general_topic_trigger']:.2%} | {d['ordinary_ai_recall']:.2%} | {d['short_ai_recall']:.2%} |")
    md += ["", "## 停止判据评估（用户既定）", "",
           f"- 普通昵称召回：A {arms['A_v8_frozen']['1pct']['ordinary_ai_recall']['recall']:.2%} vs D {arms['D_equal_replacement']['1pct']['ordinary_ai_recall']['recall']:.2%}。",
           f"- H 账号级证据触发率：A {arms['A_v8_frozen']['1pct']['human_trigger']['h_account_fpr']:.2%} vs D {arms['D_equal_replacement']['1pct']['human_trigger']['h_account_fpr']:.2%}。",
           "- 若 D 未改善普通昵称召回、或 H 触发率明显恶化 → 关闭历史负样本路线（由数字直接判定，见正文）。", "",
           "## 限制", "",
           "- 召回率为与机械规则标签的一致性，不是真实 AI 来源识别率；历史负类为时间代理标签（authorship_verified=false）。",
           "- 验证集普通昵称 AI 仅 112 行，±1 行即 ±0.9pp；单次实验、单划分。",
           "- 基准为历史参考表现；已降级，不作为验收。", ""]
    (HERE / "ablation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"composition": result["design"]["composition"],
                      "A_1pct": {"recall": arms["A_v8_frozen"]["1pct"]["all"]["recall"],
                                 "ordinary": arms["A_v8_frozen"]["1pct"]["ordinary_ai_recall"]["recall"],
                                 "h_fpr": arms["A_v8_frozen"]["1pct"]["human_trigger"]["h_account_fpr"]},
                      "D_1pct": {"recall": arms["D_equal_replacement"]["1pct"]["all"]["recall"],
                                 "ordinary": arms["D_equal_replacement"]["1pct"]["ordinary_ai_recall"]["recall"],
                                 "h_fpr": arms["D_equal_replacement"]["1pct"]["human_trigger"]["h_account_fpr"]},
                      "bench": bench_diag}, ensure_ascii=False, indent=2))


def hints_for(username):
    import re as _re
    pattern = read_json(ROOT / "fasttext-classifier/reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"]
    return _re.search(pattern, username or "")


if __name__ == "__main__":
    main()
