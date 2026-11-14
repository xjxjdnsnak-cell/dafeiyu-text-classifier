"""Version user-authorized >=30 AI labels and draw a reproducible 1% audit."""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from bili_ai.common import sha256, write_csv, write_json
from bili_ai.preprocess import normalize
from text_policy import EMOTES_PATH, POLICY_PATH, human_exclusion_matches, is_mention_only
from text_specialist_dataset import save_csv, save_jsonl

CLASSIFIER = Path(__file__).resolve().parents[2]
ROOT = CLASSIFIER.parent
RAW_EXPORTS = (
    ROOT / "out/comment-dataset-20261112/export/all_scored.jsonl",
    ROOT / "out/collector-expand-20261112/export/all_scored.jsonl",
    ROOT / "out/collector-newtopics-20261112/export/all_scored.jsonl",
    ROOT / "out/text-specialist-20261113/collection/export/all_scored.jsonl",
)
FIELDS = ("uid", "username", "text", "label")
REVIEW_FIELDS = ("序号", "样本ID", "正文", "判定", "备注")


def read_jsonl(path):
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return str(row["uid"]), normalize(row["text"])


def stable_id(identity):
    payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return "DS30-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def refresh_rule_scores(corpus, exports):
    """Later exports include expanded account/video context; preserve observations."""
    observations = defaultdict(list)
    for path in exports:
        for line, row in enumerate(read_jsonl(path), 1):
            observations[str(row["mid"]), str(row["rpid"])].append({
                "file": str(path.resolve()), "line": line, "score": int(row["score"]),
                "reasons": row.get("reasons", []), "collected_at": row.get("collected_at"),
                "export_label": row.get("label"), "rule_version": row.get("rule_version"),
                "text": normalize(row.get("training_text") or row["text"]),
            })
    result = []
    for row in corpus:
        entries = observations[str(row["uid"]), str(row["rpid"])]
        if not entries or any(entry["text"] != normalize(row["text"]) for entry in entries):
            raise ValueError("原始导出记录缺失或同一评论的正文不一致")
        updated = dict(row)
        updated.update(snapshot_rule_score=row["rule_score"], rule_score=entries[-1]["score"],
                       rule_reasons=entries[-1]["reasons"], score_observations=entries,
                       score_resolution="last_export_in_declared_source_order")
        result.append(updated)
    return result


def merge_population(base_rows, corpus):
    sources = defaultdict(list)
    originals = {}
    for row in corpus:
        if not isinstance(row["rule_score"], int):
            raise ValueError("rule_score must be an integer, not a model probability")
        sources[key(row)].append(row)
    for line, row in enumerate(base_rows, 2):
        identity = key(row)
        if identity in originals or row["label"] not in ("0", "1"):
            raise ValueError("duplicate base sample or invalid original label")
        originals[identity] = (row, line)
    records, excluded = [], []
    for identity in sorted(sources.keys() | originals.keys()):
        evidence = sources.get(identity, [])
        original, line = originals.get(identity, (None, None))
        text = original["text"] if original else evidence[-1]["text"]
        scores = [r["rule_score"] for r in evidence]
        qualifies = any(score >= 30 for score in scores)
        previous = original["label"] if original else None
        reason = "mention_only" if is_mention_only(text) else (
            "human_expression" if previous == "0" and not qualifies and human_exclusion_matches(text) else "")
        if reason:
            excluded.append({"sample_id": stable_id(identity), "uid": identity[0], "text": text,
                             "previous_label": previous, "reason": reason,
                             "base_row": original, "base_csv_line": line, "sources": evidence})
            continue
        if original is None and not qualifies:
            continue
        label = "1" if qualifies else previous
        records.append({"sample_id": stable_id(identity), "uid": identity[0], "text": text,
                        "username": original["username"] if original else evidence[-1]["username"],
                        "label": label, "previous_label": previous,
                        "label_source": "user_rule_score_ge_30" if label == "1" and previous != "1"
                                        else "preserved_existing_label",
                        "matches_new_rule": qualifies, "newly_added": original is None,
                        "base_row": original, "base_csv_line": line, "sources": evidence,
                        "rule_score_max": max(scores) if scores else None,
                        "authorship_label": "uncertain"})
    return records, excluded


def sample_ai(records, seed=20261114):
    population = sorted((r for r in records if r["label"] == "1"), key=lambda r: r["sample_id"])
    if len({r["sample_id"] for r in population}) != len(population):
        raise ValueError("duplicate population IDs")
    count = (len(population) + 99) // 100
    return random.Random(seed).sample(population, count)


def review_summary(rows, expected_ids):
    identifiers = [row["样本ID"] for row in rows]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected_ids):
        raise ValueError("复核表必须保留全部抽样 ID，不得重复、增删或替换")
    counts = Counter(row["判定"].strip() for row in rows)
    if counts.keys() - {"AI", "真人", "不确定", ""}:
        raise ValueError("判定仅可填 AI、真人、不确定，未填请留空")
    total = len(rows)
    decided = counts["AI"] + counts["真人"]
    unresolved = counts["不确定"] + counts[""]
    return {"sample_size": total, "ai": counts["AI"], "human": counts["真人"],
            "uncertain": counts["不确定"], "pending": counts[""], "unresolved": unresolved,
            "review_complete": unresolved == 0 and total > 0,
            "reviewed_false_discovery_fraction": counts["真人"] / decided if decided else None,
            "confirmed_human_fraction_of_sample": counts["真人"] / total if total else None,
            "sample_error_bounds_with_unresolved": [counts["真人"] / total,
                (counts["真人"] + unresolved) / total] if total else None,
            "metric_note": "AI集抽样中确认真人的比例；不是以所有真人为分母的真人误拦率。"
                           "无法判断与未填均单列，不计为确认AI。"}


def review_markdown(rows):
    parts = ["# AI 集随机抽样复核\n",
             f"共 {len(rows)} 条。判定可填 AI、真人、不确定；也可在聊天中直接回复序号和判定。"
             "正文来自真实评论，未作内容改写。分数、昵称与历史标签在另一个证据表中。\n"]
    for row in rows:
        longest = max((len(m[0]) for m in re.finditer(r"`+", row["正文"])), default=0)
        fence = "`" * max(3, longest + 1)
        parts.append(f"## {row['序号']}\n\n{fence}text\n{row['正文']}\n{fence}\n\n判定：　　备注：\n")
    return "\n".join(parts)


def build(source, bundle, data_output, review_output, seed=20261114):
    source, bundle, data_output, review_output = [p.resolve() for p in (source, bundle, data_output, review_output)]
    if data_output.exists() or review_output.exists() or data_output == review_output:
        raise FileExistsError("只写入两个独立的新目录，不覆盖旧数据或复核结果")
    protected = [source / "comments.csv", source / "splits.csv", bundle / "corpus.jsonl",
                 bundle / "legacy_uid_decisions.csv", bundle / "legacy_comment_judgments.jsonl",
                 POLICY_PATH, EMOTES_PATH, Path(__file__), *RAW_EXPORTS]
    hashes = {str(path): sha256(path) for path in protected}
    base_rows = read_csv(source / "comments.csv")
    snapshot = read_jsonl(bundle / "corpus.jsonl")
    corpus = refresh_rule_scores(snapshot, RAW_EXPORTS)
    records, excluded = merge_population(base_rows, corpus)
    inherited = {r["uid"]: r["split"] for r in read_csv(source / "splits.csv")}
    for row in records:
        row["historical_split"] = inherited.get(row["uid"])
        row["source_pool_roles"] = sorted({r["pool_role"] for r in row["sources"]})
    ai = [r for r in records if r["label"] == "1"]
    added = [r for r in records if r["label_source"] == "user_rule_score_ge_30"]
    sample = sample_ai(records, seed)
    ai_ids = {r["sample_id"] for r in ai}
    if not ai:
        raise ValueError("AI population is empty")
    assert all(not is_mention_only(r["text"]) for r in records)
    assert all(not human_exclusion_matches(r["text"]) for r in records if r["label"] == "0")
    retained = {key(r): r for r in records}
    assert all(key(r) in retained and retained[key(r)]["label"] == "1"
               for r in corpus if r["rule_score"] >= 30 and not is_mention_only(r["text"]))
    assert all(retained[key(r)]["label"] == "1" for r in base_rows
               if r["label"] == "1" and not is_mention_only(r["text"]))
    assert len(ai_ids) == len(ai)
    assert {r["sample_id"] for r in sample} <= ai_ids
    assert len(sample) == (len(ai) + 99) // 100

    data_output.mkdir(parents=True)
    review_output.mkdir(parents=True)
    write_csv(data_output / "comments.csv", records, FIELDS)
    save_jsonl(data_output / "ai.jsonl", ai)
    save_jsonl(data_output / "label-decisions.jsonl", records)
    save_jsonl(data_output / "moved-to-ai.jsonl", added)
    save_jsonl(data_output / "excluded.jsonl", excluded)
    for path in (POLICY_PATH, EMOTES_PATH, bundle / "legacy_uid_decisions.csv",
                 bundle / "legacy_comment_judgments.jsonl"):
        shutil.copyfile(path, data_output / path.name)
    shutil.copyfile(source / "splits.csv", data_output / "inherited-splits.csv")

    review_rows = [{"序号": i, "样本ID": r["sample_id"], "正文": r["text"], "判定": "", "备注": ""}
                   for i, r in enumerate(sample, 1)]
    save_csv(review_output / "抽样复核.csv", review_rows, REVIEW_FIELDS)
    (review_output / "抽样正文.md").write_text(review_markdown(review_rows), encoding="utf-8")
    save_jsonl(review_output / "sample.jsonl", sample)
    evidence = [{"序号": i, "样本ID": r["sample_id"], "UID": r["uid"], "昵称": r["username"],
                 "规则分最高值": r["rule_score_max"], "本次符合30分规则": r["matches_new_rule"],
                 "原标签": r["previous_label"], "标签来源": r["label_source"],
                 "原数据CSV行号": r["base_csv_line"], "历史划分": r["historical_split"],
                 "评论链接": sorted({s["url"] for s in r["sources"] if s.get("url")}),
                 "评论ID": [s["rpid"] for s in r["sources"]],
                 "原始正文": [s.get("text_raw", s["text"]) for s in r["sources"]],
                 "历史人工意见": [{"账号": s.get("manual_account_label"),
                                   "评论": s.get("legacy_comment_judgment")} for s in r["sources"]]}
                for i, r in enumerate(sample, 1)]
    save_csv(review_output / "抽样证据.csv", evidence, tuple(evidence[0]))
    manifest = {"population": str(data_output / "ai.jsonl"), "population_sha256": sha256(data_output / "ai.jsonl"),
                "population_size": len(ai), "unit": "uid_and_normalized_body",
                "seed": seed, "fraction_requested": 0.01, "size_rounding": "ceil",
                "sample_size": len(sample), "actual_fraction": len(sample) / len(ai),
                "algorithm": "random.Random(seed).sample(sorted(population, key=sample_id), ceil(N/100))",
                "python": sys.version, "sample_ids_in_review_order": [r["sample_id"] for r in sample],
                "sample_sha256": sha256(review_output / "sample.jsonl")}
    write_json(review_output / "sampling.json", manifest)
    write_json(review_output / "pending-review-summary.json",
               review_summary(review_rows, set(manifest["sample_ids_in_review_order"])))
    qualifying = [r for r in corpus if r["rule_score"] >= 30]
    eligible = [r for r in qualifying if not is_mention_only(r["text"])]
    report = {"policy": "user_authorized_rule_score_ge_30", "threshold": 30, "inclusive": True,
              "score_field": "rule_score (raw export score), never model text_score",
              "score_resolution": "last export in RAW_EXPORTS order; all observations retained",
              "data_output": str(data_output), "review_output": str(review_output),
              "source_sha256": hashes, "raw_comments": len(corpus),
              "rule_scores_refreshed": sum(r["rule_score"] != r["snapshot_rule_score"] for r in corpus),
              "qualifying_raw_comments": len(qualifying), "qualifying_mention_only_excluded": len(qualifying)-len(eligible),
              "qualifying_substantive_comments": len(eligible),
              "qualifying_unique_uid_body_samples": len({key(r) for r in eligible}),
              "score_exactly_30_substantive_comments": sum(r["rule_score"] == 30 for r in eligible),
              "before": dict(Counter(r["label"] for r in base_rows)),
              "after": dict(Counter(r["label"] for r in records)), "total_after": len(records),
              "moved_to_ai": len(added), "new_rows": sum(r["newly_added"] for r in added),
              "relabeled_human_to_ai": sum(r["previous_label"] == "0" for r in added),
              "original_ai_retained": sum(r["previous_label"] == "1" for r in ai),
              "prior_human_account_evidence_in_moved": sum(any(s.get("manual_account_label") == "human"
                    for s in r["sources"]) for r in added),
              "excluded_by_kind": dict(Counter(r["reason"] for r in excluded)),
              "human_rows_unchanged": Counter(tuple(r[f] for f in FIELDS) for r in base_rows if r["label"] == "0")
                  == Counter(tuple(r[f] for f in FIELDS) for r in records if r["label"] == "0"),
              "inherited_split_rows": len(inherited), "new_evaluation_split_created": False,
              "model_training_run": False, "authorship_evidence_added": False,
              "sample": manifest, "review_status": "pending_user_review"}
    assert all(sha256(Path(path)) == digest for path, digest in hashes.items())
    write_json(data_output / "build_report.json", report)
    write_json(review_output / "build_report.json", report)
    (data_output / "README.md").write_text(
        "# 规则分30分及以上 AI 数据新版本\n\n"
        f"以新授权 >=30 规则归入 AI；原有AI样本保留。AI {len(ai)}，人类 {len(records)-len(ai)}，合计 {len(records)}；"
        f"新增/迁入AI {len(added)}。原文件未覆盖。\n\n"
        "comments.csv 为机器使用的原文数据，label=1为AI、0为人类；不要把供Excel查看的复核表反向当成训练正文。"
        "ai.jsonl 是抽样总体，label-decisions.jsonl 和 moved-to-ai.jsonl 保留旧标签、原正文、评论ID、"
        "所有分数观测及历史人工意见。样本按已有数据粒度 UID+规范化正文去重，重复评论ID仍全部保留。\n\n"
        "规则分来自最后一次导出（顺序见 build_report.json），59条旧快照分数发生更新；没有使用模型分数。"
        "纯@无正文排除；短语名单仍只影响人类样本，不单凭短语赋AI标签。AI是用户规则标签，"
        "未增加作者使用AI的来源证据。\n\n"
        "inherited-splits.csv 只归档旧UID划分，新增样本没有被分配到训练/验证/测试集；"
        "原专项数据中的pool_role保留在来源中。此版本未作新的隔离审计，不能直接声称是新评测集。"
        "未训练或更新模型。\n\n"
        f"随机复核见 {review_output.as_posix()}/抽样正文.md 或 抽样复核.csv。\n", encoding="utf-8")
    (review_output / "README.md").write_text(
        "# 随机复核说明\n\n"
        f"在合并后的完整AI集{len(ai)}个样本中均匀无放回抽取1%，向上取整为{len(sample)}条"
        f"（实际{len(sample)/len(ai):.4%}）；种子{seed}。不是只抽新增记录，也没有挑选正文或按分数分层。\n\n"
        "打开抽样正文.md可逐条查看并在聊天中回复序号；也可在抽样复核.csv的判定列填写AI、真人、不确定，"
        "备注可留空。昵称、分数和旧标签在抽样证据.csv中，便于需要时追溯。"
        "sample.jsonl保存抽样记录，sampling.json保存抽样顺序、总体哈希及随机方法。\n\n"
        "尚未人工复核，当前误报比例未知。后续分别统计确认真人、确认AI、不确定及未填；"
        "真人/(真人+AI)只表示已判定样本中的误报比例，并同时报告真人/全部样本及未决范围。"
        "该统计不是所有真人中的误拦率，也不单独代表普通昵称这一组的表现。\n\n"
        "复核统计命令：\n\n```powershell\n"
        f"& '{(CLASSIFIER/'.venv/Scripts/python.exe').as_posix()}' -X utf8 "
        f"'{Path(__file__).as_posix()}' summarize --review-dir '{review_output.as_posix()}' "
        f"--reviews '{(review_output/'抽样复核.csv').as_posix()}'\n```\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("build")
    create.add_argument("--source", type=Path, default=CLASSIFIER / "data-augment-70-15-15-text-gray-20261114-v2")
    create.add_argument("--bundle", type=Path, default=ROOT / "out/text-specialist-20261113/bundle-v6")
    create.add_argument("--data-output", type=Path, default=CLASSIFIER / "data-rule-ge30-20261114")
    create.add_argument("--review-output", type=Path, default=ROOT / "out/rule-ge30-review-20261114")
    create.add_argument("--seed", type=int, default=20261114)
    summary = commands.add_parser("summarize")
    summary.add_argument("--review-dir", required=True, type=Path)
    summary.add_argument("--reviews", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.source, args.bundle, args.data_output, args.review_output, args.seed)
        result = {key: result[key] for key in ("after", "moved_to_ai", "relabeled_human_to_ai", "total_after",
                                               "qualifying_substantive_comments", "data_output", "review_output")}
    else:
        manifest = json.loads((args.review_dir / "sampling.json").read_text(encoding="utf-8"))
        if sha256(Path(manifest["population"])) != manifest["population_sha256"]:
            raise ValueError("抽样总体文件已变更，不能复用旧复核表")
        if sha256(args.review_dir / "sample.jsonl") != manifest["sample_sha256"]:
            raise ValueError("抽样记录已变更")
        result = review_summary(read_csv(args.reviews), set(manifest["sample_ids_in_review_order"]))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
