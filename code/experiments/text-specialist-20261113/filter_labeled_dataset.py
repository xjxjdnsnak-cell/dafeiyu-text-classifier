"""Version the existing labelled corpus after the user's text exclusions."""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from bili_ai.common import read_csv, sha256, write_csv, write_json
from bili_ai.data import FIELDS, read_splits
from text_policy import EMOTES_PATH, POLICY, POLICY_PATH, filter_labeled_rows
from text_specialist_dataset import save_csv, save_jsonl


def main():
    classifier = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=classifier / "data-augment-70-15-15-clean-uid")
    parser.add_argument("--output", type=Path, default=classifier / "data-augment-70-15-15-text-gray-20261114")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("过滤结果另存新版本，不覆盖原数据或已有结果")
    original_hashes = {name: sha256(source / name) for name in ("comments.csv", "splits.csv")}
    rows = read_csv(source / "comments.csv", FIELDS)
    mapping = read_splits(source / "splits.csv", rows)
    kept, excluded = filter_labeled_rows(rows)
    for row in excluded:
        row["split"] = mapping[row["uid"]]
    output.mkdir(parents=True)
    write_csv(output / "comments.csv", kept, FIELDS)
    retained_uids = {r["uid"] for r in kept}
    write_csv(output / "splits.csv", [{"uid": uid, "split": split} for uid, split in sorted(mapping.items())
                                      if uid in retained_uids], ("uid", "split"))
    save_jsonl(output / "excluded_rows.jsonl", excluded)
    save_csv(output / "excluded_rows.csv", excluded, (*FIELDS, "split", "exclusion_kind", "matched_expressions"))
    shutil.copyfile(POLICY_PATH, output / POLICY_PATH.name)
    shutil.copyfile(EMOTES_PATH, output / EMOTES_PATH.name)
    def counts(items):
        return {"rows": len(items), "labels": dict(Counter(r["label"] for r in items)),
                "splits": {split: dict(Counter(r["label"] for r in items if mapping[r["uid"]] == split))
                           for split in ("train", "validation", "test")}}
    report = {"source": str(source), "output": str(output), "source_sha256": original_hashes,
        "policy_version": POLICY["version"], "policy_sha256": sha256(POLICY_PATH),
        "text_policy_sha256": sha256(Path(__file__).with_name("text_policy.py")),
        "confirmed_emotes_sha256": sha256(EMOTES_PATH),
        "before": counts(rows), "after": counts(kept), "excluded": counts(excluded),
        "excluded_by_kind": dict(Counter(r["exclusion_kind"] for r in excluded)),
        "excluded_by_kind_and_label": dict(Counter(f"{r['exclusion_kind']}:label_{r['label']}" for r in excluded)),
        "human_expression_rule_counts": dict(Counter(hit for r in excluded if r["exclusion_kind"] == "human_expression"
                                                      for hit in r["matched_expressions"])),
        "labels_changed": 0, "uid_assignments_changed": 0, "model_training_run": False,
        "caution": "旧测试集经过规则筛选，评估总体和分母发生变化；不是新的独立测试集，不在本次重新报告模型准确率。"}
    assert all(sha256(source / name) == digest for name, digest in original_hashes.items())
    new_mapping = read_splits(output / "splits.csv", kept)
    assert all(new_mapping[uid] == mapping[uid] for uid in retained_uids)
    write_json(output / "build_report.json", report)
    (output / "README.md").write_text(
        "# 正文规则过滤后的新数据版本\n\n"
        "原数据和模型保持不变。本版本只移除记录，不把人类改为 AI，也不重划 UID。\n\n"
        "纯 @ 点名的记录不用于正文分类；用户列出的表达只从 label=0 人类样本移除，label=1 原有 AI 标签保留。"
        "移除记录和原标签、原划分、命中规则见 excluded_rows.jsonl / CSV。\n\n"
        f"过滤前 {len(rows)} 条，过滤后 {len(kept)} 条，移除 {len(excluded)} 条。具体分布见 build_report.json。\n\n"
        "未启动训练，也未生成新准确率。原模型元数据对应原划分及文件哈希，后续训练/评估应明确使用这一新版本。\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
