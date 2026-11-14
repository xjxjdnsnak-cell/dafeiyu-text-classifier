"""Train text-only fastText on the user's frozen labels and video holdouts."""

import argparse
import json
import random
import re
import shutil
import sys
import time
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

CLASSIFIER = Path(__file__).resolve().parents[2]
ROOT = CLASSIFIER.parent
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

import apply_user_review
from bili_ai.common import read_csv, read_json, sha256, write_csv, write_json
from bili_ai.data import FIELDS, clean_rows
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import DEFAULT_PARAMS, Predictor, load_model, positive_scores, quantize_model, train_model
from bili_ai.pipeline import _examples, _metadata, benchmark, verify_prepared
from bili_ai.preprocess import ASSETS, Preprocessor, normalize
from score30_dataset import RAW_EXPORTS, read_jsonl, review_summary
from text_policy import EMOTES_PATH, KNOWN_EMOTES, MENTION, POLICY_PATH, is_mention_only
from text_specialist_dataset import near_index, near_match, save_csv, save_jsonl

SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
REVIEW = ROOT / "out/rule-ge30-review-20261114"
DATA = CLASSIFIER / "data-text-specialist-20261114-v8"
MODELS = CLASSIFIER / "models-text-specialist-20261114-v8"
REPORTS = CLASSIFIER / "reports-text-specialist-20261114-v8"
NEW_EXPORTS = (
    ROOT / "out/collector-expand-topics-20261114/export/all_scored.jsonl",
    ROOT / "out/dsh-video-20261114/export/ai-only-ge30.jsonl",
)
V2_POLICY_PATH = Path(__file__).with_name("human_exclusion_policy.json")
PROTOCOL = CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json"
PHONE_EXPORT = Path("D:/xwechat_files/<wxid>/msg/file/2026-11/dafeiyu-comments-20261112-1719.zip")
EARLY_PROVENANCE = CLASSIFIER / "data/provenance.jsonl"
ROLES = ("train", "validation", "test")
USER_STATEMENT = "全是ai，感觉可以把数据集重新拆分，再次进行正文专项训练"
TRIALS = (
    {"name": "baseline", "weight": 1, "params": {}},
    {"name": "ordinary3", "weight": 3, "params": {}},
    {"name": "ordinary3_dim16", "weight": 3, "params": {"dim": 16, "epoch": 50}},
)


def announce(stage, **values):
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False), flush=True)


def identity(row):
    return row["uid"], normalize(row["text"])


def user_reviews(population, expected_ids):
    by_id = {row["sample_id"]: row for row in population}
    if len(by_id) != len(population) or len(expected_ids) != len(set(expected_ids)) or not set(expected_ids) <= by_id.keys():
        raise ValueError("人工复核必须精确匹配已有无重复抽样ID")
    return [{"sample_id": identifier, "uid": by_id[identifier].get("uid"),
             "text": by_id[identifier].get("text"), "review_label": "AI", "reviewer": "user",
             "review_recorded_on": "2026-11-14", "user_statement": USER_STATEMENT,
             "authorship_label": by_id[identifier].get("authorship_label", "uncertain")}
            for identifier in expected_ids if by_id[identifier]["label"] == "1"]


def template_key(text):
    text = MENTION.sub("", normalize(text)[:4096])
    text = re.sub(r"\[[^\]]*\]", lambda m: "" if m[0] in KNOWN_EMOTES else m[0], text)
    text = re.sub(r"https?://\S+|\d+", "", text)
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))


def recover_video_groups(metadata, bvid_aids, phone_lookup, provenance_lookup, raw_by_id):
    def video_key(aid):
        if not str(aid).isdigit() or int(aid) <= 0:
            raise ValueError("视频aid必须是正整数字符串")
        return "video:" + str(aid)
    videos = {video_key(bvid_aids[s["bvid"]]) for s in metadata["sources"]}
    trace = []
    for row in phone_lookup.get(identity(metadata), []):
        if str(row["replyType"]) != "1":
            raise ValueError("原手机记录不是视频评论，不能把oid当作视频aid")
        videos.add(video_key(row["oid"]))
        trace.append({"match": "phone_uid_normalized_body", "rpid": str(row["rpid"]),
                      "replyType": "1", "oid": str(row["oid"]), "archive": str(PHONE_EXPORT)})
    for provenance in provenance_lookup.get(identity(metadata), []):
        raw = raw_by_id.get((metadata["uid"], str(provenance["rpid"])))
        if raw:
            videos.add(video_key(raw["aid"]))
            trace.append({"match": "prior_import_uid_body_then_uid_rpid", "rpid": str(provenance["rpid"]),
                          "aid": str(raw["aid"]), "prior_import": provenance})
    return sorted(videos), trace


def source_contexts():
    expected = {r["name"]: r["sha256"] for r in read_json(CLASSIFIER / "data/import_report.json")["archives"]}
    if sha256(PHONE_EXPORT) != expected[PHONE_EXPORT.name]:
        raise ValueError("原始手机导出ZIP与既有导入报告哈希不一致")
    with zipfile.ZipFile(PHONE_EXPORT) as archive:
        phone_rows = [json.loads(line) for line in archive.read("comments.jsonl").decode("utf-8-sig").splitlines() if line.strip()]
    phone_lookup, provenance_lookup = defaultdict(list), defaultdict(list)
    for row in phone_rows:
        phone_lookup[str(row["mid"]), normalize(row["text"])].append(row)
    for row in read_jsonl(EARLY_PROVENANCE):
        provenance_lookup[identity(row)].append(row)
    bvid_aids, raw_by_id = {}, {}
    for path in (*RAW_EXPORTS, *NEW_EXPORTS):
        for row in read_jsonl(path):
            bvid, aid = row["bvid"], str(row["aid"])
            if bvid in bvid_aids and bvid_aids[bvid] != aid:
                raise ValueError("同一bvid出现冲突aid")
            bvid_aids[bvid] = aid
            raw_by_id[str(row["mid"]), str(row["rpid"])] = row
    return bvid_aids, phone_lookup, provenance_lookup, raw_by_id


def merge_body_lineage(members, contexts):
    videos, traces = set(), []
    for member in members:
        recovered, trace = recover_video_groups(member, *contexts)
        videos.update(recovered)
        traces.extend(trace)
    return {"videos": sorted(videos), "recovered_sources": traces,
            "associated_uids": sorted({member["uid"] for member in members}),
            "source_metadata_ids": sorted({member["sample_id"] for member in members}),
            "source_sample_ids": sorted({s["sample_id"] for member in members for s in member["sources"]}),
            "bvids": sorted({s["bvid"] for member in members for s in member["sources"]})}


def assign_video_splits(rows, seed=20261114, forbidden=None):
    stats = defaultdict(lambda: [0, 0, 0])
    totals, unknown = [0, 0, 0], [0, 0, 0]
    for row in rows:
        vector = [int(row["label"] == 0), int(row["label"] == 1), int(row["label"] == 1 and row["ordinary"])]
        totals = [a + b for a, b in zip(totals, vector)]
        if not row["videos"]:
            unknown = [a + b for a, b in zip(unknown, vector)]
        for video in set(row["videos"]):
            stats[video] = [a + b for a, b in zip(stats[video], vector)]
    counts = {role: list(unknown) if role == "train" else [0, 0, 0] for role in ROLES}
    ratios = dict(zip(ROLES, (.70, .15, .15)))
    videos = sorted(stats)
    random.Random(seed).shuffle(videos)
    videos.sort(key=lambda video: -max(n / max(t, 1) for n, t in zip(stats[video], totals)))
    forbidden = forbidden or {}
    mapping = {}
    for video in videos:
        def cost(role):
            return sum(((counts[role][j] + stats[video][j]) / max(totals[j] * ratios[role], 1) - 1) ** 2
                       - (counts[role][j] / max(totals[j] * ratios[role], 1) - 1) ** 2 for j in range(3))
        candidates = [role for role in ROLES if role not in forbidden.get(video, ())] or list(ROLES)
        chosen = min(candidates, key=cost)
        mapping[video] = chosen
        counts[chosen] = [a + b for a, b in zip(counts[chosen], stats[video])]
    return mapping


def partition_rows(rows, video_mapping):
    result = []
    uid_roles, template_roles = defaultdict(set), defaultdict(set)
    for original in rows:
        row = dict(original)
        proposed = {video_mapping[v] for v in row["videos"]}
        role = next(iter(proposed)) if len(proposed) == 1 else "train" if not proposed else "quarantine"
        row.update(proposed_split=role, split=role, template=template_key(row["text"]),
                   quarantine_reasons=["source_videos_cross_split"] if role == "quarantine" else [])
        if role != "quarantine":
            for uid in row.get("associated_uids", [row["uid"]]):
                uid_roles[uid].add(role)
            template_roles[row["template"]].add(role)
        result.append(row)
    for row in result:
        if row["split"] not in ("train", "quarantine"):
            if any(len(uid_roles[uid]) > 1 for uid in row.get("associated_uids", [row["uid"]])):
                row["quarantine_reasons"].append("uid_cross_split")
            if len(template_roles[row["template"]]) > 1:
                row["quarantine_reasons"].append("template_cross_split")
            if row["quarantine_reasons"]:
                row["split"] = "quarantine"
    indices = {role: near_index({r["template"] for r in result if r["split"] == role}) for role in ROLES}
    for row in result:
        if row["split"] in ("validation", "test") and any(
                near_match(row["template"], indices[role]) for role in ROLES if role != row["split"]):
            row["quarantine_reasons"].append("near_template_cross_split_heuristic")
            row["split"] = "quarantine"
    return result


def leakage_audit(rows):
    report = {}
    for i, a in enumerate(ROLES):
        left = [r for r in rows if r["split"] == a]
        for b in ROLES[i + 1:]:
            right = [r for r in rows if r["split"] == b]
            counts = {"uids": len({uid for r in left for uid in r.get("associated_uids", [r["uid"]])}
                                  & {uid for r in right for uid in r.get("associated_uids", [r["uid"]])}),
                      "videos": len({v for r in left for v in r["videos"]} & {v for r in right for v in r["videos"]}),
                      "templates": len({r["template"] for r in left} & {r["template"] for r in right}),
                      "model_visible_body": len({normalize(r["text"])[:4096] for r in left}
                                                & {normalize(r["text"])[:4096] for r in right})}
            if any(counts.values()):
                raise ValueError(f"划分泄漏: {a}/{b}: {counts}")
            report[f"{a}/{b}"] = counts
    return report


def weighted_training_lines(rows, lines, weight):
    pairs = [(line, row["sample_id"]) for row, line in zip(rows, lines, strict=True)
             for _ in range(weight if row["label"] == 1 and row["ordinary"] else 1)]
    random.Random(42).shuffle(pairs)
    return [line for line, _ in pairs], [identifier for _, identifier in pairs]


def validation_rank(ordinary_recall, metrics, model_bytes):
    return ordinary_recall, metrics["recall"], metrics["precision"], -model_bytes


def groups(rows, scores, threshold):
    def describe(indices):
        if not indices:
            return None
        y = [rows[i]["label"] for i in indices]
        result = binary_metrics(y, [scores[i] for i in indices], threshold)
        if not any(y):
            result["recall"] = None
        result["rows"] = len(indices)
        return result
    result = {"all": describe(list(range(len(rows)))),
              "ordinary": describe([i for i, r in enumerate(rows) if r["ordinary"]])}
    for low, high, name in ((1, 20, "1-20"), (21, 60, "21-60"), (61, sys.maxsize, "61+")):
        result[f"ordinary_length_{name}"] = describe([i for i, r in enumerate(rows)
                                                      if r["ordinary"] and low <= len(r["text"]) <= high])
    return result


def build_dataset():
    if any(p.exists() for p in (DATA, MODELS, REPORTS)):
        raise FileExistsError("数据、模型、报告必须使用三个新目录，不覆盖任何已有版本")
    sampling = read_json(REVIEW / "sampling.json")
    # v5 源扩充后 ai.jsonl 哈希必然变化；复核绑定样本 ID 而非文件哈希：
    # 61 条抽样必须在新的 AI 总体中逐一存在，且 uid+正文与原始抽样一致。
    sample_rows = {r["sample_id"]: r for r in read_jsonl(REVIEW / "sample.jsonl")}
    population = read_jsonl(SOURCE / "ai.jsonl")
    population_by_id = {r["sample_id"]: r for r in population}
    if len(population_by_id) != len(population):
        raise ValueError("新源 ai.jsonl 存在重复样本ID")
    for identifier in sampling["sample_ids_in_review_order"]:
        row = population_by_id.get(identifier)
        if row is None:
            raise ValueError(f"复核样本不在扩充后的AI总体中: {identifier}")
        reference = sample_rows.get(identifier)
        if reference is None or (str(row["uid"]), normalize(row["text"])) != (str(reference["uid"]), normalize(reference["text"])):
            raise ValueError(f"复核样本正文在新源中发生变化: {identifier}")
    corrections = apply_user_review.correction_map()
    false_block_ids = apply_user_review.reviewed_sample_ids()
    protected = [SOURCE / name for name in ("comments.csv", "label-decisions.jsonl", "ai.jsonl")]
    protected += [REVIEW / "sampling.json", REVIEW / "sample.jsonl", PROTOCOL, POLICY_PATH, EMOTES_PATH,
                  ASSETS / "preprocess.json", ASSETS / "jieba.dict.txt", Path(__file__),
                  V2_POLICY_PATH, *apply_user_review.review_csvs(),
                  CLASSIFIER / "models-clean-dual-comparison/text/model.bin", PHONE_EXPORT,
                  EARLY_PROVENANCE, CLASSIFIER / "data/import_report.json", *RAW_EXPORTS, *NEW_EXPORTS]
    hashes = {str(p): sha256(p) for p in protected}
    originals = read_csv(SOURCE / "comments.csv", FIELDS)
    metadata = read_jsonl(SOURCE / "label-decisions.jsonl")
    by_key = {identity(r): r for r in metadata}
    metadata_by_body = defaultdict(list)
    for meta in metadata:
        metadata_by_body[normalize(meta["text"])].append(meta)
    reviews = user_reviews(read_jsonl(SOURCE / "ai.jsonl"), sampling["sample_ids_in_review_order"])
    if len(reviews) != sampling["sample_size"]:
        raise ValueError("AI抽样确认中出现非AI旧标签，须显式处理")
    reviewed = {r["sample_id"] for r in reviews}
    cleaned, cleaning = clean_rows(originals)
    pattern = read_json(PROTOCOL)["ordinary_nickname_pattern"]
    hints = re.compile(pattern, re.I)
    contexts = source_contexts()
    rows = []
    for row in cleaned:
        meta = by_key[identity(row)]
        if is_mention_only(row["text"]):
            raise ValueError("源标注集仍有纯@记录")
        members = metadata_by_body[normalize(row["text"])]
        lineage = merge_body_lineage(members, contexts)
        rows.append({**row, "sample_id": meta["sample_id"],
                     **lineage, "video_recovered_from_early_import": not bool(meta["sources"]),
                     "ordinary": any(not hints.search(member["username"]) for member in members),
                     "user_reviewed_ai": bool(set(lineage["source_metadata_ids"]) & reviewed),
                     "original_label_source": meta["label_source"],
                     "source_metadata": str(SOURCE / "label-decisions.jsonl")})
    before = {r["sample_id"]: r["label"] for r in rows}
    for sample_id, item in corrections.items():
        if str(before[sample_id]) != "0":
            raise ValueError(f"复核第{item}条原标签不是人类: {sample_id}")
        if sample_id not in false_block_ids:
            raise ValueError(f"复核第{item}条不在误拦复核表中: {sample_id}")
    kept, gray = apply_user_review.apply_corrections(rows, corrections)
    for row in kept:
        row["label"] = int(row["label"])
        row["user_corrected_ai"] = row["sample_id"] in corrections
    gray = [{**row, "exclusion_kind": "user_gray_phrase",
             "note": "命中用户灰区表达；只排除出人类样本，不自动改为AI；原标签保留"} for row in gray]
    reviewed_videos = sorted({video for row in rows if row["sample_id"] in false_block_ids for video in row["videos"]})
    if not reviewed_videos:
        raise ValueError("误拦复核样本没有任何视频来源，无法执行开发侧约束")
    video_mapping = assign_video_splits(kept, seed=20261114,
                                        forbidden={video: {"test"} for video in reviewed_videos})
    rows = partition_rows(kept, video_mapping)
    overlap = leakage_audit(rows)
    retained = [r for r in rows if r["split"] in ROLES]
    removed = []
    body_groups = defaultdict(list)
    for row in originals:
        body_groups[normalize(row["text"])].append(row)
    for group in body_groups.values():
        conflicting = len({r["label"] for r in group}) > 1
        for row in group if conflicting else group[1:]:
            removed.append({"reason": "conflicting_same_body_labels" if conflicting else "duplicate_same_body",
                            "original": row, "source_metadata": by_key[identity(row)]})
    for folder in (DATA, MODELS, REPORTS):
        folder.mkdir(parents=True)
    write_csv(DATA / "comments.csv", retained, FIELDS)
    uid_mapping = {r["uid"]: r["split"] for r in retained}
    write_csv(DATA / "splits.csv", [{"uid": uid, "split": split} for uid, split in sorted(uid_mapping.items())], ("uid", "split"))
    save_jsonl(DATA / "sample-lineage.jsonl", rows)
    save_jsonl(DATA / "quarantine.jsonl", [r for r in rows if r["split"] == "quarantine"])
    save_jsonl(DATA / "cleaning-exclusions.jsonl", removed)
    save_jsonl(DATA / "user-reviewed-ai.jsonl", reviews)
    save_jsonl(DATA / "gray-excluded.jsonl", gray)
    decisions = []
    for index, batch in enumerate(apply_user_review.BATCHES, 1):
        ai_items = set(batch["ai_items"])
        for row in apply_user_review.load_review_rows(batch["review_csv"]):
            decisions.append({"批次": index, **row,
                              "用户结论": "AI（改标）" if int(row["序号"]) in ai_items
                              else "未指认为AI（维持人类标签）"})
    save_csv(DATA / "user-corrections-20261114.csv", decisions, tuple(decisions[0]))
    for name in ("legacy_uid_decisions.csv", "legacy_comment_judgments.jsonl", POLICY_PATH.name, EMOTES_PATH.name):
        shutil.copyfile(SOURCE / name, DATA / name)
    forms = read_csv(REVIEW / "抽样复核.csv", ("样本ID", "判定"))
    if [f["样本ID"] for f in forms] != sampling["sample_ids_in_review_order"]:
        raise ValueError("原抽样表ID顺序发生变化")
    confirmed = [{**r, "判定": "AI", "备注": "用户于2026-11-14确认：全是ai"} for r in forms]
    save_csv(DATA / "user-reviewed-sample.csv", confirmed, tuple(confirmed[0]))
    prior_review = REVIEW / "reviewed-by-user-20261114.csv"
    if not prior_review.exists():
        save_csv(prior_review, confirmed, tuple(confirmed[0]))
    elif read_csv(prior_review, ("样本ID", "判定")) != confirmed:
        raise ValueError("既有人工复核文件已变更，不覆盖")
    write_json(DATA / "user-review-summary.json", review_summary(confirmed, reviewed))

    pre = Preprocessor()
    counts = {}
    for role in ROLES:
        subset = [r for r in retained if r["split"] == role]
        if {r["label"] for r in subset} != {0, 1} or not any(r["ordinary"] and r["label"] == 1 for r in subset):
            raise ValueError(f"{role}缺少两类或普通昵称AI，无法按协议验收")
        counts[role] = {"rows": len(subset), "uids": len({r["uid"] for r in subset}),
                        "labels": dict(Counter(str(r["label"]) for r in subset)),
                        "ordinary_labels": dict(Counter(str(r["label"]) for r in subset if r["ordinary"]))}
        write_csv(DATA / "processed" / f"{role}.csv", subset, FIELDS)
        lines = [f"__label__{r['label']} {pre.encode(r['text'], '', 'text')}" for r in subset]
        for row, line in zip(subset, lines):
            assert line.split(" ", 1)[1] == pre.encode(row["text"], row["username"], "text")
            assert all(token.startswith("txt_") for token in line.split()[1:])
        folder = DATA / "processed/text"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{role}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if role == "train":
            for weight in (1, 3):
                weighted, ids = weighted_training_lines(subset, lines, weight)
                target = DATA / "processed/weighted-text" / f"train-weight{weight}.txt"
                target.parent.mkdir(exist_ok=True)
                target.write_text("\n".join(weighted) + "\n", encoding="utf-8")
                write_json(DATA / f"training-members-weight{weight}.json", ids)
    manifest = {"schema": 1, "seed": 20261114, "split_rule": "greedy video 70/15/15; reviewed videos kept off test; UID/template/near-template holdout quarantine",
                "split_source": "video allocation redrawn for user-correction round (seed 20261114)", "comments_sha256": sha256(DATA / "comments.csv"),
                "splits_sha256": sha256(DATA / "splits.csv"), "dictionary_sha256": pre.config["dictionary_sha256"],
                "preprocess_sha256": sha256(ASSETS / "preprocess.json"), "cleaning": cleaning, "splits": counts,
                "files": {p.relative_to(DATA / "processed").as_posix(): sha256(p)
                          for p in (DATA / "processed").rglob("*") if p.is_file()}}
    write_json(DATA / "processed/manifest.json", manifest)
    write_json(DATA / "protocol.json", {"video_split_seed": 20261114, "target_ratios": [.7, .15, .15],
        "ordinary_nickname_pattern": pattern, "trials": TRIALS, "training_defaults": DEFAULT_PARAMS,
        "max_validation_fpr": .01, "secondary_reporting_fpr": .02,
        "selection_order": ["ordinary_ai_recall", "overall_recall", "precision", "smaller_bytes"],
        "selection_splits": ["train", "validation"], "test_used_for_selection": False})
    split_report = {"source_sha256": hashes, "cleaning": cleaning, "video_assignment": video_mapping,
                    "video_counts": dict(Counter(video_mapping.values())), "partitions": counts,
                    "quarantine_rows": len(rows) - len(retained),
                    "quarantine_reasons": dict(Counter(reason for r in rows for reason in r["quarantine_reasons"])),
                    "overlap": overlap, "user_reviewed_ai": len(reviews),
                    "user_corrections": {"relabelled_to_ai": len(corrections),
                                         "batches": [{"batch": index, "statement": batch["statement"],
                                                      "recorded_on": batch["recorded_on"],
                                                      "ai_items": list(batch["ai_items"]),
                                                      "reviewed_rows": len(apply_user_review.load_review_rows(batch["review_csv"]))}
                                                     for index, batch in enumerate(apply_user_review.BATCHES, 1)],
                                         "reviewed_false_blocks": len(false_block_ids),
                                         "reviewed_videos": reviewed_videos,
                                         "reviewed_videos_allowed_roles": ["train", "validation"],
                                         "corrected_row_splits": dict(Counter(r["split"] for r in rows if r["user_corrected_ai"]))},
                    "gray_excluded": {"rows": len(gray),
                                      "by_phrase": dict(Counter(m for r in gray for m in r["human_exclusion_matches"])),
                                      "policy_version": apply_user_review.json.loads(V2_POLICY_PATH.read_text(encoding="utf-8"))["version"],
                                      "action": "excluded from human samples; never promoted to AI"},
                    "video_id_policy": "collector aid and phone replyType1 oid share canonical video:<aid>; all deduplicated origins unioned",
                    "source_expansion": {"source": "data-rule-ge30-20261114-r2",
                        "policy": "v1 source + 2026-11-14 topics batch (all labels kept) + dsh official video AI-only (user: 只记录ai评论)",
                        "new_rows_human": sum(1 for r in metadata if r.get("label_source") == "weak_rule_human_new_collection"),
                        "new_rows_ai": sum(1 for r in metadata if r.get("label_source") == "user_rule_score_ge_30" and r.get("newly_added"))},
                    "source_integrity_repair": "all original UID/video origins merged for duplicate bodies; holdout redrawn because reviewed videos moved to development",
                    "video_recovered_from_early_import": sum(r["video_recovered_from_early_import"] for r in rows),
                    "unknown_video_train_only": sum(not r["videos"] and r["split"] == "train" for r in rows),
                    "near_duplicates": "prefix/suffix candidate + >=0.90 edit similarity heuristic, not semantic independence",
                    "prior_data_status": "previously collected/labeled archive, freshly split for this new training run"}
    write_json(DATA / "split-report.json", split_report)
    announce("dataset_ready", partitions=counts, quarantined=split_report["quarantine_rows"])
    return rows, manifest, hashes


def predict_text(model_path, text):
    if not normalize(text) or is_mention_only(text):
        return {"label": None, "ai_style_score": None, "blocked": False, "reason": "insufficient_body"}
    return Predictor(model_path, mode="text", blacklist=None).predict_one(text=text)


def run():
    rows, manifest, protected = build_dataset()
    processed, _ = verify_prepared(DATA)
    corrections = apply_user_review.correction_map()
    false_block_ids = apply_user_review.reviewed_sample_ids()
    subsets = {role: [r for r in rows if r["split"] == role] for role in ROLES}
    test_ids = {r["sample_id"] for r in subsets["test"]}
    if test_ids & false_block_ids:
        raise ValueError("用户复核过的误拦样本进入了测试集，必须留在开发侧（训练/验证）")
    corrected = [r for r in rows if r["user_corrected_ai"]]
    if len(corrected) != len(corrections) or any(r["label"] != 1 for r in corrected):
        raise ValueError("用户改判的AI样本未全部保留为AI或被隔离")
    if any(r["split"] == "test" for r in corrected):
        raise ValueError("改判样本进入了测试集")
    _, validation_y, validation_features = _examples(processed, "text", "validation")
    trials, best_rank, best = [], None, None
    for config in TRIALS:
        start = time.perf_counter()
        path = MODELS / "candidates" / f"{config['name']}.bin"
        params = {**DEFAULT_PARAMS, **config["params"]}
        train_file = processed / "weighted-text" / f"train-weight{config['weight']}.txt"
        model = train_model(train_file, path, params)
        scores = positive_scores(model, validation_features)
        selected = select_threshold(validation_y, scores, .01)
        grouped = groups(subsets["validation"], scores, selected["threshold"])
        metadata = _metadata("text", model, path, manifest, params, selected)
        metadata.update(training_ordinary_ai_weight=config["weight"], training_input=str(train_file),
                        training_input_sha256=sha256(train_file), validation_groups=grouped,
                        training_seconds=time.perf_counter() - start)
        write_json(str(path) + ".json", metadata)
        trial = {"name": config["name"], "path": str(path), "metadata": metadata}
        trials.append(trial)
        rank = validation_rank(grouped["ordinary"]["recall"], selected["metrics"], path.stat().st_size)
        if best_rank is None or rank > best_rank:
            best_rank, best = rank, trial
        announce("trial_complete", name=config["name"], ordinary_recall=grouped["ordinary"]["recall"],
                 recall=selected["metrics"]["recall"], fpr=selected["metrics"]["fpr"], seconds=metadata["training_seconds"])
        del model

    folder = MODELS / "text"
    folder.mkdir()
    original = folder / "model.bin"
    shutil.copyfile(best["path"], original)
    shutil.copyfile(best["path"] + ".json", str(original) + ".json")
    shutil.copytree(ASSETS, folder / "assets")
    for policy in (POLICY_PATH, EMOTES_PATH):
        shutil.copyfile(policy, folder / policy.name)
    raw_metadata = best["metadata"]
    quantized = folder / "model.ftz"
    quant = quantize_model(original, Path(raw_metadata["training_input"]), quantized, cutoff=20000, dsub=2)
    quant_scores = positive_scores(quant, validation_features)
    quant_selected = select_threshold(validation_y, quant_scores, .01)
    quant_groups = groups(subsets["validation"], quant_scores, quant_selected["threshold"])
    quant_metadata = _metadata("text", quant, quantized, manifest, raw_metadata["params"], quant_selected)
    quant_metadata.update(original_model_sha256=sha256(original), validation_groups=quant_groups,
                          training_ordinary_ai_weight=raw_metadata["training_ordinary_ai_weight"],
                          training_input=raw_metadata["training_input"], training_input_sha256=raw_metadata["training_input_sha256"],
                          quantization={"cutoff": 20000, "dsub": 2, "retrain": False, "qnorm": True, "qout": False})
    write_json(str(quantized) + ".json", quant_metadata)
    del quant
    recommended = "ftz" if validation_rank(quant_groups["ordinary"]["recall"], quant_selected["metrics"], quantized.stat().st_size) > best_rank else "bin"
    evaluation_entries = [("baseline", Path(trials[0]["path"])), ("selected_original", original), ("selected_quantized", quantized)]
    frozen = {"selected_trial": best["name"], "recommended_format": recommended,
              "recommended_model": str(folder / f"model.{recommended}"), "test_used_for_selection": False,
              "models": {}, "trials": trials}
    for name, path in evaluation_entries:
        model = load_model(path)
        metadata = read_json(str(path) + ".json")
        scores = positive_scores(model, validation_features)
        frozen["models"][name] = {"path": str(path), "sha256": sha256(path),
            "working_points": {str(cap): select_threshold(validation_y, scores, cap) for cap in (.01, .02)}}
        del model
    write_json(MODELS / "selection-frozen-before-test.json", frozen)
    announce("selection_frozen", trial=best["name"], recommended_format=recommended)

    _, test_y, test_features = _examples(processed, "text", "test")
    reports, prediction_rows = [], []
    for name, path in evaluation_entries:
        model = load_model(path)
        scores = positive_scores(model, test_features)
        working_points = frozen["models"][name]["working_points"]
        for cap, selected in working_points.items():
            grouped = groups(subsets["test"], scores, selected["threshold"])
            reports.append({"name": name, "validation_fpr_cap": float(cap), "threshold": selected["threshold"],
                            "test": grouped, "model_bytes": path.stat().st_size, "model_sha256": sha256(path)})
        threshold = working_points["0.01"]["threshold"]
        prediction_rows.extend({"model": name, "sample_id": r["sample_id"], "uid": r["uid"],
            "text": r["text"], "label": r["label"], "ordinary": r["ordinary"], "score": score,
            "threshold": threshold, "predicted": int(score >= threshold)} for r, score in zip(subsets["test"], scores))
        del model
    save_csv(REPORTS / "test-predictions.csv", prediction_rows, tuple(prediction_rows[0]))
    save_jsonl(REPORTS / "test-predictions.jsonl", prediction_rows)
    recommended_path = folder / f"model.{recommended}"
    predictor = Predictor(recommended_path, mode="text", blacklist=None)
    for row in subsets["test"][:30]:
        ordinary_result = predictor.predict_one(text=row["text"], username="普通昵称", uid="1")
        altered_result = predictor.predict_one(text=row["text"], username="大肥鱼AI机器人", uid="999999999")
        assert ordinary_result["ai_style_score"] == altered_result["ai_style_score"]
    assert predict_text(recommended_path, "回复 @某人：[doge]")["label"] is None
    perf_rows = subsets["test"][:200]
    perf = benchmark(predictor.model, predictor.pre, perf_rows, "text", test_features[:200], repeats=3)
    assert all(sha256(Path(path)) == digest for path, digest in protected.items())
    verify_prepared(DATA)
    result = {"selection": {k: frozen[k] for k in ("selected_trial", "recommended_format", "recommended_model", "test_used_for_selection")},
              "selection_file_sha256": sha256(MODELS / "selection-frozen-before-test.json"),
              "data": read_json(DATA / "split-report.json"), "results": reports,
              "runtime_checks": {"nickname_uid_invariance_real_examples": 30, "mention_only_abstains": True,
                                 "only_text_features": True, "selected_model_load_verified": True},
              "latency": {"test_subset_rows": len(perf_rows), **perf},
              "source_hashes_unchanged": True,
              "source_integrity_repair": {"duplicate_body_origins_merged": True,
                  "note": "重复正文的全部原账号/视频来源已合并；本轮v5训练源扩充为 data-rule-ge30-20261114-r2"
                          "（+14个话题视频与dsh官方视频AI-only），79条已复核样本所在视频全部留在开发侧，"
                          "留出集按同一协议重新划分，未沿用冻结分配。"},
              "user_corrections": {"relabelled_to_ai": len(corrections),
                  "reviewed_false_blocks": len(false_block_ids),
                  "reviewed_videos_development_only": True,
                  "batches": [{"batch": index, "statement": batch["statement"], "recorded_on": batch["recorded_on"],
                               "ai_items": list(batch["ai_items"])}
                              for index, batch in enumerate(apply_user_review.BATCHES, 1)]},
              "limitations": ["Rule/user labels measure label agreement; not verified AI authorship.",
                              "Freshly split existing archive, not a new prospective source dataset.",
                              "The 1% cap is validation empirical FPR; test/deployment need not meet it.",
                              "Approximate duplicate removal is heuristic; only a few held-out videos.",
                              "Desktop timing only; no Android update or hardware test."]}
    write_json(REPORTS / "evaluation.json", result)
    write_report(result)
    announce("complete", report=str(REPORTS / "evaluation.md"), model=str(recommended_path))


def write_report(report):
    corrections = report["data"]["user_corrections"]
    gray = report["data"]["gray_excluded"]
    batch_text = "；".join(
        f"批次{item['batch']}（{item['recorded_on']}，用户原话“{item['statement']}”）：第{('、'.join(str(number) for number in item['ai_items']))}条改判AI"
        for item in corrections["batches"])
    lines = ["# 正文专项重训结果", "", "用户确认61条随机样本全为AI，复核记录已单独保存。其余仍遵从既定用户/规则标签；"
             "下列是标签一致性指标，不是有来源日志验证的AI作者识别准确率。", "",
             f"本轮训练源扩充为 data-rule-ge30-20261114-r2：并入 2026-11-14 采集的 14 个话题视频（6,198 条，"
             f"新人类行按扩充目的保留）与 dsh 官方入驻视频的 AI-only 354 条（用户要求只记录AI评论）。", "",
             f"用户累计复核六批测试误拦共{corrections['reviewed_false_blocks']}条：{batch_text}。"
             f"这些复核样本所在的{len(corrections['reviewed_videos'])}个视频全部移出测试集、只用于训练/验证，留出集按同一贪心协议重新划分。"
             f"另有{gray['rows']}条命中灰区表达（{('、'.join(sorted(gray['by_phrase'])))}）的人类候选只排除、不改标。", "",
             "模型输入仅为正文；昵称只用于分组与训练权重，UID只用于划分和追溯。", "",
             f"选中候选：{report['selection']['selected_trial']}；推荐格式：{report['selection']['recommended_format']}。"
             "全部选择在验证集完成并写入冻结清单后，才对测试集评分。", "",
             "|集合|人类|AI|普通昵称人类|普通昵称AI|", "|---|---:|---:|---:|---:|"]
    for role, row in report["data"]["partitions"].items():
        lines.append(f"|{role}|{row['labels'].get('0',0)}|{row['labels'].get('1',0)}|"
                     f"{row['ordinary_labels'].get('0',0)}|{row['ordinary_labels'].get('1',0)}|")
    lines += ["", f"清理：{report['data']['cleaning']}；另隔离{report['data']['quarantine_rows']}条跨集合/近重复记录。"
              f"视频分配：{report['data']['video_counts']}。最终UID、已知视频、模板与模型可见正文交集均为0；"
              f"缺失视频来源的训练样本{report['data']['unknown_video_train_only']}条。", "",
              "|模型|验证误拦预算|测试整体召回|测试普通昵称AI检出|测试普通昵称AI召回|测试人类误拦|测试人类误拦率|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for result in report["results"]:
        all_m, ordinary = result["test"]["all"], result["test"]["ordinary"]
        lines.append(f"|{result['name']}|{result['validation_fpr_cap']:.0%}|{all_m['recall']:.2%}|"
                     f"{ordinary['confusion_matrix'][1][1]}/{ordinary['support']['ai_style']}|{ordinary['recall']:.2%}|"
                     f"{all_m['confusion_matrix'][0][1]}/{all_m['support']['normal']}|{all_m['fpr']:.2%}|")
    recommended_name = "selected_quantized" if report["selection"]["recommended_format"] == "ftz" else "selected_original"
    selected = next(r for r in report["results"] if r["name"] == recommended_name and r["validation_fpr_cap"] == .01)
    lines += ["", "## 推荐模型的普通昵称长度分组", "", "|正文长度|AI检出|召回|人类误拦|", "|---|---:|---:|---:|"]
    for label in ("1-20", "21-60", "61+"):
        row = selected["test"][f"ordinary_length_{label}"]
        if row:
            recall = f"{row['recall']:.2%}" if row["recall"] is not None else "无AI样本"
            lines.append(f"|{label}|{row['confusion_matrix'][1][1]}/{row['support']['ai_style']}|{recall}|"
                         f"{row['confusion_matrix'][0][1]}/{row['support']['normal']}|")
    interval = selected["test"]["all"]["fpr_wilson_95"]
    lines += ["", f"推荐模型阈值{selected['threshold']:.17g}；模型{selected['model_bytes']/1024:.1f}KiB，"
              f"冻结分词词典另占{(ASSETS/'jieba.dict.txt').stat().st_size/1024/1024:.2f}MiB。"
              f"测试误拦率Wilson95%区间：[{interval[0]:.2%}, {interval[1]:.2%}]（逐评论近似，账号/视频相关性仍存在）。", "",
              "验证1%预算不保证测试或线上为1%；规则筛选过的人类负类也不能代表所有真实评论。"
              "本次有新留出划分，但来自已经使用过的采集档案，不称为前瞻性外部验收。", "",
              f"推荐模型：{report['selection']['recommended_model']}。"
              "模型入口需排除纯@；本实验predict子命令已执行此输入门控。", "",
              f"桌面CPU预热单条端到端均值{report['latency']['end_to_end_single']['mean_ms']:.3f}ms；"
              "仅测测试集前200条，含预处理/评分，不含启动及文件I/O；不是Android实测。", "",
              "训练三组仅比较新数据同一划分；不能将旧模型在旧测试集上的成绩与本次直接相减当作提升。"
              "没有修改APK或昵称模型。", "",
              "本轮v8：新增第六批复核（第7条改判AI，其余维持人类），累计30条改判，"
              "79条已复核样本所在视频全部留在开发侧，留出集重新划分（种子20261114）。"
              "没有用测试分数调参或选模型；v1–v7输出保留为过程记录。"]
    (REPORTS / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (MODELS / "README.md").write_text(
        "# 新正文模型\n\n" + f"推荐使用{text_path(report)}；原始与量化版本分别为text/model.bin和text/model.ftz，"
        "各自阈值见同名.json。分词词典/配置在text/assets。\n\n"
        "仅正文，UID与昵称不进入模型。纯@无实质正文必须在评分前排除；不要用原始模型对这类输入直接作AI判断。\n\n"
        "当前工作区Python推理入口：\n\n```powershell\n"
        f"& '{(CLASSIFIER/'.venv/Scripts/python.exe').as_posix()}' -X utf8 '{Path(__file__).as_posix()}' predict "
        f"--model '{report['selection']['recommended_model'].replace(chr(92), '/')}' --text '待判定的正文'\n```\n\n"
        f"报告：{(REPORTS/'evaluation.md').as_posix()}；新数据：{DATA.as_posix()}。"
        "旧模型、旧数据与APK没有更新。\n", encoding="utf-8")
    (DATA / "README.md").write_text(
        "# 正文专项新划分（v2：用户改判后重建）\n\n"
        "comments.csv与splits.csv为本轮实际参与训练/验证/测试的数据。sample-lineage.jsonl保留每条清洗后记录的去向、"
        "源标签决策文件位置及原评论ID；quarantine.jsonl是隔离集合，不训练、不评价；cleaning-exclusions.jsonl保留冲突/重复原记录；"
        "gray-excluded.jsonl是命中用户灰区表达（v2名单，含“这句最戳我”与“不是X，是Y”变体）的人类候选，只排除、不改标。\n\n"
        "user-reviewed-ai.jsonl保存61条用户复核；user-corrections-20261114.csv保存各批测试误拦复核的结论"
        "（累计30条改判AI，其余维持人类）；这些样本所在视频全部不进测试集。"
        "未将风格确认改成作者发布日志证据。旧737个UID与138条评论人工意见文件原样复制。\n\n"
        "processed仅生成正文特征。weighted-text两份训练文件分别为普通昵称AI权重1/3，"
        "只重复真实训练记录，不生成或改写正文。training-members文件可核对每条重复记录只来自训练集。"
        "视频不明者只入训练。详见protocol.json、split-report.json与processed/manifest.json。\n", encoding="utf-8")


def text_path(report):
    return "text/model." + report["selection"]["recommended_format"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--text", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        print(json.dumps(predict_text(args.model, args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
