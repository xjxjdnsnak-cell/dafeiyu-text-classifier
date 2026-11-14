"""Build provenance-preserving review data; never infer authorship from style."""

import argparse
import csv
import difflib
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from bili_ai.common import read_csv, read_json, sha256, write_json
from bili_ai.model import load_model, positive_scores
from bili_ai.pipeline import _check_metadata, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize
from text_policy import EMOTES_PATH, MENTION, POLICY, POLICY_PATH, apply_text_policy, filter_labeled_rows, human_exclusion_matches, is_mention_only

CLASSIFIER = Path(__file__).resolve().parents[2]
ROOT = CLASSIFIER.parent
sys.path.insert(0, str(CLASSIFIER / "scripts"))
from build_expanded_gray import gray_reasons

OLD_SOURCES = (
    "out/comment-dataset-20261112/export/all_scored.jsonl",
    "out/collector-expand-20261112/export/all_scored.jsonl",
    "out/collector-newtopics-20261112/export/all_scored.jsonl",
)
STYLE_LABELS = {"unreviewed", "human_like", "ai_like", "mixed", "uncertain"}


def read_uid_reviews(path):
    decisions = {}
    for row in read_csv(path, ("UID", "判定")):
        uid, value = row["UID"].strip(), row["判定"].strip()
        if not uid.isdigit() or int(uid) <= 0 or value not in ("AI", "人类"):
            raise ValueError("账号人工判断格式异常，不把不确定或未知值改成人类")
        if uid in decisions and decisions[uid]["判定"] != value:
            raise ValueError("同 UID 的人工判断冲突，需要显式解决")
        decisions[uid] = row
    return decisions


def evidence_fields(score, export_label, manual_account_label, legacy_comment_label):
    return {"rule_score": score, "raw_export_label": export_label,
        "weak_comment_label": "ai" if score > 55 else "human" if export_label == "human" else "review",
        "manual_account_label": manual_account_label or "unreviewed",
        "legacy_comment_judgment": legacy_comment_label or "unreviewed",
        "body_style_label": "unreviewed", "style_notes": "", "style_reviewer": "",
        "authorship_label": "uncertain", "authorship_evidence": [],
        "authorship_status": "not_verified_from_rule_or_account_or_style"}


def apply_body_reviews(records, annotations):
    output = {row["sample_id"]: dict(row) for row in records}
    seen = set()
    for item in annotations:
        sample_id, label = item["sample_id"].strip(), item["body_style_label"].strip()
        if sample_id not in output or sample_id in seen or label not in STYLE_LABELS or item.get("style_reviewer") == "qa_preview":
            raise ValueError("正文复核 ID 未知/重复，或风格标签无效")
        seen.add(sample_id)
        row = output[sample_id]
        if is_mention_only(row.get("text", "")) and label == "ai_like":
            raise ValueError("纯 @ 没有有效正文，不能标为 AI 风格")
        if row["body_style_label"] not in ("unreviewed", label):
            raise ValueError("已有正文风格判断冲突，不能静默覆盖")
        row.update(body_style_label=label, style_notes=item.get("style_notes", ""),
                   style_reviewer=item.get("style_reviewer", ""))
    return [apply_text_policy(row) for row in output.values()]


def template_text(text):
    text = MENTION.sub("", normalize(text))
    text = re.sub(r"https?://\S+|\[[^\]]*\]|\d+", "", text)
    return "".join(ch for ch in text if not ch.isspace()
                   and not unicodedata.category(ch).startswith(("P", "S")))


def near_index(values):
    index = defaultdict(set)
    for text in values:
        if len(text) >= 20:
            index[("start", text[:12])].add(text)
            index[("end", text[-12:])].add(text)
    return index


def near_match(text, index):
    if len(text) < 20:
        return False
    candidates = index.get(("start", text[:12]), set()) | index.get(("end", text[-12:]), set())
    return any(min(len(text), len(other)) / max(len(text), len(other)) >= 0.85
               and difflib.SequenceMatcher(None, text, other, autojunk=False).ratio() >= 0.90
               for other in candidates)


def assign_roles(records, planned, known_uids=(), known_templates=()):
    records = [apply_text_policy(row) for row in records]
    old = [row for row in records if row["is_old_source"]]
    fresh = [row for row in records if not row["is_old_source"]]
    old_uids = {r["uid"] for r in old} | set(known_uids)
    old_videos = {r["bvid"] for r in old}
    old_templates = {r["template_text"] for r in old} | set(known_templates)
    old_near = near_index(old_templates)
    uid_roles, template_roles = defaultdict(set), defaultdict(set)
    for row in fresh:
        if row["bvid"] not in planned:
            raise ValueError("新视频未在采集前分配用途")
        if row["text_model_eligible"]:
            uid_roles[row["uid"]].add(planned[row["bvid"]])
            template_roles[row["template_text"]].add(planned[row["bvid"]])
    near_by_role = {role: near_index({r["template_text"] for r in fresh if planned[r["bvid"]] == role and r["text_model_eligible"]})
                    for role in ("validation", "test")}
    output = []
    for row in records:
        result = dict(row)
        reasons = []
        if not row["text_model_eligible"]:
            role = "excluded_text"
            if not row["is_old_source"]:
                result["planned_split"] = planned[row["bvid"]]
        elif row["is_old_source"]:
            role = "development"
        else:
            intended = planned[row["bvid"]]
            other = "test" if intended == "validation" else "validation"
            if row["uid"] in old_uids:
                reasons.append("uid_seen_in_old_data")
            if row["bvid"] in old_videos:
                reasons.append("video_seen_in_old_data")
            if len(uid_roles[row["uid"]]) > 1:
                reasons.append("uid_bridges_validation_test")
            if row["template_text"] in old_templates:
                reasons.append("normalized_template_seen_in_old_data")
            elif near_match(row["template_text"], old_near):
                reasons.append("near_template_seen_in_old_data")
            if len(template_roles[row["template_text"]]) > 1:
                reasons.append("template_bridges_validation_test")
            elif near_match(row["template_text"], near_by_role[other]):
                reasons.append("near_template_bridges_validation_test")
            role = "quarantine" if reasons else intended
            result["planned_split"] = intended
        result.update(pool_role=role, quarantine_reasons=reasons)
        output.append(result)
    return output


def length_group(length):
    return "1-20" if length <= 20 else "21-60" if length <= 60 else "61+"


def match_controls(positives, controls, per_positive=2):
    by_video = defaultdict(list)
    for row in controls:
        if not is_mention_only(row["text"]) and not human_exclusion_matches(row["text"]):
            by_video[row["bvid"]].append(row)
    used, pairs = set(), []
    for positive in positives:
        if is_mention_only(positive["text"]):
            continue
        length = len(positive["text"])
        candidates = [row for row in by_video[positive["bvid"]]
                      if row["sample_id"] not in used and row["uid"] != positive["uid"]
                      and length_group(len(row["text"])) == length_group(length)
                      and abs(len(row["text"]) - length) <= max(4, int(length * 0.25))]
        candidates.sort(key=lambda row: (
            0 if row.get("legacy_comment_judgment") == "human" else
            1 if row.get("manual_account_label") == "human" else 2,
            abs(len(row["text"]) - length), row["sample_id"]))
        for control in candidates[:per_positive]:
            used.add(control["sample_id"])
            pairs.append({"positive_candidate_id": positive["sample_id"], "control_id": control["sample_id"],
                "bvid": positive["bvid"], "positive_length": length, "control_length": len(control["text"]),
                "length_difference": abs(len(control["text"]) - length),
                "control_evidence": control.get("manual_account_label", "unreviewed"),
                "labels_are_pending": True})
    return pairs


def csv_cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and (value.lstrip().startswith(("=", "+", "-", "@"))
                                  or value.isdigit() and len(value) > 15):
        return "'" + value
    return value


def save_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key, "")) for key in fields})


def save_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_legacy_judgments(root):
    judgments = {}
    specs = [
        ("data-augment-70-15-15-tsfix-final/timestampfix_final_report.json", (("changed", "ai"), ("rejected_candidates", "human"))),
        ("data-augment-70-15-15-final4/revert_report.json", (("reverted", "human"),)),
        ("data-augment-70-15-15-final6/human_revert_report.json", (("reverted", "human"),)),
    ]
    for relative, groups in specs:
        obj = read_json(root / "fasttext-classifier" / relative)
        for group, label in groups:
            for row in obj[group]:
                judgments[(str(row["uid"]), normalize(row["text"]))] = {"label": label, "source": relative, "case": group}
    report = read_json(root / "fasttext-classifier/data-augment-70-15-15-final9/revert_report.json")
    assert "all 110" in report["action"] and "human" in report["action"]
    original = read_json(root / "fasttext-classifier/data-augment-70-15-15-final8/at_rule_v2_report.json")
    for group in ("changed_rows", "bare_rows", "manual_rows"):
        for row in original[group]:
            judgments[(str(row["uid"]), normalize(row["text"]))] = {
                "label": "human", "source": "data-augment-70-15-15-final9/revert_report.json",
                "case": f"all 110 confirmed human; row in final8/{group}"}
    return judgments


def load_records(root, sources, manual, legacy, hints):
    observations = {}
    for relative, is_old in sources:
        with (root / relative).open(encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                raw = json.loads(line)
                uid, rpid = str(raw["mid"]), str(raw["rpid"])
                text = normalize(raw.get("training_text") or raw["text"])
                if not text:
                    continue
                key = (uid, rpid)
                source = {"file": relative, "line": line_number, "raw_ref": raw["raw_ref"],
                          "collected_at": raw["collected_at"]}
                if key in observations:
                    if observations[key]["text"] != text:
                        raise ValueError("原始评论存在多版本正文，需显式处理")
                    observations[key]["provenance"].append(source)
                    continue
                account = raw.get("account") or {}
                name = normalize(account.get("uname", ""))
                decision = manual.get(uid)
                legacy_comment = legacy.get((uid, text))
                row = {"sample_id": "TXT-" + hashlib.sha256(f"{uid}:{rpid}".encode()).hexdigest()[:20],
                    "uid": uid, "rpid": rpid, "bvid": raw["bvid"], "video_title": raw["video_title"],
                    "url": raw["url"], "username": name, "text_raw": raw["text"], "text": text,
                    "text_mentions_masked": MENTION.sub("<MENTION>", text), "length": len(text),
                    "length_group": length_group(len(text)),
                    "nickname_group": "ordinary" if not hints.search(name) else "has_ai_name_hint",
                    "body_pattern_tags": gray_reasons(text), "template_text": template_text(text),
                    "template_id": hashlib.sha256(template_text(text).encode()).hexdigest(),
                    "is_old_source": is_old, "provenance": [source], "text_score": None,
                    "manual_account_review": decision, "legacy_comment_review": legacy_comment,
                    "rule_reasons": raw.get("reasons", []), "candidate_tags": []}
                row.update(evidence_fields(int(raw["score"]), raw["label"],
                    ("ai" if decision["判定"] == "AI" else "human") if decision else "",
                    legacy_comment["label"] if legacy_comment else ""))
                observations[key] = row
    return list(observations.values())


def add_development_scores(records, model_path, data):
    _, manifest = verify_prepared(data)
    metadata = _check_metadata(model_path, manifest)
    if metadata["mode"] != "text":
        raise ValueError("难例挖掘只使用既有正文模型")
    model, pre = load_model(model_path), Preprocessor()
    bodies = dict.fromkeys(row["text"] for row in records if row["pool_role"] == "development")
    values = positive_scores(model, [pre.encode(text, "", "text") for text in bodies])
    scores = dict(zip(bodies, values, strict=True))
    for row in records:
        if row["pool_role"] == "development":
            row["text_score"] = float(scores[row["text"]])


def select_candidates(records, threshold):
    development = [r for r in records if r["pool_role"] == "development" and not is_mention_only(r["text"])]
    queues = defaultdict(list)
    for row in development:
        tags = []
        ai_hint = row["weak_comment_label"] == "ai" or row["manual_account_label"] == "ai" or row["legacy_comment_judgment"] == "ai"
        human_hint = row["human_label_eligible"] and (row["manual_account_label"] == "human" or row["legacy_comment_judgment"] == "human")
        if row["nickname_group"] == "ordinary" and ai_hint:
            tags.append("ordinary_ai_candidate")
        if ai_hint and row["text_score"] < threshold:
            tags.append("possible_model_miss")
        if (human_hint or row["weak_comment_label"] == "human") and row["text_score"] >= threshold:
            tags.append("possible_false_block")
        if human_hint and row["body_pattern_tags"]:
            tags.append("human_style_challenge")
        if human_hint and row["nickname_group"] != "ordinary":
            tags.append("ai_hint_nickname_human_candidate")
        if row["manual_account_label"] in ("ai", "human") and row["manual_account_label"] != row["source_weak_comment_label"]:
            tags.append("manual_rule_disagreement")
        if row["legacy_comment_judgment"] == "ai" or row["legacy_comment_judgment"] == "human" and row["human_label_eligible"]:
            tags.append("legacy_comment_judgment")
        row["candidate_tags"] = tags
        for tag in tags:
            queues[tag].append(row)
    for name, rows in queues.items():
        rows.sort(key=lambda r: (0 if r["nickname_group"] == "ordinary" else 1,
            0 if r["manual_account_label"] != "unreviewed" else 1,
            -r["text_score"] if name == "possible_false_block" else r["text_score"], r["sample_id"]))
    selected_ids = set()
    for name, rows in queues.items():
        limit = len(rows) if name in ("ordinary_ai_candidate", "possible_false_block", "legacy_comment_judgment") else 150
        selected_ids.update(r["sample_id"] for r in rows[:limit])
    positives = queues["ordinary_ai_candidate"]
    controls = [r for r in development if r["human_label_eligible"] and r["manual_account_label"] != "ai"
                and r["legacy_comment_judgment"] != "ai"
                and (r["manual_account_label"] == "human" or r["legacy_comment_judgment"] == "human"
                     or r["weak_comment_label"] == "human")]
    pairs = match_controls(positives, controls)
    selected_ids.update(pair["control_id"] for pair in pairs)
    for row in development:
        if row["sample_id"] in {pair["control_id"] for pair in pairs}:
            row["candidate_tags"].append("matched_control_candidate")
    candidates = [r for r in records if r["sample_id"] in selected_ids
                  or r["pool_role"] in ("validation", "test")]
    return candidates, queues, pairs


def partition_audit(records):
    result = {}
    for left, right in (("development", "validation"), ("development", "test"), ("validation", "test")):
        a = [r for r in records if r["pool_role"] == left]
        b = [r for r in records if r["pool_role"] == right]
        counts = {key: len({r[key] for r in a} & {r[key] for r in b})
                  for key in ("uid", "bvid", "template_id")}
        if any(counts.values()):
            raise ValueError(f"隔离失败: {left}/{right}: {counts}")
        result[f"{left}_vs_{right}"] = counts
    return result


def build(output):
    parent = ROOT / "out/text-specialist-20261113"
    seeds = read_json(parent / "seed-manifest.json")
    planned = {row["bvid"]: row["planned_split"] for row in seeds["videos"]}
    manual_path = Path("E:/human-review-uid-decisions (2).csv")
    manual = read_uid_reviews(manual_path)
    legacy = load_legacy_judgments(ROOT)
    previous_protocol = read_json(CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json")
    hints = re.compile(previous_protocol["ordinary_nickname_pattern"], re.I)
    sources = [(rel, True) for rel in OLD_SOURCES] + [("out/text-specialist-20261113/collection/export/all_scored.jsonl", False)]
    hashes = {rel: sha256(ROOT / rel) for rel, _ in sources}
    hashes[str(manual_path)] = sha256(manual_path)
    hashes["out/text-specialist-20261113/seed-manifest.json"] = sha256(parent / "seed-manifest.json")
    hashes[POLICY_PATH.relative_to(ROOT).as_posix()] = sha256(POLICY_PATH)
    hashes[EMOTES_PATH.relative_to(ROOT).as_posix()] = sha256(EMOTES_PATH)
    hashes[Path(__file__).with_name("text_policy.py").relative_to(ROOT).as_posix()] = sha256(Path(__file__).with_name("text_policy.py"))
    model_data_path = CLASSIFIER / "data-augment-70-15-15-clean-uid/comments.csv"
    model_rows = read_csv(model_data_path, ("uid", "text", "label"))
    hashes[model_data_path.relative_to(ROOT).as_posix()] = sha256(model_data_path)
    for relative in (
        "fasttext-classifier/data-augment-70-15-15-tsfix-final/timestampfix_final_report.json",
        "fasttext-classifier/data-augment-70-15-15-final4/revert_report.json",
        "fasttext-classifier/data-augment-70-15-15-final6/human_revert_report.json",
        "fasttext-classifier/data-augment-70-15-15-final8/at_rule_v2_report.json",
        "fasttext-classifier/data-augment-70-15-15-final9/revert_report.json",
        "fasttext-classifier/experiments/text-specialist-20261113/review_template.html",
    ):
        hashes[relative] = sha256(ROOT / relative)
    records = load_records(ROOT, sources, manual, legacy, hints)
    old_videos = {r["bvid"] for r in records if r["is_old_source"]}
    if old_videos & planned.keys():
        raise ValueError("预先选择的新视频已出现在旧语料")
    known_uids = {r["uid"] for r in model_rows} | set(manual)
    known_templates = {template_text(r["text"]) for r in model_rows} | {template_text(text) for _, text in legacy}
    records = assign_roles(records, planned, known_uids, known_templates)
    model_path = CLASSIFIER / "models-clean-dual-comparison/text/model.bin"
    model_hash = sha256(model_path)
    add_development_scores(records, model_path, CLASSIFIER / "data-augment-70-15-15-clean-uid")
    frozen = read_json(CLASSIFIER / "reports-text-fusion-20261113-r3/frozen-selection.json")
    threshold = frozen["0.02"]["candidates"]["single"]["single_threshold"]
    candidates, queues, pairs = select_candidates(records, threshold)
    audit = partition_audit(records)
    output.mkdir(parents=True, exist_ok=False)
    save_jsonl(output / "corpus.jsonl", records)
    save_jsonl(output / "candidates.jsonl", candidates)
    mention_only_rows = [r for r in records if not r["text_model_eligible"]]
    human_gray_rows = [r for r in records if r["human_exclusion_matches"] and (
        r["source_weak_comment_label"] == "human" or r["manual_account_label"] == "human" or r["legacy_comment_judgment"] == "human")]
    save_jsonl(output / "excluded-mention-only.jsonl", mention_only_rows)
    save_jsonl(output / "excluded-human-expressions.jsonl", human_gray_rows)
    save_jsonl(output / "legacy_comment_judgments.jsonl", [
        {"uid": uid, "text": text, **decision} for (uid, text), decision in legacy.items()])
    shutil.copyfile(manual_path, output / "legacy_uid_decisions.csv")
    evidence_fields_list = ("sample_id", "pool_role", "uid", "rpid", "username", "bvid", "url", "length",
        "manual_account_label", "legacy_comment_judgment", "weak_comment_label", "rule_score",
        "text_score", "nickname_group", "body_pattern_tags", "candidate_tags", "authorship_label", "provenance",
        "source_weak_comment_label", "text_model_eligible", "human_label_eligible", "text_exclusion_reason", "human_exclusion_matches")
    save_csv(output / "evidence.csv", candidates, evidence_fields_list)
    save_csv(output / "excluded-mention-only.csv", mention_only_rows, (*evidence_fields_list, "text"))
    save_csv(output / "excluded-human-expressions.csv", human_gray_rows, (*evidence_fields_list, "text"))
    body_fields = ("sample_id", "text", "body_style_label", "style_notes", "style_reviewer")
    for role in ("development", "validation", "test"):
        rows = [r for r in candidates if r["pool_role"] == role]
        rows.sort(key=lambda r: hashlib.sha256(("blind-v1:" + r["sample_id"]).encode()).hexdigest())
        save_csv(output / f"body-review-{role}.csv", rows, body_fields)
    source_fields = ("sample_id", "authorship_label", "evidence_type", "evidence_url_or_path", "source_notes", "source_reviewer")
    save_csv(output / "source-review.csv", candidates, source_fields)
    save_csv(output / "matched-controls.csv", pairs, ("positive_candidate_id", "control_id", "bvid", "positive_length", "control_length", "length_difference", "control_evidence", "labels_are_pending"))
    queue_dir = output / "queues"
    queue_dir.mkdir()
    for name, rows in queues.items():
        save_csv(queue_dir / f"{name}.csv", rows, (*evidence_fields_list, "text"))
    role_counts = Counter(r["pool_role"] for r in records)
    report = {"source_hashes": hashes, "model_sha256": model_hash, "builder_sha256": sha256(Path(__file__)),
        "raw_unique_comments": len(records), "old_comments": sum(r["is_old_source"] for r in records),
        "fresh_comments": sum(not r["is_old_source"] for r in records), "roles": dict(role_counts),
        "candidates": len(candidates), "candidate_roles": dict(Counter(r["pool_role"] for r in candidates)),
        "queue_counts": {key: len(value) for key, value in queues.items()}, "matched_control_pairs": len(pairs),
        "positive_candidates_without_control": len({r["sample_id"] for r in queues["ordinary_ai_candidate"]}
            - {p["positive_candidate_id"] for p in pairs}),
        "manual_uid_decisions_preserved": len(manual),
        "model_and_manual_uids_in_leakage_guard": len(known_uids),
        "manual_uid_decisions": dict(Counter(r["判定"] for r in manual.values())),
        "legacy_comment_decisions_preserved": len(legacy),
        "legacy_comment_decisions_matched_in_raw": sum((r["uid"], r["text"]) in legacy for r in records),
        "quarantine_reasons": dict(Counter(reason for r in records for reason in r["quarantine_reasons"])),
        "partition_overlap": audit,
        "prospective_rows_model_scored": sum(r["text_score"] is not None for r in records if not r["is_old_source"]),
        "ordinary_prospective_counts": {role: sum(r["pool_role"] == role and r["nickname_group"] == "ordinary" for r in records) for role in ("validation", "test")},
        "prospective_length_counts": {role: dict(Counter(r["length_group"] for r in records if r["pool_role"] == role)) for role in ("validation", "test")},
        "authorship_label_counts": dict(Counter(r["authorship_label"] for r in candidates)),
        "body_style_label_counts": dict(Counter(r["body_style_label"] for r in candidates))}
    report.update(mention_only_excluded=len(mention_only_rows),
        mention_only_previous_weak_ai=sum(r["source_weak_comment_label"] == "ai" for r in mention_only_rows),
        human_expression_excluded=len(human_gray_rows),
        human_expression_rule_counts=dict(Counter(hit for r in human_gray_rows for hit in r["human_exclusion_matches"])),
        human_expression_unique_rules=len(set(POLICY["source_phrases"])))
    protocol = {"primary_target": "comment authorship supported by source evidence, not rule agreement",
        "secondary_target": "human-reviewed body style consistency; report separately from authorship",
        "budgets": [0.01, 0.02], "threshold_selection": "validation only; freeze before test scoring",
        "required_groups": ["ordinary nickname", "all nicknames", "1-20 chars", "21-60 chars", "61+ chars"],
        "required_metrics": ["detected AI / eligible AI", "misblocked human / eligible human", "missed AI count", "uncertain/excluded count", "review coverage", "95% error-rate interval with UID-dependence caveat"],
        "recommended_sample_targets_not_current_counts": {"normal_validation": 1000, "normal_test": 1000, "ordinary_ai_test": 200},
        "split_policy": "old samples development only; new videos fixed before capture; raw-history/model-data/manual-review UIDs and overlapping templates quarantined, never moved into training on held-out videos",
        "template_screen": "normalized same text; near text >=20 chars sharing first/last 12 normalized chars and SequenceMatcher ratio >=0.90; heuristic does not detect all paraphrases",
        "unknown_policy": "retain uncertainty; style, account verdict and model score cannot certify authorship",
        "text_eligibility": "mention-only bodies excluded from text training/review/acceptance; user expression matches cannot be human-labelled negatives, but do not become AI labels; always report exclusions and coverage",
        "gate": "no formal authorship score without source-verified labels of both classes; no style score before body annotations",
        "sampling_limit": seeds["sampling"]}
    write_json(output / "audit.json", report)
    write_json(output / "acceptance-protocol.json", protocol)
    write_json(output / "field-definitions.json", {
        "manual_account_label": "原有 UID 人工判断；不等于这条正文的生成来源或风格",
        "legacy_comment_judgment": "原有逐评论用户判断；保留出处，不自动迁移为风格标签或新的来源核验",
        "weak_comment_label": "正文准入后的规则弱标签；纯 @ 或命中人类排除表达的原 human 改为 review，不改成 AI",
        "source_weak_comment_label": "应用新正文规则之前的规则弱标签，保留原始线索",
        "human_label_eligible": "是否可用于人类标签；用户表达命中后为 false，历史人工判断另存不覆盖",
        "text_model_eligible": "纯 @ 无有效正文为 false，单独留档",
        "body_style_label": {"values": sorted(STYLE_LABELS), "meaning": "逐评论表达风格，不能确认作者身份"},
        "authorship_label": "ai/human/uncertain，需独立逐评论来源证据；本批均保持 uncertain",
        "pool_role": "development/validation/test/quarantine/excluded_text；隔离和无正文记录不用于本轮训练或验收",
        "text_score": "仅旧 development 的模型难例挖掘分数，不是真值；新批次不评分",
        "provenance": "原始 JSONL 路径及行号、响应路径、采集时间，可追溯真实原文"})
    write_json(output / "readiness.json", {"authorship_acceptance_ready": False, "body_style_acceptance_ready": False,
        "reason": "No new per-comment authorship proof was supplied; body-style annotations pending",
        "source_verified_ai": 0, "source_verified_human": 0, "body_reviewed": 0,
        "preserved_legacy_judgments_are_not_new_source_verification": True})
    make_review_page(output, candidates)
    assert sha256(model_path) == model_hash
    assert all(sha256(Path(rel) if Path(rel).is_absolute() else ROOT / rel) == digest for rel, digest in hashes.items())
    write_readme(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def write_readme(output, report):
    text = f"""# 正文专项数据：候选与验收准备

新增真实采集 {report['fresh_comments']} 条；与旧语料合并追溯共 {report['raw_unique_comments']} 条。选出 {report['candidates']} 条供正文复核。

当前正文策略：纯 @ 点名（可带标点或已确认的 B 站表情）没有有效正文，移到 `excluded-mention-only.jsonl`；未知方括号文字保留为正文。表情 token 由本地原始响应的 content.emote 元数据确认。用户列出的表达从人类可用样本剔除，见 `excluded-human-expressions.jsonl`。这些记录保留原文、原账号/逐评论人工判断与原规则标签，不自动标为 AI。单字“稳”也按用户要求命中；xx/xxx 是可变内容。

本包没有训练模型，也没有把规则、账号人工判断或模型预测变成正文风格真值。用户确认没有可对应评论 ID 的 AI 运行日志；作者来源未能核验的全部保留 uncertain。

## 先看与复核

- `review.html`：本地正文复核页，默认隐藏账号证据；按钮记录风格后可查看出处，判断自动保存在本机，可导出 CSV。
- `body-review-development.csv`：只看正文填写 body_style_label，作者昵称、规则分和模型分数均隐藏。
- `body-review-validation.csv` / `body-review-test.csv`：新视频中的隔离样本；用于预留验收，正文标签同样待复核。
- 风格值：`human_like`（更像自然交流）、`ai_like`（呈现明显生成式话术）、`mixed`（引用/混合）、`uncertain`（无法判断）。未审保留 `unreviewed`。这些值只描述风格。
- 写简短 style_notes；极短句、引用 AI 回答、玩梗和角色扮演拿不准时保留不确定。
- `source-review.csv`：独立填写逐评论来源证据；不因正文“像 AI”就填确定作者身份。
- `evidence.csv`：复核正文后再对照作者、评论链接、原始出处、账号人工判断和规则线索。验证/测试样本没有模型评分。

## 已保留的判断

`legacy_uid_decisions.csv` 原样保存 {report['manual_uid_decisions_preserved']} 个 UID 的用户判断。`legacy_comment_judgments.jsonl` 保留 {report['legacy_comment_decisions_preserved']} 条明确的历史逐评论修正，标明来源；这些不会被机械规则覆盖，也不会自动变成“正文像真人/AI”的风格标签。

## 难例与匹配对照

`queues/ordinary_ai_candidate.csv` 包含普通昵称且有规则或旧人工线索的真实评论，属于候选。`possible_model_miss.csv` / `possible_false_block.csv` 用既有正文模型挖掘旧开发数据的分歧，不证明它们一定被误判。`human_style_challenge.csv` 包含旧人工账号/逐评论人类判断且出现引号、破折号等模式的内容。

`matched-controls.csv` 给出 {report['matched_control_pairs']} 对同视频、同长度组、长度差不超过 max(4字,25%) 的对照候选，优先选已有人工人类判断；标签仍需复核。无法匹配的正类候选不拼凑跨话题文本。

## 隔离和验收

旧数据全部归 development。六个新视频在采集前固定 validation/test 用途；新视频中已见 UID、跨集合 UID、同文/近似模板进入 quarantine，不转入训练。具体数量和隔离审计见 `audit.json`。近似模板检查是启发式，不能承诺排除所有复述。

隔离检查同时使用现有模型数据和旧人工记录中出现的 UID/模板，覆盖只存在于模型数据、未出现在原始采集归档中的账号。查看出处后再修改的风格判断属于知情复核，不能无条件视为严格盲审金标准。

新验收按固定 1% / 2% 验证误拦预算选阈值，冻结后再评分测试集；单列普通昵称和三个长度组，报告不确定样本与覆盖率。来源核验和正文风格分别验收，不能用风格一致性冒充作者身份准确率。完整规则见 `acceptance-protocol.json`；当前状态见 `readiness.json`。

## 复核导入

```powershell
# 从 fasttext-classifier 目录运行；新输出路径必须不存在
.venv/Scripts/python.exe experiments/text-specialist-20261113/text_specialist_dataset.py import-body --bundle '{output.resolve().as_posix().replace(chr(39), chr(39)*2)}' --reviews <填写后的正文CSV> --output <新的JSONL路径>
```

导入只按 sample_id 更新风格标签，忽略 CSV 中被编辑的正文；不会改账号、来源或原始内容，也不会覆盖已有冲突结论。

## 文件说明

`corpus.jsonl` 保留全部规范化记录及原文、原始 JSONL 行号和响应路径。`candidates.jsonl` 是复核子集；模型只对旧 development 数据评分。CSV 为 Excel 显示对公式起始字符和长 UID 加前置单引号；JSONL 保存原始精确值，复核导入依赖 sample_id。

新采集是六个 AI 话题视频的有界定向样本，实际部分视频没有可返回评论；不是全站随机样本，不代表线上 AI 比例。原始采集及停止原因见上级 collection/export/summary.json。没有使用合成评论或外部 AI 标注服务。
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def make_review_page(output, candidates):
    fields = ("sample_id", "text", "pool_role", "url", "username", "uid", "manual_account_label",
              "legacy_comment_judgment", "authorship_label", "body_style_label")
    data = [{key: row[key] for key in fields} for row in candidates]
    version = hashlib.sha256("".join(sorted(row["sample_id"] for row in data)).encode()).hexdigest()[:16]
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    html = template.replace("__DATA__", payload).replace("__VERSION__", version)
    (output / "review.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("build")
    create.add_argument("--output", type=Path, default=ROOT / "out/text-specialist-20261113/bundle")
    load = sub.add_parser("import-body")
    load.add_argument("--bundle", type=Path, required=True)
    load.add_argument("--reviews", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.output.exists():
            raise FileExistsError("拒绝覆盖既有专项数据目录")
        build(args.output)
    else:
        if args.output.exists():
            raise FileExistsError("拒绝覆盖既有复核结果")
        records = [json.loads(line) for line in (args.bundle / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        annotations = read_csv(args.reviews, ("sample_id", "body_style_label"))
        result = apply_body_reviews(records, annotations)
        save_jsonl(args.output, result)
        print(json.dumps({"rows": len(result), "reviewed": sum(r["body_style_label"] != "unreviewed" for r in result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
