"""CSV validation, provenance-preserving archive import and UID splits."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .common import read_csv, write_csv, write_json, sha256
from .preprocess import normalize, Preprocessor, MODES

FIELDS = ("uid", "username", "text", "label")
SPLITS = ("train", "validation", "test")


def validate_uid(uid):
    if not isinstance(uid, str) or not re.fullmatch(r"[0-9]{1,32}", uid.strip()):
        raise ValueError(f"无效 UID（必须是十进制字符串，禁止浮点/科学计数）: {uid!r}")
    return uid.strip().lstrip("0") or "0"


def clean_rows(rows):
    groups = defaultdict(list)
    audit = {"input_rows": len(rows), "empty_rows_removed": 0, "duplicate_rows_removed": 0, "conflicting_rows_removed": 0}
    for i, row in enumerate(rows, 2):
        if not set(FIELDS).issubset(row):
            raise ValueError(f"第 {i} 行缺少 {FIELDS}")
        uid = validate_uid(row["uid"])
        label = str(row["label"]).strip()
        if label not in ("0", "1"):
            raise ValueError(f"第 {i} 行 label 标签必须为 0 或 1")
        text, nick = normalize(row["text"]), normalize(row["username"])
        if not text and not nick:
            audit["empty_rows_removed"] += 1
            continue
        # Text-only ablation must not see the same normalized body in two splits.
        key = ("text", text) if text else ("nickname", nick)
        groups[key].append({"uid": uid, "username": nick, "text": text, "label": int(label)})
    cleaned = []
    for group in groups.values():
        if len({r["label"] for r in group}) > 1:
            audit["conflicting_rows_removed"] += len(group)
        else:
            cleaned.append(group[0])
            audit["duplicate_rows_removed"] += len(group) - 1
    audit["clean_rows"] = len(cleaned)
    audit["unique_uids"] = len({r["uid"] for r in cleaned})
    audit["labels"] = dict(Counter(str(r["label"]) for r in cleaned))
    return cleaned, audit


def assign_splits(rows, seed=42):
    groups = defaultdict(set)
    for row in rows:
        groups[validate_uid(row["uid"])].add(int(row["label"]))
    strata = defaultdict(list)
    for uid, labels in groups.items():
        strata[tuple(sorted(labels))].append(uid)
    rng = np.random.RandomState(seed)
    mapping = {}
    for stratum in sorted(strata):
        ordered = sorted(strata[stratum])
        shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
        exact = np.array([0.8, 0.1, 0.1]) * len(shuffled)
        counts = np.floor(exact).astype(int)
        for i in sorted(range(3), key=lambda i: (-(exact[i]-counts[i]), i))[:len(shuffled)-int(counts.sum())]:
            counts[i] += 1
        if len(shuffled) >= 3:
            for i in range(3):
                if counts[i] == 0:
                    counts[int(np.argmax(counts))] -= 1
                    counts[i] += 1
        start = 0
        for split, count in zip(SPLITS, counts, strict=True):
            for uid in shuffled[start:start+count]:
                mapping[uid] = split
            start += count
    return mapping


def read_splits(path, rows):
    mapping = {}
    for row in read_csv(path, ("uid", "split")):
        uid, split = validate_uid(row["uid"]), row["split"].strip()
        if uid in mapping:
            raise ValueError(f"splits.csv 重复 duplicate UID: {uid}")
        if split not in SPLITS:
            raise ValueError(f"无效 split {split!r}；请使用 train/validation/test")
        mapping[uid] = split
    missing = {r["uid"] for r in rows} - mapping.keys()
    if missing:
        raise ValueError(f"splits.csv 缺失 missing {len(missing)} 个 UID，例如 {sorted(missing)[:3]}")
    return mapping


def read_row_splits(path, rows):
    """Row-level split: uid,text,split. The same UID may appear in several splits."""
    mapping = {}
    for row in read_csv(path, ("uid", "text", "split")):
        key = (validate_uid(row["uid"]), normalize(row["text"]))
        split = row["split"].strip()
        if key in mapping:
            raise ValueError(f"row-splits.csv 重复行: {key[0]}")
        if split not in SPLITS:
            raise ValueError(f"无效 split {split!r}；请使用 train/validation/test")
        mapping[key] = split
    expected = [(r["uid"], normalize(r["text"])) for r in rows]
    if len(expected) != len(set(expected)):
        raise ValueError("清洗后仍存在相同的 uid+text，无法用行级划分")
    missing = set(expected) - mapping.keys()
    extra = mapping.keys() - set(expected)
    if missing or extra:
        raise ValueError(f"row-splits.csv 与清洗后数据不一致：缺失 {len(missing)}，多余 {len(extra)}")
    return mapping


def import_archives(paths, data_dir, unreviewed_ai=False):
    data_dir = Path(data_dir)
    if (data_dir / "comments.csv").exists():
        raise ValueError("comments.csv 已存在，请指定新的 data-dir，避免覆盖人工标签")
    selected, review, provenance, archives = [], [], [], []
    for path in paths:
        path = Path(path)
        digest = sha256(path)
        archives.append({"name": path.name, "sha256": digest})
        with ZipFile(path) as z:
            members = [n for n in z.namelist() if n.endswith("/all_scored.jsonl") or n == "all_scored.jsonl"]
            members += [n for n in z.namelist() if n == "comments.jsonl" or n.endswith("/comments.jsonl")]
            if not members:
                raise ValueError(f"{path.name}: 没有支持的原始 JSONL；不会把无 UID 的 fastText 文件当作数据源")
            for member in members:
                if z.getinfo(member).file_size > 256 * 1024 * 1024:
                    raise ValueError("JSONL 文件过大")
                for line in z.read(member).decode("utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    raw = json.loads(line)
                    uid = validate_uid(str(raw.get("uid", raw.get("mid", ""))))
                    account = raw.get("account") or {}
                    row = {"uid": uid, "username": raw.get("username", raw.get("name", account.get("uname", ""))), "text": raw.get("text", "")}
                    label = raw.get("label")
                    source = raw.get("label_source", raw.get("source", "unspecified"))
                    if label == "unreviewed" and unreviewed_ai:
                        label, source = "ai", "user_confirmed_unreviewed_ai"
                    if label in ("ai", "1", 1, "human", "0", 0):
                        row["label"] = 1 if label in ("ai", "1", 1) else 0
                        selected.append(row)
                    else:
                        row["label"] = ""
                        review.append({**row, "original_label": raw.get("label", ""), "label_source": source})
                    provenance.append({**row, "label_source": source, "source_archive": path.name, "archive_sha256": digest, "member": member, "rpid": str(raw.get("rpid", ""))})
    write_csv(data_dir / "comments.csv", selected, FIELDS)
    write_csv(data_dir / "review.csv", review, (*FIELDS, "original_label", "label_source"))
    with (data_dir / "provenance.jsonl").open("w", encoding="utf-8") as f:
        for row in provenance:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"archives": archives, "selected_rows": len(selected), "selected_labels": dict(Counter(str(r["label"]) for r in selected)), "excluded_unreviewed": len(review), "unreviewed_ai": unreviewed_ai, "label_sources": dict(Counter(r["label_source"] for r in provenance if r["label"] != "")), "warning": "规则弱标签与用户确认标签混合；指标不是 AI 真实性的准确率。review 不会自动变成正类。"}
    write_json(data_dir / "import_report.json", report)
    return report


def prepare(data_dir, splits_path=None, row_splits_path=None):
    data_dir = Path(data_dir)
    source = data_dir / "comments.csv"
    rows, audit = clean_rows(read_csv(source, FIELDS))
    if not rows:
        raise ValueError("没有可用训练数据")
    if splits_path and row_splits_path:
        raise ValueError("splits 与 row-splits 只能提供一个")
    pre = Preprocessor()
    processed = data_dir / "processed"
    if row_splits_path:
        row_path = Path(row_splits_path)
        if not row_path.exists():
            raise ValueError(f"指定行级划分文件不存在: {row_path}")
        row_mapping = read_row_splits(row_path, rows)
        split_of = lambda row: row_mapping[(row["uid"], normalize(row["text"]))]
        canonical_splits = data_dir / "row_splits.csv"
        if row_path.resolve() != canonical_splits.resolve():
            write_csv(canonical_splits, read_csv(row_path, ("uid", "text", "split")), ("uid", "text", "split"))
        split_source = "row-level provided"
        split_rule = "random comment rows; label-stratified 70/15/15; same UID may cross splits"
        splits_sha = sha256(canonical_splits)
    else:
        path = Path(splits_path) if splits_path else data_dir / "splits.csv"
        existing = path.exists()
        if splits_path and not existing:
            raise ValueError(f"指定划分文件不存在: {path}")
        mapping = read_splits(path, rows) if existing else assign_splits(rows)
        if not existing:
            write_csv(path, ({"uid": u, "split": s} for u, s in sorted(mapping.items())), ("uid", "split"))
        canonical_splits = data_dir / "splits.csv"
        if path.resolve() != canonical_splits.resolve():
            write_csv(canonical_splits, ({"uid": u, "split": s} for u, s in sorted(mapping.items())), ("uid", "split"))
        split_of = lambda row: mapping[row["uid"]]
        split_source = "provided" if existing else "generated"
        split_rule = "sorted_UIDs; label-set strata; RandomState42; largest remainder; min1 if stratum>=3 (plan one compatible)"
        splits_sha = sha256(canonical_splits)
    counts = {}
    for split in SPLITS:
        subset = [r for r in rows if split_of(r) == split]
        if not subset or {r["label"] for r in subset} != {0, 1}:
            raise ValueError(f"{split} 必须同时包含 0/1 两类。请补充数据或显式提供划分。")
        counts[split] = {"rows": len(subset), "uids": len({r["uid"] for r in subset}), "labels": dict(Counter(str(r["label"]) for r in subset))}
        write_csv(processed / f"{split}.csv", subset, FIELDS)
        for mode in MODES:
            file = processed / mode / f"{split}.txt"
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("".join(f"__label__{r['label']} {pre.encode(r['text'], r['username'], mode)}\n" for r in subset), encoding="utf-8")
    manifest = {"schema": 1, "seed": 42, "split_rule": split_rule, "split_source": split_source, "comments_sha256": sha256(source), "splits_sha256": splits_sha, "preprocess_sha256": sha256(pre.assets / "preprocess.json"), "dictionary_sha256": pre.config["dictionary_sha256"], "cleaning": audit, "splits": counts, "files": {str(p.relative_to(processed)).replace('\\', '/'): sha256(p) for p in processed.rglob('*') if p.is_file() and p.name != "manifest.json"}}
    write_json(processed / "manifest.json", manifest)
    return manifest
