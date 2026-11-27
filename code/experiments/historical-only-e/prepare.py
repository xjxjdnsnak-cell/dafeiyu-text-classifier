"""Prepare isolated E training data and freeze both validation working points before scoring."""

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CODE = REPO / "code"
sys.path[:0] = [
    str(CODE / "src"),
    str(CODE / "experiments/text-training-20261114"),
    str(CODE / "experiments/text-specialist-20261113"),
]
from bili_ai.common import sha256
from bili_ai.metrics import select_threshold
from bili_ai.model import DEFAULT_PARAMS, load_model, positive_scores, train_model
from bili_ai.preprocess import ASSETS, Preprocessor, normalize
from text_policy import is_mention_only
from train_text import near_index, near_match, template_key, weighted_training_lines


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def lines(path):
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            yield json.loads(line)


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf8")


def table(path, data, keys=None):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys or list(data[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def identity(r):
    return str(r["uid"]), normalize(r["text"])


def length_group(text):
    return "1–20" if len(text) <= 20 else "21–60" if len(text) <= 60 else "61+"


def tagged(row, uid, video, reply):
    return dict(
        row,
        ids=set(map(str, uid)),
        vids=set(video),
        replies=set(map(str, reply)),
        body=normalize(row["text"])[:4096],
        tpl=template_key(row["text"]),
    )


class Index:
    def __init__(self, records):
        self.uids, self.videos, self.replies = set(), set(), set()
        self.bodies, self.templates = set(), set()
        for r in records:
            self.uids.update(r["ids"])
            self.videos.update(r["vids"])
            self.replies.update(r["replies"])
            self.bodies.add(r["body"])
            self.templates.add(r["tpl"])
        self.near = near_index(self.templates)

    def hits(self, r):
        return [
            name
            for name, hit in (
                ("uid", self.uids & r["ids"]),
                ("video", self.videos & r["vids"]),
                ("rpid", self.replies & r["replies"]),
                ("body", r["body"] in self.bodies),
                ("template", r["tpl"] in self.templates),
                ("near_template", r["tpl"] not in self.templates and near_match(r["tpl"], self.near)),
            )
            if hit
        ]


def raw_metadata(w, corpus):
    manifest = read(corpus / "manifest.json")["sources"]
    paths = [w / p for p in manifest["raw_exports"]] + [corpus / "collection/export/all_scored.jsonl"]
    aid, evidence = {}, defaultdict(dict)
    for path in paths:
        source = "new_collection" if path == paths[-1] else "existing_corpus"
        for r in lines(path):
            key = (
                source,
                str(r["mid"]),
                str(r["rpid"]),
                r["bvid"],
                str(r["ctime"]),
                normalize(r.get("training_text") or r["text"]),
            )
            evidence[key].setdefault("files", []).append(path.relative_to(w).as_posix())
            assert aid.setdefault(r["bvid"], str(r["aid"])) == str(r["aid"])
    return aid, evidence, paths


def domain_map(out, corpus):
    domains = {}
    for name, filename, domain in [
        ("collector-expand-20261112", "videos-19.txt", "普通区"),
        ("collector-newtopics-20261112", "videos-13.txt", "普通区"),
        ("collector-expand-20261112", "videos-ai11-deepen.txt", "AI/插件"),
    ]:
        for bv in (out / name / filename).read_text(encoding="utf-8-sig").split():
            domains[bv] = domain
    for r in read(out / "collector-expand-topics-20261114/videos-discovered.json"):
        domains[r["bvid"]] = "AI/插件" if r["ai_topic"] else "普通区"
    for r in read(out / "text-specialist-20261113/seed-manifest.json")["videos"]:
        domains[r["bvid"]] = "AI/插件"
    domains["BV1i24P6rEmt"] = "AI/插件"
    for r in read(corpus / "video-discovery.json")["picks"]:
        if r["topic"] == "ai_domain":
            domains[r["bvid"]] = "AI/插件"
    return domains


def train(w, dest):
    out = w / "out"
    cls = w / "fasttext-classifier"
    data = cls / "data-text-specialist-20261114-v8"
    proc = data / "processed"
    corpus = out / "pre-fish-human-corpus-20261114"
    retro = out / "confirmation-retrospective-20261123"
    primary_path = retro / "game-supplement/合并评论.json"
    hist_path = corpus / "historical-human-candidates.csv"
    base_path = REPO / "model/text/model.bin"
    meta = read(base_path.with_suffix(".bin.json"))
    assert meta["params"] == DEFAULT_PARAMS and sha256(base_path) == meta["model_sha256"]
    assert all(version(k) == v for k, v in meta["versions"].items())
    assert sha256(ASSETS / "preprocess.json") == meta["preprocess_sha256"]
    assert sha256(ASSETS / "jieba.dict.txt") == meta["dictionary_sha256"]
    assert sha256(primary_path) == read(retro / "analysis/private/预测封存.json")["member_sha256"]
    for filename, digest in meta["dataset"]["files"].items():
        assert sha256(proc / filename) == digest
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.iterdir()):
        raise ValueError("Use an empty private output directory.")
    aid, evidence, raw_paths = raw_metadata(w, corpus)
    domains = domain_map(out, corpus)
    aid_domains = {num: domains[bv] for bv, num in aid.items() if bv in domains}

    def group(r):
        found = {
            aid_domains.get(v.removeprefix("video:"), "未知") for v in r["vids"] if v.startswith("video:")
        }
        return next(iter(found)) if len(found) == 1 else "混合/未知"

    primary = read(primary_path)
    for vid in {r["bvid"] for r in primary}:
        key = hashlib.sha256(vid.encode()).hexdigest() + ".json"
        archives = [base / "private/videos" / key for base in (retro, retro / "game-supplement")]
        capture = next(read(file) for file in archives if file.exists() and read(file)["pages"])
        ids = {
            str(p["params"]["oid"])
            for p in capture["pages"]
            if "oid" in p["params"] and p["kind"] == "reply"
        }
        assert len(ids) == 1 and capture["bvid"] == vid
        aid[vid] = ids.pop()
    decisions_path = cls / "data-rule-ge30-20261114-r2/label-decisions.jsonl"
    replies = {
        r["sample_id"]: {str(s["rpid"]) for s in r["sources"] if s.get("rpid")}
        for r in lines(decisions_path)
    }
    lineage = {}
    all_by_identity = {}
    for r in lines(data / "sample-lineage.jsonl"):
        ids = set(r["associated_uids"]) | {r["uid"]}
        rp = {str(s["rpid"]) for s in r["recovered_sources"] if s.get("rpid")}
        for sid in r["source_metadata_ids"]:
            rp.update(replies.get(sid, ()))
        record = tagged(r, ids, r["videos"] + r["bvids"], rp)
        record["domain"] = group(record)
        all_by_identity[identity(r)] = record
        if r["split"] != "quarantine":
            lineage[identity(r)] = record
    splits = {}
    for split in ("train", "validation", "test"):
        splits[split] = [lineage[identity(r)] for r in rows(proc / (split + ".csv"))]
        assert all(r["split"] == split for r in splits[split])
    old = splits["train"]
    negative = [r for r in old if r["label"] == 0]
    positive = [r for r in old if r["label"] == 1]
    assert (len(negative), len(positive)) == (8530, 4431)
    assert Counter(r["label"] for r in splits["test"]) == {0: 1684, 1: 697}
    hint = re.compile(
        read(cls / "reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"],
        re.IGNORECASE,
    )
    for i, r in enumerate(sorted(primary, key=lambda r: r["rpid"]), 1):
        r.update(code=f"H{i:04}", sample_id=f"H{i:04}", ordinary=not bool(hint.search(r["nickname"] or "")))
    paper = [tagged(r, [r["uid"]], [r["bvid"], "video:" + aid[r["bvid"]]], [r["rpid"]]) for r in primary]
    indexes = {k: Index(v) for k, v in splits.items()}
    isolation = {}
    for name, right, left in (
        ("train_validation", "validation", "train"),
        ("train_test", "test", "train"),
        ("validation_test", "test", "validation"),
    ):
        counts = Counter(h for r in splits[right] for h in indexes[left].hits(r))
        isolation[name] = dict(counts)
        assert not counts, (name, counts)
    for r in paper:
        r["overlap"] = "|".join(k + "/" + h for k, idx in indexes.items() for h in idx.hits(r))
    isolation["paper_vs_legacy"] = dict(Counter(h for r in paper for h in r["overlap"].split("|") if h))
    # Protect the entire paper frame including its unlabeled and any overlapping members.
    protect = Index(splits["validation"] + splits["test"] + paper)
    old_bodies = {r["body"] for r in old}
    old_replies = {i for r in old for i in r["replies"]}
    old_by_identity = {identity(r) for r in old}
    benchmark = rows(out / "prospective-benchmark-20261114/benchmark/comments.csv")
    bench_uid, bench_bv, bench_body = (
        {r["uid"] for r in benchmark},
        {r["bvid"] for r in benchmark},
        {normalize(r["text"])[:4096] for r in benchmark},
    )
    pool = rows(hist_path)
    assert len(pool) == len({r["candidate_id"] for r in pool}) == 17477
    audit, eligible = [], []
    rng = random.Random(42)
    rng.shuffle(pool)
    seen_body, seen_reply = set(), set()
    for r in pool:
        key = (r["source"], r["uid"], r["rpid"], r["bvid"], r["ctime_unix"], normalize(r["text"]))
        assert key in evidence and 0 < int(r["ctime_unix"]) < 1790784000
        assert datetime.fromisoformat(r["ctime_beijing"]).timestamp() == int(r["ctime_unix"])
        assert (
            r["label_source"] == "historical_timestamp"
            and r["proposed_label"] == "0"
            and r["authorship_verified"] == "False"
        )
        ancestor = all_by_identity.get(identity(r), {})
        rec = tagged(
            r,
            {r["uid"]} | ancestor.get("ids", set()),
            {r["bvid"], "video:" + aid[r["bvid"]]} | ancestor.get("vids", set()),
            {r["rpid"]} | ancestor.get("replies", set()),
        )
        rec.update(label=0, ordinary=False, sample_id=r["candidate_id"], domain=group(rec))
        why = []
        if identity(r) in old_by_identity or rec["replies"] & old_replies or rec["body"] in old_bodies:
            why.append("original_training_row_or_body")
        if ancestor.get("label") == 1 or (r["rule_score"] and int(r["rule_score"]) >= 30):
            why.append("positive_conflict")
        why.extend(protect.hits(rec))
        if rec["ids"] & bench_uid or rec["vids"] & bench_bv or rec["body"] in bench_body:
            why.append("legacy_benchmark")
        if r["training_pool_eligible"] != "True":
            why.append("corpus_ineligible")
        if is_mention_only(r["text"]) or not rec["body"]:
            why.append("empty_or_mention_only")
        if not why and (rec["body"] in seen_body or r["rpid"] in seen_reply):
            why.append("duplicate_eligible_body_or_rpid")
        if not why:
            seen_body.add(rec["body"])
            seen_reply.add(r["rpid"])
            eligible.append(rec)
        audit.append(
            {
                "candidate_id": r["candidate_id"],
                "source": r["source"],
                "rpid": r["rpid"],
                "uid": r["uid"],
                "bvid": r["bvid"],
                "ctime": r["ctime_beijing"],
                "raw_export": evidence[key]["files"][0],
                "domain": rec["domain"],
                "length": length_group(r["text"]),
                "status": "|".join(why) or "eligible",
            }
        )
    bucket = lambda r: (r["domain"], length_group(r["text"]))
    targets = Counter(map(bucket, negative))
    groups = defaultdict(list)
    for r in eligible:
        groups[bucket(r)].append(r)
    selected = []
    remainder = []
    for cat, group_rows in sorted(groups.items()):
        n = min(targets[cat], len(group_rows))
        selected.extend(group_rows[:n])
        remainder.extend(group_rows[n:])
    rng.shuffle(remainder)
    selected += remainder[: len(negative) - len(selected)]
    assert len(selected) == min(len(negative), len(eligible))
    selected_ids = {r["sample_id"] for r in selected}
    for r in audit:
        r["selected"] = int(r["candidate_id"] in selected_ids)
    table(dest / "candidate-audit.csv", audit)
    distribution = []
    for cat in sorted(set(targets) | set(groups)):
        distribution.append(
            {
                "domain": cat[0],
                "length": cat[1],
                "original": targets[cat],
                "eligible": len(groups[cat]),
                "selected": sum(bucket(r) == cat for r in selected),
            }
        )
    table(dest / "matching.csv", distribution)
    chosen = iter(sorted(selected, key=lambda r: r["candidate_id"]))
    # Fill original negative positions and retain relative order of positives. If short,
    # spread empty positions across the original negative slots with the frozen seed.
    slots = set(random.Random(42).sample(range(len(negative)), len(selected)))
    training, j = [], 0
    for r in old:
        if r["label"] == 1:
            training.append(r)
        else:
            if j in slots:
                training.append(next(chosen))
            j += 1
    for r in selected:
        assert not protect.hits(r)
    assert {r["sample_id"] for r in training if r["label"]} == {r["sample_id"] for r in positive}
    pre = Preprocessor()
    e_lines = [f"__label__{r['label']} " + pre.encode(r["text"], "", "text") for r in training]
    e_lines, ids = weighted_training_lines(training, e_lines, 1)
    trainfile = dest / "E-train.txt"
    trainfile.write_text("\n".join(e_lines) + "\n", encoding="utf8")
    a_members = read(data / "training-members-weight1.json")
    assert (
        weighted_training_lines(old, (proc / "text/train.txt").read_text(encoding="utf8").splitlines(), 1)[
            1
        ]
        == a_members
    )
    manifest = []

    def member(r, role, order=""):
        return {
            "split": role,
            "order": order,
            "sample_id": r["sample_id"],
            "uid": r["uid"],
            "associated_uids": "|".join(sorted(r["ids"])),
            "videos": "|".join(sorted(r["vids"])),
            "rpids": "|".join(sorted(r["replies"])),
            "text": r["text"],
            "label": r.get("label", ""),
            "ordinary": int(r["ordinary"]),
            "domain": r.get("domain", r.get("stratum", "")),
            "length": length_group(r["text"]),
            "source": r.get("source", ""),
            "overlap": r.get("overlap", ""),
        }

    for group_name, records, order in (("A_train", old, a_members), ("E_train", training, ids)):
        lookup = {r["sample_id"]: r for r in records}
        manifest.extend(member(lookup[sid], group_name, i) for i, sid in enumerate(order, 1))
    for group_name, records in (
        ("validation", splits["validation"]),
        ("auxiliary", splits["test"]),
        ("paper", paper),
    ):
        manifest.extend(member(r, group_name) for r in records)
    table(dest / "split-members.csv", manifest)
    sources = [
        primary_path,
        hist_path,
        data / "sample-lineage.jsonl",
        decisions_path,
        *raw_paths,
        out / "cross-model-20261123/盲评裁决与抽样权重.csv",
        out / "cross-model-20261123/survey_calibration.py",
    ]
    freeze = {
        "params": DEFAULT_PARAMS,
        "versions": meta["versions"],
        "model_A": meta["model_sha256"],
        "preprocessing": {k: meta[k] for k in ("dictionary_sha256", "preprocess_sha256")},
        "original_training_input": meta["training_input_sha256"],
        "E_training_input": sha256(trainfile),
        "source_sha256": {p.relative_to(w).as_posix(): sha256(p) for p in sources},
        "v8_split_files": meta["dataset"]["files"],
        "outputs": {
            p.name: sha256(p)
            for p in (dest / "split-members.csv", dest / "matching.csv", dest / "candidate-audit.csv")
        },
        "isolation": isolation,
        "candidates": len(pool),
        "sources": dict(Counter(r["source"] for r in pool)),
        "candidate_time_range": [
            min(r["ctime_beijing"] for r in pool),
            max(r["ctime_beijing"] for r in pool),
        ],
        "duplicates": {
            k: len(pool) - len({f(r) for r in pool})
            for k, f in (
                ("rpid", lambda r: r["rpid"]),
                ("uid_body", identity),
                ("visible_body", lambda r: normalize(r["text"])[:4096]),
            )
        },
        "eligible": len(eligible),
        "used": len(selected),
        "positive": len(positive),
        "selected_sources": dict(Counter(r["source"] for r in selected)),
        "cutoff_exclusive": "2026-10-01T00:00:00+08:00",
        "sample_seed": 42,
        "design": "Use A/E joint prediction cells nested within original sampling strata; keep unlabeled unknown. "
        "Primary target excludes preflagged conflicts; full-frame descriptive sensitivity if needed.",
        "model_input": "Original body; frozen jieba HMM=False; empty/mention-only pass, AP score -1.",
        "validation": "Original v8 validation, recall then precision/F1 then higher threshold under empirical cap.",
        "budgets": [0.01, 0.02],
    }
    save(dest / "before-training.json", freeze)
    print("Preflight:", len(eligible), len(training), isolation, flush=True)
    e_model = train_model(trainfile, dest / "E-model.bin", meta["params"])
    a_model = load_model(base_path)
    val = splits["validation"]
    val_x = [
        line.split(" ", 1)[1]
        for line in (proc / "text/validation.txt").read_text(encoding="utf8").splitlines()
    ]
    y = [r["label"] for r in val]
    selection = {}
    val_rows = [{"sample_id": r["sample_id"], "label": r["label"]} for r in val]
    for name, model in (("A", a_model), ("E", e_model)):
        scores = positive_scores(model, val_x)
        selection[name] = {str(b): select_threshold(y, scores, b) for b in freeze["budgets"]}
        for row, score in zip(val_rows, scores, strict=True):
            row[name] = score
    for b, old_point in read(REPO / "evaluation/v8/selection-frozen-before-test.json")["models"][
        "selected_original"
    ]["working_points"].items():
        if b in selection["A"]:
            assert selection["A"][b]["threshold"] == old_point["threshold"]
    table(dest / "validation-scores.csv", val_rows)
    freeze.update(
        model_E=sha256(dest / "E-model.bin"),
        working_points=selection,
        validation_scores=sha256(dest / "validation-scores.csv"),
        before_training=sha256(dest / "before-training.json"),
    )
    save(dest / "frozen.json", freeze)
    print("Thresholds frozen", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    train(args.workspace.resolve(), args.output.resolve())
