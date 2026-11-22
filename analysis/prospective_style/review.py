"""Build two blind packets. Read returned xlsx files, adjudicate fields, and report results."""

import argparse
import csv
import hashlib
import io
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from collect import MENTIONS, P, freeze, read

DIMS = ("style", "intent", "promotion")
HEADERS = ("编号", "评论正文", "文本风格", "对话意图", "推广表达")
METHODS = ("plain", "guarded", "learned", "v8", "v9", "v8.1", "v9.1", "trained_mask_raw", "trained_mask")


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def packet(folder):
    raw, scores = read(folder / "frame.json"), read(folder / "scores.json")
    if [r["rpid"] for r in raw] != [s["id"] for s in scores]:
        raise ValueError("Predictions do not align with frozen frame")
    kept = [r for r, s in zip(raw, scores) if r["arm"] == "natural" or s["eligible"] and s["plain"]]
    for who in "AB":
        sequence = sorted(kept, key=lambda r: digest(P["seed"] + who + r["rpid"]))
        key = {f"{who}{i:04}": r["rpid"] for i, r in enumerate(sequence, 1)}
        freeze(folder / f"key-{who}.json", key)
        freeze(
            folder / f"packet-{who}.json",
            [[code, MENTIONS.sub("@某人", r["text"])] for code, r in zip(key, sequence, strict=True)],
        )
    return kept


def read_sheet(path, folder, who):
    if path.suffix.lower() == ".json":
        from web_packet import read_web

        return read_web(path, folder, who)
    from openpyxl import load_workbook

    key = read(folder / f"key-{who}.json")
    display = dict(read(folder / f"packet-{who}.json"))
    book = load_workbook(path, read_only=True, data_only=False)
    try:
        rows = iter(book["盲评"].values)
        for row in rows:
            if tuple(row[:5]) == HEADERS:
                break
        else:
            raise ValueError("Missing blind review header")
        labels = {}
        for row in rows:
            if not any(v is not None for v in row[:5]):
                continue
            code, body, *values = row[:5]
            if code not in key or body != display[code] or key[code] in labels:
                raise ValueError(f"Unexpected, duplicate or edited {who} review row {code}")
            for dim, value in zip(DIMS, values, strict=True):
                if value not in P[dim]:
                    raise ValueError(f"Missing/invalid {dim} in {code}")
            labels[key[code]] = dict(zip(DIMS, values, strict=True))
    finally:
        book.close()
    if set(labels) != set(key.values()):
        raise ValueError(f"Incomplete {who} packet")
    return labels


def agreement(a, b, dim):
    labels = P[dim]
    confusion = [[sum(a[i][dim] == x and b[i][dim] == y for i in a) for y in labels] for x in labels]
    n = len(a)
    same = sum(confusion[i][i] for i in range(len(labels)))
    expected = sum(sum(confusion[i]) * sum(row[i] for row in confusion) for i in range(len(labels))) / n**2
    return {
        "n": n,
        "labels": labels,
        "confusion": confusion,
        "agreement": same / n,
        "kappa": (same / n - expected) / (1 - expected) if expected < 1 else None,
        "uncertain_A": sum(confusion[-1]) / n,
        "uncertain_B": sum(r[-1] for r in confusion) / n,
    }


def ratio(a, b):
    return a / b if b else None


def quantile(sorted_values, p):
    idx = (len(sorted_values) - 1) * p
    lo, hi = math.floor(idx), math.ceil(idx)
    return sorted_values[lo] + (idx - lo) * (sorted_values[hi] - sorted_values[lo])


def paired_ci(rows, videos):
    # Keep all selected natural videos, including zero-member ordinary-short clusters.
    strata = defaultdict(list)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["bvid"]].append(r)
    for v in videos:
        rr = buckets[v["bvid"]]
        h = [r for r in rr if r["style"] == P["style"][0]]
        a = [r for r in rr if r["style"] == P["style"][1]]
        strata[v["pool"]].append(
            (
                len(h),
                sum(int(r["guarded"]) - int(r["learned"]) for r in h),
                len(a),
                sum(int(r["learned"]) - int(r["guarded"]) for r in a),
            )
        )
    rng = random.Random(20261122)
    dist = [[], []]
    for _ in range(5000):
        draw = [c for clusters in strata.values() for c in rng.choices(clusters, k=len(clusters))]
        nh, dh, na, da = map(sum, zip(*draw))
        if nh:
            dist[0].append(1000 * dh / nh)
        if na:
            dist[1].append(da / na)

    def interval(values):
        values.sort()
        return {
            "ci95": [quantile(values, 0.025), quantile(values, 0.975)] if values else None,
            "valid": len(values),
            "undefined": 5000 - len(values),
        }

    return [interval(x) for x in dist]


def summarize(rows, methods):
    h = [r for r in rows if r["style"] == P["style"][0]]
    a = [r for r in rows if r["style"] == P["style"][1]]

    def metrics(method):
        fp, tp = sum(r[method] for r in h), sum(r[method] for r in a)
        result = {
            "fp": fp,
            "tp": tp,
            "human_blocks_per_1000": ratio(1000 * fp, len(h)),
            "recall": ratio(tp, len(a)),
            "precision": ratio(tp, tp + fp),
            "f1": ratio(2 * tp, len(a) + tp + fp),
        }
        key = {"plain": "masked_score", "trained_mask_raw": "trained_mask_score"}.get(
            method, method + "_score"
        )
        if method != "trained_mask" and rows and key in rows[0]:
            groups = defaultdict(lambda: [0, 0])
            for label, subset in enumerate((h, a)):
                for r in subset:
                    value = r[key] if r[key] is not None else -1
                    groups[value][label] += 1
            seen_pos = seen_total = 0
            ap = 0.0
            for _, (neg, pos) in sorted(groups.items(), reverse=True):
                seen_pos += pos
                seen_total += neg + pos
                ap += pos * seen_pos / seen_total
            result["average_precision"] = ratio(ap, len(a))
        return result

    return {
        "rows": len(rows),
        "human_style": len(h),
        "ai_style": len(a),
        "uncertain": len(rows) - len(h) - len(a),
        "models": {m: metrics(m) for m in methods},
        "fewer_human_blocks_per_1000": ratio(
            1000 * sum(int(r["guarded"]) - int(r["learned"]) for r in h), len(h)
        ),
        "recall_delta": ratio(sum(int(r["learned"]) - int(r["guarded"]) for r in a), len(a)),
    }


def evaluate(folder, path_a, path_b, decisions):
    a, b = read_sheet(path_a, folder, "A"), read_sheet(path_b, folder, "B")
    if a.keys() != b.keys():
        raise ValueError("Different annotation populations")
    frame = {r["rpid"]: r for r in read(folder / "frame.json")}
    scores = {r["id"]: r for r in read(folder / "scores.json")}
    agree = {dim: agreement(a, b, dim) for dim in DIMS}
    for arm in ("natural", "diagnostic"):
        subset = {k: v for k, v in a.items() if frame[k]["arm"] == arm}
        agree[arm] = {d: agreement(subset, b, d) for d in DIMS} if subset else {}
    # No predictions in the human adjudication file.
    conflicts = {}
    consensus = {k: {} for k in a}
    for k in sorted(a):
        for d in DIMS:
            if a[k][d] == b[k][d]:
                consensus[k][d] = a[k][d]
            else:
                code = digest(f"adjudicate-{k}-{d}")[:16]
                conflicts[code] = (k, d)
    adjudication = folder / "adjudication.csv"
    if conflicts and not decisions:
        f = io.StringIO(newline="")
        w = csv.writer(f)
        w.writerow(("code", "text", "field", "A", "B", "decision"))
        for code, (k, d) in conflicts.items():
            body = MENTIONS.sub("@某人", frame[k]["text"])
            if body.lstrip().startswith(("=", "+", "-", "@")):
                body = "'" + body
            w.writerow((code, body, d, a[k][d], b[k][d], ""))
        template = f.getvalue().encode("utf-8-sig")
        if adjudication.exists() and adjudication.read_bytes() != template:
            raise ValueError("Existing adjudication differs; pass completed copy with --adjudicated")
        if not adjudication.exists():
            adjudication.write_bytes(template)
        freeze(folder / "agreement.json", agree)
        return "Awaiting human adjudication"
    if decisions:
        with decisions.open(encoding="utf-8-sig", newline="") as f:
            resolved = list(csv.DictReader(f))
        if len(resolved) != len(conflicts) or {r["code"] for r in resolved} != set(conflicts):
            raise ValueError("Adjudication coverage differs")
        for r in resolved:
            k, d = conflicts[r["code"]]
            if r["decision"] not in P[d] or (r["field"], r["A"], r["B"]) != (d, a[k][d], b[k][d]):
                raise ValueError("Invalid adjudication")
            consensus[k][d] = r["decision"]
    methods = list(METHODS)
    extra = {}
    probabilities = {}
    for name in ("qwen", "roberta"):
        path = folder / (name + ".json")
        if path.exists():
            predictions = read(path)
            if [r["id"] for r in predictions] != list(frame):
                raise ValueError(f"{name} coverage differs")
            methods.append(name)
            extra[name] = {r["id"]: r["blocked"] for r in predictions}
            if name == "roberta":
                probabilities["roberta_score"] = {r["id"]: r["score"] for r in predictions}
    full = [
        dict(
            frame[k], **scores[k], **consensus[k], **{m: vv[k] for m, vv in (extra | probabilities).items()}
        )
        for k in a
    ]
    natural = [r for r in full if r["arm"] == "natural"]
    diagnostic = [r for r in full if r["arm"] == "diagnostic"]
    videos = [v for v in read(folder / "selection.json") if v["arm"] == "natural"]
    result = {}
    for name, rows in [
        ("natural", natural),
        ("ordinary_short", [r for r in natural if r["ordinary_short"]]),
    ]:
        result[name] = summarize(rows, methods)
        result[name]["paired_ci"] = paired_ci(rows, videos)
    result["diagnostic_counts"] = {
        "rows": len(diagnostic),
        "by_intent_style_decisions": dict(
            Counter(
                "|".join([r["intent"], r["style"], str(r["guarded"]), str(r["learned"])])
                for r in diagnostic
            )
        ),
    }
    for arm, rows in (("natural", natural), ("diagnostic", diagnostic)):
        eligible = [r for r in rows if r["eligible"]]
        by_policy = {}
        for policy in ("rule", "request"):
            requests = [r for r in eligible if r["intent"] == P["intent"][0]]
            statements = [r for r in eligible if r["intent"] == P["intent"][1]]
            by_policy[policy] = {
                "requests": len(requests),
                "recognized_requests": sum(r[policy] for r in requests),
                "recognized_requests_previously_blocked": sum(r[policy] and r["plain"] for r in requests),
                "nonrequests": len(statements),
                "wrongly_recognized_nonrequests": sum(r[policy] for r in statements),
            }
        result[arm + "_intent"] = {
            "eligible": len(eligible),
            "uncertain": sum(r["intent"] == P["intent"][2] for r in eligible),
            "policies": by_policy,
        }
    freeze(folder / "agreement.json", agree)
    freeze(folder / "result.json", result)
    return "Finalized after complete labels"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["packet", "evaluate"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--a", type=Path)
    ap.add_argument("--b", type=Path)
    ap.add_argument("--adjudicated", type=Path)
    args = ap.parse_args()
    if args.action == "packet":
        print("Rows per reviewer:", len(packet(args.folder)))
    else:
        if not args.a or not args.b:
            ap.error("evaluate requires --a and --b")
        print(evaluate(args.folder, args.a, args.b, args.adjudicated))
