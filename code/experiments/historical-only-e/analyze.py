"""Estimate frozen operating points and paired uncertainty using the paper's sampling design."""

import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(path, data):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def div(n, d):
    n, d = np.broadcast_arrays(n, d)
    return np.divide(n, d, out=np.full(n.shape, np.nan), where=d != 0)


def confusion(tp, fp, fn, tn):
    return {
        "recall": div(tp, tp + fn),
        "fpr": div(fp, fp + tn),
        "precision": div(tp, tp + fp),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
        "f1": div(2 * tp, 2 * tp + fp + fn),
    }


def observed(records, model, budget):
    part = [r for r in records if r["label"] != ""]
    y = np.array([int(r["label"]) for r in part], dtype=int)
    w = np.array([float(r["weight"]) for r in part])
    hit = np.array([int(r[f"{model}_{budget}"]) for r in part], dtype=int)
    totals = [w[(y == a) & (hit == b)].sum() for a, b in ((1, 1), (0, 1), (1, 0), (0, 0))]
    return confusion(*totals)


def ap_and_reps(records, model, videos=None, draws=None):
    judged = [r for r in records if r["label"] != ""]
    y = np.array([int(r["label"]) for r in judged])
    if len(set(y)) < 2:
        return np.nan, None
    w = np.array([float(r["weight"]) for r in judged])
    score = np.array([float(r[model + "_score"]) for r in judged])
    ap = average_precision_score(y, score, sample_weight=w)
    if videos is None:
        return ap, None
    vid_index = {v: j for j, v in enumerate(videos)}
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]
    weights = draws[:, [vid_index[judged[i]["bvid"]] for i in order]] * w[order]
    pos = (weights * y[order]).cumsum(1)[:, ends]
    neg = (weights * (1 - y[order])).cumsum(1)[:, ends]
    precision = np.divide(pos, pos + neg, out=np.zeros_like(pos), where=(pos + neg) > 0)
    samples = div((np.diff(pos, axis=1, prepend=0) * precision).sum(1), pos[:, -1])
    samples[neg[:, -1] == 0] = np.nan
    assert np.isclose(ap, samples[0])
    return ap, samples[1:]


def percentiles(lo, hi):
    valid = np.isfinite(lo) & np.isfinite(hi)
    if not valid.any():
        return np.nan, np.nan, 0
    return np.quantile(lo[valid], 0.025), np.quantile(hi[valid], 0.975), int(valid.sum())


def accuracy_range(ai, human, empty, blocked):
    tp, fn = ai[:, blocked].sum(1), ai[:, ~blocked].sum(1)
    fp, tn = human[:, blocked].sum(1), human[:, ~blocked].sum(1)
    total = tp + fn + fp + tn + empty.sum(1)
    return div(tp + tn, total), div(tp + tn + empty.sum(1), total)


def delta_bounds(lower, upper, sign):
    # The recall (or FPR) difference uses a shared positive (or negative)
    # denominator. Collapse cells with equal decision change before taking corners.
    ls = np.stack([lower[:, sign == j].sum(1) for j in (-1, 0, 1)], axis=1)
    us = np.stack([upper[:, sign == j].sum(1) for j in (-1, 0, 1)], axis=1)
    values = []
    for bits in itertools.product((0, 1), repeat=3):
        x = np.where(bits, us, ls)
        values.append(div(x[:, 2] - x[:, 0], x.sum(1)))
    values = np.stack(values)
    valid = np.isfinite(values).any(0)
    lo = np.min(np.where(np.isfinite(values), values, np.inf), axis=0)
    hi = np.max(np.where(np.isfinite(values), values, -np.inf), axis=0)
    lo[~valid] = hi[~valid] = np.nan
    return lo, hi


def packing(head, name, design, lo, hi, fl=np.nan, fh=np.nan, vl=np.nan, vh=np.nan, nrep=0):
    return dict(
        **head,
        metric=name,
        design_point=design,
        calibrated_lo=lo,
        calibrated_hi=hi,
        fixed95_lo=fl,
        fixed95_hi=fh,
        video95_lo=vl,
        video95_hi=vh,
        valid_video_reps=nrep,
    )


def calibrated(survey, part, budget, videos, draws, header):
    records = []
    for r in part:
        records.append(
            {
                "old": bool(int(r["certainty"])),
                "pool": (r["pool"], r["sampling_v9_M"]),
                "video": r["bvid"],
                "style": r["final_style"] or None,
                "blocked": {m: int(r[m + "_" + budget]) for m in ("A", "E")},
            }
        )
    group = survey.cells(records, ("A", "E"), videos)
    est = survey.estimates(group, draws)
    keys, raw, ai, human, empty = est
    bounds = survey.cell_intervals(raw)
    result = []
    designs = {m: observed(part, m, budget) for m in ("A", "E")}
    for slot, m in enumerate(("A", "E")):
        head = dict(**header, model=m)
        b = np.array([bool(k[2 + slot]) for k in keys])
        summary = survey.model_summary(est, slot)
        for metric, original in (
            ("recall", "AI风格召回率"),
            ("fpr", "真人风格误拦率"),
            ("precision", "精确率"),
            ("f1", "F1"),
        ):
            item = summary["metrics"][original]
            video = item.get("video_interval") or (np.nan, np.nan)
            result.append(
                packing(
                    head,
                    metric,
                    designs[m][metric],
                    *item["empty_cell_range"],
                    *item["fixed_frame_interval"],
                    *video,
                    item.get("valid_reps", 0),
                )
            )
        lo, hi = accuracy_range(ai, human, empty, b)
        al, au, hl, hu = bounds.T
        low_right = al[b].sum() + hl[~b].sum()
        high_right = au[b].sum() + hu[~b].sum()
        fixed = (
            div(low_right, low_right + hu[b].sum() + au[~b].sum()),
            div(high_right, high_right + hl[b].sum() + al[~b].sum()),
        )
        result.append(
            packing(
                head, "accuracy", designs[m]["accuracy"], lo[0], hi[0], *fixed, *percentiles(lo[1:], hi[1:])
            )
        )
    sign = np.array([int(k[3]) - int(k[2]) for k in keys])
    for field, counts, idx in (("recall", ai, 2), ("fpr", human, 3)):
        lo, hi = delta_bounds(counts, counts + empty, sign)
        tests = max(1, sum(0 < n < N for N, n, *_ in raw))
        exact = np.array(
            [survey.hg_bounds(int(cell[0]), int(cell[1]), int(cell[idx]), 0.025 / tests) for cell in raw]
        )
        fl, fh = delta_bounds(exact[:, 0][None, :], exact[:, 1][None, :], sign)
        result.append(
            packing(
                dict(**header, model="E-A"),
                field,
                designs["E"][field] - designs["A"][field],
                lo[0],
                hi[0],
                fl[0],
                fh[0],
                *percentiles(lo[1:], hi[1:]),
            )
        )
    return result, [
        dict(
            **header,
            certainty=int(k[0]),
            pool=k[1][0],
            sampling_v9_M=k[1][1],
            A=k[2],
            E=k[3],
            N=int(N),
            n=int(n),
            cell_ai=int(a),
            cell_human=int(h),
        )
        for k, (N, n, a, h) in zip(keys, raw)
    ]


def groups(data):
    yield "all", "全部", data
    yield "ordinary", "普通昵称", [r for r in data if r["ordinary"] == "1"]
    for col in ("length", "pool", "topic"):
        if col == "topic" and data[0]["dataset"] == "auxiliary":
            continue
        for v in sorted({r[col] for r in data}):
            yield col, v, [r for r in data if r[col] == v]


def analyze(w, dest):
    study = w / "out/cross-model-20261123"
    sys.path.insert(0, str(study))
    import survey_calibration as survey

    locked = json.loads((dest / "frozen.json").read_text(encoding="utf8"))
    assert (
        hashlib.sha256((study / "survey_calibration.py").read_bytes()).hexdigest()
        == locked["source_sha256"]["out/cross-model-20261123/survey_calibration.py"]
    )
    input_file = dest / "predictions.csv"
    audit = json.loads((dest / "evaluation-audit.json").read_text(encoding="utf8"))
    assert hashlib.sha256(input_file.read_bytes()).hexdigest() == audit["predictions_sha256"]
    data = read(input_file)
    paper = [r for r in data if r["dataset"] == "paper"]
    history = [r for r in data if r["dataset"] == "auxiliary"]
    videos, draws = survey.video_draws([{"video": r["bvid"], "topic": r["topic"]} for r in paper])
    assert len(videos) == 180 and len(draws) == 5001
    results, all_cells = [], []
    for name, dataset in (
        ("primary", [r for r in paper if not r["overlap"]]),
        ("full_sensitivity", paper),
        ("auxiliary", history),
    ):
        for group, val, part in groups(dataset):
            if name == "full_sensitivity" and group not in ("all", "ordinary"):
                continue
            header = {
                "dataset": name,
                "group": group,
                "value": val,
                "frame": len(part),
                "judged": sum(r["label"] != "" for r in part),
                "ai": sum(r["label"] == "1" for r in part),
                "human": sum(r["label"] == "0" for r in part),
            }
            ap_samples = {}
            for m in ("A", "E"):
                ap, samples = (
                    ap_and_reps(part, m, videos, draws) if name != "auxiliary" else ap_and_reps(part, m)
                )
                ap_samples[m] = samples
                bounds = percentiles(samples, samples) if samples is not None else (np.nan, np.nan, 0)
                results.append(
                    packing(
                        dict(**header, budget="continuous", model=m),
                        "AP",
                        ap,
                        np.nan,
                        np.nan,
                        vl=bounds[0],
                        vh=bounds[1],
                        nrep=bounds[2],
                    )
                )
            if all(s is not None for s in ap_samples.values()):
                delta = ap_samples["E"] - ap_samples["A"]
                lo, hi, n = percentiles(delta, delta)
                results.append(
                    packing(
                        dict(**header, budget="continuous", model="E-A"),
                        "AP",
                        results[-1]["design_point"] - results[-2]["design_point"],
                        np.nan,
                        np.nan,
                        vl=lo,
                        vh=hi,
                        nrep=n,
                    )
                )
            for b in ("1pct", "2pct"):
                head = dict(**header, budget=b)
                if name != "auxiliary":
                    res, cells = calibrated(survey, part, b, videos, draws, head)
                    results += res
                    all_cells += cells
                else:
                    for m in ("A", "E"):
                        obs = observed(part, m, b)
                        for k, v in obs.items():
                            results.append(packing(dict(**head, model=m), k, v, np.nan, np.nan))
    for r in results:
        for key, v in list(r.items()):
            if isinstance(v, (np.ndarray, float, np.floating)):
                r[key] = float(v) if np.isfinite(v) else ""
    write(dest / "metrics.csv", results)
    write(dest / "poststrata.csv", all_cells)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.workspace.resolve(), args.output.resolve())
