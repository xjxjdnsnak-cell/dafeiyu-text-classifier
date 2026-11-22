"""Independently compare recorded frame keys against history, and check sampling/decision flow."""

import argparse
import sys
from collections import Counter
from pathlib import Path

from collect import canonical, freeze, history, read
from score import contrasts


def main(workspace, folder):
    sys.path.insert(0, str(workspace / "out/new-video-blind-20261121/private"))
    import collect_blind as legacy

    uids, keys, videos, aids, ids, visible = history(workspace, legacy)
    rows = read(folder / "frame.json")
    scores = read(folder / "scores.json")
    selected = read(folder / "selection.json")
    flow = read(folder / "flow.json")
    if len(selected) != len(flow) or len({r["bvid"] for r in selected}) != len(selected):
        raise ValueError("Video coverage differs")
    if {v["bvid"] for v in selected} != {v["bvid"] for v in flow}:
        raise ValueError("Video ledger is incomplete")
    if any(v["bvid"] in videos or v["aid"] in aids for v in selected):
        raise ValueError("Video leakage")
    n = Counter(r["bvid"] for r in rows)
    if any(v["kept"] != n[v["bvid"]] for v in flow):
        raise ValueError("Comment counts differ")
    if len({r["rpid"] for r in rows}) != len(rows) or [r["rpid"] for r in rows] != [
        s["id"] for s in scores
    ]:
        raise ValueError("Prediction coverage differs")

    def index(part):
        return [
            {r["bvid"] for r in part},
            {r["uid"] for r in part},
            {canonical(r["text"])[0] for r in part},
            {canonical(r["text"])[1] for r in part},
        ]

    natural = [r for r in rows if r["arm"] == "natural"]
    diagnostic = [r for r in rows if r["arm"] == "diagnostic"]
    hist = [videos, uids, keys, keys]
    comparisons = {}
    for name, left, right in [
        ("history_natural", hist, index(natural)),
        ("history_diagnostic", hist, index(diagnostic)),
        ("natural_diagnostic", index(natural), index(diagnostic)),
    ]:
        c = {
            k: len(a & b) for k, a, b in zip(("video", "account", "full", "body"), left, right, strict=True)
        }
        c["cross_text"] = len((left[2] | left[3]) & (right[2] | right[3]))
        if any(c.values()):
            raise ValueError(f"Overlap in {name}: {c}")
        comparisons[name] = c
    if any(r["rpid"] in ids or legacy.normalize(r["text"])[:4096] in visible for r in rows):
        raise ValueError("History ID/visible text overlap")
    totals = {}
    categories = {}
    for path in legacy.PREVIOUS.joinpath("raw").glob("*.json"):
        if path.name.endswith(".meta.json"):
            continue
        data = read(path).get("data")
        if isinstance(data, dict):
            for v in data.get("result") or []:
                if isinstance(v, dict) and v.get("bvid") and v.get("typename"):
                    categories[v["bvid"]] = v["typename"]
    for arm in ("natural", "diagnostic"):
        pairs = [(r, s) for r, s in zip(rows, scores, strict=True) if r["arm"] == arm]
        totals[arm] = {
            "videos": sum(v["arm"] == arm for v in selected),
            "nonempty_videos": len({r["bvid"] for r, s in pairs}),
            "video_categories": dict(Counter(categories[v["bvid"]] for v in selected if v["arm"] == arm)),
            "categories_with_comments": sorted({categories[r["bvid"]] for r, s in pairs}),
            "rows": len(pairs),
            "ordinary_short": sum(r["ordinary_short"] for r, s in pairs),
            "eligible": sum(s["eligible"] for r, s in pairs),
            "blocked_and_eligible": sum(s["eligible"] and s["plain"] for r, s in pairs),
            "decisions_differ": sum(s["guarded"] != s["learned"] for r, s in pairs),
            "methods": {
                m: sum(s[m] for r, s in pairs)
                for m in (
                    "v8",
                    "v9",
                    "plain",
                    "guarded",
                    "learned",
                    "v8.1",
                    "v9.1",
                    "trained_mask_raw",
                    "trained_mask",
                )
            },
            "ordinary_short_blocks": {
                m: sum(r["ordinary_short"] and s[m] for r, s in pairs)
                for m in ("plain", "guarded", "learned")
            },
        }
    cx = contrasts()
    data = read(folder / "contrast-scores.json")
    if [r["rpid"] for r in cx] != [s["id"] for s in data]:
        raise ValueError("Contrast IDs differ")
    pairs = []
    for i in range(0, len(data), 4):
        a, b, c, d = data[i : i + 4]
        if [r["target"] for r in cx[i : i + 4]] != [1, 1, 0, 0]:
            raise ValueError("Contrast grouping differs")
        pairs.append(
            {
                "topic": cx[i]["topic"],
                "rule_both": a["rule"] and not c["rule"],
                "learned_both": a["request"] and not c["request"],
                "masked_max_entity_score_diff": max(
                    abs(x["masked_score"] - y["masked_score"]) for x, y in ((a, b), (c, d))
                ),
                "learned_entity_flips": sum(x["request"] != y["request"] for x, y in ((a, b), (c, d))),
            }
        )
    freeze(folder / "audit.json", {"comparisons": comparisons, "arms": totals, "contrasts": pairs})
    renames = list(zip(data[::2], data[1::2], strict=True))
    freeze(
        folder / "contrast-rename.json",
        {
            "comparisons": len(renames),
            "flips": {
                m: sum(a[m] != b[m] for a, b in renames)
                for m in ("v8", "v9", "plain", "guarded", "learned")
            },
            "max_absolute_score_change": {
                m: max(abs(a[m] - b[m]) for a, b in renames)
                for m in ("v8_score", "v9_score", "masked_score")
            },
        },
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("folder", type=Path)
    a = ap.parse_args()
    main(a.workspace.resolve(), a.folder.resolve())
