"""Score both frozen models; verify the final study labels against the source returns."""

import argparse
import hashlib
import math
from collections import Counter
from pathlib import Path

from prepare import (
    REPO,
    Preprocessor,
    is_mention_only,
    load_model,
    positive_scores,
    read,
    rows,
    save,
    sha256,
    table,
)

STYLES = {"真人风格": 0, "AI风格": 1}


def label_evidence(study, paper):
    keys = study / "private"
    ret = keys / "return-AC"
    draw = read(keys / "draw.json")
    exported = {r["code"]: r for r in rows(study / "盲评裁决与抽样权重.csv")}
    public = {r["code"]: r for r in rows(study / "public-supplement/labels.csv")}
    full = {r["code"]: r for r in rows(study / "逐条六路.csv")}
    assert len(full) == len(paper) == 1233 and len(public) == len(exported) == 230
    judgments = {}
    for who, count in (("A", 200), ("C", 224)):
        key = read(keys / f"key-{who}.json")
        form = read(ret / f"短评风格复核-{who}-结果-{count}.json")
        assert key["fingerprint"] == form["fingerprint"] and key["ids"].keys() == form["labels"].keys()
        judgments[who] = {rid: form["labels"][code][0] for code, rid in key["ids"].items()}
    by_a, by_c = judgments.values()
    assert set(by_a) == set(draw["members"]) and set(by_c) == set(by_a) | set(draw["repeat"])
    disagreements = sorted(
        (i for i in by_a if by_a[i] != by_c[i]),
        key=lambda i: hashlib.sha256(("judge/" + i).encode()).digest(),
    )
    judge = read(ret / "短评风格裁决-J-结果-5.json")
    assert len(disagreements) == 5 and judge["fingerprint"] == read(ret / "待裁决.json")["fingerprint"]
    final = draw["old"] | by_a
    for j, rid in enumerate(disagreements, 1):
        final[rid] = judge["labels"][f"J{j:03}"][0]
    assert len(final) == 230 and set(draw["old"]).isdisjoint(by_a)
    video_codes = {v: f"V{i:03}" for i, v in enumerate(sorted({r["bvid"] for r in paper}), 1)}
    strata = Counter()
    for r in paper:
        rid, code = r["rpid"], r["code"]
        v = full[code]
        assert (r["topic"], video_codes[r["bvid"]]) == (v["topic"], v["video"])
        r.update(
            pool=v["pool"],
            sampling_v9_M=v["v9_M_block"],
            batch=v["batch"],
            certainty=int(rid in draw["old"]),
        )
        if not r["certainty"]:
            strata[(r["pool"], r["sampling_v9_M"])] += 1
        if rid not in final:
            assert code not in exported
            continue
        rec = exported[code]
        old = rid in draw["old"]
        h = draw["members"].get(rid, {"N": 1, "n": 1})
        if not old:
            assert (h["pool"], str(h["M"])) == (r["pool"], r["sampling_v9_M"])
        assert rec["final_style"] == final[rid] and rec["source"] == ("原短表" if old else "分层抽样")
        assert rec["video"] == v["video"] and rec["pool"] == v["pool"] and rec["topic"] == v["topic"]
        assert (int(rec["N"]), int(rec["n"])) == (h["N"], h["n"])
        pub = public[code]
        assert pub["style"] == ("ai" if STYLES[final[rid]] else "human") and (pub["N"], pub["n"]) == (
            rec["N"],
            rec["n"],
        )
        r.update(
            label=STYLES[final[rid]],
            label_source=rec["source"],
            N=h["N"],
            n=h["n"],
            weight=h["N"] / h["n"],
            final_style=final[rid],
        )
    assert math.isclose(sum(float(r["weight"] or 0) for r in paper), 1233)
    for (pool, m), N in strata.items():
        part = [
            r
            for r in paper
            if not r["certainty"] and (r["pool"], r["sampling_v9_M"]) == (pool, m) and r["label"] != ""
        ]
        assert all(r["N"] == N and r["n"] == len(part) for r in part)
    return {
        "unique": len(final),
        "certainty": len(draw["old"]),
        "probability": len(by_a),
        "repeat": len(draw["repeat"]),
        "adjudicated": len(disagreements),
        "style": dict(Counter(final.values())),
        "strata": draw["cells"],
        "repeat_matches": sum(by_c[i] == draw["old"][i] for i in draw["repeat"]),
        "source_hashes": {
            p.name: sha256(p)
            for p in [
                keys / "draw.json",
                keys / "key-A.json",
                keys / "key-C.json",
                *ret.glob("*结果*.json"),
                study / "盲评裁决与抽样权重.csv",
                study / "逐条六路.csv",
            ]
        },
    }


def score(w, dest):
    freeze = read(dest / "frozen.json")
    base = REPO / "model/text/model.bin"
    assert sha256(base) == freeze["model_A"] and sha256(dest / "E-model.bin") == freeze["model_E"]
    assert sha256(dest / "split-members.csv") == freeze["outputs"]["split-members.csv"]
    for path, digest in freeze["source_sha256"].items():
        assert sha256(w / path) == digest
    members = rows(dest / "split-members.csv")
    paper = []
    aux = []
    for r in members:
        if r["split"] not in ("auxiliary", "paper"):
            continue
        bvs = [v for v in r["videos"].split("|") if v.startswith("BV")]
        entry = {
            "dataset": r["split"],
            "code": r["sample_id"],
            "rpid": r["rpids"],
            "uid": r["uid"],
            "bvid": bvs[0] if len(bvs) == 1 else r["videos"],
            "text": r["text"],
            "ordinary": int(r["ordinary"]),
            "length": r["length"],
            "topic": r["domain"],
            "overlap": r["overlap"],
            "label": r["label"],
            "final_style": "",
            "label_source": "",
            "N": "",
            "n": "",
            "weight": "",
            "pool": "",
            "sampling_v9_M": "",
            "batch": "",
            "certainty": "",
        }
        if r["split"] == "paper":
            paper.append(entry)
        else:
            entry.update(label=int(r["label"]), weight=1, label_source="v8历史标签", pool=r["domain"])
            aux.append(entry)
    audit = label_evidence(w / "out/cross-model-20261123", paper)
    data = paper + aux
    pre = Preprocessor()
    valid = [i for i, r in enumerate(data) if r["text"].strip() and not is_mention_only(r["text"])]
    feats = [pre.encode(data[i]["text"], "", "text") for i in valid]
    for model, path in (("A", base), ("E", dest / "E-model.bin")):
        measured = dict(zip(valid, positive_scores(load_model(path), feats), strict=True))
        for i, r in enumerate(data):
            s = measured.get(i, -1.0)
            r[model + "_score"] = s
            r["scorable"] = int(i in measured)
            for b, key in ((0.01, "1pct"), (0.02, "2pct")):
                r[model + "_" + key] = int(s >= freeze["working_points"][model][str(b)]["threshold"])
    reference = rows(w / "out/v81-v91-results-20261120/test-scores.csv")
    assert len(reference) == len(aux) == 2381
    for r, old in zip(aux, reference, strict=True):
        assert (r["uid"], r["text"], r["label"]) == (old["uid"], old["text"], int(old["label"]))
        assert abs(r["A_score"] - float(old["v8_score"])) < 1e-12 and r["A_1pct"] == int(old["v8_hit"])
    table(dest / "predictions.csv", data)
    differences = []
    for r in data:
        for key in ("1pct", "2pct"):
            a, e = r["A_" + key], r["E_" + key]
            if a == e:
                continue
            label = r["label"]
            change = ("新增误杀" if e else "撤销误杀") if label == 0 else ("新增检出" if e else "新增漏检")
            if label == "":
                change = "未标注新增拦截" if e else "未标注撤拦"
            differences.append(dict(budget=key, change=change, **r))
    table(dest / "changes.csv", differences)
    audit.update(
        predictions_sha256=sha256(dest / "predictions.csv"),
        frozen_sha256=sha256(dest / "frozen.json"),
        old_holdout_scores_reproduced=len(reference),
        primary_excluded=[r["code"] for r in paper if r["overlap"]],
        excluded_labels=dict(Counter(r["final_style"] or "未标注" for r in paper if r["overlap"])),
    )
    save(dest / "evaluation-audit.json", audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("workspace", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    score(args.workspace.resolve(), args.output.resolve())
