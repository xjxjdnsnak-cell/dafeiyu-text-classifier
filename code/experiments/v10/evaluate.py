"""Replay frozen working points. Only aggregate counts and split fingerprints leave the workspace."""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from statistics import mean

from inputs import SPLITS, TEACHER, check, digest, save, split
from views import CORNER, MENTION, broad, training_views
from predict_masked_v9 import ASSETS, ROOT, MaskedV9Predictor, Predictor
from predict_v10 import MODEL, V10Predictor


def read(path):
    return json.loads(path.read_bytes())


def score(net, texts):
    unique = list(dict.fromkeys(texts))
    results = net.predict_many([{"text": t} for t in unique])
    table = {t: {"score": r["ai_style_score"], "blocked": r["blocked"]}
             for t, r in zip(unique, results, strict=True)}
    return [table[t] for t in texts]


def confusion(labels, values):
    cells = Counter((y, int(r["blocked"])) for y, r in zip(labels, values, strict=True))
    return dict(n=len(labels), tp=cells[1, 1], fp=cells[0, 1], fn=cells[1, 0], tn=cells[0, 0],
                unknown=len(labels)-sum(cells[y, p] for y in (1, 0) for p in (1, 0)))


def paired(a, b):
    changes = Counter((x["blocked"], y["blocked"]) for x, y in zip(a, b, strict=True))
    comparable = [(x["score"], y["score"]) for x, y in zip(a, b, strict=True)
                  if None not in (x["score"], y["score"])]
    return dict(n=len(a), before=sum(x["blocked"] for x in a), after=sum(y["blocked"] for y in b),
                withdrawn=changes[True, False], added=changes[False, True], numeric_pairs=len(comparable),
                mean_abs_score_delta=mean(abs(x-y) for x, y in comparable) if comparable else None)


def weak_test(processed, nets):
    original_rows = split(processed, "train")
    val_rows = split(processed, "validation")
    test = split(processed, "test")
    # Audit after fitting. Count exact encoded duplicates newly introduced by augmentation.
    encoder = nets["v10"].pre
    old, aug = set(), set()
    rng = random.Random(42)
    for row in original_rows:
        variants = training_views(row["text"], rng.randrange(4))
        orig = encoder.encode(variants[0], "", "text")
        old.add(orig)
        for body in set(variants):
            features = encoder.encode(body, "", "text")
            aug.add(orig if features == "field_empty" else features)

    def overlap(rows):
        enc = [encoder.encode(r["text"], "", "text") for r in rows]
        return [i for i, e in enumerate(enc) if e in old], [i for i, e in enumerate(enc) if e in aug-old]

    seen, new = overlap(test)
    old_val, new_val = overlap(val_rows)
    excluded = set(seen + new)
    labels = [int(row["label"]) for row in test]
    # v8 groups duplicates by all source nicknames, including non-representative rows.
    lineage = processed.parent / "sample-lineage.jsonl"
    with lineage.open(encoding="utf-8-sig") as f:
        held = [r for line in f if (r := json.loads(line))["split"] == "test"]
    ordinary = {(r["uid"], r["text"]): r["ordinary"] for r in held}
    if len(ordinary) != len(test):
        raise ValueError("Test rows and lineage differ")
    groups = {"all": list(range(len(test))),
              "ordinary": [i for i, r in enumerate(test) if ordinary[r["uid"], r["text"]]],
              "short": [i for i, r in enumerate(test) if len(r["text"]) <= 20],
              "unseen_encoded_body": [i for i in range(len(test)) if i not in excluded]}
    outputs = {}
    v8 = Predictor(ROOT / "model/text/model.bin", mode="text", assets=ASSETS)
    for name, net in {"v8": v8, **nets}.items():
        rows = score(net, [r["text"] for r in test])
        outputs[name] = {g: confusion([labels[i] for i in ids], [rows[i] for i in ids])
                         for g, ids in groups.items()}
    return {"models": outputs, "lineage_sha256": digest(lineage), "encoded_overlap_rows": {
        "test_original": len(seen), "test_added": len(new),
        "val_original": len(old_val), "val_added": len(new_val)}}


STYLE = {"AI风格": 1, "真人风格": 0, "不确定": None}


def pilot(out, nets):
    pilot_dir = out / "a-pilot-counterfactual-20261122"
    source = read(pilot_dir / "inputs.json")
    texts = {r["code"]: r["text"] for r in source["stimuli"] if r["variant"] == "raw"}
    records = source["cohort"]
    manifest = read(out / "a-pilot-closure-20261122/b120-manifest.json")
    final = read(out / ("prospective-style-private-20261122/returns/accepted/"
                        "b120-20261122/人工裁决回收/最终标签-120.json"))
    adjudicated = {r["id"]: final["labels_by_b_code"][r["code"]]["style"] for r in manifest["rows"]
                   if r["arm"] == "natural"}
    if len(records) != 308 or len(adjudicated) != 108:
        raise ValueError("Unexpected pilot cohort")
    bodies = [texts[r["code"]] for r in records]
    mentions = [i for i, t in enumerate(bodies) if MENTION.search(t)]
    selected = [i for i, r in enumerate(records) if r["id"] in adjudicated]
    neural = read(pilot_dir / "roberta-replay.json")
    if neural["weights"] != TEACHER["model.safetensors"]:
        raise ValueError("Unexpected pilot teacher")
    archived = {(r["code"], r["variant"]): r["roberta"] for r in neural["rows"]}
    result = {}
    for name in [*nets, "roberta"]:
        def predictions(ids, variant="raw"):
            if name == "roberta":
                return [archived[records[i]["code"], variant] for i in ids]
            forms = {"neutral": "@小林", "project": "@大肥鱼", "drop": ""}
            return score(nets[name], [bodies[i] if variant == "raw" else MENTION.sub(forms[variant], bodies[i])
                                      for i in ids])

        raw = predictions(range(len(records)))
        original = [raw[i] for i in mentions]
        result[name] = {
            "A_natural": confusion([STYLE[r["style"]] for r in records], raw),
            "adjudicated_natural": confusion([STYLE[adjudicated[records[i]["id"]]] for i in selected],
                                             [raw[i] for i in selected]),
            "mentions": {k: paired(original, predictions(mentions, k)) for k in ("neutral", "project", "drop")}}
    return result


def retrospective(out, nets, private):
    retro = out / "confirmation-retrospective-20261123"
    members = retro / "game-supplement/合并评论.json"
    records_dir = retro / "analysis/private"
    archive = records_dir / "冻结逐条预测.json"
    lock = read(records_dir / "预测封存.json")
    check(members, lock["member_sha256"])
    check(archive, lock["predictions_sha256"])
    rows, reference = read(members), read(archive)
    keys = [r["rpid"] for r in rows]
    teacher = read(out / "cross-model-20261123/roberta.json")
    if (len(set(keys)) != len(keys) or len(keys) != 1233
            or keys != [r["rpid"] for r in reference] or keys != [r["rpid"] for r in teacher["rows"]]
            or teacher["receipt"]["weights"] != TEACHER["model.safetensors"]):
        raise ValueError("Frozen inputs do not align")
    labels = {r["rpid"]: r["consensus"] for r in read(records_dir / "returns/两表对照/逐条对应.json")}
    base = [r["text"] for r in rows]
    forms = {"M": base, "MQ": [t.translate(CORNER) for t in base], "MB": [broad(t) for t in base]}
    dropped = [i for i, r in enumerate(reference) if r["arms"]["M"]["blocked"] and not r["arms"]["MB"]["blocked"]]
    known = [i for i in dropped if labels.get(keys[i]) == "AI风格"]
    if (len(dropped), len(known)) != (77, 19):
        raise ValueError("Unexpected diagnostic subset")
    result = {}
    for name in [*nets, "roberta"]:
        if name == "roberta":
            arms = {k: [r["arms"][k] for r in teacher["rows"]] for k in forms}
        else:
            arms = {k: score(nets[name], texts) for k, texts in forms.items()}
        if name == "v9":
            for k, values in arms.items():
                if any(r != frozen["arms"][k] for r, frozen in zip(values, reference, strict=True)):
                    raise ValueError(f"v9 replay differs from frozen arm {k}")
        result[name] = {k: paired(arms["M"], arms[k]) for k in ("MQ", "MB")}
        for k, ids in (("v9_withdrawn_77", dropped), ("double_review_ai_19", known)):
            result[name][k] = paired([arms["M"][i] for i in ids], [arms["MB"][i] for i in ids])
        if name == "v10":
            save(private, [{"rpid": key, **{k: v[i] for k, v in arms.items()}} for i, key in enumerate(keys)])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("workspace", type=Path, help="Local study workspace with historical private inputs")
    p.add_argument("controls", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("private_rows", type=Path)
    p.add_argument("--candidate", type=Path, default=MODEL, help="Trained v10 model.bin")
    a = p.parse_args()
    out = a.workspace / "out"
    processed = a.workspace / "fasttext-classifier/data-text-specialist-20261114-v8/processed"
    nets = {"v9": MaskedV9Predictor(),
            **{s: V10Predictor(a.controls / s / "model.bin") for s in ("raw_hard", "aug_hard")},
            "v10": V10Predictor(a.candidate)}
    result = {"split_sha256": SPLITS, "model_thresholds": {k: v.threshold for k, v in nets.items()},
              "controls": {k: nets[k].metadata for k in ("raw_hard", "aug_hard")},
              "weak_test": weak_test(processed, nets),
              "pilot": pilot(out, nets), "retrospective": retrospective(out, nets, a.private_rows)}
    result["v10_sha256"] = nets["v10"].metadata["model_sha256"]
    result["private_replay_sha256"] = digest(a.private_rows)
    save(a.results, result)


if __name__ == "__main__":
    main()
