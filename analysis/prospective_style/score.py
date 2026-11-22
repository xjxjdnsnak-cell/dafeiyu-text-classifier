"""Run locally frozen predictors. Raw comments and row-level outputs stay private."""

import argparse
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

from collect import HERE, P, freeze, read


def contrasts():
    rows = []
    for i, (topic, request, statement) in enumerate(read(HERE / "contrasts.json")):
        for target, sentence in [(1, request), (0, statement)]:
            for entity, mention in enumerate(("小林", "大肥鱼")):
                rows.append(
                    {
                        "rpid": f"pair-{i}-{target}-{entity}",
                        "text": f"@{mention} {sentence}",
                        "topic": topic,
                        "target": target,
                        "entity": entity,
                        "arm": "contrast",
                    }
                )
    return rows


def main(workspace, folder, intent_python):
    archive = workspace / "out"
    repo = HERE.parents[1]
    guard_repo = archive / "v9-direct-request-candidate"
    for directory, revision in [(guard_repo, "3f218c7"), (archive / "mention-aware-training", "6d6b528")]:
        for path in (
            ["code/direct_request.py", "code/predict_masked_v9.py"]
            if directory == guard_repo
            else ["code/predict_mention_trained.py"]
        ):
            expected = subprocess.check_output(
                ["git", "-C", directory, "rev-parse", f"{revision}:{path}"]
            ).strip()
            actual = subprocess.check_output(["git", "-C", directory, "hash-object", path]).strip()
            if expected != actual:
                raise ValueError(f"Frozen predictor differs: {path}")
    sys.path[:0] = [str(guard_repo / "code"), str(archive / "mention-aware-training/code")]
    # The frozen predictor adds the matching feature extraction modules to sys.path.
    # isort: off
    from predict_masked_v9 import MaskedV9Predictor, is_direct_request, MODEL, THRESHOLD
    from predict_mention_trained import MentionTrainedPredictor
    from bili_ai.model import load_model, positive_scores
    from bili_ai.common import sha256
    from bili_ai.preprocess import Preprocessor
    from structure_features import encode_structure_augmented
    from text_policy import is_mention_only
    # isort: on

    records = read(folder / "frame.json") + contrasts()
    inputs = [{"text": r["text"]} for r in records]
    masked = MaskedV9Predictor().predict_many(inputs)
    guard = MaskedV9Predictor(protect_direct_requests=True).predict_many(inputs)
    train_mask = MentionTrainedPredictor(
        archive / "mention-training-results-20261120/candidate"
    ).predict_many(inputs)
    legacy = archive / "v81-v91-results-20261120"
    meta = read(legacy / "frozen-before-test.json")["models"]
    pre = Preprocessor()
    extra = {}
    for name, model in [
        ("v8", repo / "model/text/model.bin"),
        ("v9", MODEL),
        ("v8.1", legacy / "v8.1/model.bin"),
        ("v9.1", legacy / "v9.1/model.bin"),
    ]:
        if sha256(model) != meta[name]["model_sha256"]:
            raise ValueError(f"Frozen weights differ: {name}")
        valid = [i for i, r in enumerate(records) if r["text"].strip() and not is_mention_only(r["text"])]
        lines = [
            (
                pre.encode(records[i]["text"], "", "text")
                if name.startswith("v8")
                else encode_structure_augmented(records[i]["text"], pre)
            )
            for i in valid
        ]
        scores = dict(zip(valid, positive_scores(load_model(model), lines), strict=True))
        extra[name] = [scores.get(i) for i in range(len(records))]
    weights = archive / "address-intent-private/model"
    if (
        sha256(weights / "intent.safetensors") != P["models"]["intent_sha256"]
        or THRESHOLD != P["models"]["masked_v9"]
    ):
        raise ValueError("Frozen parameters differ")
    with tempfile.TemporaryDirectory(dir=folder) as tmp:
        inp, out = Path(tmp) / "input.csv", Path(tmp) / "scores.csv"
        with inp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "text"])
            writer.writeheader()
            writer.writerows({"id": r["rpid"], "text": r["text"]} for r in records)
        subprocess.run(
            [intent_python, repo / "analysis/address_intent/pilot.py", "score", inp, weights, out],
            check=True,
        )
        with out.open(encoding="utf-8-sig", newline="") as f:
            intents = list(csv.DictReader(f))
    results = []
    for i, (r, a, b, c, t) in enumerate(zip(records, masked, guard, train_mask, intents, strict=True)):
        if t["id"] != r["rpid"] or t["text"] != r["text"]:
            raise ValueError("Intent scoring order differs")
        request = t["predicted_request"] == "True"
        results.append(
            dict(
                id=r["rpid"],
                eligible=bool(t["intent_score"]),
                request=request,
                intent_score=float(t["intent_score"]) if t["intent_score"] else None,
                rule=is_direct_request(r["text"]),
                plain=a["blocked"],
                guarded=b["blocked"],
                learned=a["blocked"] and not request,
                masked_score=a["ai_style_score"],
                trained_mask=c["blocked"],
                trained_mask_score=c["ai_style_score"],
                trained_mask_raw=c["model_label"] == 1,
                **{n: s[i] is not None and s[i] >= meta[n]["threshold"] for n, s in extra.items()},
                **{n + "_score": s[i] for n, s in extra.items()},
            )
        )
    freeze(folder / "scores.json", results[: len(results) - len(contrasts())])
    freeze(folder / "contrast-scores.json", results[-len(contrasts()) :])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("private_output", type=Path)
    ap.add_argument("intent_python")
    a = ap.parse_args()
    main(a.workspace.resolve(), a.private_output.resolve(), a.intent_python)
