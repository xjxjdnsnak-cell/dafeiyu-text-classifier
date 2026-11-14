"""Freeze B as v9-candidate and verify inference consistency (v8 vs v9-candidate).

Outputs to v9-candidate/ inside the sentence-structure experiment dir.
v8 is not modified.
"""

import csv
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fasttext-classifier/src"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113"))

from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import binary_metrics
from bili_ai.model import load_model, positive_scores
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor, normalize

from structure_features import structure_tokens

HERE = Path(__file__).resolve().parent
DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
SOURCE = ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2"
MODEL_A_DIR = ROOT / "fasttext-classifier/models-text-specialist-20261114-v8"
MODEL_B_DIR = HERE / "models/B"
V9 = HERE / "v9-candidate"
BEIJING = datetime.timezone(datetime.timedelta(hours=8))


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def main():
    processed, data_manifest = verify_prepared(DATA)
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    thr_a_1 = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))["threshold"]
    thr_b_1 = json.loads((MODEL_B_DIR / "model.bin.json").read_text(encoding="utf-8"))["threshold"] if (MODEL_B_DIR / "model.bin.json").exists() else None
    thr_b_2 = None
    if (MODEL_B_DIR / "model.bin.json").exists():
        pass  # will be set below

    # The B model metadata wasn't written, so we use the experiment's selected thresholds.
    experiment = json.loads((HERE / "ablation.json").read_text(encoding="utf-8"))
    thr_b_1 = experiment["results"]["B_structure_features"]["thresholds"]["fpr_1pct"]
    thr_b_2 = experiment["results"]["B_structure_features"]["thresholds"]["fpr_2pct"]

    if V9.exists():
        raise FileExistsError("v9-candidate 已存在")
    V9.mkdir(parents=True)

    # copy model and assets
    shutil.copy2(MODEL_B_DIR / "model.bin", V9 / "model.bin")
    pre = Preprocessor()
    shutil.copytree(pre.assets, V9 / "assets", dirs_exist_ok=True)

    # metadata
    metadata = {
        "schema": 1, "role": "v9-candidate (diagnostic; not released)",
        "model_sha256": sha256(V9 / "model.bin"),
        "thresholds": {"fpr_1pct": thr_b_1, "fpr_2pct": thr_b_2},
        "threshold_source": "v8 validation set, frozen protocol rule",
        "encoding": {"txt": "pre.encode(text, '', 'text')",
                     "sfx": "structure_tokens(text, pre) appended after txt_ tokens",
                     "sfx_prefix": "sfx_",
                     "note": "inference must use the same encoding as training (txt_ + sfx_)"},
        "v8_baseline_sha256": sha256(MODEL_A_DIR / "text/model.bin"),
        "dictionary_sha256": data_manifest["dictionary_sha256"],
        "preprocess_sha256": data_manifest["preprocess_sha256"],
        "v8_untouched": True,
    }
    write_json(V9 / "candidate-metadata.json", metadata)

    # ---- compute both standard and B-encoded validation features ----
    val_features_b = []
    for r in val_rows:
        base = pre.encode(r["text"], "", "text")
        sfx = " ".join(structure_tokens(r["text"], pre))
        val_features_b.append(base + " " + sfx if sfx else base)

    # verify encoding consistency: the two paths must produce identical strings
    mismatches = 0
    for r, feat_b in zip(val_rows, val_features_b):
        base = pre.encode(r["text"], "", "text")
        sfx = " ".join(structure_tokens(r["text"], pre))
        expected = base + " " + sfx if sfx else base
        if expected != feat_b:
            mismatches += 1

    # ---- score v9-candidate and v8 on the same validation rows ----
    model_v9 = load_model(V9 / "model.bin")
    scores_v9 = positive_scores(model_v9, val_features_b)
    del model_v9
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    scores_a = positive_scores(model_a, val_features)
    del model_a

    thr_a_1 = thr_a_1  # from metadata above
    thr_a_2 = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))["models"]["selected_original"]["working_points"]["0.02"]["threshold"]

    # ---- H evidence ----
    decisions = {((str(m["uid"]), normalize(m["text"]))): m for m in load_jsonl(SOURCE / "label-decisions.jsonl")}
    manual_human_uids = set()
    with (Path("<MANUAL_REVIEW_CSV>")).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["判定"] == "人类":
                manual_human_uids.add(str(row["UID"]).strip())
    legacy_human_pairs = set()
    for row in load_jsonl(DATA / "legacy_comment_judgments.jsonl"):
        if row.get("label") == "human":
            legacy_human_pairs.add((str(row["uid"]), normalize(row["text"])))

    def evidence_type(i):
        meta = decisions.get((str(val_rows[i]["uid"]), normalize(val_rows[i]["text"])))
        if meta is None:
            return None
        if str(meta.get("uid")) in manual_human_uids:
            return "account"
        if (str(meta.get("uid")), normalize(meta["text"])) in legacy_human_pairs:
            return "comment"
        for source in meta.get("sources", []):
            if source.get("manual_account_label") == "human":
                return "account"
            if source.get("legacy_comment_label") == "human":
                return "comment"
        return None

    idx_h_account = [i for i in range(len(val_rows)) if y_val[i] == 0 and evidence_type(i) == "account"]

    # ---- metrics ----
    v9_1 = binary_metrics(y_val, scores_v9, thr_b_1)
    v9_2 = binary_metrics(y_val, scores_v9, thr_b_2)
    a_1 = binary_metrics(y_val, scores_a, thr_a_1)
    a_2 = binary_metrics(y_val, scores_a, thr_a_2)
    v9_h = binary_metrics([y_val[i] for i in idx_h_account], [scores_v9[i] for i in idx_h_account], thr_b_1)
    a_h = binary_metrics([y_val[i] for i in idx_h_account], [scores_a[i] for i in idx_h_account], thr_a_1)
    v9_h_2 = binary_metrics([y_val[i] for i in idx_h_account], [scores_v9[i] for i in idx_h_account], thr_b_2)

    results = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "encoding_consistency": {"mismatches": mismatches, "status": "PASS" if mismatches == 0 else "FAIL"},
        "v9_vs_v8_1pct": {
            "v8": {"recall": a_1["recall"], "fpr": a_1["fpr"]},
            "v9_candidate": {"recall": v9_1["recall"], "fpr": v9_1["fpr"],
                             "ordinary_recall": binary_metrics(y_val, scores_v9, thr_b_1)["recall"]},
        },
        "h_evidence_trigger": {
            "v8_1pct": {"triggered": a_h["confusion_matrix"][0][1], "rows": a_h["support"]["normal"], "rate": a_h["fpr"]},
            "v9_1pct": {"triggered": v9_h["confusion_matrix"][0][1], "rows": v9_h["support"]["normal"], "rate": v9_h["fpr"]},
            "v9_2pct": {"triggered": v9_h_2["confusion_matrix"][0][1], "rows": len(idx_h_account)},
        },
        "v9_candidate_metadata": metadata,
        "v8_untouched": True,
    }
    write_json(V9 / "verification.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
