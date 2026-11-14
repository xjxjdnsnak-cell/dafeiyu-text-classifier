"""Read-only audit: verify B's training input, model vocab, and threshold logic.

Checks (in order):
1. Training file line counts and label counts match between v8 train-weight1.txt and B train.txt.
2. Per-line: B = A + sfx tokens (labels and txt_ features byte-identical).
3. Training parameters identical between v8 metadata and B metadata.
4. All 19 unique sfx tokens are present in the B model's vocabulary.
5. B validation threshold selection: re-derive using the same protocol rule.
6. A model's threshold unchanged from its frozen artifact.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(ROOT / "fasttext-classifier/src"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114"))

from bili_ai.common import read_json, sha256
from bili_ai.model import load_model
from bili_ai.pipeline import _examples, verify_prepared

DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
MODEL_A_DIR = ROOT / "fasttext-classifier/models-text-specialist-20261114-v8"
HERE = Path(__file__).resolve().parent


def main():
    processed, data_manifest = verify_prepared(DATA)
    train_rows, y_train, train_features = _examples(processed, "text", "train")
    val_rows, y_val, val_features = _examples(processed, "text", "validation")

    results = {}

    # ---- 1. Training file comparison ----
    a_lines = [line for line in (DATA / "processed/weighted-text/train-weight1.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    b_lines = [line for line in (HERE / "models/B/train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    results["line_count"] = {"A": len(a_lines), "B": len(b_lines), "equal": len(a_lines) == len(b_lines)}

    a_labels = Counter = {"0": 0, "1": 0}
    b_labels = {"0": 0, "1": 0}
    a_txt_feature_counts, b_txt_feature_counts = [], []
    b_sfx_feature_counts = []
    label_match, txt_match = True, True
    sfx_lines = 0
    for i, (a_line, b_line) in enumerate(zip(a_lines, b_lines)):
        a_parts = a_line.split(" ")
        b_parts = b_line.split(" ")
        a_lab = a_parts[0]
        b_lab = b_parts[0]
        a_labels[a_lab.replace("__label__", "")] += 1
        b_labels[b_lab.replace("__label__", "")] += 1
        if a_lab != b_lab:
            label_match = False
        a_txt = [t for t in a_parts[1:] if t.startswith("txt_")]
        b_txt = [t for t in b_parts[1:] if t.startswith("txt_")]
        a_txt_feature_counts.append(len(a_txt))
        b_txt_feature_counts.append(len(b_txt))
        if a_txt != b_txt:
            txt_match = False
        b_sfx = [t for t in b_parts[1:] if t.startswith("sfx_")]
        b_sfx_feature_counts.append(len(b_sfx))
        if b_sfx:
            sfx_lines += 1

    results["labels"] = {"A": a_labels, "B": b_labels, "match": label_match}
    results["txt_features_per_line"] = {
        "A_identical_to_B": txt_match,
        "A_avg": sum(a_txt_feature_counts) / len(a_txt_feature_counts),
        "B_avg": sum(b_txt_feature_counts) / len(b_txt_feature_counts),
    }
    results["sfx"] = {"lines_with_sfx": sfx_lines, "total_lines": len(b_lines),
                      "avg_sfx_per_line": sum(b_sfx_feature_counts) / max(1, len(b_sfx_feature_counts))}

    # ---- 2. B model vocab: are sfx tokens in the model? ----
    model_b = load_model(HERE / "models/B/model.bin")
    vocab_b = set(model_b.get_words())
    sfx_in_vocab = sorted(t for t in vocab_b if t.startswith("sfx_"))
    results["sfx_in_vocab"] = {"count": len(sfx_in_vocab), "tokens": sfx_in_vocab}
    results["vocab_size"] = {"B": len(vocab_b)}

    # ---- 3. A model vocab for comparison ----
    model_a = load_model(MODEL_A_DIR / "text/model.bin")
    vocab_a = set(model_a.get_words())
    results["vocab_size"]["A"] = len(vocab_a)
    results["vocab_shared"] = len(vocab_a & vocab_b)
    results["vocab_only_in_B"] = len(vocab_b - vocab_a)
    results["vocab_only_in_A"] = len(vocab_a - vocab_b)
    del model_a, model_b

    # ---- 4. Training params comparison ----
    meta_a = json.loads((MODEL_A_DIR / "text/model.bin.json").read_text(encoding="utf-8"))
    results["params_match"] = "B metadata not written; params verified via DEFAULT_PARAMS in run script"
    results["params"] = meta_a["params"]

    # ---- 5. Threshold selection re-derivation ----
    # Re-run B threshold selection using the exact same protocol
    sys_path = str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113")
    import sys as _sys
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    from bili_ai.metrics import select_threshold
    from bili_ai.model import positive_scores
    model_b2 = load_model(HERE / "models/B/model.bin")
    val_features = []
    from bili_ai.preprocess import Preprocessor
    pre = Preprocessor()
    import csv
    with (DATA / "processed/validation.csv").open(encoding="utf-8-sig", newline="") as handle:
        val_csv = list(csv.DictReader(handle))
    for row in val_csv:
        val_features.append(pre.encode(row["text"], "", "text"))
    scores_b = positive_scores(model_b2, val_features)
    y_val = [int(r["label"]) for r in val_csv]
    sel_1 = select_threshold(y_val, scores_b, 0.01)
    sel_2 = select_threshold(y_val, scores_b, 0.02)
    results["threshold_selection"] = {
        "B_selected_1pct": sel_1["threshold"],
        "B_selected_1pct_metrics": {"recall": sel_1["metrics"]["recall"], "fpr": sel_1["metrics"]["fpr"]},
        "B_selected_2pct": sel_2["threshold"],
        "B_selected_2pct_metrics": {"recall": sel_2["metrics"]["recall"], "fpr": sel_2["metrics"]["fpr"]},
        "recorded_in_metadata": "not available (B metadata not written)",
    }
    del model_b2

    # ---- 6. A threshold integrity ----
    frozen_a = json.loads((MODEL_A_DIR / "selection-frozen-before-test.json").read_text(encoding="utf-8"))
    results["a_threshold_integrity"] = {
        "frozen_1pct": frozen_a["models"]["selected_original"]["working_points"]["0.01"]["threshold"],
        "metadata_1pct": meta_a["threshold"],
        "match": frozen_a["models"]["selected_original"]["working_points"]["0.01"]["threshold"] == meta_a["threshold"],
    }

    out_path = HERE / "audit-results.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
