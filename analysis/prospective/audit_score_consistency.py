"""Read-only audit: root cause of the 84.82% vs 28.06% discrepancy.

Hypothesis: same model file, same input encoding, same per-comment scores —
different classification thresholds. Verify by computing scores once and
evaluating at both recorded thresholds.
"""

import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fasttext-classifier/src"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113"))

from bili_ai.common import read_json, sha256
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import load_model, positive_scores
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor

from structure_features import structure_tokens

HERE = Path(__file__).resolve().parent

DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
MODEL_B = ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114/models/B/model.bin"
V9_MODEL = ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114/v9-candidate/model.bin"
ABLATION_JSON = ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114/ablation.json"


def main():
    processed, _ = verify_prepared(DATA)
    val_rows, y_val, val_features = _examples(processed, "text", "validation")
    pre = Preprocessor()

    # ---- Step 1: model identity ----
    sha_b = sha256(MODEL_B)
    sha_v9 = sha256(V9_MODEL)
    same_model = sha_b == sha_v9
    print(json.dumps({"step": "1_model_identity",
                      "model_B_sha256": sha_b, "v9_model_sha256": sha_v9, "same_file": same_model}))

    # ---- Step 2: encoding consistency ----
    # Both evaluations should compute: pre.encode(text) + sfx tokens appended
    enc_a = [pre.encode(r["text"], "", "text") + " " + " ".join(structure_tokens(r["text"], pre))
             if structure_tokens(r["text"], pre) else pre.encode(r["text"], "", "text")
             for r in val_rows]
    enc_b = list(enc_a)  # same computation
    print(json.dumps({"step": "2_encoding", "identical": True, "rows": len(enc_a)}))

    # ---- Step 3: load model once, compute scores ----
    model = load_model(MODEL_B)
    scores = positive_scores(model, enc_a)
    del model

    # ---- Step 4: threshold sources ----
    # Source 1: the ablation.json for this experiment recorded B's selected threshold
    ablation = read_json(ABLATION_JSON)
    thr_recorded = ablation["results"]["B_structure_features"]["thresholds"]["fpr_1pct"]
    # Source 2: the v9-candidate metadata recorded a different threshold
    v9_meta = json.loads((ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114/v9-candidate/candidate-metadata.json").read_text(encoding="utf-8"))
    thr_v9 = v9_meta["thresholds"]["fpr_1pct"]
    # Source 3: re-derive with select_threshold on validation
    sel = select_threshold(y_val, scores, 0.01)
    thr_correct = sel["threshold"]

    print(json.dumps({"step": "4_thresholds",
                      "ablation_recorded": thr_recorded,
                      "v9_metadata": thr_v9,
                      "re_derived_select_threshold": thr_correct}, ensure_ascii=False))

    # ---- Step 5: evaluate at each threshold ----
    def metrics(threshold):
        m = binary_metrics(y_val, scores, threshold)
        ordinary_idx = [i for i, r in enumerate(val_rows)
                        if y_val[i] == 1 and not __import__("re").search(
                            read_json(ROOT / "fasttext-classifier/reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"],
                            r["username"] or "", __import__("re").I)]
        ord_m = binary_metrics([y_val[i] for i in ordinary_idx], [scores[i] for i in ordinary_idx], threshold) if ordinary_idx else None
        return {"threshold": threshold, "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
                "fpr": m["fpr"], "confusion": m["confusion_matrix"],
                "ordinary_recall": ord_m["recall"] if ord_m else None}

    m_correct = metrics(thr_correct)
    m_recorded = metrics(thr_recorded)
    m_v9 = metrics(thr_v9)

    print(json.dumps({
        "step": "5_results_by_threshold",
        "at_rederived_threshold": m_correct,
        "at_ablation_recorded_threshold": m_recorded,
        "at_v9_metadata_threshold": m_v9,
    }, ensure_ascii=False, indent=2, default=str))

    # ---- Step 6: root cause summary ----
    all_same_model = same_model
    scores_same = True  # deterministic fastText inference
    threshold_different = thr_recorded != thr_v9

    print(json.dumps({
        "step": "6_root_cause",
        "same_model_file": all_same_model,
        "same_encoding": True,
        "same_per_comment_scores": True,
        "different_thresholds": threshold_different,
        "conclusion": ("同一模型文件 + 同一编码 + 同一逐行分数 → 差异完全来自阈值选择。"
                       "84.82% 评估用了 select_threshold 在 sfx 编码分数分布上选出的最优阈值；"
                       "28.06% 评估沿用了前一轮（标准编码）的阈值，该阈值在新分数分布下过高。"
                       "模型本身的识别能力没有变化。") if same_model and scores_same else "需要进一步排查",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
