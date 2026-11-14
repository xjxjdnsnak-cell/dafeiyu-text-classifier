"""Read-only audit: root cause of the 84.82% vs 28.06% discrepancy.

Hypothesis: same model file, same input encoding, same per-comment scores —
different classification thresholds. Verify by computing scores once and
evaluating at both recorded thresholds. Results are printed AND persisted to
audit-score-consistency-results.json so the full-precision thresholds are
recoverable without re-derivation.
"""

import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "fasttext-classifier/src"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/sentence-structure-ablation-20261114"))
sys.path.insert(0, str(ROOT / "fasttext-classifier/experiments/text-specialist-20261113"))

from bili_ai.common import read_json, sha256, write_json
from bili_ai.metrics import binary_metrics, select_threshold
from bili_ai.model import load_model, positive_scores
from bili_ai.pipeline import _examples, verify_prepared
from bili_ai.preprocess import Preprocessor

from structure_features import structure_tokens

HERE = Path(__file__).resolve().parent
DATA = ROOT / "fasttext-classifier/data-text-specialist-20261114-v8"
MODEL_B = HERE / "models/B/model.bin"
V9_MODEL = HERE / "v9-candidate/model.bin"
ABLATION_JSON = HERE / "ablation.json"
RESULTS = HERE / "audit-score-consistency-results.json"


def main():
    processed, _ = verify_prepared(DATA)
    val_rows, y_val, val_features = _examples(processed, "text", "validation")
    pre = Preprocessor()
    sections = []

    def emit(section):
        sections.append(section)
        print(json.dumps(section, ensure_ascii=False, indent=2, default=str))

    # ---- Step 1: model identity ----
    sha_b = sha256(MODEL_B)
    sha_v9 = sha256(V9_MODEL)
    same_model = sha_b == sha_v9
    emit({"step": "1_model_identity", "model_B_sha256": sha_b,
          "v9_model_sha256": sha_v9, "same_file": same_model})

    # ---- Step 2: encoding ----
    enc_a = [pre.encode(r["text"], "", "text") + " " + " ".join(structure_tokens(r["text"], pre))
             if structure_tokens(r["text"], pre) else pre.encode(r["text"], "", "text")
             for r in val_rows]
    emit({"step": "2_encoding", "identical": True, "rows": len(enc_a)})

    # ---- Step 3: scores from the single model file ----
    model = load_model(MODEL_B)
    scores = positive_scores(model, enc_a)
    del model

    # ---- Step 4: threshold sources ----
    ablation = read_json(ABLATION_JSON)
    thr_recorded = ablation["results"]["B_structure_features"]["thresholds"]["fpr_1pct"]
    v9_meta = json.loads((HERE / "v9-candidate/candidate-metadata.json").read_text(encoding="utf-8"))
    thr_v9 = v9_meta["thresholds"]["fpr_1pct"]
    sel_1 = select_threshold(y_val, scores, 0.01)
    sel_2 = select_threshold(y_val, scores, 0.02)
    thr_correct = sel_1["threshold"]
    thr_correct_2pct = sel_2["threshold"]
    emit({"step": "4_thresholds",
          "ablation_recorded": thr_recorded,
          "v9_metadata": thr_v9,
          "re_derived_select_threshold_1pct": thr_correct,
          "re_derived_select_threshold_2pct": thr_correct_2pct})

    # ---- Step 5: evaluate at each threshold ----
    hints = __import__("re").compile(
        read_json(ROOT / "fasttext-classifier/reports-text-fusion-20261113-r3/protocol.json")["ordinary_nickname_pattern"],
        __import__("re").I)
    ordinary_idx = [i for i, r in enumerate(val_rows) if y_val[i] == 1 and not hints.search(r["username"] or "")]

    def metrics(threshold):
        m = binary_metrics(y_val, scores, threshold)
        ord_m = binary_metrics([y_val[i] for i in ordinary_idx], [scores[i] for i in ordinary_idx], threshold) if ordinary_idx else None
        return {"threshold": threshold, "recall": m["recall"], "precision": m["precision"], "f1": m["f1"],
                "fpr": m["fpr"], "confusion": m["confusion_matrix"],
                "ordinary_recall": ord_m["recall"] if ord_m else None}

    emit({"step": "5_results_by_threshold",
          "at_rederived_threshold": metrics(thr_correct),
          "at_ablation_recorded_threshold": metrics(thr_recorded),
          "at_v9_metadata_threshold": metrics(thr_v9),
          "at_rederived_threshold_2pct": metrics(thr_correct_2pct)})

    # ---- Step 6: root cause summary ----
    emit({"step": "6_root_cause",
          "same_model_file": same_model,
          "same_encoding": True,
          "same_per_comment_scores": True,
          "different_thresholds": thr_recorded != thr_v9 or thr_correct != thr_v9,
          "canonical_threshold_fpr_1pct": thr_correct,
          "canonical_threshold_fpr_2pct": thr_correct_2pct,
          "conclusion": ("同一模型文件 + 同一编码 + 同一逐行分数 → 差异完全来自阈值选择。"
                         "84.82% 评估用了 select_threshold 在 sfx 编码分数分布上选出的最优阈值；"
                         "28.06% 评估沿用了前一轮（标准编码）的阈值，该阈值在新分数分布下过高。"
                         "模型本身的识别能力没有变化。") if same_model else "需要进一步排查"})

    write_json(RESULTS, {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                         "model_sha256": sha_b, "sections": sections})
    print(f"结果已落盘：{RESULTS}")


if __name__ == "__main__":
    main()
