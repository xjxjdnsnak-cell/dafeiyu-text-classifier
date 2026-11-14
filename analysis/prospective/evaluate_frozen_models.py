"""Evaluate v5..v8 on the FROZEN prospective benchmark. No tuning of any kind.

Each model is scored with its own deployed configuration: model.bin + the 1%
validation threshold stored in its metadata, plus the 2% working point from its
round's selection-frozen-before-test.json. The benchmark comments.csv hash is
verified against the manifest, and disjointness against the labeled corpus is
re-asserted before scoring.
"""

import datetime
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER = ROOT / "fasttext-classifier"
ROOT = CLASSIFIER.parent
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

from bili_ai.common import sha256, read_csv, write_json
from bili_ai.metrics import binary_metrics
from bili_ai.model import load_model, positive_scores
from bili_ai.preprocess import Preprocessor, normalize

HERE = Path(__file__).resolve().parent
BENCH = HERE / "benchmark"
SOURCE = CLASSIFIER / "data-rule-ge30-20261114-r2"
ROUNDS = {"v5": "models-text-specialist-20261114-v5", "v6": "models-text-specialist-20261114-v6",
          "v7": "models-text-specialist-20261114-v7", "v8": "models-text-specialist-20261114-v8"}
REPORT = HERE / "evaluation.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def verify_disjointness(rows):
    corpus_uids, corpus_texts, corpus_videos = set(), set(), set()
    for row in load_jsonl(SOURCE / "label-decisions.jsonl"):
        corpus_uids.add(str(row["uid"]))
        corpus_texts.add(normalize(row["text"]))
        for source in row.get("sources", []):
            if source.get("bvid"):
                corpus_videos.add(source["bvid"])
    hits = Counter()
    for row in rows:
        if str(row["uid"]) in corpus_uids:
            hits["uid"] += 1
        if normalize(row["text"]) in corpus_texts:
            hits["text"] += 1
        if row["bvid"] in corpus_videos:
            hits["video"] += 1
    if hits:
        raise ValueError(f"基准集与训练语料出现交集: {dict(hits)}")
    return {"uids": len(corpus_uids), "texts": len(corpus_texts), "videos": len(corpus_videos)}


def subgroup(indices, y, scores, threshold):
    metrics = binary_metrics([y[i] for i in indices], [scores[i] for i in indices], threshold)
    metrics["ai_rows"] = sum(y[i] for i in indices)
    metrics["human_rows"] = len(indices) - metrics["ai_rows"]
    if not metrics["ai_rows"]:
        metrics["recall"] = None
    return metrics


def main():
    manifest = json.loads((BENCH / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("frozen"):
        raise ValueError("基准集未冻结")
    rows = read_csv(BENCH / "comments.csv", ("uid", "username", "text", "label", "score", "bvid", "pool"))
    digest = sha256(BENCH / "comments.csv")
    if digest != manifest["comments_sha256"]:
        raise ValueError("基准集文件与冻结清单哈希不一致")
    corpus_info = verify_disjointness(rows)
    protocol = json.loads((CLASSIFIER / "reports-text-fusion-20261113-r3/protocol.json").read_text(encoding="utf-8"))
    hints = re.compile(protocol["ordinary_nickname_pattern"], re.I)
    pre = Preprocessor()
    features = [pre.encode(r["text"], "", "text") for r in rows]
    y = [int(r["label"]) for r in rows]
    ordinary = [not hints.search(r["username"] or "") for i, r in enumerate(rows)]
    short = [len(r["text"]) <= 20 for r in rows]
    idx_all = list(range(len(rows)))
    idx_ordinary_ai = [i for i in idx_all if ordinary[i] and y[i] == 1]
    idx_short_ai = [i for i in idx_all if short[i] and y[i] == 1]
    idx_short_human = [i for i in idx_all if short[i] and y[i] == 0]
    idx_len_human = {"1_20": [i for i in idx_all if y[i] == 0 and len(rows[i]["text"]) <= 20],
                     "21_60": [i for i in idx_all if y[i] == 0 and 21 <= len(rows[i]["text"]) <= 60],
                     "61p": [i for i in idx_all if y[i] == 0 and len(rows[i]["text"]) >= 61]}

    rounds = {}
    for tag, models_dir in ROUNDS.items():
        base = CLASSIFIER / models_dir
        metadata = json.loads((base / "text/model.bin.json").read_text(encoding="utf-8"))
        frozen = json.loads((base / "selection-frozen-before-test.json").read_text(encoding="utf-8"))
        if frozen["recommended_format"] != "bin":
            raise ValueError(f"{tag}: 推荐格式不是 bin，评估脚本按 bin 部署配置编写")
        thr1 = metadata["threshold"]
        thr2 = frozen["models"]["selected_original"]["working_points"]["0.02"]["threshold"]
        model = load_model(base / "text/model.bin")
        scores = []
        for start in range(0, len(features), 4000):
            scores.extend(positive_scores(model, features[start:start + 4000]))
        del model
        entry = {"model": str(base / "text/model.bin"), "model_sha256": sha256(base / "text/model.bin"),
                 "thresholds": {"fpr_1pct": thr1, "fpr_2pct": thr2},
                 "trial": frozen["selected_trial"], "source_of_thresholds": "validation (frozen before each round's test); unchanged here"}
        for cap, threshold in (("1pct", thr1), ("2pct", thr2)):
            entry[cap] = {
                "all": binary_metrics(y, scores, threshold),
                "ordinary_ai_recall": subgroup(idx_ordinary_ai, y, scores, threshold),
                "short_ai_recall": subgroup(idx_short_ai, y, scores, threshold),
                "short_human_fpr": subgroup(idx_short_human, y, scores, threshold),
                "length_1_20_human_fpr": subgroup(idx_len_human["1_20"], y, scores, threshold),
                "length_21_60_human_fpr": subgroup(idx_len_human["21_60"], y, scores, threshold),
                "length_61p_human_fpr": subgroup(idx_len_human["61p"], y, scores, threshold),
            }
        rounds[tag] = entry
        print(json.dumps({"round": tag, "thr1": thr1, "thr2": thr2,
                          "recall_1pct": entry["1pct"]["all"]["recall"],
                          "fpr_1pct": entry["1pct"]["all"]["fpr"],
                          "ordinary_1pct": entry["1pct"]["ordinary_ai_recall"]["recall"],
                          "short_1pct": entry["1pct"]["short_ai_recall"]["recall"]}, ensure_ascii=False), flush=True)

    result = {"benchmark": {"comments_sha256": digest, "rows": len(rows),
                            "labels": dict(Counter(str(v) for v in y)),
                            "videos": manifest["videos"], "uids": manifest["uids"],
                            "frozen_at": manifest["frozen_at"],
                            "ordinary_ai_rows": sum(ordinary[i] and y[i] == 1 for i in range(len(rows))),
                            "short_ai_rows": sum(short[i] and y[i] == 1 for i in range(len(rows))),
                            "disjointness_recheck": corpus_info,
                            "manifest_sha256": sha256(BENCH / "manifest.json")},
              "protocol": {"thresholds": "each model's own frozen validation thresholds (1% cap deployed default, 2% secondary); no re-tuning, no test-based selection",
                           "labels": manifest["policy"],
                           "frozen_at": manifest["frozen_at"]},
              "evaluated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "rounds": rounds}
    write_json(REPORT, result)
    print(json.dumps({"report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
