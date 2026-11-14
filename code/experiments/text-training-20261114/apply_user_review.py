"""User corrections from the false-block reviews.

Two review batches exist; each names specific 误拦 rows as AI. Explicit user AI
labels override gray patterns; pattern hits only remove rows from the human pool
and never promote anything to AI.
"""

import csv
import json
import sys
from pathlib import Path

CLASSIFIER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CLASSIFIER / "experiments/text-specialist-20261113"))

from text_policy import compile_rules, human_exclusion_matches

REVIEW_CSV = CLASSIFIER / "reports-text-specialist-20261114" / "真人误拦复核.csv"
POLICY_PATH = Path(__file__).with_name("human_exclusion_policy.json")
BATCHES = (
    {"review_csv": REVIEW_CSV,
     "ai_items": (5, 8, 10, 11, 12, 15),
     "statement": "5 8 10 11 12 15都是ai",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v2" / "真人误拦复核.csv",
     "ai_items": (2, 3, 5, 10, 13),
     "statement": "2 3 5 10 13都是ai",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v3" / "真人误拦复核.csv",
     "ai_items": (1, 2),
     "statement": "第1、2条是ai（第3条未指认，维持人类）",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v5" / "真人误拦复核-2pct.csv",
     "ai_items": (1, 2, 3, 4, 5, 6, 13, 18, 20),
     "statement": "1 2 3 4 5 6 13 18 20都是ai",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v6" / "真人误拦复核-2pct.csv",
     "ai_items": (1, 2, 3, 5, 8, 9, 11),
     "statement": "1 2 3 5 8 9 11都是ai",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v7" / "真人误拦复核.csv",
     "ai_items": (7,),
     "statement": "7是ai",
     "recorded_on": "2026-11-14"},
    {"review_csv": CLASSIFIER / "reports-text-specialist-20261114-v8" / "真人误拦复核.csv",
     "ai_items": (),
     "statement": "确实是真人误拦（18条全部维持人类）",
     "recorded_on": "2026-11-14"},
)
RULES = compile_rules(json.loads(POLICY_PATH.read_text(encoding="utf-8")))


def load_review_rows(path=None):
    path = Path(path) if path else REVIEW_CSV
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    items = [int(row["序号"]) for row in rows]
    if sorted(items) != list(range(1, len(rows) + 1)):
        raise ValueError(f"误拦复核表序号必须从1连续编号: {path.name}")
    if len({row["样本ID"] for row in rows}) != len(rows):
        raise ValueError(f"误拦复核表样本ID重复: {path.name}")
    return rows


def correction_map():
    """sample_id -> batch metadata for every user-named AI row across batches."""
    result = {}
    for index, batch in enumerate(BATCHES, 1):
        by_item = {int(row["序号"]): row["样本ID"] for row in load_review_rows(batch["review_csv"])}
        for item in batch["ai_items"]:
            sample_id = by_item[item]
            if sample_id in result:
                raise ValueError(f"样本在多个复核批次中出现: {sample_id}")
            result[sample_id] = {"batch": index, "item": item,
                                 "statement": batch["statement"], "recorded_on": batch["recorded_on"]}
    return result


def reviewed_sample_ids():
    """Every row the user has seen and ruled on, AI-named or not."""
    result = set()
    for batch in BATCHES:
        result.update(row["样本ID"] for row in load_review_rows(batch["review_csv"]))
    return result


def review_csvs():
    return [batch["review_csv"] for batch in BATCHES]


def apply_corrections(rows, corrections):
    """Return (kept, gray). Corrections promote named rows to AI; other human rows
    matching the user's gray expressions only move to gray, keeping label 0."""
    by_id = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id in by_id:
            raise ValueError(f"rows 内 sample_id 重复: {sample_id}")
        by_id[sample_id] = row
    unknown = [sample_id for sample_id in corrections if sample_id not in by_id]
    if unknown:
        raise ValueError(f"复核样本ID不存在: {sorted(unknown)[:3]}")
    kept, gray = [], []
    for original in rows:
        row = dict(original)
        sample_id = row.get("sample_id")
        label = row.get("label")
        if sample_id in corrections:
            detail = corrections[sample_id]
            detail = detail if isinstance(detail, dict) else {"item": detail, "statement": BATCHES[0]["statement"],
                                                              "recorded_on": BATCHES[0]["recorded_on"]}
            row["previous_effective_label"] = label
            row["label"] = "1" if isinstance(label, str) else 1
            row["user_correction"] = {"decision": "AI", "reviewer": "user", **detail}
            kept.append(row)
        elif str(label) == "0" and human_exclusion_matches(row.get("text", ""), rules=RULES):
            row["human_exclusion_matches"] = human_exclusion_matches(row.get("text", ""), rules=RULES)
            row["exclusion_kind"] = "user_gray_phrase"
            gray.append(row)
        else:
            kept.append(row)
    return kept, gray
