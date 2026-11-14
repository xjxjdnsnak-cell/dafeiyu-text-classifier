"""User-specified text eligibility; gray human exclusions are never AI labels."""

import json
import re
import unicodedata
from pathlib import Path

from bili_ai.preprocess import normalize

MENTION = re.compile(r"@[^\s:：,，。!！?？\[\]【】]+")
POLICY_PATH = Path(__file__).with_name("human_exclusion_policy.json")
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
EMOTES_PATH = Path(__file__).with_name("known_bilibili_emotes.json")
KNOWN_EMOTES = {normalize(value) for value in json.loads(EMOTES_PATH.read_text(encoding="utf-8"))["tokens"]}


def _compact(text):
    return "".join(ch for ch in text if not ch.isspace()
                   and not unicodedata.category(ch).startswith("P"))


def compile_rules(policy):
    return [(phrase, re.compile(policy["pattern_overrides"][phrase], re.I)
             if phrase in policy["pattern_overrides"] else _compact(normalize(phrase)))
            for phrase in dict.fromkeys(policy["source_phrases"])]


_RULES = compile_rules(POLICY)


def is_mention_only(text):
    text = normalize(text)
    if not MENTION.search(text):
        return False
    text = re.sub(r"^回复\s*(?=@)", "", text)
    remaining = MENTION.sub("", text)
    remaining = re.sub(r"\[[^\]]*\]", lambda match: "" if match[0] in KNOWN_EMOTES else match[0], remaining)
    return not any(ch.isalnum() for ch in remaining)


def human_exclusion_matches(text, rules=None):
    body = MENTION.sub("", normalize(text))
    compact = _compact(body)
    return [phrase for phrase, rule in (_RULES if rules is None else rules)
            if (rule in compact if isinstance(rule, str) else rule.search(body))]


def apply_text_policy(row):
    result = dict(row)
    text = row.get("text", "")
    mention_only = is_mention_only(text)
    matches = human_exclusion_matches(text)
    original = row.get("source_weak_comment_label", row.get("weak_comment_label", "review"))
    result.update(source_weak_comment_label=original,
        text_model_eligible=not mention_only,
        text_exclusion_reason="mention_only" if mention_only else "",
        human_label_eligible=not mention_only and not matches,
        human_exclusion_matches=matches)
    result["weak_comment_label"] = "review" if mention_only or (matches and original == "human") else original
    return result


def filter_labeled_rows(rows):
    kept, excluded = [], []
    for row in rows:
        if row["label"] not in ("0", "1"):
            raise ValueError("只过滤现有二分类标签，不为未知来源补标签")
        matches = human_exclusion_matches(row["text"])
        kind = "mention_only" if is_mention_only(row["text"]) else "human_expression" if row["label"] == "0" and matches else ""
        if kind:
            excluded.append({**row, "exclusion_kind": kind, "matched_expressions": matches})
        else:
            kept.append(dict(row))
    return kept, excluded
