"""Shared boundary for the body-only fastText candidates."""
import re

MENTION = re.compile(r"@([^\s@:：，,。！!?？]{1,24})")


def mask_mentions(text):
    return MENTION.sub("@某人", text)
