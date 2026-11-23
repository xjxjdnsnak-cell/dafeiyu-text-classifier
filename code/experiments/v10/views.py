"""Label-independent, paired training inputs. Keep mentions intact during punctuation edits."""
import re
import sys
import unicodedata
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE))
from body_mentions import MENTION

# Keep URLs and bracketed Bilibili emotes intact in training views.
PROTECTED = re.compile(r"https?://[^\s]+|\[[^\[\]\n]{1,32}\]|" + MENTION.pattern)
CORNER = str.maketrans("", "", "「」")
TYPOGRAPHY = str.maketrans({"「": "“", "」": "”", "。": ".", "，": ",", "；": ";", "：": ":", "—": "-"})


def outside(text, pattern, edit):
    pieces = []
    end = 0
    for m in pattern.finditer(text):
        pieces.extend((edit(text[end:m.start()]), m[0]))
        end = m.end()
    return "".join([*pieces, edit(text[end:])])


def broad(text, protected=MENTION):
    return outside(text, protected, lambda s: "".join(
        " " if unicodedata.category(c).startswith("P") else c for c in s))


def without_names(text):
    return MENTION.sub(" ", text)


def usable(text):
    return any(c.isalnum() for c in without_names(text))


def training_views(text, index):
    # A row contributes the same four slots, even when a transformation is a no-op.
    if index % 4 < 2:
        punct = broad(text, PROTECTED)
    elif index % 4 == 2:
        punct = outside(text, PROTECTED, lambda s: s.translate(CORNER))
        if punct == text:
            # Wrapping a whole mention can change its parse; bracket body words only.
            punct = outside(text, PROTECTED, lambda s: re.sub(
                r"[\u4e00-\u9fff]{2,6}", lambda m: "「" + m[0] + "」", s, count=1))
    else:
        punct = outside(text, PROTECTED, lambda s: s.translate(TYPOGRAPHY))
    if not usable(punct):
        punct = text

    def toggle(t):
        if MENTION.search(text):
            removed = without_names(t)
            return removed if usable(removed) else t
        return "@小林 " + t if usable(t) else t

    return text, punct, toggle(text), toggle(punct)


def teacher_pair(text):
    stripped = broad(without_names(text), PROTECTED)
    return text, stripped if usable(stripped) else text
