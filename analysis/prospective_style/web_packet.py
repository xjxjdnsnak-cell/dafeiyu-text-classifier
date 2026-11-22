"""Build separate offline review pages from the same frozen blind packets as Excel."""

import argparse
import hashlib
import json
from pathlib import Path

from collect import P, read

HERE = Path(__file__).resolve().parent
FORMAT = "blind-review-v1"
DIMS = ("style", "intent", "promotion")


def fingerprint(packet):
    canonical = json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def build(folder, dest):
    template = (HERE / "blind_review.html").read_text(encoding="utf-8")
    dest.mkdir(parents=True, exist_ok=True)
    for who in "AB":
        packet = read(folder / f"packet-{who}.json")
        payload = {
            "who": who,
            "format": FORMAT,
            "fingerprint": fingerprint(packet),
            "packet": packet,
            "choices": [P[d] for d in DIMS],
        }
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "<", r"\u003c"
        )
        page = template.replace("__RATER__", who).replace("__PACKET__", data)
        (dest / f"独立盲评-{who}.html").write_text(page, encoding="utf-8")


def read_web(path, folder, who):
    packet = read(folder / f"packet-{who}.json")
    key = read(folder / f"key-{who}.json")
    data = read(path)
    if (data.get("format"), data.get("who"), data.get("fingerprint")) != (
        FORMAT,
        who,
        fingerprint(packet),
    ):
        raise ValueError("Web review belongs to another packet or reviewer")
    labels = data.get("labels")
    if not isinstance(labels, dict) or set(labels) != set(key):
        raise ValueError("Incomplete web review")
    result = {}
    for code, values in labels.items():
        if not isinstance(values, list) or len(values) != len(DIMS):
            raise ValueError(f"Invalid web review row {code}")
        row = dict(zip(DIMS, values, strict=True))
        if any(value not in P[d] for d, value in row.items()):
            raise ValueError(f"Missing/invalid label in {code}")
        result[key[code]] = row
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    build(args.folder, args.dest)
