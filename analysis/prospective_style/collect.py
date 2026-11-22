"""Freeze a pilot frame from the project's local historical sampling archives."""

import argparse
import csv
import hashlib
import http.client
import json
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "protocol.json").read_bytes())
MENTIONS = re.compile(r"@[^\s@,，:：。!！?？、;；]{1,64}")
PREFIX = re.compile(r"^\s*回复\s*(?=@)")


def canonical(text):
    text = unicodedata.normalize("NFKC", text).casefold()

    def compact(value):
        return re.sub(r"[\W_]", "", value)

    return compact(text), compact(MENTIONS.sub("", PREFIX.sub("", text)))


def read(path):
    return json.loads(path.read_bytes())


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen file differs: {path.name}")
        return
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def history(root, legacy):
    uids, bodies, visible, videos, aids, ids = legacy.old_inventory()
    out = root / "out"

    def add(row):
        for key in ("mid", "uid"):
            if row.get(key):
                uids.add(str(row[key]))
        if row.get("text"):
            bodies.add(legacy.normalize(row["text"]))
        for key, values in [("bvid", videos), ("aid", aids), ("rpid", ids)]:
            if row.get(key):
                values.add(str(row[key]))

    for batch in ("new-video-blind-20261121", "direct-request-blind-20261121"):
        path = out / batch / "private"
        for video in read(path / "completion.json")["videos"]:
            add(video)
            for ref in read(path / (video["bvid"] + ".json"))["refs"]:
                stack = list(read(path / ref)["data"]["replies"] or [])
                while stack:
                    row = stack.pop()
                    add({"mid": row["mid"], "text": row["content"]["message"], "rpid": row["rpid"]})
                    stack.extend(row.get("replies") or [])
    files = [
        out / "address-intent-private" / "annotation.csv",
        out / "address-intent-private" / "train.csv",
        out / "address-intent-private" / "逐条对照.csv",
    ]
    for folder in ("mention-training-results-20261120", "v81-v91-results-20261120"):
        files.extend((out / folder).glob("*scores.csv"))
        files.append(out / folder / "training-ledger.csv")
    for file in files:
        with file.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                add(row)
    keys = {key for value in bodies for key in canonical(value)}
    return uids, keys, videos, aids, ids, visible


def choose(files, old_bvs, old_aids):
    pools = {}
    for file in sorted(files):
        for video in read(file):
            if video["bvid"] not in old_bvs and str(video["aid"]) not in old_aids:
                pools.setdefault(video["pool"], {}).setdefault(video["query"], []).append(video)
    used, owners, selected = set(), set(), []
    for pool, queries in sorted(pools.items()):
        if len(queries) != 6:
            raise ValueError(f"Missing topic strata in {pool}")
        all_options = [v for group in queries.values() for v in group]
        for i, (query, options) in enumerate(sorted(queries.items())):
            sort_key = lambda v: (-v["metadata_replies"], v["bvid"])
            ranked = sorted(options, key=sort_key) + sorted(all_options, key=sort_key)
            for j in range(2):
                v = next(v for v in ranked if v["bvid"] not in used and v["up_mid"] not in owners)
                used.add(v["bvid"])
                owners.add(v["up_mid"])
                selected.append(dict(v, slot=query, arm=("natural", "diagnostic")[(i + j) % 2]))
    if len(selected) != 24:
        raise ValueError("Expected 24 video slots")
    return selected


def reply_page(client, params):
    from collect_comments import FetchError

    for attempt in range(4):
        try:
            return client.get("x/v2/reply", params)
        except (http.client.RemoteDisconnected, FetchError) as error:
            transient = isinstance(error, http.client.RemoteDisconnected) or "重试失败" in str(error)
            if not transient or attempt == 3:
                raise
            time.sleep(4 * (attempt + 1))


def main(root, dest):
    dest.mkdir(exist_ok=True, parents=True)
    freeze(dest / "protocol.json", P)
    archive = root / "out"
    sys.path.insert(0, str(archive / "new-video-blind-20261121/private"))
    import collect_blind as legacy
    from collect_comments import FetchError, normalize_comment

    uids, keys, videos, aids, ids, visible = history(root, legacy)
    selected = choose(legacy.PREVIOUS.glob("search-*.json"), videos, aids)
    freeze(dest / "selection.json", selected)
    print(
        "Locked",
        len(selected),
        "video slots; historical index sizes",
        list(map(len, (videos, aids, uids, keys, ids))),
        flush=True,
    )
    client = legacy.prior.client()
    client.folder = dest
    start, stop = (int(datetime.fromisoformat(P[k]).timestamp()) for k in ("start", "stop"))
    ordinary = re.compile(
        read(root / "fasttext-classifier/data-text-specialist-20261114-v8/protocol.json")[
            "ordinary_nickname_pattern"
        ],
        re.IGNORECASE,
    )
    rows, stats = [], []
    for arm in ("natural", "diagnostic"):
        arm_uids, arm_keys = set(), set()
        for v in (v for v in selected if v["arm"] == arm):
            found, drops, refs = {}, Counter(), []
            try:
                for pn in range(1, P["pages"] + 1):
                    data, ref = reply_page(
                        client, {"oid": v["aid"], "type": 1, "pn": pn, "ps": P["page_size"], "sort": 0}
                    )
                    if not isinstance(data, dict) or "replies" not in data:
                        raise ValueError("Unexpected reply page schema")
                    refs.append(ref)
                    roots = data["replies"] or []
                    stack = list(roots)
                    while stack:
                        raw = stack.pop()
                        row = normalize_comment(raw, v, ref)
                        if row:
                            found.setdefault(row["rpid"], row)
                        stack.extend(raw.get("replies") or [])
                    if len(roots) < P["page_size"]:
                        break
            except FetchError as exc:
                if not any(code in str(exc) for code in ("code=-404", "code=12002", "code=62002")):
                    raise
                if refs:
                    raise
                drops["unavailable_video"] += 1
            kept = 0
            for rid, r in sorted(found.items()):
                full, body = canonical(r["text"])
                visible_body = legacy.normalize(r["text"])[:4096]
                if not start <= (r["ctime"] or 0) < stop:
                    reason = "outside_window"
                elif not full or not body:
                    reason = "empty_key"
                elif r["mid"] in uids:
                    reason = "seen_uid"
                elif full in keys or body in keys or visible_body in visible:
                    reason = "seen_text"
                elif rid in ids:
                    reason = "seen_id"
                else:
                    name = (r["account"].get("uname") or "").strip()
                    short = bool(name and not ordinary.search(name) and 1 <= len(r["text"]) <= 20)
                    rows.append(
                        {
                            "rpid": rid,
                            "bvid": v["bvid"],
                            "uid": r["mid"],
                            "text": r["text"],
                            "pool": v["pool"],
                            "arm": arm,
                            "ordinary_short": short,
                        }
                    )
                    arm_uids.add(r["mid"])
                    arm_keys.update((full, body))
                    ids.add(rid)
                    kept += 1
                    continue
                drops[reason] += 1
            stats.append(
                {
                    "bvid": v["bvid"],
                    "arm": arm,
                    "pool": v["pool"],
                    "found": len(found),
                    "refs": refs,
                    "excluded": dict(drops),
                    "kept": kept,
                }
            )
            print(arm, len(stats), "/24", kept, flush=True)
        uids.update(arm_uids)
        keys.update(arm_keys)
    freeze(dest / "frame.json", rows)
    freeze(dest / "flow.json", stats)
    print("Frozen comments", len(rows), hashlib.sha256((dest / "frame.json").read_bytes()).hexdigest())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace", type=Path)
    ap.add_argument("private_output", type=Path)
    args = ap.parse_args()
    main(args.workspace.resolve(), args.private_output.resolve())
