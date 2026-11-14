#!/usr/bin/env python3
"""User-selected BiliBili videos -> resumable, traceable weak-label fastText corpus."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
RULE_VERSION = "liyupi/dafeiyu-cleaner@32866038316f81d77557f20a213b8c3723bf9ce0"
API = "https://api.bilibili.com/"
BV_RE = re.compile(r"BV[1-9A-HJ-NP-Za-km-z]{10}")
RISK_CODES = {-352, -412, -509, -101, 12002}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_videos(text: str):
    videos = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in re.split(r"[\s,，;；]+", line):
            if not token:
                continue
            if BV_RE.fullmatch(token):
                bv = token
            else:
                parsed = urllib.parse.urlsplit(token)
                if parsed.scheme not in ("https", "http") or parsed.hostname not in (
                    "www.bilibili.com", "bilibili.com", "m.bilibili.com"
                ):
                    raise ValueError(f"请提供 BV 号或 bilibili.com/video/BV… 链接：{token}")
                match = re.fullmatch(r"/video/(BV[1-9A-HJ-NP-Za-km-z]{10})/?", parsed.path)
                if not match:
                    raise ValueError(f"无法识别的视频链接：{token}")
                bv = match[1]
            if bv not in videos:
                videos.append(bv)
    if not videos:
        raise ValueError("没有读到 BV 号")
    return videos


class Blocked(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


class Client:
    def __init__(self, folder: Path, delay: float, cookie: str = ""):
        self.folder = folder
        self.delay = delay
        self.last = 0.0
        self.requests = 0
        self.cached = 0
        self.headers = {
            "User-Agent": "Mozilla/5.0 BiliPaiDafeiyuCollector/1.0",
            "Referer": "https://www.bilibili.com/",
        }
        if cookie:
            self.headers["Cookie"] = cookie
        self.auth_mode = "authenticated" if cookie else "guest"

    def wait(self):
        remaining = self.delay - (time.monotonic() - self.last)
        if remaining > 0:
            time.sleep(remaining)
        self.last = time.monotonic()

    def get(self, endpoint, params):
        url = API + endpoint + "?" + urllib.parse.urlencode(params)
        # Authenticated reads must not reuse a truncated guest page.
        key = hashlib.sha256((self.auth_mode + ":" + url).encode()).hexdigest()
        path = self.folder / "raw" / (key + ".json")
        if path.exists():
            payload = json.loads(path.read_bytes())
            if payload.get("code") == 0:
                self.cached += 1
                return payload.get("data"), str(path.relative_to(self.folder))
        for attempt in range(3):
            self.wait()
            try:
                req = urllib.request.Request(url, headers=self.headers)
                with urllib.request.urlopen(req, timeout=25) as response:
                    body = response.read(16 * 1024 * 1024)
                self.requests += 1
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
                dump(path.with_suffix(".meta.json"), {"url": url, "fetched_at": now(), "authentication": self.auth_mode})
                payload = json.loads(body)
                code = payload.get("code")
                if code in RISK_CODES:
                    raise Blocked(f"接口要求登录或限制访问（code={code}）；已保存断点，不继续请求。")
                if code != 0:
                    raise FetchError(f"API code={code}: {payload.get('message', '')}")
                return payload.get("data"), str(path.relative_to(self.folder))
            except urllib.error.HTTPError as exc:
                if exc.code in (401, 403, 412, 429):
                    raise Blocked(f"HTTP {exc.code}，已停止请求并保留断点。") from exc
                if exc.code < 500:
                    raise FetchError(f"HTTP {exc.code}") from exc
                if attempt == 2:
                    raise FetchError(f"HTTP {exc.code}，重试后仍失败") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                if attempt == 2:
                    raise FetchError(f"请求重试失败：{type(exc).__name__}") from exc
            time.sleep(2 ** (attempt + 1))


def db_open(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / "collector.sqlite3")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS comments(rpid TEXT PRIMARY KEY, bvid TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS profiles(mid TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS videos(bvid TEXT PRIMARY KEY, payload TEXT NOT NULL);
    """)
    return db


def as_int(value):
    try:
        return int(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def account_from(member, profile=False):
    member = member or {}
    vip = member.get("vip") or {}
    sign = member.get("sign")
    return {
        "uname": member.get("name" if profile else "uname"),
        "sign": sign or "",
        # Empty signatures in comment payloads may be placeholders. Verify public card before negatives.
        "signatureKnown": isinstance(sign, str) and (profile or bool(sign.strip())),
        "level": as_int((member.get("level_info") or {}).get("current_level")),
        "verify": as_int((member.get("official_verify") or {}).get("type")),
        "vip": as_int(vip.get("vipType", vip.get("type"))),
        "avatar_url": member.get("face" if profile else "avatar"),
    }


def normalize_comment(reply, video, raw_ref, inherited_root="0"):
    rid = str(reply.get("rpid_str") or reply.get("rpid") or "0")
    mid = str(reply.get("mid_str") or reply.get("mid") or (reply.get("member") or {}).get("mid") or "0")
    if rid == "0" or mid == "0":
        return None
    root = str(reply.get("root_str") or reply.get("root") or inherited_root)
    return {
        "bvid": video["bvid"], "aid": str(video["aid"]), "up_mid": str(video["up_mid"]),
        "video_title": video["title"], "rpid": rid, "root": root,
        "parent": str(reply.get("parent_str") or reply.get("parent") or "0"),
        "mid": mid, "text": str((reply.get("content") or {}).get("message") or ""),
        "ctime": as_int(reply.get("ctime")), "collected_at": now(),
        "like": as_int(reply.get("like")), "reply_count": as_int(reply.get("rcount")) or 0,
        "account": {"mid": mid, **account_from(reply.get("member"))},
        "url": f"https://www.bilibili.com/video/{video['bvid']}/#reply{rid}",
        "raw_ref": raw_ref,
    }


def store_replies(db, replies, video, raw_ref, limit, inherited_root="0"):
    added = 0
    for reply in replies:
        if not isinstance(reply, dict):
            continue
        row = normalize_comment(reply, video, raw_ref, inherited_root)
        if row:
            if db.execute("SELECT 1 FROM comments WHERE rpid=?", (row["rpid"],)).fetchone() is None:
                if db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] >= limit:
                    break
                db.execute("INSERT INTO comments VALUES(?,?,?)", (row["rpid"], row["bvid"], json.dumps(row, ensure_ascii=False)))
                added += 1
            added += store_replies(db, reply.get("replies") or [], video, raw_ref, limit, row["rpid"] if row["root"] == "0" else row["root"])
    db.commit()
    return added


def roots_from(data):
    by_id = {}
    top = data.get("top") or {}
    top_rows = list(top.values()) if isinstance(top, dict) else []
    for row in (data.get("replies") or []) + (data.get("hots") or []) + (data.get("top_replies") or []) + top_rows:
        if isinstance(row, dict) and (row.get("rpid_str") or row.get("rpid")):
            by_id[str(row.get("rpid_str") or row["rpid"])] = row
    return list(by_id.values())


def collect_video(client, db, bv, args):
    data, raw_ref = client.get("x/web-interface/view", {"bvid": bv})
    if not isinstance(data, dict) or not data.get("aid"):
        raise FetchError("视频接口没有返回 aid")
    video = {"bvid": bv, "aid": str(data["aid"]), "title": data.get("title", ""),
             "up_mid": str((data.get("owner") or {}).get("mid", "0")),
             "reported_total": (data.get("stat") or {}).get("reply"), "raw_ref": raw_ref,
             "root_pages": 0, "child_pages": 0, "roots_seen": 0, "stop_reasons": [], "fetched_at": now()}
    previous = db.execute("SELECT payload FROM videos WHERE bvid=?", (bv,)).fetchone()
    previous = json.loads(previous[0]) if previous else {}
    def save():
        video["observed_comments"] = db.execute("SELECT COUNT(*) FROM comments WHERE bvid=?", (bv,)).fetchone()[0]
        for key in ("root_pages", "child_pages", "roots_seen"):
            video[key] = max(video[key], previous.get(key, 0))
        db.execute("INSERT OR REPLACE INTO videos VALUES(?,?)", (bv, json.dumps(video, ensure_ascii=False)))
        db.commit()
    seen_roots = set()
    other_count = db.execute("SELECT COUNT(*) FROM comments WHERE bvid<>?", (bv,)).fetchone()[0]
    video_limit = min(args.max_comments, other_count + args.per_video_comments)
    try:
        for page in range(1, args.root_pages + 1):
            payload, raw_ref = client.get("x/v2/reply", {"oid": video["aid"], "type": 1, "pn": page, "ps": 20, "sort": args.sort})
            if not isinstance(payload, dict):
                raise FetchError("评论数据结构异常")
            video["root_pages"] = page
            rows = roots_from(payload)
            ids = {str(r.get("rpid_str") or r.get("rpid")) for r in rows}
            primary = payload.get("replies") or []
            if not primary and not ids.difference(seen_roots):
                video["stop_reasons"].append("server_empty_page_coverage_unknown")
                break
            if ids and not ids.difference(seen_roots):
                video["stop_reasons"].append("repeated_page_no_new_root")
                break
            fresh = [r for r in rows if str(r.get("rpid_str") or r.get("rpid")) not in seen_roots]
            seen_roots.update(ids)
            video["roots_seen"] = len(seen_roots)
            store_replies(db, rows, video, raw_ref, video_limit)
            for root in fresh:
                root_id = str(root.get("rpid_str") or root.get("rpid"))
                count = as_int(root.get("rcount")) or 0
                if count <= len(root.get("replies") or []):
                    continue
                seen_children = set()
                for child_page in range(1, min(args.child_pages, math.ceil(count / 20)) + 1):
                    if db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] >= video_limit:
                        break
                    sub, sub_ref = client.get("x/v2/reply/reply", {"oid": video["aid"], "type": 1, "root": root_id, "pn": child_page, "ps": 20})
                    video["child_pages"] += 1
                    children = (sub or {}).get("replies") or []
                    child_ids = {str(r.get("rpid_str") or r.get("rpid")) for r in children}
                    if not children or not child_ids.difference(seen_children):
                        video["stop_reasons"].append(f"child_empty_or_repeated:{root_id}:{child_page}")
                        break
                    seen_children.update(child_ids)
                    store_replies(db, children, video, sub_ref, video_limit, root_id)
                if count > args.child_pages * 20:
                    video["stop_reasons"].append(f"child_page_limit:{root_id}")
            save()
            print(f"  {bv} 根列表 {page} 页，已保存 {video['observed_comments']} 条（含楼中楼）", flush=True)
            if db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] >= video_limit:
                video["stop_reasons"].append("per_video_or_total_comment_limit")
                break
        else:
            video["stop_reasons"].append("root_page_limit")
    except (FetchError, Blocked, KeyboardInterrupt) as exc:
        video["stop_reasons"].append(type(exc).__name__ + ":" + str(exc))
        raise
    finally:
        save()
    return video


def score_rows(rows, profiles):
    node = shutil.which("node")
    if not node:
        raise RuntimeError("未找到 Node.js，请安装 Node.js 后重试")
    run = subprocess.run([node, str(HERE / "score_comments.cjs")],
                         input=json.dumps({"comments": rows, "profiles": profiles}, ensure_ascii=False),
                         encoding="utf-8", capture_output=True, check=False)
    if run.returncode:
        raise RuntimeError("评分失败：" + run.stderr[-1000:])
    return json.loads(run.stdout)


def blue_ratio(body):
    from PIL import Image
    with Image.open(io.BytesIO(body)) as image:
        pixels = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
        blue = 0
        for rv, gv, bv in pixels.getdata():
            r, g, b = rv / 255, gv / 255, bv / 255
            mx, mn = max(r, g, b), min(r, g, b)
            if mx == 0 or (mx - mn) / mx <= 0.35 or mx <= 0.18 or mx != b:
                continue
            h = 60 * (4 + (r - g) / (mx - mn))
            if 205 <= h <= 250:
                blue += 1
        return blue / 1024


def avatar_for(url, folder):
    if not url:
        return None
    parsed = urllib.parse.urlsplit(url.split("@", 1)[0])
    if parsed.scheme not in ("http", "https") or not (parsed.hostname or "").endswith(".hdslb.com"):
        return None
    normalized = urllib.parse.urlunsplit(("https", parsed.netloc, parsed.path, "", "")) + "@48w_48h.png"
    path = folder / "avatars" / (hashlib.sha256(normalized.encode()).hexdigest() + ".png")
    try:
        if not path.exists():
            # CDN images use their own short delay; no cookies are sent to image hosts.
            time.sleep(0.15)
            with urllib.request.urlopen(urllib.request.Request(normalized, headers={"User-Agent": "Mozilla/5.0"}), timeout=10) as response:
                body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return blue_ratio(path.read_bytes())
    except (OSError, ValueError, ImportError, urllib.error.URLError):
        return None


def enrich(client, db, rows, args):
    profiles = {mid: json.loads(payload) for mid, payload in db.execute("SELECT mid,payload FROM profiles")}
    initial = score_rows(rows, profiles)
    by_uid = {}
    for row in initial:
        by_uid.setdefault(row["mid"], row)
    requested = 0
    # Negatives need a verified (possibly empty) signature; positives >=80 already satisfy the threshold.
    candidates = sorted(by_uid.values(), key=lambda r: (r["uid_max_score"] >= args.ai_min, r["uid_max_score"]))
    for row in candidates:
        mid = row["mid"]
        profile = profiles.setdefault(mid, {})
        if "signature" not in row["missing_features"]:
            continue
        if row["score"] >= args.ai_min and not any(k in row["missing_features"] for k in ("uname", "level", "verify", "vip")):
            continue
        if requested >= args.profile_limit:
            break
        try:
            data, ref = client.get("x/web-interface/card", {"mid": mid})
            requested += 1
            card = (data or {}).get("card")
            if isinstance(card, dict) and str(card.get("mid")) == mid:
                fields = account_from(card, profile=True)
                profile.update({k: v for k, v in fields.items() if v is not None})
                profile["profile_raw_ref"] = ref
                db.execute("INSERT OR REPLACE INTO profiles VALUES(?,?)", (mid, json.dumps(profile, ensure_ascii=False)))
                db.commit()
        except FetchError as exc:
            requested += 1
            print(f"  UID {mid} 资料跳过：{exc}", flush=True)
        if requested % 25 == 0:
            print(f"  已核对 {requested} 个公开账号资料", flush=True)
    if not args.no_avatars:
        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            print("  未安装 Pillow：跳过头像，正常样本按缺失头像的最高加分保护。", flush=True)
            return profiles
        rescored = score_rows(rows, profiles)
        pending = {}
        for row in rescored:
            if "avatar" not in row["missing_features"]:
                continue
            # Avatars only add points. Fetch when they can change either admission threshold.
            if (row["score"] < args.ai_min <= row["score"] + 22 or
                    row["score"] < args.human_max <= row["score"] + 22):
                pending.setdefault(row["mid"], row["account"].get("avatar_url"))
        for index, (mid, url) in enumerate(list(pending.items())[:args.avatar_limit], 1):
            ratio = avatar_for(url, client.folder)
            if ratio is not None:
                profile = profiles.setdefault(mid, {})
                profile["avatarBlue"] = ratio
                db.execute("INSERT OR REPLACE INTO profiles VALUES(?,?)", (mid, json.dumps(profile, ensure_ascii=False)))
                db.commit()
            if index % 50 == 0:
                print(f"  已分析 {index} 个可能改变分档的头像", flush=True)
    return profiles


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = "".join("" if unicodedata.category(c) == "Cf" else
                   " " if unicodedata.category(c).startswith("C") else c for c in text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace("__label__", "label:")


def classify(row, ai_min=80, human_max=30):
    if not normalize_text(row["text"]):
        return "review", "empty_text"
    if any(k in row["missing_features"] for k in ("uname", "level", "verify", "vip")):
        return "review", "missing_account_metadata"
    if row["score"] >= ai_min:
        return "ai", "comment_score_at_least_ai_min"
    if row["score"] < human_max:
        if row.get("text_score", 0) >= human_max or any(
            r.get("tag") in ("name", "sign") and r.get("w", 0) >= 25 for r in row.get("reasons", [])
        ):
            return "review", "strong_ai_features_masked_by_deductions"
        if row["uid_max_score"] >= human_max:
            return "review", "same_uid_has_suspicious_comments"
        if row["score_upper_bound"] >= human_max or row["uid_score_upper_bound"] >= human_max:
            return "review", "missing_features_can_cross_human_threshold"
        return "human", "comment_and_uid_upper_bound_below_human_max"
    return "review", "middle_score"


def export_dataset(db, folder, args, run_status):
    raw_rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM comments ORDER BY bvid,rpid")]
    profiles = {mid: json.loads(payload) for mid, payload in db.execute("SELECT mid,payload FROM profiles")}
    rows = score_rows(raw_rows, profiles)
    for row in rows:
        row["label"], row["selection_reason"] = classify(row, args.ai_min, args.human_max)
        row["label_source"] = "offline_rules_weak_label"
        row["rule_version"] = RULE_VERSION
        row["training_text"] = normalize_text(row["text"])
    groups = defaultdict(list)
    for row in rows:
        if row["label"] != "review":
            groups[row["training_text"]].append(row)
    train = []
    conflicts = []
    duplicates = 0
    for text, group in sorted(groups.items()):
        labels = {r["label"] for r in group}
        if len(labels) > 1:
            for row in group:
                row["label"] = "review"
                row["selection_reason"] = "same_text_conflicting_labels"
                conflicts.append(row)
            continue
        train.append((group[0]["label"], text))
        duplicates += len(group) - 1
    target = folder / "export"
    target.mkdir(parents=True, exist_ok=True)
    prior_summary_path = target / "summary.json"
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8")) if prior_summary_path.exists() else {}
    def jsonl(name, records):
        with (target / name).open("w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    jsonl("all_scored.jsonl", rows)
    for label in ("ai", "human", "review"):
        jsonl(label + ".jsonl", (r for r in rows if r["label"] == label))
    jsonl("conflicts.jsonl", conflicts)
    with (target / "fasttext_train.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for label, text in train:
            stream.write(f"__label__{label} {text}\n")
    # Character segmentation can merge texts that differed only in word spacing.
    char_groups = defaultdict(set)
    for label, text in train:
        char_groups[" ".join(c for c in text if not c.isspace())].add(label)
    char_train = [(next(iter(labels)), text) for text, labels in sorted(char_groups.items()) if len(labels) == 1]
    with (target / "fasttext_train_char.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for label, text in char_train:
            stream.write(f"__label__{label} {text}\n")
    videos = [json.loads(r[0]) for r in db.execute("SELECT payload FROM videos ORDER BY bvid")]
    run_config = folder / "run_config.json"
    requested_videos = json.loads(run_config.read_text(encoding="utf-8")).get("videos", []) if run_config.exists() else []
    summary = {
        "created_at": now(), "status": run_status, "rule_version": RULE_VERSION,
        "source_collection_status": prior_summary.get("source_collection_status", prior_summary.get("status"))
            if run_status == "export_only_cached_snapshot" else run_status,
        "rule_sha256": hashlib.sha256((HERE / "upstream-detector.js").read_bytes()).hexdigest(),
        "thresholds": {"ai_min_inclusive": args.ai_min, "human_max_exclusive": args.human_max},
        "comments": len(rows), "unique_uids": len({r["mid"] for r in rows}),
        "record_labels": dict(Counter(r["label"] for r in rows)),
        "training_unique_texts": dict(Counter(label for label, _ in train)),
        "training_char_unique_texts": dict(Counter(label for label, _ in char_train)),
        "char_text_conflict_groups_removed": sum(len(labels) > 1 for labels in char_groups.values()),
        "duplicate_training_texts_removed": duplicates, "conflicting_records": len(conflicts),
        "review_reasons": dict(Counter(r["selection_reason"] for r in rows if r["label"] == "review")),
        "missing_features": dict(Counter(k for r in rows for k in r["missing_features"])),
        "history_boost": 0, "videos": videos, "requested_videos": requested_videos,
        "videos_with_comments": len({r["bvid"] for r in rows}),
        "unattempted_videos": [bv for bv in requested_videos if bv not in {v["bvid"] for v in videos}],
        "caveats": ["规则分数不是 AI 概率；human 是低风险弱标签，仍可能有漏网 AI。",
                    "游客接口可能截断根评论，页数和条数上限也会截断；不声称完整评论集。",
                    "按单条评论评分录入 AI；正常还要求同 UID 的聚合风险上限 <30。",
                    "资料缺失时跳过该规则；正常候选按缺失签名+40、头像+22的上界保护。",
                    "强文本/昵称/签名特征被认证、会员或UP主减分压低时，暂不标正常。",
                    "训练集按规范化正文去重，跨标签同文排除；AI/正常条数未人为配平。",
                    "请按 UID 分组划分训练/验证集，避免同一账号的表达风格泄漏。"],
    }
    dump(target / "summary.json", summary)
    (target / "README.txt").write_text(
        "大肥鱼评论数据集（自动规则弱标注）\n"
        f"AI: 单条规则分数 >= {args.ai_min}；human: 单条分数 < {args.human_max}，且账号风险/缺失特征上界也 < {args.human_max}。\n"
        "分数不是 AI 概率。请抽样复核，尤其是普通讨论 AI / 大肥鱼的真人评论。\n"
        "fasttext_train.txt 是规范化原文；fasttext_train_char.txt 额外按单字分词，适合先做中文基线。\n"
        "all_scored.jsonl 保留全部原文、来源、评分原因和缺失资料；ai/human/review 为对应子集。\n"
        "训练文本已去重，矛盾标签进入 conflicts.jsonl，不写入训练集。\n"
        "采集范围见 summary.json。raw/、collector.sqlite3、avatars/ 支持回溯和断点续采。\n"
        "规则来源：https://github.com/liyupi/dafeiyu-cleaner，固定提交 32866038316f81d77557f20a213b8c3723bf9ce0，MIT。\n",
        encoding="utf-8")
    checksums = []
    for path in sorted(target.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (target / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("comments", "unique_uids", "record_labels", "training_unique_texts")}, ensure_ascii=False), flush=True)
    print("导出目录：" + str(target.resolve()), flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="*", help="BV 号或完整视频链接，可多个")
    p.add_argument("--input", type=Path, help="UTF-8 视频清单")
    p.add_argument("--output", type=Path, default=HERE / "dataset")
    p.add_argument("--root-pages", type=int, default=10, help="每视频最多根列表页数，默认10")
    p.add_argument("--child-pages", type=int, default=3, help="每楼最多楼中楼页数，默认3")
    p.add_argument("--max-comments", type=int, default=5000, help="库内原始评论总上限")
    p.add_argument("--per-video-comments", type=int, default=300, help="每视频原始评论上限，避免首个视频占满整批")
    p.add_argument("--profile-limit", type=int, default=500, help="每次最多核对的公开资料数")
    p.add_argument("--avatar-limit", type=int, default=500)
    p.add_argument("--delay", type=float, default=1.2, help="API 请求最短间隔秒数")
    p.add_argument("--sort", type=int, choices=(0, 1, 2), default=2, help="0时间/1点赞/2回复数；游客可用范围有限")
    p.add_argument("--ai-min", type=int, default=80)
    p.add_argument("--human-max", type=int, default=30, help="严格小于，默认30")
    p.add_argument("--no-avatars", action="store_true")
    p.add_argument("--export-only", action="store_true", help="不请求网络，按现有缓存重新导出")
    p.add_argument("--cookie-file", type=Path, help="可选：用户自行提供的 Cookie 文本；不保存、不输出")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not (0 < args.human_max < args.ai_min <= 99):
        raise ValueError("需要 0 < human-max < ai-min <= 99")
    if min(args.root_pages, args.child_pages, args.max_comments, args.per_video_comments) < 1 or min(args.profile_limit, args.avatar_limit) < 0:
        raise ValueError("页数、条数需为正数；资料/头像限额不能为负")
    if args.delay < 0.5:
        raise ValueError("API 间隔不能低于0.5秒")
    folder = args.output.resolve()
    db = db_open(folder)
    status = "export_only_cached_snapshot" if args.export_only else "bounded_collection_finished"
    code = 0
    failure_message = None
    try:
        if not args.export_only:
            text = "\n".join(args.videos)
            if args.input:
                text += "\n" + args.input.read_text(encoding="utf-8-sig")
            videos = read_videos(text)
            cookie = args.cookie_file.read_text(encoding="utf-8-sig").strip() if args.cookie_file else os.environ.get("BILIBILI_COOKIE", "").strip()
            client = Client(folder, args.delay, cookie)
            dump(folder / "run_config.json", {"videos": videos, "root_pages": args.root_pages,
                 "child_pages": args.child_pages, "max_comments": args.max_comments,
                 "per_video_comments": args.per_video_comments,
                 "profile_limit": args.profile_limit, "avatar_limit": args.avatar_limit,
                 "ai_min": args.ai_min, "human_max": args.human_max, "started_at": now(),
                 "authentication": "user_supplied_cookie" if cookie else "guest"})
            for index, bv in enumerate(videos, 1):
                print(f"[{index}/{len(videos)}] {bv}", flush=True)
                try:
                    collect_video(client, db, bv, args)
                except FetchError as exc:
                    print(f"  本视频未完成：{exc}", flush=True)
                    status = "partial_api_errors"
                    previous = db.execute("SELECT payload FROM videos WHERE bvid=?", (bv,)).fetchone()
                    record = json.loads(previous[0]) if previous else {"bvid": bv, "observed_comments": 0}
                    record["last_error"] = str(exc)
                    db.execute("INSERT OR REPLACE INTO videos VALUES(?,?)", (bv, json.dumps(record, ensure_ascii=False)))
                    db.commit()
                if db.execute("SELECT COUNT(*) FROM comments").fetchone()[0] >= args.max_comments:
                    status = "comment_limit_reached"
                    break
            rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM comments")]
            print(f"采集快照 {len(rows)} 条，开始核对资料和评分。", flush=True)
            enrich(client, db, rows, args)
            print(f"本次 API 网络请求 {client.requests}，缓存复用 {client.cached}。", flush=True)
    except Blocked as exc:
        print(str(exc), file=sys.stderr, flush=True)
        status, code = "stopped_by_login_or_rate_limit", 2
        failure_message = str(exc)
    except KeyboardInterrupt:
        print("已中断，导出已有样本；下次同一输出目录继续。", flush=True)
        status, code = "interrupted", 130
    finally:
        try:
            config_path = folder / "run_config.json"
            if not args.export_only and config_path.exists():
                config = json.loads(config_path.read_text(encoding="utf-8"))
                config.update(last_status=status, finished_at=now(), last_error=failure_message)
                dump(config_path, config)
            export_dataset(db, folder, args, status)
        finally:
            db.close()
    return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print("错误：" + str(error), file=sys.stderr)
        raise SystemExit(1)
