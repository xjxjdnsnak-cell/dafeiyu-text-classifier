"""Discover pre-outbreak videos for the historical human-negative corpus.

Searches Bilibili (wbi-signed) for AI/programming/tech topics, keeps only videos
published (by their own pubdate, Beijing) before 2026-10-01, excludes every
video the project has ever collected, and prefers videos with substantial
historical replies. Topic groups are recorded per video for distribution stats.
"""

import datetime
import hashlib
import json
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/dafeiyu"))

from collect_comments import Client  # noqa: E402

HERE = Path(__file__).resolve().parent
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
MIXIN_TAB = (46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 27,
             11, 46, 22, 4, 28, 36, 1, 32, 49, 55, 19, 50, 29, 54, 15, 6,
             16, 2, 44, 8, 19, 47, 12, 1, 27, 33, 52, 47, 38, 49, 53, 15,
             38, 47, 12, 1, 27, 33, 52, 47, 38, 49, 53, 15, 38, 47, 12, 1)
PUBDATE_MAX = int(datetime.datetime(2026, 9, 30, 23, 59, 59,
                                    tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp())
PUBDATE_MIN = int(datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc).timestamp())
KEYWORDS = (
    ("DeepSeek 教程", "ai_domain"), ("AI Agent", "ai_domain"), ("MCP 教程", "ai_domain"),
    ("ChatGPT 插件", "ai_domain"), ("AI 工具 推荐", "ai_domain"), ("大模型 部署", "ai_domain"),
    ("GitHub 开源项目", "dev"), ("编程 教程", "dev"), ("软件安装 教程", "dev"),
    ("插件 推荐", "dev"), ("科技数码 评测", "general"), ("软件 推荐", "general"),
    ("互联网 技术", "general"), ("程序员 日常", "dev"),
)
PER_KEYWORD = 2
MIN_REVIEW = 300


def load_jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def ever_collected_bvs():
    covered = set()
    for name in ("collector_videos.txt", "videos-19.txt", "videos-13.txt", "videos-ai11-deepen.txt"):
        for base in (ROOT / "tools/dafeiyu", ROOT / "out/collector-expand-20261112"):
            path = base / name
            if path.exists():
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    token = line.split("#", 1)[0].strip()
                    if token.startswith("BV"):
                        covered.add(token)
    exports = json.loads((HERE / "exports.json").read_text(encoding="utf-8"))["raw_exports"]
    for export in exports:
        path = Path(export)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bvid"):
                covered.add(row["bvid"])
    for line in (CLASSIFIER_LABELS := ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2/label-decisions.jsonl").read_text(encoding="utf-8").splitlines():
        for source in json.loads(line).get("sources", []):
            if source.get("bvid"):
                covered.add(source["bvid"])
    for name in ("videos.txt",):
        path = ROOT / "out/collector-expand-topics-20261114" / name
        if path.exists():
            covered |= {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip().startswith("BV")}
    for name in ("videos-dsh.txt",):
        path = ROOT / "out/dsh-video-20261114" / name
        if path.exists():
            covered |= {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip().startswith("BV")}
    for name in ("videos-ai.txt", "videos-topics.txt"):
        path = ROOT / "out/prospective-benchmark-20261114" / name
        if path.exists():
            covered |= {line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip().startswith("BV")}
    return covered


def mixin_key(client):
    data, _ = client.get("x/web-interface/nav", {})
    wbi = (data or {}).get("wbi_img") or {}
    raw = Path(urllib.parse.urlparse(wbi.get("img_url", "")).path).stem + \
        Path(urllib.parse.urlparse(wbi.get("sub_url", "")).path).stem
    if not raw:
        raise SystemExit("nav 未返回 wbi 密钥")
    return "".join(raw[i] for i in MIXIN_TAB)[:32]


def signed_params(key, params):
    params = {**params, "wts": int(time.time())}
    clean = {k: str(v).replace("!", "").replace("'", "") for k, v in sorted(params.items())}
    query = urllib.parse.urlencode(clean)
    return {**clean, "w_rid": hashlib.md5((query + key).encode()).hexdigest()}


def main():
    cookie_path = Path.home() / ".codex/local-keys/bilibili-comment-collector/cookie.txt"
    cookie = cookie_path.read_text(encoding="utf-8-sig").strip() if cookie_path.exists() else ""
    if not cookie:
        raise SystemExit("没有找到小号 cookie，请先在采集器里登录")
    client = Client(HERE, 1.5, cookie)
    client.headers["User-Agent"] = BROWSER_UA
    covered = ever_collected_bvs()
    key = mixin_key(client)
    picks, seen = [], set()
    for keyword, topic in KEYWORDS:
        try:
            data, _ = client.get("x/web-interface/wbi/search/type",
                                 signed_params(key, {"search_type": "video", "keyword": keyword,
                                                     "page": 1, "page_size": 30}))
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"keyword": keyword, "error": str(exc)}, ensure_ascii=False), flush=True)
            continue
        added = 0
        for item in (data or {}).get("result") or []:
            bvid = item.get("bvid", "")
            pubdate = int(item.get("pubdate", 0))
            review = int(item.get("review", 0))
            if not bvid or bvid in seen or bvid in covered:
                continue
            if not (PUBDATE_MIN <= pubdate <= PUBDATE_MAX) or review < MIN_REVIEW:
                continue
            seen.add(bvid)
            picks.append({"bvid": bvid, "keyword": keyword, "topic": topic,
                          "title": item.get("title", "").replace('<em class="keyword">', "").replace("</em>", ""),
                          "review": review, "pubdate": pubdate,
                          "pubdate_beijing": datetime.datetime.fromtimestamp(pubdate, datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")})
            added += 1
            if added >= PER_KEYWORD:
                break
        print(json.dumps({"keyword": keyword, "accepted_total": len(picks)}, ensure_ascii=False), flush=True)
    by_topic = {}
    for pick in picks:
        by_topic.setdefault(pick["topic"], []).append(pick)
    if len(picks) < 15:
        raise SystemExit(f"合格视频仅 {len(picks)} 个，少于 15，中止（不放宽时间条件）")
    (HERE / "videos.txt").write_text("\n".join(p["bvid"] for p in picks) + "\n", encoding="utf-8")
    (HERE / "video-discovery.json").write_text(json.dumps(
        {"picks": picks, "excluded_ever_collected": len(covered), "pubdate_rule": "video pubdate in [2023-01-01, 2026-09-30] Beijing; comment-level ctime filter applied later",
         "min_review": MIN_REVIEW}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"picked": len(picks),
                      "by_topic": {k: len(v) for k, v in by_topic.items()},
                      "min_pubdate": min(p["pubdate_beijing"] for p in picks),
                      "max_pubdate": max(p["pubdate_beijing"] for p in picks),
                      "total_review": sum(p["review"] for p in picks)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
