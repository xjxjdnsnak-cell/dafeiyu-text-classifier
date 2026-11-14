"""Find one NEW dsh-plugin video with many comments via wbi-signed Bilibili search.

The collector covers comment APIs but not search; wbi signing here follows the
public algorithm (nav keys -> mixin key -> md5). Read-only; the cookie stays local.
"""

import hashlib
import json
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/dafeiyu"))
sys.path.insert(0, str(ROOT / "out/collector-expand-topics-20261114"))

from collect_comments import Client  # noqa: E402
from discover import covered_bvs  # noqa: E402

OUT = Path(__file__).resolve().parent
MIXIN_TAB = (46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 27,
             11, 46, 22, 4, 28, 36, 1, 32, 49, 55, 19, 50, 29, 54, 15, 6,
             16, 2, 44, 8, 19, 47, 12, 1, 27, 33, 52, 47, 38, 49, 53, 15,
             38, 47, 12, 1, 27, 33, 52, 47, 38, 49, 53, 15, 38, 47, 12, 1)
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
KEYWORDS = ("大肥鱼 插件", "DSH 插件", "DeepSeek Harness", "大肥鱼 教程")


def mixin_key(client):
    data, _ = client.get("x/web-interface/nav", {})
    wbi = (data or {}).get("wbi_img") or {}
    raw = Path(urllib.parse.urlparse(wbi.get("img_url", "")).path).stem + \
        Path(urllib.parse.urlparse(wbi.get("sub_url", "")).path).stem
    if not raw:
        raise SystemExit("nav 未返回 wbi 密钥")
    return "".join(raw[i] for i in MIXIN_TAB)[:32]


def signed_params(client, key, params):
    params = {**params, "wts": int(time.time())}
    clean = {k: str(v).replace("!", "").replace("'", "") for k, v in sorted(params.items())}
    query = urllib.parse.urlencode(clean)
    return {**clean, "w_rid": hashlib.md5((query + key).encode()).hexdigest()}


def main():
    cookie_path = Path.home() / ".codex/local-keys/bilibili-comment-collector/cookie.txt"
    cookie = cookie_path.read_text(encoding="utf-8-sig").strip() if cookie_path.exists() else ""
    if not cookie:
        raise SystemExit("没有找到小号 cookie，请先在采集器里登录")
    client = Client(OUT, 1.5, cookie)
    client.headers["User-Agent"] = BROWSER_UA
    covered = covered_bvs()
    key = mixin_key(client)
    candidates, seen = [], set()
    for keyword in KEYWORDS:
        try:
            data, _ = client.get("x/web-interface/wbi/search/type",
                                 signed_params(client, key, {"search_type": "video", "keyword": keyword,
                                                             "page": 1, "page_size": 30}))
        except Exception as exc:  # noqa: BLE001 - one keyword failing is not fatal
            print(json.dumps({"keyword": keyword, "error": str(exc)}, ensure_ascii=False), flush=True)
            continue
        for item in (data or {}).get("result") or []:
            bvid = item.get("bvid", "")
            if not bvid or bvid in seen:
                continue
            seen.add(bvid)
            title = item.get("title", "").replace('<em class="keyword">', "").replace("</em>", "")
            candidates.append({"bvid": bvid, "title": title, "review": item.get("review", 0),
                               "play": item.get("play", 0), "author": item.get("author", ""),
                               "pubdate": item.get("pubdate", 0), "keyword": keyword,
                               "covered": bvid in covered})
        print(json.dumps({"keyword": keyword, "results": len(candidates)}, ensure_ascii=False), flush=True)
    fresh = [c for c in candidates if not c["covered"] and c["review"]]
    fresh.sort(key=lambda c: -c["review"])
    if not fresh:
        raise SystemExit("没有找到新的 dsh 插件视频")
    picks = fresh[:5]
    (OUT / "dsh-candidates.json").write_text(json.dumps({"candidates": fresh[:20]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "videos-dsh.txt").write_text(picks[0]["bvid"] + "\n", encoding="utf-8")
    print(json.dumps({"picked": picks[0]["bvid"], "title": picks[0]["title"], "review": picks[0]["review"],
                      "author": picks[0]["author"], "alternatives": picks[1:]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
