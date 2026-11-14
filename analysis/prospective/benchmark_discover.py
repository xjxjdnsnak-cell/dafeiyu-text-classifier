"""Discover fresh videos for the frozen prospective benchmark.

Exclusion set = every video that has ever been collected anywhere: the four
collector lists, every bvid in the r2 label-decisions sources, and every bvid
appearing in any raw export file. The four dsh/AI-topic candidates below come
from the earlier wbi search (never collected); their exclusion is re-verified.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/dafeiyu"))
sys.path.insert(0, str(ROOT / "out/dsh-video-20261114"))

from collect_comments import Client  # noqa: E402

OUT = Path(__file__).resolve().parent
AI_CANDIDATES = (
    ("BV126nt6GE4W", "什么是鱼灾？B站大肥鱼一周爆发始末全复盘【附安全养鱼指南】"),
    ("BV1GJnZ6MEFw", "【拒绝废话】用DSH必须下载的八个插件"),
    ("BV1REx46NEc4", "【全网最全】DeepSeek Harness超详细教程！50分钟吃透DSH"),
    ("BV11qJ96LELG", "dsh神级插件！集合全网免费模型资源全自动调用"),
)
REGIONS = ((1, "动画"), (3, "音乐"), (4, "游戏"), (5, "娱乐"), (23, "电影"), (36, "知识"),
           (129, "舞蹈"), (155, "时尚"), (160, "生活"), (181, "影视"), (177, "纪录片"),
           (171, "电竞"), (188, "科幻"), (11, "电视剧"), (119, "鬼畜"), (13, "番剧"))
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def ever_collected_bvs():
    historical = set()
    for name in ("collector_videos.txt", "videos-19.txt", "videos-13.txt", "videos-ai11-deepen.txt"):
        for base in (ROOT / "tools/dafeiyu", ROOT / "out/collector-expand-20261112"):
            path = base / name
            if path.exists():
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    token = line.split("#", 1)[0].strip()
                    if token.startswith("BV"):
                        historical.add(token)
    for export in (*json.load(open(Path(__file__).with_name("exports.json"), encoding="utf-8"))["raw_exports"],
                   str(ROOT / "fasttext-classifier/data-rule-ge30-20261114-r2/label-decisions.jsonl")):
        path = Path(export)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("bvid"):
                historical.add(row["bvid"])
            for source in row.get("sources", []):
                if source.get("bvid"):
                    historical.add(source["bvid"])
    return historical


def main():
    cookie_path = Path.home() / ".codex/local-keys/bilibili-comment-collector/cookie.txt"
    cookie = cookie_path.read_text(encoding="utf-8-sig").strip() if cookie_path.exists() else ""
    client = Client(OUT, 1.2, cookie)
    client.headers["User-Agent"] = BROWSER_UA
    historical = ever_collected_bvs()
    conflicts = [bvid for bvid, _ in AI_CANDIDATES if bvid in historical]
    if conflicts:
        raise SystemExit(f"AI候选已出现在历史采集，违反基准要求: {conflicts}")
    covered = historical | {bvid for bvid, _ in AI_CANDIDATES}
    manifest = {"excluded_videos": len(covered)}
    manifest["ai_pool"] = [{"bvid": bvid, "title": title} for bvid, title in AI_CANDIDATES]
    picks = []
    for rid, name in REGIONS:
        try:
            data, _ = client.get("x/web-interface/ranking/v2", {"rid": rid, "type": "all"})
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"region": rid, "skipped": str(exc)}, ensure_ascii=False), flush=True)
            continue
        candidates = [a for a in (data.get("list") or [])
                      if a.get("bvid") and a["bvid"] not in covered
                      and all(a["bvid"] != p["bvid"] for p in picks)]
        candidates.sort(key=lambda a: -(a.get("stat", {}).get("reply") or 0))
        best = next((a for a in candidates if (a.get("stat", {}).get("reply") or 0) >= 400), None) or (candidates[0] if candidates else None)
        if best is None:
            continue
        picks.append({"bvid": best["bvid"], "region": f"{rid}:{name}", "title": best.get("title", ""),
                      "reply": best.get("stat", {}).get("reply", 0)})
        print(json.dumps({"topic_pick": best["bvid"], "region": name, "reply": best.get("stat", {}).get("reply", 0)}, ensure_ascii=False), flush=True)
    if len(picks) < 8:
        raise SystemExit(f"话题池只有 {len(picks)} 个，太少")
    manifest["topic_pool"] = picks
    (OUT / "videos-ai.txt").write_text("\n".join(p["bvid"] for p in manifest["ai_pool"]) + "\n", encoding="utf-8")
    (OUT / "videos-topics.txt").write_text("\n".join(p["bvid"] for p in picks) + "\n", encoding="utf-8")
    (OUT / "discovery.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ai_pool": len(manifest["ai_pool"]), "topic_pool": len(picks),
                      "excluded_videos": len(covered), "api_requests": client.requests}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
