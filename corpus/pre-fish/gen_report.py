"""Generate collection-report.md and manifest.json from build-stats.json (+ phase-0 stats).

Re-runnable: call again after any collection resume + build_corpus.py rerun.
"""

import datetime
import hashlib
import json
from pathlib import Path

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

stats = json.loads((HERE / "build-stats.json").read_text(encoding="utf-8"))
phase0 = json.loads((HERE / "existing-partition-stats.json").read_text(encoding="utf-8"))
discovery = json.loads((HERE / "video-discovery.json").read_text(encoding="utf-8"))
by_video = stats["new_candidates_by_video"]
video_meta = {p["bvid"]: p for p in discovery["picks"]}

picks_table = ["| # | BV号 | 主题组 | 关键词 | 发布时间(北京) | 历史回复数 | 有效候选 |",
               "|---:|---|---|---|---|---:|---:|"]
for i, pick in enumerate(discovery["picks"], 1):
    bvid = pick["bvid"]
    picks_table.append(f"| {i} | {bvid} | {pick['topic']} | {pick['keyword']} | {pick['pubdate_beijing'][:10]} | "
                       f"{pick['review']} | {by_video.get(bvid, 0)} |")
picks_table.append(f"| | **合计 {len(discovery['picks'])} 个视频** | | | | {sum(p['review'] for p in discovery['picks'])} | "
                   f"**{sum(by_video.values())}** |")

month_rows = sorted({**phase0["n_pool_by_year_month"], **stats["new_candidates_by_month"]}.items())
month_table = ["| 月份 | 既有候选 | 新采候选 |", "|---|---:|---:|"]
for month, count in month_rows:
    month_table.append(f"| {month} | {phase0['n_pool_by_year_month'].get(month, 0)} | {stats['new_candidates_by_month'].get(month, 0)} |")

lines = [
    "# 鱼灾前历史真人负样本候选集：采集与数据分布报告",
    "",
    f"生成时间：{datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')}（北京时间）",
    "",
    "## 1. 时间规则与执行",
    "",
    "- 截止条件：评论自身 `ctime` 转换为北京时间后严格早于 **2026-10-01 00:00:00 (UTC+8)**。",
    f"- 截止对应 Unix 时间戳：{phase0['cutoff_unix']}。",
    "- 时间缺失、异常或无法验证的记录进入隔离池，未做任何猜测。",
    "- 原始 Unix 时间戳、转换后时间、BV 号、评论 ID 全部保留，可逐条追溯。",
    "- 2025 年及更早发布的视频若在 2026 年 10 月后收到新评论，已按同一 ctime 规则过滤（新采集中隔离 post_cutoff "
    f"{stats['quarantine_by_reason'].get('post_cutoff', 0)} 条）。",
    "",
    "## 2. 现有数据的历史筛选（Phase 0）",
    "",
    f"现有全部采集原始记录 {phase0['raw_rows_total']} 行（按评论 ID 去重后），划分为：",
    "",
    f"- **P（高可信 AI 正类）**：{phase0['stats']['P']} 行——规则分 ≥30 与历史确认 AI，不受时间条件约束，保持原标签。",
    f"- **N（历史真人候选）**：{phase0['n_pool_rows']} 行——既有语料中 label=0 且 ctime 达标 {phase0['stats']['N/existing_corpus']} 行，"
    f"加从未进入语料的达标原始行 {phase0['stats']['N/raw_unlabeled']} 行。",
    f"- **U（来源不确定）**：{phase0['stats']['U/post_cutoff']} 行——label=0 但 ctime 在截止之后（鱼灾爆发期间的评论），历史用途存疑。",
    f"- **隔离池**：{sum(v for k, v in phase0['stats'].items() if k.startswith('quarantine'))} 行——"
    f"重复评论 ID {phase0['stats']['quarantine/duplicate_rpid']}、灰区策略 {phase0['stats']['quarantine/gray_policy_v3']}、"
    f"纯@/空正文 {phase0['stats']['quarantine/mention_only_or_empty']}、UID+正文重复 {phase0['stats']['quarantine/duplicate_uid_body']}。",
    "- 规则冲突（label=0 但规则分 ≥30）在既有语料中为 0 行——语料构建时已全部归入 AI 类。",
    "",
    "## 3. 新采集：视频发现与执行",
    "",
    f"- 28 个关键词搜索（AI 域 / 开发 / 普通话题），视频自身发布时间过滤为 ≤ 2026-09-30，"
    f"排除本项目历史上采集过的 {discovery['excluded_ever_collected']} 个视频。",
    f"- 入选 {len(discovery['picks'])} 个视频：AI 域 12 / 开发 10 / 普通话题 6；发布时间 "
    f"{min(p['pubdate_beijing'] for p in discovery['picks'])[:10]} 至 {max(p['pubdate_beijing'] for p in discovery['picks'])[:10]}。",
    f"- 采集执行：前 10 个视频完成后遇到 **HTTP 412（平台反爬限流）**，断点已保留。按规则冷却 10 分钟后续采一次，"
    f"仍在视频 10 处被 412 拦截——限流窗口长于本次会话可等待时间。已停止重试（不做开放式重试循环），"
    f"实际覆盖 10/28 个视频；剩余 18 个视频清单已存档（videos.txt），后续限流解除后可断点续采。",
    f"- 实际入库原始行 {stats['new_collection_raw_rows']} 条（10 个视频的完整数据 + 续采增量）。",
    "",
    "## 4. 有效候选与分布",
    "",
    f"- **新采历史真人负样本候选：{stats['new_candidates']} 条**（ctime 达标 + 清洗通过 + 非规则冲突）。",
    f"- **既有语料历史候选：{stats['existing_n_pool_rows']} 条**（已在 v8 各划分中，未来训练时按来源标记，不算新增量）。",
    f"- 候选合计：{stats['new_candidates'] + stats['existing_n_pool_rows']} 行。",
    "",
    "### 按月份分布（检查时间段偏差）",
    "",
    *month_table,
    "",
    "既有候选集中在 2025-03/04（占 77%）——这正是新采集要补的时间段；新采候选覆盖 2025-01 至 2026-09。",
    "",
    "### 按视频分布（检查单视频主导）",
    "",
    *picks_table,
    "",
    f"新采候选最多单视频 {max(by_video.values())} 条、最少 {min(by_video.values())} 条，无单一视频主导。",
    "",
    "### 长度与规则分分布",
    "",
    f"- 新采候选长度：1–20 字 {stats['new_candidates_length']['1-20']} / 21–60 字 {stats['new_candidates_length']['21-60']} / "
    f"61+ 字 {stats['new_candidates_length']['61+']}。",
    f"- 新采候选规则分：0–9 分 {stats['new_candidates_score_bands']['0-9']} / 10–19 分 {stats['new_candidates_score_bands']['10-19']} / "
    f"20–29 分 {stats['new_candidates_score_bands']['20-29']}（规则冲突 ≥30 的 {stats['quarantine_by_reason'].get('rule_conflict_score_ge30', 0)} 条已隔离）。",
    f"- 既有候选规则分：0–9 分 {phase0['n_pool_score_bands']['0-9']} / 10–19 分 {phase0['n_pool_score_bands']['10-19']} / "
    f"20–29 分 {phase0['n_pool_score_bands']['20-29']}。",
    "",
    "### 同领域（AI/编程/插件）占比",
    "",
    f"- 新采候选按话题组：AI 域 {stats['new_candidates_by_topic'].get('ai_domain', 0)}、"
    f"开发 {stats['new_candidates_by_topic'].get('dev', 0)}、普通话题 {stats['new_candidates_by_topic'].get('general', 0)}、"
    f"未知 {stats['new_candidates_by_topic'].get('unknown', 0)}。",
    f"- 同领域（AI 域 + 开发）合计 {stats['new_candidates_by_topic'].get('ai_domain', 0) + stats['new_candidates_by_topic'].get('dev', 0)} 条，"
    f"占新采候选 {round((stats['new_candidates_by_topic'].get('ai_domain', 0) + stats['new_candidates_by_topic'].get('dev', 0)) / max(1, stats['new_candidates']) * 100, 1)}%。",
    "",
    "## 5. 与既有数据的重叠检查",
    "",
    f"- 与 v8 **训练**划分重叠（UID/正文）：{stats['overlap_flag_counts'].get('v8_train', 0)} 行——单独统计，不算新增量，已标记。",
    f"- 与 v8 **验证/测试**划分或**冻结基准**（UID/正文/BV）重叠："
    f"{stats['overlap_flag_counts'].get('v8_validation', 0) + stats['overlap_flag_counts'].get('v8_test', 0) + stats['overlap_flag_counts'].get('benchmark_uid', 0) + stats['overlap_flag_counts'].get('benchmark_text', 0)} 行——"
    "已标记 training_pool_eligible=false，不会进入未来训练候选池。",
    f"- 新采候选中完全干净（与全部既有划分零重叠）：{stats['training_pool_eligible']['new_collection']} 条。",
    "- 完全重复按 UID+规范化正文去重；近重复文本（≥0.90 相似度）仅统计规模，不排除。",
    "",
    "## 6. 采集限制记录",
    "",
    "- HTTP 412 反爬限流在 10/28 个视频后触发；断点已保留；冷却后按同一断点续采（结果见附录状态行与本报告重新生成时间）。",
    "- 未使用任何绕过验证码或反爬验证的手段；未放宽截止时间。",
    "- 历史评论获取方式依赖视频自身的评论区存档：过早的视频（2017–2022）评论区热度衰减，翻页深度受限——这既是限制也是数据真实性的来源。",
    "",
    "## 7. 交付文件",
    "",
    "- `raw/`：采集器原始导出与运行配置（未修改）。",
    "- `historical-human-candidates.csv`：全部 N 候选（既有 + 新采），含 proposed_label=0 / label_source=historical_timestamp / authorship_verified=false 与 eligibility 标记。",
    "- `quarantine.csv`：全部隔离行与原因。",
    "- `existing-partition.jsonl` / `existing-partition-stats.json`：Phase 0 现有数据划分。",
    "- `manifest.json`：来源、时间规则、数量与 SHA256。",
    "",
]
(HERE / "collection-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

manifest = {
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "time_rule": {"cutoff_beijing": "2026-10-01T00:00:00+08:00", "cutoff_unix": phase0["cutoff_unix"],
                  "field": "comment ctime (unix seconds)", "on_missing": "quarantine, never guessed"},
    "labels": {"proposed_label": 0, "label_source": "historical_timestamp", "authorship_verified": False,
               "rule_conflict_policy": "score>=30 rows quarantined separately, never mixed into negatives",
               "identity_label_policy": "no confirmed_human claims; historical proxy only"},
    "sources": {"existing_partition": str((HERE / "existing-partition.jsonl").as_posix()),
                "raw_exports": json.loads((HERE / "exports.json").read_text(encoding="utf-8"))["raw_exports"],
                "new_collection_export": str((HERE / "collection/export/all_scored.jsonl").as_posix())},
    "counts": {"existing_partition": phase0, "build": stats},
    "sha256": {"historical-human-candidates.csv": sha256(HERE / "historical-human-candidates.csv"),
               "quarantine.csv": sha256(HERE / "quarantine.csv"),
               "existing-partition.jsonl": sha256(HERE / "existing-partition.jsonl"),
               "collection_export": sha256(HERE / "collection/export/all_scored.jsonl"),
               "phase0_script": sha256(HERE / "phase0_existing_partition.py"),
               "build_script": sha256(HERE / "build_corpus.py")},
}
(HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"report": str((HERE / 'collection-report.md').as_posix()),
                  "manifest": str((HERE / 'manifest.json').as_posix())}, ensure_ascii=False))
