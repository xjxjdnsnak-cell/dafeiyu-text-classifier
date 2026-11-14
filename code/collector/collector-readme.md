# 大肥鱼评论批量采集工具

本地 Python + Node.js 工具，输入你指定的 BV 号或完整视频链接，采集视频主评论和楼中楼，生成 fastText 数据集。无需 DeepSeek API，也不需要重新安装 APK。

## 当前筛选规则

- **AI 正样本：单条评论的规则分数 ≥80。** 不把同账号的一条高分评论直接传播为该账号所有评论的 AI 标签。
- **正常负样本：单条分数 <30，且该 UID 在本次全部视频中的聚合风险、缺失特征上界也 <30。** 同 UID 有可疑发言时，低分短回复也先进入待审核。
- 30～79 分、资料不足、同文标签矛盾的评论进入待审核，不写入训练集。
- 缺失等级不当成 Lv0；签名未核实时，对正常样本预留最高 +40 分，头像未分析时预留最高 +22 分。缺失昵称、等级、认证或会员字段的样本先待审核。
- 如果正文分数本身已达 30，或昵称/签名有至少 +25 的明确风险项，但被认证、UP 主等减分压到 30 以下，也暂不标成正常。
- 每个视频按 UID 聚合去重后计算重复发言、点名等风险；**历史识别加分关闭**，避免采集越多分数越高。
- 分数不是 AI 概率；`human` 表示低风险自动弱标签，仍需抽样复核，不能保证都是真人。规则特征包含账号资料，因此标签不等同于仅看正文就能作出的判断。

## 使用

需要 Python 3.10+ 和 Node.js 18+。先安装扫码和头像依赖：

```powershell
python -m pip --isolated install --index-url https://pypi.org/simple -r requirements-collector.txt
```

修改 `collector_videos.txt`，一行一个 BV 号或 `https://www.bilibili.com/video/BV…/` 链接。已填入你提供的 11 个视频。短链请先打开取得完整 BV 链接。

双击 `Start-Collector.cmd` 即可采集；默认每个视频最多 300 条，根列表最多 10 页，每楼最多 3 页（60 条），整库最多 5000 条。API 请求至少间隔 1.2 秒，单连接，不并发轰炸接口。

首次扫码登录：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\collect-comments.ps1 -Login
```

用 B 站小号扫描打开的二维码，在手机确认。已有本地登录时，脚本会自动复用。登录凭据保存在 `%USERPROFILE%\.codex\local-keys\bilibili-comment-collector\cookie.txt`，不会进入源码包或数据集。删除该文件即可停止脚本使用此登录；也可在 B 站账号设备管理中退出该会话。二维码有效期约 3 分钟，超时请重新运行。

指定视频、增加限额继续采集：

```powershell
.\collect-comments.ps1 -Videos BV1jhn16GEcv,BV1DY4v6DEtz -RootPages 50 -ChildPages 10 -PerVideoComments 2000 -MaxComments 20000 -ProfileLimit 2000
```

更细参数可直接用 Python：

```powershell
python collect_comments.py --input collector_videos.txt --output dataset --root-pages 20 --child-pages 5 --per-video-comments 1000 --max-comments 11000 --profile-limit 1500 --avatar-limit 1000 --cookie-file "$env:USERPROFILE\.codex\local-keys\bilibili-comment-collector\cookie.txt"
```

同一输出目录会复用成功响应和已核对的资料，并按评论 ID 去重，中断后重新运行即可继续。**需要提高触及的页数/条数限额才能取得更深的评论。** 已缓存的页面保持为当时快照；想要一份新的实时快照，请换一个输出目录。游客与登录响应分开缓存，登录后会重新请求受游客限制的页面。接口要求登录、触发限制或 HTTP 429/412 时停止请求并导出现有样本，不绕过验证。

只按缓存重新评分导出，不联网：

```powershell
.\collect-comments.ps1 -ExportOnly
```

## 输出

`dataset/export/`：

| 文件 | 用途 |
|---|---|
| `fasttext_train.txt` | `__label__ai` / `__label__human`，每行一个规范化正文 |
| `fasttext_train_char.txt` | 同一批样本按单字空格分词，可用于中文 fastText 基线 |
| `ai.jsonl`、`human.jsonl` | 自动标注样本的原文、评分、原因、视频、UID、评论 ID |
| `review.jsonl` | 中间分、同 UID 可疑、资料缺失等待审核样本 |
| `all_scored.jsonl` | 全部去重评论，原文不改写，附评分和缺失特征 |
| `conflicts.jsonl` | 正常/AI 同文冲突；这些文本已从训练集排除 |
| `summary.json` | 各视频采集页数、条数、停止原因、分档数、去重统计、规则版本 |
| `SHA256SUMS.txt` | 导出文件 SHA-256 |

`raw/` 保存接口原始 JSON 和请求 URL/时间（不含请求头、Cookie）。`collector.sqlite3` 保存去重记录和资料；`avatars/` 缓存实际分析过的头像。请保留这些文件，才能回溯、复核和断点续采。

训练文件去掉零宽字符、统一空白和 Unicode 全角形态，防止正文伪造 `__label__` 或多行；按规范化正文去重，跨标签冲突不进入训练集。JSONL 保留原文。AI/正常样本不人为配平；请先按 UID 分组划分训练/验证集，再在训练集内调整平衡，避免同账号评论泄漏到验证集。

按字分词文件还会单独去重并剔除分词后形成的跨标签冲突；该视图的数量见 `training_char_unique_texts`，可能小于原文版。

头像仅在可能改变 30/80 分档时下载，按原规则的 32×32 HSV 蓝色占比计算。Pillow 缩放与浏览器 Canvas 像素可能略有差异；无法读取时跳过，并应用正常样本的缺失上界保护。公开资料和头像的单次核对数量也有上限，未核对完的低分样本可能暂存待审核；提高 `--profile-limit`、`--avatar-limit` 重新运行即可。

游客接口可能只返回少量根评论，即使服务器总数很大。登录也不保证每个视频的所有评论都可访问。工具不把空页当作“完整抓取成功”，实际覆盖范围以 `summary.json` 为准。

## 验证和来源

```powershell
python -m unittest discover -s . -p test_collector.py -v
```

覆盖 29/30、79/80 边界、UID 保护、未知等级、正文标签注入、原始 JavaScript 样例得分/原因、重复 ID、游客/登录缓存隔离和冲突标签导出。测试使用临时目录，不混入真实采集库。

规则直接运行 [鱼皮的大肥鱼消除计划 detector.js](https://github.com/liyupi/dafeiyu-cleaner/blob/32866038316f81d77557f20a213b8c3723bf9ce0/dafeiyu-cleaner/src/detector.js)，固定提交 `32866038316f81d77557f20a213b8c3723bf9ce0`，未修改上游规则文件，MIT 许可见 `LICENSE`。这是研究用弱标注数据，不应把某个账号或某一条评论的标签公开表述为已证实的 AI 身份。
