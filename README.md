# 大肥鱼 正文专项 AI 评论分类器（fastText，仅正文特征）

离线运行的 B 站 AI 风格评论分类器。**模型只看正文**——UID 与昵称不进入特征（昵称仅用于分组和训练采样权重），目标是识别挂在普通昵称下的 AI 评论。

- 当前发布模型：v8（`model/text/model.bin`，fastText supervised，dim32 / epoch30 / lr0.5 / wordNgrams2 / minn2-maxn4 / bucket20000 / softmax / seed42 / thread1）
- 阈值：`model/text/model.bin.json`（0.9700783414714084，按验证集 1% 人类误拦预算选出）；量化版 `model.ftz` 附带独立阈值
- 训练数据：19,317 行规则/人工标签（AI 6,393 / 人类 12,249，规则分 ≥30 记 AI、灰区短语双向排除、纯 @ 排除），来自 60 个公开视频的采集档案，**数据集本体不在本仓库**（含真实用户评论与 UID）
- 运行依赖：`fasttext-wheel==0.9.2`、`jieba==0.42.1`、`numpy==1.26.4`（见 `code/requirements.txt`）

## 测试集表现（v8，2,381 行留出账号评论；1,684 人类 / 697 AI）

| 验证误拦预算 | 整体召回 | 普通昵称AI召回 | 人类误拦 |
|---:|---:|---:|---:|
| 1% | 52.51% | 45/122（36.89%） | 11（0.65%） |
| 2% | 67.72% | 73/122（59.84%） | 18（1.07%） |

分组：61 字以上召回 36.84%、误拦 1；21–60 字 46.81%、误拦 0；**1–20 字 11.11%、误拦 11**——短文本（弹幕玩梗式短评）是当前主要弱点。

误拦率 0.65%（Wilson 95% [0.37%, 1.17%]，逐评论近似）。标签是规则分 + 人工裁决的弱标签，以上是**与标签体系的一致性**，不是经作者身份验证的识别准确率。

## 使用

```python
import sys
sys.path.insert(0, "code/src")
from bili_ai.model import Predictor

p = Predictor("model/text/model.bin", mode="text", blacklist=None)
result = p.predict_one(text="待判定的评论正文")
print(result["ai_style_score"], result["predicted_label"])
```

注意：模型入口必须**先排除纯 @ 无正文的输入**（`bili_ai.preprocess.is_mention_only`），这类输入不给 AI 判断；评分前需按 `assets/preprocess.json` 与冻结词典（`assets/jieba.dict.txt`）做同样预处理。灰区名单 `policy/human_exclusion_policy.json` 用于**训练侧人类样本排除**，不是线上判 AI 的规则。

## 目录

```
report/    六轮“误拦复核→用户裁决→改判→重划分重训”的完整对比报告（先读这个）
model/     当前发布模型（bin + ftz）与冻结元数据（阈值、sha256、来源哈希）
evaluation/v1..v8  每轮评估（aggregate 指标、冻结清单、划分报告；不含评论原文）
review/    每轮误拦复核表（用户逐条裁决记录）、61 行 AI 抽查、改判汇总 CSV
policy/    灰区排除名单 v3（53 条）与 B 站表情名单
code/      训练/评估/划分/复核流水线源码与测试（27 项）；采集器脚本
```

## 审计要点

1. 每轮测试集在阈值冻结后才读取：见各轮 `evaluation/<tag>/selection-frozen-before-test.json` 与 `code/experiments/text-training-20261114/train_text.py` 的 `run()`。
2. 划分按视频贪心 70/15/15（种子 20261114），UID/正文模板/近重复模板跨集合的行全部隔离，泄漏审计要求四维交集为 0：见各轮 `evaluation/<tag>/split-report.json` 的 `overlap`。
3. 用户裁决过的样本所在视频永久退出测试池：`split-report.json` 的 `user_corrections.reviewed_videos`。
4. 六轮绝对数字**不可跨轮比较**（测试集每轮重画、误拦修正改变训练分布），机制分析见 `report/` 第 6 节。
5. 本仓库刻意**不含**：训练语料本体、逐行 lineage、隔离与灰区明细（均含真实用户评论与 UID）；也不含任何 cookie / API key。报告中的全部数字可由 `code/` 在本地数据上复现。

## 已知限制

- 短文本（≤20 字）召回 11%，误拦集中在弹幕玩梗式短评（已确认为真人）。
- 评估档案来自已使用过的采集批次，重新划分 ≠ 前瞻性新来源验收。
- 普通昵称 AI 测试池随复核循环逐轮变薄（122 条），后续需补采新来源。
- 训练侧存在“误拦→用户裁决→改判入 AI 类”的反馈回路，属流程固有，已在报告中明示。
