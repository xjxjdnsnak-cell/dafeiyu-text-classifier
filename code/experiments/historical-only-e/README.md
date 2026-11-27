# v8 纯历史负样本 E 组消融

E 保留 v8 的 4,431 条训练正样本。指定的 17,477 条历史候选中，7,334 条通过来源、时间、旧训练行排除、跨集合隔离和正文去重。E 共 11,765 条训练样本，比等量对照目标少 1,196 条。负样本中 AI / 插件视频的占比由 15.4% 增至 63.6%。

| 验证误杀预算 | 独立盲评 AI 风格召回 A / E | 独立盲评真人风格误杀 A / E | 旧留出召回 A / E | 旧留出误杀 A / E |
|---|---|---|---|---|
| 1% | 54.11–54.36% / 2.28–2.29% | 0.86–1.09% / 1.84–1.85% | 52.51% / 4.59% | 0.65% / 0.06% |
| 2% | 69.90–70.63% / 12.48–13.41% | 0.93–1.98% / 2.63–2.99% | 67.72% / 21.38% | 1.07% / 0.30% |

盲评数值的范围来自无标注后分层单元。严格隔离框有 1,215 条，228 条获最终人工风格标签。固定抽样框中 E 减 A 的成对召回差 95% 区间分别为 −67.74 至 −32.66、−73.20 至 −32.85 个百分点。真人风格误杀差的区间均包含零。原 2,381 条留出评论按历史标签单独核算。当前结果不支持优先在其他架构复现或替换正式模型。

## 本地复算

下列命令从保存原始语料的项目根目录运行。分类器仓库位于 `out/historical-only-e-study`。将输出目录改为一个空目录。准备和评分使用 v8 的冻结 Python 环境，统计使用带 NumPy、SciPy 和 scikit-learn 的研究环境。

```powershell
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = (Resolve-Path fasttext-classifier/scripts).Path
$studyScripts = 'out/historical-only-e-study/code/experiments/historical-only-e'
$studyOutput = 'out/ae-reproduction'
& fasttext-classifier/.venv/Scripts/python.exe "$studyScripts/prepare.py" . $studyOutput
& fasttext-classifier/.venv/Scripts/python.exe "$studyScripts/score.py" . $studyOutput
python "$studyScripts/analyze.py" . $studyOutput
```

`prepare.py` 从受限本地源核验归档与原始输入，先固定成员，再用原 v8 训练函数训练 E，分别校准 A/E 两个验证工作点。它读取论文成员身份与正文做隔离，只对标签文件计算校验和。`score.py` 校验冻结件，对接原抽样密钥、双评回执及裁决表，复核每个旧留出 A 分数，写出逐条配对预测和改判。`analyze.py` 调用论文的单元格、超几何区间和整视频重抽函数。它在原采样层内按两模型的拦截决定后分层，用同一套设计权重计算 AP。解读主比较时查看 `metrics.csv` 中 `dataset=primary` 的配对结果。`full_sensitivity` 包含模板交叉的 18 条。`auxiliary` 使用原 v8 的弱标签。

原始训练数据、评阅回执、模型、逐条预测与带身份字段的审计表从本地工作区读取或写入本地输出目录。正式发布路径使用原 v8 权重。
