# TinyBERT 4L 中文正文分类器（逐通道 INT8 量化 ONNX）

基于 `huawei-noah/TinyBERT_4L_zh`（revision `152bf15d86715b41dd89c1e03cf5664963d9b005`）在 v8 同源遮名正文划分上微调 4 轮（`TB-Hard`，纯交叉熵监督），导出 ONNX 后对 `MatMul` 与 `Gather` 做 **逐通道有符号动态 INT8 量化（`per_channel=True, QuantType.QInt8`）**，保留 `/pooler/` 与 `/classifier/` 为 FP32。

此模型作为轻量级神经正文模型发布在 `models/tinybert-perchannel-int8/`，**不影响现有 fastText v8 (`model/text/model.bin`) 的发布状态**。

## 冻结工件

- 模型文件：`models/tinybert-perchannel-int8/tb-hard-perchannel-qint8.onnx`（11,966,761 字节 / 11.41 MB）
- 模型 SHA-256：`72af82be617ab6b01c5373a546bb3e103418decc03079a83dfd626a7242289aa`
- 分词器文件：`models/tinybert-perchannel-int8/tokenizer.json`（SHA-256：`a7a002f10a1b86cfec8d77a14bfdbde8c556499091dca2d1523e94d3de1a6c6b`）
- 冻结阈值：`0.9905704259872437`（按 2,764 条验证集 1% 误拦预算选定）
- 预处理：仅正文；`@([^\s@:：，,。！!?？]{1,24})` 替换为 `@某人`；纯 `@` 提及（含已知表情）在评分前跳过；最大长度 128 token。

## 量化方案对比（基于同一 TB-Hard FP32 权重）

所有方案在 2,764 条验证集上各自按 1% 误拦预算重选阈值后，评估 228 条风格盲评（设计权重加权）：

| 方案 | 模型体积 | 验证召回 (1% 预算) | 与 FP32 分差 P95 | 盲评加权召回 | 盲评加权误拦 | 盲评 AP |
|---|---:|---:|---:|---:|---:|---:|
| FP32 基线 | 43.64 MB | 92.04% | 0.0000 | 86.25% (121/129) | 1.58% (4/99) | 95.73% |
| 旧逐张量 QUInt8 | 11.36 MB | 89.84% | 0.0569 | 82.79% (118/129) | 1.46% (3/99) | 95.19% |
| **逐通道 QInt8（入选）** | **11.41 MB** | **90.89%** | **0.0119** | **86.25% (121/129)** | **1.17% (2/99)** | **95.54%** |
| FP16（LayerNorm/Softmax 留 FP32） | 21.91 MB | 92.04% | 0.0001 | 86.25% (121/129) | 1.58% (4/99) | 95.73% |

历史弱标签留出集（2,379 条分词去重后）：逐通道 QInt8 在冻结阈值下召回 **56.38%**（393/697）、误拦 **0.12%**（2/1,682）、AP **93.13%**。

> 注：228 条风格盲评为既有复用基准，仅用于模型迭代对比，不等同于独立前瞻验收。

## 安卓真机实测（arm64-v8a，ONNX Runtime CPU）

在天玑 9400 手机上使用 100 条真实评论动态长度（不补齐、`batch=1` 逐条推理）实测：

| 线程数 | 加载耗时 | 单条 P50 | 单条 P95 | 一页 20 条合计 | 进程内存增量 |
|---:|---:|---:|---:|---:|---:|
| 1 线程 | 88.3 ms | 3.23 ms | 8.21 ms | 94.3 ms | +46.8 MB |
| 2 线程 | 70.8 ms | 2.64 ms | 6.25 ms | 73.8 ms | +46.9 MB |
| 4 线程 | 71.4 ms | 1.93 ms | 4.33 ms | 51.1 ms | +47.3 MB |

- 补齐到 128 token 做 `batch=20`（4 线程）耗时 94.2 ms，比动态长度逐条慢约 1.8 倍，因此移动端采用动态长度逐条推理。
- FP16 在手机 CPU 4 线程下一页 20 条需 263.6 ms（慢于 FP32 的 108.7 ms），故淘汰。

## 使用方法

```bash
pip install onnxruntime transformers numpy
python code/predict_tinybert_onnx.py "待判定的评论正文"
```

或在 Python 中调用：

```python
from code.predict_tinybert_onnx import TinyBertOnnxPredictor

predictor = TinyBertOnnxPredictor()
print(predictor.predict_one("待判定的评论正文"))
```
