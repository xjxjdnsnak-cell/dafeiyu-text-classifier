# 依赖和数据说明

- fastText / fasttext-wheel 0.9.2：MIT。https://github.com/facebookresearch/fastText 与 https://github.com/messense/fasttext-wheel 。原始模型/量化格式及原生训练推理来自 fastText。
- jieba 0.42.1：MIT。https://github.com/fxsjy/jieba 。固定分词词典来自此版本包内 dict.txt；JS 仅移植 HMM=False 的 DAG 分词逻辑。版权和许可证见 src/bili_ai/assets/jieba-LICENSE.txt。
- fasttext.wasm 1.0.1：MIT，第三方预编译 fastText WebAssembly 核心。https://github.com/DreamOfIce/fasttext.wasm 。vendor/LICENSE 保留原许可，vendor/provenance.json 保留分发包和文件校验。未打包其语言识别模型，不将该二进制称作官方构建。
- NumPy 1.26.4：BSD-3-Clause。https://numpy.org 。许可证保留在离线 wheel 中。
- pytest：MIT，仅测试使用。https://docs.pytest.org 。许可证保留在离线 wheel 中。
- 本次评论来自用户提供的两个 ZIP，本项目不声明拥有评论著作权，也未发布到外部服务。原始规则弱标签来源在第一包中标为 liyupi/dafeiyu-cleaner@32866038316f81d77557f20a213b8c3723bf9ce0（MIT）；本项目不执行该规则，只读取既有标签。

开发时下载依赖与阅读官方文档会使用网络；已安装后的所有训练、量化、评估、Python/JS/WASM 推理和本地自测不调用外网或外部 AI API。部署目录的 fetch 仅加载随包文件。
