import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import __version__
from .common import read_csv, read_json, write_json
from .data import import_archives, prepare
from .deployment import export_bundle
from .dual import DualPredictor
from .model import Predictor, update_blacklist
from .pipeline import train, quantize, evaluate
from .preprocess import MODES


def parser():
    p = argparse.ArgumentParser(description="离线 B 站 AI 风格评论分类器（fastText + jieba）")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    imp = sub.add_parser("import-data", help="导入指定双 ZIP，保留来源和待复核记录")
    imp.add_argument("archives", type=Path, nargs="+")
    imp.add_argument("--data-dir", type=Path, default=Path("data"))
    imp.add_argument("--unreviewed-ai", action="store_true", help="经用户确认，把 app 的 unreviewed 标为 1；collector 的 review 仍排除")
    prep = sub.add_parser("prepare", help="清洗、UID 或行级划分、转换三套 fastText 文件")
    prep.add_argument("--data-dir", type=Path, default=Path("data"))
    prep.add_argument("--splits", type=Path, help="读取方案一的 uid,split CSV，禁止缺失/重复 UID")
    prep.add_argument("--row-splits", type=Path, help="行级划分 uid,text,split；同一 UID 可跨集合")
    for command in ("train", "quantize", "evaluate"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--data-dir", type=Path, default=Path("data"))
        cmd.add_argument("--models-dir", type=Path, default=Path("models"))
        cmd.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
        if command in ("train", "quantize"):
            cmd.add_argument("--max-fpr", type=float, default=0.01, help="验证集经验 FPR 上限，默认 0.01")
        if command == "train":
            cmd.add_argument("--tune", action="store_true", help="仅基于 train/validation 的 3 组小参数搜索")
            cmd.add_argument("--dim", type=int, default=32)
            cmd.add_argument("--bucket", type=int, default=20000)
            cmd.add_argument("--epoch", type=int, default=30)
            cmd.add_argument("--lr", type=float, default=0.5)
            cmd.add_argument("--word-ngrams", type=int, default=2)
            cmd.add_argument("--minn", type=int, default=2)
            cmd.add_argument("--maxn", type=int, default=4)
        elif command == "quantize":
            cmd.add_argument("--cutoff", type=int, default=20000)
            cmd.add_argument("--dsub", type=int, default=2)
        else:
            cmd.add_argument("--reports-dir", type=Path, default=Path("reports"))
            cmd.add_argument("--repeats", type=int, default=3)
    for command in ("predict", "batch"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--model", type=Path, default=Path("models/combined/model.ftz"))
        cmd.add_argument("--mode", choices=MODES, help="通常从模型元数据读取")
        cmd.add_argument("--threshold", type=float, help="手动覆盖验证集阈值（评分 >= 阈值判为 1）")
        cmd.add_argument("--blacklist", type=Path, default=Path("data/blacklist.txt"))
        cmd.add_argument("--assets", type=Path, help="冻结词典/规则目录")
        if command == "predict":
            cmd.add_argument("--text", default="")
            cmd.add_argument("--username", default="")
            cmd.add_argument("--uid", default="")
        else:
            cmd.add_argument("--input", type=Path, required=True, help="CSV 或 JSONL；uid/username/text 均可省略")
            cmd.add_argument("--output", type=Path, required=True, help="JSONL 输出")
            cmd.add_argument("--batch-size", type=int, default=64)
    for command in ("dual-predict", "dual-batch"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--text-model", type=Path, default=Path("models-dual/text/model.ftz"))
        cmd.add_argument("--nickname-model", type=Path, default=Path("models-dual/nickname/model.ftz"))
        cmd.add_argument("--config", type=Path, default=Path("models-dual/dual.json") if Path("models-dual/dual.json").exists() else None, help="双通道配置；未显式给出的阈值从这里读取")
        cmd.add_argument("--text-threshold", type=float, help="覆盖正文通道阈值")
        cmd.add_argument("--nickname-threshold", type=float, help="覆盖昵称通道阈值")
        cmd.add_argument("--blacklist", type=Path, default=Path("data/blacklist.txt"))
        cmd.add_argument("--assets", type=Path)
        if command == "dual-predict":
            cmd.add_argument("--text", default="")
            cmd.add_argument("--username", default="")
            cmd.add_argument("--uid", default="")
        else:
            cmd.add_argument("--input", type=Path, required=True, help="CSV 或 JSONL；uid/username/text 均可省略")
            cmd.add_argument("--output", type=Path, required=True, help="JSONL 输出")
            cmd.add_argument("--batch-size", type=int, default=64)
    black = sub.add_parser("blacklist", help="独立本地 UID 黑名单")
    black.add_argument("action", choices=("add", "remove", "list"))
    black.add_argument("uid", nargs="?")
    black.add_argument("--file", type=Path, default=Path("data/blacklist.txt"))
    exp = sub.add_parser("export", help="导出模型/词典/规则/阈值/JS/WASM/验证向量")
    exp.add_argument("--data-dir", type=Path, default=Path("data"))
    exp.add_argument("--models-dir", type=Path, default=Path("models"))
    exp.add_argument("--output", type=Path, default=Path("deploy/bundle"))
    exp.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    return p


def _batch(predictor, input_path, output_path, batch_size):
    if batch_size < 1:
        raise ValueError("batch-size 必须 >=1")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("预测输出不得覆盖输入")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    def records():
        if input_path.suffix.lower() == ".csv":
            import csv
            with input_path.open(encoding="utf-8-sig", newline="") as f:
                yield from csv.DictReader(f)
        else:
            with input_path.open(encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("批量输入每行必须是 JSON 对象")
                        yield record
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, prefix=".prediction-", suffix=".tmp", delete=False)
    temp = Path(handle.name)
    try:
        with handle as f:
            batch = []
            for row in records():
                batch.append(row)
                if len(batch) >= batch_size:
                    for result in predictor.predict_many(batch):
                        f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                        count += 1
                    batch.clear()
            for result in predictor.predict_many(batch):
                f.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                count += 1
        temp.replace(output_path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return {"rows": count, "output": str(output_path.resolve())}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        if args.command == "import-data":
            result = import_archives(args.archives, args.data_dir, args.unreviewed_ai)
        elif args.command == "prepare":
            result = prepare(args.data_dir, args.splits, args.row_splits)
        elif args.command == "train":
            params = {"dim": args.dim, "bucket": args.bucket, "epoch": args.epoch, "lr": args.lr, "wordNgrams": args.word_ngrams, "minn": args.minn, "maxn": args.maxn}
            metadata = train(args.data_dir, args.models_dir, args.modes, params, args.tune, args.max_fpr)
            result = {m: {"model_bytes": v["model_bytes"], "threshold": v["threshold"], "validation": v["validation"]["metrics"], "params": v["params"]} for m, v in metadata.items()}
        elif args.command == "quantize":
            metadata = quantize(args.data_dir, args.models_dir, args.modes, args.cutoff, args.dsub, args.max_fpr)
            result = {m: {"model_bytes": v["model_bytes"], "threshold": v["threshold"], "validation": v["validation"]["metrics"]} for m, v in metadata.items()}
        elif args.command == "evaluate":
            report = evaluate(args.data_dir, args.models_dir, args.reports_dir, args.modes, args.repeats)
            result = {"reports": str(args.reports_dir.resolve()), "results": [{"mode": r["mode"], "format": r["format"], "metrics": r["metrics"], "model_bytes": r["model_bytes"]} for r in report["results"]]}
        elif args.command in ("predict", "batch"):
            predictor = Predictor(args.model, args.mode, args.threshold, args.blacklist, args.assets)
            result = predictor.predict_one(args.text, args.username, args.uid) if args.command == "predict" else _batch(predictor, args.input, args.output, args.batch_size)
        elif args.command in ("dual-predict", "dual-batch"):
            text_threshold, nickname_threshold = args.text_threshold, args.nickname_threshold
            if args.config:
                config = read_json(args.config)
                if text_threshold is None:
                    text_threshold = config["text_threshold"]
                if nickname_threshold is None:
                    nickname_threshold = config["nickname_threshold"]
            predictor = DualPredictor(args.text_model, args.nickname_model, text_threshold, nickname_threshold, args.blacklist, args.assets)
            result = predictor.predict_one(args.text, args.username, args.uid) if args.command == "dual-predict" else _batch(predictor, args.input, args.output, args.batch_size)
        elif args.command == "blacklist":
            result = {"uids": update_blacklist(args.file, args.action, args.uid)}
        elif args.command == "export":
            result = export_bundle(args.data_dir, args.models_dir, args.output, modes=args.modes)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, KeyError, TypeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
