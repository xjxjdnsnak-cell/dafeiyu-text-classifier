"""Export an offline, checksummed inference bundle and cross-runtime vectors."""

import shutil
import sys
from pathlib import Path

from .common import read_csv, write_json, sha256
from .data import FIELDS
from .model import positive_scores, load_model
from .pipeline import verify_prepared, _check_metadata, check_model_pair
from .preprocess import ASSETS, MODES, Preprocessor


def deployment_assets():
    candidates = [Path(__file__).resolve().parents[2] / "deploy", Path(sys.prefix) / "share/bili-ai/deploy"]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in ("preprocess.mjs", "classifier.mjs", "verify.mjs", "verify-node.mjs", "verify.html")):
            return candidate
    raise ValueError("缺少部署脚本，请重新安装包含 share/bili-ai/deploy 的本项目 wheel")


def export_bundle(data_dir, models_dir, output, deploy_dir=None, modes=MODES):
    processed, manifest = verify_prepared(data_dir)
    output = Path(output)
    if not modes or any(mode not in MODES for mode in modes):
        raise ValueError("无效导出特征模式")
    for mode in modes:
        folder = Path(models_dir) / mode
        check_model_pair(folder / "model.bin", folder / "model.ftz", manifest)
    if output.resolve() == Path(models_dir).resolve():
        raise ValueError("导出目录不得是模型目录")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS, output / "assets", dirs_exist_ok=True)
    pre = Preprocessor()
    rows = read_csv(processed / "test.csv", FIELDS)
    samples = [{"text": r["text"], "username": r["username"], "kind": "real_test_input"} for r in rows]
    # Edge fixtures are software tests, never training or evaluation samples.
    samples += [{"text": text, "username": nick, "kind": "software_edge_fixture"} for text, nick in [
        ("", ""), ("你好\n__label__1 世界", "鱼鱼111"), ("ＡＢＣ　QQ+插件 123.45% #tag", "Ｆｏｏ鱼００７"),
        ("emoji🐟👩‍💻\u200b与换行\r\n中文", "🐟🐟🐟"), ("扩展汉字𠮷野家 与 é", "XZZ-二号机"),
        ("只有评论", ""), ("", "只有昵称"), ("南京市长江大桥 研究生命起源", "aaa111鱼鱼"),
        ("前缀\t正文\u00a0结束\ufeff标签", "用户_ABC-123"), ("重复" * 2200, "鱼" * 70)
    ]]
    for sample in samples:
        sample["features"] = {mode: pre.encode(sample["text"], sample["username"], mode) for mode in MODES}
        sample["scores"] = {}
    entries = []
    for mode in modes:
        for suffix in ("bin", "ftz"):
            source = Path(models_dir) / mode / f"model.{suffix}"
            metadata = _check_metadata(source, manifest)
            dest = output / f"{mode}.{suffix}"
            shutil.copyfile(source, dest)
            shutil.copyfile(str(source) + ".json", str(dest) + ".json")
            model = load_model(source)
            scores = positive_scores(model, [s["features"][mode] for s in samples])
            key = f"{mode}.{suffix}"
            for sample, score in zip(samples, scores, strict=True):
                sample["scores"][key] = score
            entries.append({"file": key, "mode": mode, "threshold": metadata["threshold"], "bytes": dest.stat().st_size, "sha256": metadata["model_sha256"]})
            del model
    write_json(output / "golden.json", samples)
    deploy_dir = Path(deploy_dir) if deploy_dir else deployment_assets()
    for name in ("preprocess.mjs", "classifier.mjs", "verify.mjs", "verify-node.mjs", "verify.html"):
        shutil.copyfile(deploy_dir / name, output / name)
    vendor = deploy_dir / "vendor"
    for name in ("fasttext.mjs", "fasttext.wasm", "LICENSE", "provenance.json"):
        if not (vendor / name).is_file():
            raise ValueError("缺少离线 WASM 文件；源码安装请先运行 scripts/vendor_wasm.py")
        (output / "vendor").mkdir(exist_ok=True)
        shutil.copyfile(vendor / name, output / "vendor" / name)
    write_json(output / "bundle.json", {"schema": 1, "models": entries, "dictionary_bytes": (ASSETS / "jieba.dict.txt").stat().st_size, "golden_cases": len(samples), "note": "WASM runtime must be bundled locally. No CDN or external API at inference. UID blacklist stays on each device and is not exported.", "android_status": "not validated on Android hardware"})
    managed = [output / "golden.json", output / "bundle.json"]
    managed += [output / name for name in ("preprocess.mjs", "classifier.mjs", "verify.mjs", "verify-node.mjs", "verify.html")]
    managed += list((output / "assets").glob("*"))
    managed += [output / "vendor" / name for name in ("fasttext.mjs", "fasttext.wasm", "LICENSE", "provenance.json")]
    for entry in entries:
        managed += [output / entry["file"], output / (entry["file"] + ".json")]
    files = {p.relative_to(output).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in managed if p.is_file()}
    write_json(output / "checksums.json", files)
    return {"output": str(output.resolve()), "files": len(files), "total_bytes": sum(v["bytes"] for v in files.values()), "golden_cases": len(samples)}
