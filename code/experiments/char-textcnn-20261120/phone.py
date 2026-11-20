"""Benchmark raw text to score in an Android browser worker over a USB loopback port."""
import argparse
import functools
import json
import shutil
import subprocess
import tarfile
import tempfile
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from torch import nn
from compare import CharCNN, SEEDS, batch_ids, body, encode, save

PAGE = """<!doctype html><meta charset="utf-8"><title>正文模型单条计时</title>
<p id="msg">正在本机核对分数、预热并计时…</p><script type="module">
const worker = new Worker('./phone.mjs', {type:'module'});
async function done(data) {await fetch('./result',{method:'POST',body:JSON.stringify(data)});
 document.querySelector('#msg').textContent=data.error?'运行出错':'计时完成，可以关闭此页。';worker.terminate();}
worker.onmessage=e=>done(e.data); worker.onerror=e=>done({error:e.message});
</script>"""


def prepare(site, results, ft_assets, npm_archive):
    vocab = json.loads((results / "vocab.json").read_text(encoding="utf-8"))
    model = CharCNN(len(vocab) + 2)
    model.load_state_dict(torch.load(results / f"seed{SEEDS[0]}.pt", weights_only=True, map_location="cpu"))
    frozen = json.loads((results / "freeze.json").read_text(encoding="utf-8"))
    model = nn.Sequential(model.eval(), nn.Sigmoid()).eval()
    onnx_path = results / "cnn-seed17.onnx"
    torch.onnx.export(model, torch.ones(1, 10, dtype=torch.long), str(onnx_path), dynamo=False,
                      input_names=["ids"], output_names=["score"], opset_version=17,
                      dynamic_axes={"ids": {1: "width"}})
    shutil.copy(onnx_path, site / "cnn.onnx")
    shutil.copy(results / "vocab.json", site)
    # Compare Android WASM against the same float32 CPU forward and retain the GPU
    # decisions separately. cuDNN uses TF32 on the training/evaluation host.
    sample = json.loads((results / "phone-sample.private.json").read_text(encoding="utf-8"))
    torch.set_num_threads(1)
    with torch.inference_mode():
        for row in sample:
            ids = encode([body(row)], vocab)
            row["cnn_cuda"] = row["cnn"]
            row["cnn"] = model(batch_ids(ids, "cpu")).item() if len(ids[0]) else None
    save(site / "sample.json", sample)
    save(site / "thresholds.json", {"fasttext": frozen["fasttext_threshold"],
                                    "cnn": frozen["cnn"][str(SEEDS[0])]["threshold"]})
    for name in ("manifest.json", "inference.mjs", "preprocess.mjs", "assets/preprocess.json",
                 "assets/known_bilibili_emotes.json", "assets/jieba.dict.txt", "v9.bin",
                 "vendor/fasttext.mjs", "vendor/fasttext.wasm"):
        target = site / "ft" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ft_assets / name, target)
    (site / "ort").mkdir()
    with tarfile.open(npm_archive) as archive:
        for name in ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
            (site / "ort" / name).write_bytes(archive.extractfile("package/dist/" + name).read())
    shutil.copy(Path(__file__).with_name("phone.mjs"), site)
    (site / "index.html").write_text(PAGE, encoding="utf-8")


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {".mjs": "text/javascript", ".wasm": "application/wasm"}

    def end_headers(self):
        for key, value in (("Cross-Origin-Opener-Policy", "same-origin"), ("Cross-Origin-Embedder-Policy","require-corp"),
                           ("Cache-Control","no-store")):
            self.send_header(key, value)
        super().end_headers()

    def do_POST(self):
        if self.path != "/result":
            self.send_error(404)
            return
        value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.end_headers()
        self.server.result = value


def benchmark(site, results, adb, serial):
    def command(*parts):
        return subprocess.check_output([str(adb), "-s", serial, *parts], text=True, encoding="utf-8").strip()
    with ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site))) as server:
        port = "tcp:" + str(server.server_port)
        server.result = None
        server.timeout = 1
        command("reverse", "--no-rebind", port, port)
        try:
            android = command("shell","getprop","ro.build.version.release")
            device = command("shell","getprop","ro.product.model")
            soc = command("shell","getprop","ro.soc.model")
            version = command("shell","dumpsys","package","com.android.chrome")
            chrome = next(s.strip() for s in version.splitlines() if "versionName=" in s)
            command("shell","am","start","-a","android.intent.action.VIEW","-d",
                    f"http://127.0.0.1:{server.server_port}/","-p","com.android.chrome")
            deadline = time.monotonic() + 150
            while server.result is None and time.monotonic() < deadline:
                server.handle_request()
            if server.result is None:
                raise TimeoutError("USB loopback browser benchmark did not report completion")
            if "error" in server.result:
                raise RuntimeError(server.result["error"])
            report = dict(server.result, android=android, device=device, soc=soc, chrome=chrome,
                          ort_web="1.24.3", onnx_bytes=(site/"cnn.onnx").stat().st_size,
                          vocab_bytes=(site/"vocab.json").stat().st_size,
                          fasttext_bytes=(site/"ft/v9.bin").stat().st_size)
            save(results / "phone.json", report)
            print(json.dumps(report, ensure_ascii=False))
        finally:
            command("reverse", "--remove", port)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("results", "client_assets", "npm_archive", "adb"):
        p.add_argument(name, type=Path)
    p.add_argument("serial")
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp)
        prepare(site, args.results, args.client_assets, args.npm_archive)
        benchmark(site, args.results, args.adb, args.serial)
