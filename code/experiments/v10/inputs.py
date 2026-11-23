"""Checksums of the historical partitions and the fitted teacher."""
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

SPLITS = {
    "train": "d69a7baf883af5a888b7ef4e9d91db30a7260184fd08c9d3efd804fcb90de56c",
    "validation": "f55bca4ba28f14b5716da9231e891c828166659321fac6366e3085182ebef3cd",
    "test": "887512cfacf8ee352f5fd6d63fb2a102302a1954f2e6a11ea905c1bcdb50b76c",
}
TEACHER = {
    "model.safetensors": "b401c041631282fe25fdb49af761b1deb422d4bdb4a1ac411a43bff56b477c25",
    "config.json": "2840557d70c41bc33142d80ffb66649e570310488ed003000a3e6d1e0a246067",
    "tokenizer.json": "1d3622ca59494120ff0b540cbd13c86ce05e675f5d470cf4e73fe4caa55d1980",
    "tokenizer_config.json": "b3287ec27402e1f4a8081da3a1ee45a9b9d5738292282f22f6eb93b210814713",
}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected SHA-256 for {path.name}")


def split(directory, name):
    path = directory / f"{name}.csv"
    check(path, SPLITS[name])
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        tmp.write_bytes(payload)
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Existing output differs: {path}")
    finally:
        tmp.unlink(missing_ok=True)
