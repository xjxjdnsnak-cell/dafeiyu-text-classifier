"""Privacy sweep over the git delta (robust path parsing, skips itself)."""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent
out = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--porcelain", "-uall"],
                     cwd=REPO, capture_output=True).stdout.decode("utf-8")
paths = []
for line in out.splitlines():
    if not line.strip():
        continue
    path = line[3:].strip().strip('"')
    if path and Path(path).name != "privacy_sweep.py":
        paths.append(path)

suspicious = []
regexes = {
    "SESSDATA/cookie": re.compile(r"SESSDATA|bili_jct|DedeUserID", re.I),
    "token/key": re.compile(r"(api[_-]?key|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", re.I),
    "uid_or_username_field": re.compile(r'"uid"\s*:\s*"|"mid"\s*:\s*"|"username"\s*:\s*"'),
    "user_name": re.compile(r"xsjhxs", re.I),
    "wxid": re.compile(r"wxid_[a-z0-9]{6,}", re.I),
}
plain = ["E:/human-review", "E:" + chr(92) + "human-review", "D:/xwechat", "D:" + chr(92) + "xwechat"]
for p in paths:
    path = REPO / p
    if path.suffix in (".bin", ".ftz"):
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue
    for name, pattern in regexes.items():
        hits = pattern.findall(text)
        if hits:
            suspicious.append((p, name, len(hits)))
    for needle in plain:
        if needle in text:
            suspicious.append((p, "plain:" + needle, text.count(needle)))
for row in suspicious:
    print(row)
print("checked files:", len(paths), "| flags:", len(suspicious))
