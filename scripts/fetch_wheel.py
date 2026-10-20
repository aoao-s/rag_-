"""Parallel range downloader for the torch CPU wheel.

Why not curl: the mirror stalls mid-transfer and curl keeps the connection open
forever. This script uses socket timeouts + per-range resume, so a stalled range
is aborted and retried from the last byte received.

Usage:  .venv\\Scripts\\python.exe scripts\\fetch_wheel.py
"""

from __future__ import annotations

import hashlib
import sys
import threading
import time
import urllib.request
from pathlib import Path

URL = (
    "https://mirrors.aliyun.com/pypi/packages/5f/ba/"
    "607d013b55b9fd805db2a5c2662ec7551f1910b4eef39653eeaba182c5b2/"
    "torch-2.5.1-cp312-cp312-win_amd64.whl"
)
SHA256 = "73e58e78f7d220917c5dbfad1a40e09df9929d3b95d25e57d9f8558f84c9a11c"
TOTAL = 212_926_464
PARTS = 8
CHUNK = 256 * 1024
TIMEOUT = 25  # seconds without a byte -> abort and resume

ROOT = Path(r"F:\heima\RAG\RAGpro_智能教育系统")
WHEELS = ROOT / "data" / "wheels"
PARTS_DIR = WHEELS / "parts"
OUT = WHEELS / "torch-2.5.1-cp312-cp312-win_amd64.whl"

stop = threading.Event()
lock = threading.Lock()
done_bytes = 0


def ranges() -> list[tuple[int, int]]:
    size = -(-TOTAL // PARTS)
    out = []
    for i in range(PARTS):
        s = i * size
        e = min(s + size - 1, TOTAL - 1)
        out.append((s, e))
    return out


def fetch(index: int, start: int, end: int) -> None:
    global done_bytes
    path = PARTS_DIR / f"p{index}"
    want = end - start + 1
    fails = 0
    while not stop.is_set():
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return
        if have > want:
            path.write_bytes(path.read_bytes()[:want])
            return
        req = urllib.request.Request(URL, headers={"Range": f"bytes={start + have}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                with path.open("ab") as fh:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        fh.write(block)
                        with lock:
                            done_bytes += len(block)
            fails = 0
        except Exception as exc:  # noqa: BLE001 - network errors of all kinds
            fails += 1
            print(f"[p{index}] retry {fails}: {type(exc).__name__} {exc}", flush=True)
            if fails > 60:
                print(f"[p{index}] giving up after {fails} failures", flush=True)
                stop.set()
                return
            time.sleep(min(1 + fails, 8))


def report() -> None:
    while not stop.is_set():
        time.sleep(10)
        with lock:
            got = done_bytes
        print(f"progress: {got / 1e6:6.1f} / {TOTAL / 1e6:.1f} MB ({got / TOTAL * 100:5.1f}%)", flush=True)


def main() -> int:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    WHEELS.mkdir(parents=True, exist_ok=True)
    global done_bytes
    done_bytes = sum(p.stat().st_size for p in PARTS_DIR.glob("p*"))

    threads = [threading.Thread(target=report, daemon=True)]
    for i, (s, e) in enumerate(ranges()):
        threads.append(threading.Thread(target=fetch, args=(i, s, e), daemon=True))
    for t in threads:
        t.start()
    for t in threads[1:]:
        t.join()
    stop.set()

    if any(not (PARTS_DIR / f"p{i}").exists() for i in range(PARTS)):
        print("missing parts, rerun")
        return 1

    with OUT.open("wb") as out:
        for i, (s, e) in enumerate(ranges()):
            data = (PARTS_DIR / f"p{i}").read_bytes()
            want = e - s + 1
            if len(data) != want:
                print(f"part {i} size {len(data)} != {want}, rerun")
                return 1
            out.write(data)
            print(f"part {i}: {len(data) / 1e6:.1f} MB", flush=True)

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"total {OUT.stat().st_size / 1e6:.1f} MB  sha256 {digest}", flush=True)
    if digest != SHA256:
        print("sha256 mismatch - abort")
        return 1
    print("sha256 ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
