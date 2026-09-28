#!/usr/bin/env python3
"""W40 v8: Ternary-Bonsai-2-27B on a Kaggle T4 — the CUDA truth for a Mac-famous model.

prism-ml's Ternary-Bonsai-2-27B (3.46M HF downloads on the GGUF repo) is
benchmarked everywhere on Apple Silicon (card: ~47 tok/s on an M5 Max); the
coverage claim to test is what it does on the *other* cheap hardware — a free
NVIDIA T4. Wrinkles discovered in scouting, all load-bearing:

  - STOCK llama.cpp CANNOT load these GGUFs: PQ2_0/PTQ1_0 are fork-specific
    GGML types (142/143, open upstream feature request 2026-09-18), and the
    demo repo explicitly warns that a Q2_0 path loads "without the required
    transforms and produces gibberish". The PrismML fork with the custom
    ternary hybrid-attention kernels is mandatory.
  - No CUDA toolchain needed anyway: the fork pins PREBUILT binaries at
    release `prism-b10743-adfffbe` (the Bonsai-demo's pinned version),
    including linux-cuda tarballs — sidesteps the W38 "no usable nvcc on
    Kaggle" lesson entirely.

Rows (all -ngl 99, fp16 KV, default flags):
  1  Ternary-Bonsai-2-27B-PTQ1_0  (1.75 bpw ternary packing, 5.95 GB)
  2  Ternary-Bonsai-2-27B-PQ2_0   (2.13 bpw ternary packing, 7.21 GB)
  3  unsloth Qwen3.8-27B UD-IQ2_XXS (7.27 GB — conventional quant at the
     same footprint; the card claims ternary retains 98.2% of FP16 quality
     where conventional IQ2 collapses to 72.59 on their bench)

Method: llama-bench (pp512 / tg128, -r 3) from the pinned fork build —
clean-context numbers, no chat-template pollution (the W38 v2 lesson) — with
a background nvidia-smi sampler for peak VRAM, plus a short llama-cli smoke
generation per model to prove coherent decoding (not just a clean load).
"""

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request

T0 = time.time()
REL = "prism-b10743-adfffbe"
REL_URL = (f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{REL}/"
           f"llama-{REL}-bin-linux-cuda-12.4-x64.tar.gz")
BIN = "/tmp/llama"
MODELS = "/tmp/models"
OUT = "/kaggle/working/out"
BFORK = "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main"
UQ = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main"
ROWS = [
    ("PTQ1_0_ternary_1.75bpw", f"{BFORK}/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
     "Ternary-Bonsai-2-27B-PTQ1_0.gguf", 5.95),
    ("PQ2_0_ternary_2.13bpw", f"{BFORK}/Ternary-Bonsai-2-27B-PQ2_0.gguf",
     "Ternary-Bonsai-2-27B-PQ2_0.gguf", 7.21),
    ("Qwen3.8-27B_UD-IQ2_XXS_conventional", f"{UQ}/Qwen3.8-27B-UD-IQ2_XXS.gguf",
     "Qwen3.8-27B-UD-IQ2_XXS.gguf", 7.27),
]

RES = {"run": "ternary-bonsai2-27b-t4", "surface": "kaggle",
       "llama_build": REL, "rows": []}


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def save():
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/results_v8_ternary.json", "w") as f:
        json.dump(RES, f, indent=2)


def expected_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length") or 0)


def wget_or_die(url, dest, min_bytes=1e6):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    want = expected_size(url)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_bytes and (
            not want or os.path.getsize(dest) >= want):
        log(f"cached: {os.path.basename(dest)}")
        return
    t0 = time.time()
    for attempt in range(1, 9):
        rc = subprocess.run(["wget", "-q", "-c", "--tries=3", "--timeout=120",
                             "--waitretry=10", "-O", dest, url]).returncode
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        if rc == 0 and size >= min_bytes and (not want or size >= want):
            log(f"download {os.path.basename(dest)} {size/1e9:.2f} GB in {time.time()-t0:.1f}s")
            return
        log(f"download attempt {attempt} incomplete rc={rc} {size/1e9:.2f}/{want/1e9:.2f} GB — resuming")
        time.sleep(10)
    sys.exit(f"download failed after retries: {url}")


def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True,
                          timeout=timeout)


class VramSampler:
    """Peak-VRAM sampler around a bench run (v1's 1 Hz sampling read only
    ~3 GB for 5.5-7 GB models — v2 samples 2x/s and keeps the reading list
    for the first bench to diagnose)."""

    def __init__(self, keep_trace=False):
        self.peak = 0
        self.trace = []
        self._keep = keep_trace
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
                for line in out.splitlines():
                    mb = int(line.strip())
                    self.peak = max(self.peak, mb)
                    if self._keep:
                        self.trace.append(mb)
            except Exception:
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=5)


def bench(binary, gguf, keep_trace=False):
    """llama-bench pp512/tg128 with VRAM sampling; parses this fork's LONG
    table (v1 lesson): one row per test — | model | size | params | backend |
    ngl | test | t/s | with values like '302.20 ± 20.76'."""
    cmd = [f"{BIN}/llama-bench", "-m", gguf, "-ngl", "99", "-p", "512",
           "-n", "128", "-r", "3"]
    with VramSampler(keep_trace=keep_trace) as v:
        t0 = time.time()
        r = subprocess.run(cmd, text=True, capture_output=True, timeout=1500)
        wall = round(time.time() - t0, 1)
    rec = {"cmd": " ".join(cmd), "exit": r.returncode, "wall_s": wall,
           "vram_peak_gb": round(v.peak / 1024, 2)}
    if keep_trace:
        rec["vram_trace_mb_head"] = v.trace[:40]
        rec["vram_trace_n"] = len(v.trace)
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-500:]
        return rec
    for line in r.stdout.splitlines():
        if not line.strip().startswith("|") or "llama-bench" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7 or cells[5] not in ("pp512", "tg128"):
            continue
        m = re.match(r"([0-9.]+)", cells[6])
        if not m:
            continue
        rec[f"{cells[5]}_tps"] = float(m.group(1))
        rec["backend"] = cells[3]
        rec["ngl"] = cells[4]
    # steady reading right after the run (weights still resident)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        rec["vram_end_gb"] = round(int(out.splitlines()[0]) / 1024, 2)
    except Exception:
        pass
    rec["bench_raw"] = r.stdout[-800:]
    return rec


def smoke(gguf):
    """Coherence probe: a real generation through llama-cli (thinking model,
    so just record the head — we are checking readable tokens, not answers)."""
    p = ('<|im_start|>user\nName the capital of France in one word.'
         '<|im_end|>\n<|im_start|>assistant\n')
    cmd = [f"{BIN}/llama-cli", "-m", gguf, "-ngl", "99", "-c", "2048",
           "-n", "200", "--temp", "0.0", "-p", p, "--no-display-prompt",
           "--single-turn", "-e"]
    t0 = time.time()
    r = subprocess.run(cmd, text=True, capture_output=True, timeout=600)
    out = (r.stdout or "").strip()
    return {"exit": r.returncode, "wall_s": round(time.time() - t0, 1),
            "n_chars": len(out),
            "head": re.sub(r"\s+", " ", out[:220]),
            "has_think_tag": "<think>" in out or "</think>" in out}


def main():
    log("=== env probe ===")
    g = sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
    RES["gpu"] = g.stdout.strip()
    log(f"gpu: {RES['gpu']}")
    save()

    log("=== pinned fork binaries ===")
    wget_or_die(REL_URL, "/tmp/llama.tar.gz", min_bytes=1e8)
    sh(f"mkdir -p {BIN} && tar xzf /tmp/llama.tar.gz -C {BIN} --strip-components=1")
    found = sorted(os.listdir(BIN))
    RES["binaries"] = found
    log(f"binaries: {found}")
    if "llama-bench" not in found:
        sys.exit("llama-bench missing from the pinned release tarball")
    vb = sh(f"{BIN}/llama-bench --version")
    RES["bench_version"] = (vb.stdout or vb.stderr).strip()[:200]
    log(f"llama-bench: {RES['bench_version']}")
    save()

    log("=== downloads (resume-loop) ===")
    for label, url, fname, size_gb in ROWS:
        wget_or_die(url, f"{MODELS}/{fname}", min_bytes=size_gb * 1e9 * 0.9)
    save()

    for i, (label, url, fname, size_gb) in enumerate(ROWS):
        gguf = f"{MODELS}/{fname}"
        log(f"=== row: {label} ===")
        rec = {"key": label, "file": fname,
               "size_gb": round(os.path.getsize(gguf) / 1e9, 2)}
        rec.update(bench(BIN, gguf, keep_trace=(i == 0)))
        log(f"bench: pp={rec.get('pp512_tps')} tg={rec.get('tg128_tps')} "
            f"wall={rec.get('wall_s')}s vram_peak={rec.get('vram_peak_gb')}GB "
            f"vram_end={rec.get('vram_end_gb')}GB")
        if rec["exit"] == 0:
            rec["smoke"] = smoke(gguf)
            log(f"smoke: exit={rec['smoke']['exit']} chars={rec['smoke']['n_chars']} "
                f"head={rec['smoke']['head'][:80]}")
        RES["rows"].append(rec)
        save()

    RES["success"] = all(r.get("pp512_tps") for r in RES["rows"])
    RES["total_s"] = round(time.time() - T0, 1)
    save()
    log(f"=== DONE success={RES['success']} total={RES['total_s']}s ===")
    print("=== RESULTS_V8_JSON ===", flush=True)
    print(json.dumps(RES, indent=2), flush=True)
    sys.exit(0 if RES["success"] else 1)


if __name__ == "__main__":
    main()
