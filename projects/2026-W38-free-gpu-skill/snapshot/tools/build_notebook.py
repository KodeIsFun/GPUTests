#!/usr/bin/env python3
"""Build colab/run-llm-t4.ipynb from the cell sources below.

Edit the cells here and re-run this script — never hand-edit the .ipynb.
Every code cell must TERMINATE (no blocking input(), no server in the
foreground): the notebook is executed headlessly by `colab run` for
verification, and each cell prints a grep-able marker (OK: install, OK:
download, OK: load, OK: chat, OK: api, API URL:) used by that verification.
"""
import json
from pathlib import Path

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = []

CELLS.append((MD, """\
# Run an LLM on this free GPU — and get an OpenAI-compatible API URL

One notebook, five minutes, no accounts beyond the Google one you are signed
in with. It installs llama.cpp (prebuilt CUDA wheel — the source build is a
known trap), downloads an **ungated** GGUF model, verifies the weights really
landed on the GPU, runs a chat demo, and (optional, on by default) serves an
**OpenAI-compatible API on a public URL** via a Cloudflare quick tunnel.

**Before Run all:** `Runtime → Change runtime type → T4 GPU → Save`
(the next cell checks this for you). Works on CPU too — just ~10× slower.

| Model (`MODEL` below) | File | Decode on T4 (measured) |
|---|---|---|
| `qwen3-4b` | 2.5 GB | ~63 tok/s |
| `qwen3-8b` (default) | 5.0 GB | ~40 tok/s |
| `qwen3-14b` | 9.0 GB | ~23 tok/s |
| `gpt-oss-20b` | 12.1 GB | ~59 tok/s (MoE; template caveat near the end) |

Measured on a free Kaggle T4, 2026-09: <https://www.kaggle.com/code/emdadh/gguf-on-t4>
"""))

CELLS.append((CODE, """\
# ---- Config: edit these lines and Run all ------------------------------
MODEL = "qwen3-8b"        # qwen3-4b | qwen3-8b | qwen3-14b | gpt-oss-20b
SERVE_API = True          # OpenAI-compatible API on a public URL
TUNNEL = "cloudflared"    # "cloudflared" (no account) | "ngrok" (needs token)
NGROK_AUTHTOKEN = ""      # only needed when TUNNEL == "ngrok"
N_CTX = 4096
QUESTION = "Name three planets in the solar system and one fact about each."
# ------------------------------------------------------------------------
print("Config:", MODEL, "| serve:", SERVE_API, "| tunnel:", TUNNEL)
"""))

CELLS.append((CODE, """\
import subprocess

def nvidia_smi():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        name, total = [p.strip() for p in out.stdout.splitlines()[0].split(",")]
        return {"name": name, "vram_mb": float(total)}
    except Exception:
        return None

GPU = nvidia_smi()
if GPU:
    HAS_GPU = True
    print(f"OK: GPU runtime detected — {GPU['name']} with {int(GPU['vram_mb'])} MB VRAM")
else:
    HAS_GPU = False
    print("NOTICE: no GPU attached. The notebook still works on CPU (~10x slower).")
    print("Fix: Runtime -> Change runtime type -> T4 GPU -> Save, then Run all.")
"""))

CELLS.append((CODE, """\
# llama.cpp with CUDA, from the prebuilt wheel index. The from-source build
# dies on these images with an unreadable error — do not debug it, wheel it.
import sys, time

t0 = time.time()
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet",
     "llama-cpp-python[server]", "huggingface_hub",
     "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"],
    capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    raise SystemExit("install failed — see the log above")
print(f"install: {time.time()-t0:.0f} s (prebuilt cu124 wheel)")

import llama_cpp
print("llama-cpp-python", llama_cpp.__version__)
supports = getattr(llama_cpp.llama_cpp, "llama_supports_gpu_offload", lambda: None)()
if HAS_GPU and supports is False:
    raise SystemExit("wheel reports NO GPU offload — refusing to continue on CPU-by-accident")
print("OK: install" + (" (GPU offload supported)" if supports else ""))
"""))

CELLS.append((CODE, """\
# Ungated catalog: no Hugging Face license walls, nothing to click.
import os, time
from huggingface_hub import hf_hub_download

CATALOG = {  # ordered small -> large; also the OOM-fallback order
    "qwen3-4b":   ("bartowski/Qwen_Qwen3-4B-GGUF",   "Qwen_Qwen3-4B-Q4_K_M.gguf",   2.50),
    "qwen3-8b":   ("bartowski/Qwen_Qwen3-8B-GGUF",   "Qwen_Qwen3-8B-Q4_K_M.gguf",   5.03),
    "qwen3-14b":  ("bartowski/Qwen_Qwen3-14B-GGUF",  "Qwen_Qwen3-14B-Q4_K_M.gguf",  9.00),
    "gpt-oss-20b":("ggml-org/gpt-oss-20b-GGUF",      "gpt-oss-20b-MXFP4.gguf",     12.11),
}
if MODEL not in CATALOG:
    raise SystemExit(f"MODEL must be one of {list(CATALOG)}")

order = list(CATALOG)
i = order.index(MODEL)
candidates = order[i:] + list(reversed(order[:i]))  # requested first, then smaller

t0 = time.time()
model_path = hf_hub_download(repo_id=CATALOG[MODEL][0], filename=CATALOG[MODEL][1])
size_gb = os.path.getsize(model_path) / 1e9
print(f"downloaded {MODEL}: {size_gb:.2f} GB in {time.time()-t0:.0f} s")
print("OK: download")
"""))

CELLS.append((CODE, """\
# Fit precheck (honest margins), load with offload-log proof, OOM fallback.
import io, time
from contextlib import redirect_stderr
from llama_cpp import Llama

VRAM_ATTEMPT_RESERVE_MB = 1500  # KV at 4k ctx + CUDA context + buffers
loaded, used_model = None, None

for name in candidates:
    if HAS_GPU and GPU and name == MODEL:
        need = CATALOG[name][2] * 1024 + VRAM_ATTEMPT_RESERVE_MB
        if need > GPU["vram_mb"]:  # fresh runtime: total ~ free
            print(f"skip {name}: needs ~{int(need)} MB, card has "
                  f"{int(GPU['vram_mb'])} MB — trying smaller")
            continue
    fetch = model_path if name == MODEL else hf_hub_download(
        repo_id=CATALOG[name][0], filename=CATALOG[name][1])
    cap = io.StringIO()
    t0 = time.time()
    try:
        with redirect_stderr(cap):
            llm = Llama(model_path=fetch, n_gpu_layers=-1, n_ctx=N_CTX, verbose=True)
    except Exception as e:
        print(f"load failed for {name}: {type(e).__name__} — trying smaller")
        continue
    print(f"loaded {name} in {time.time()-t0:.1f} s")
    offload = [ln.strip() for ln in cap.getvalue().splitlines()
               if "offloaded" in ln.lower()]
    if offload:
        print("offload proof:", offload[0])
    elif HAS_GPU:
        print("WARNING: no offload line found — speed below will tell the truth")
    loaded, used_model, model_path = llm, name, fetch
    break

if loaded is None:
    raise SystemExit("every candidate failed to load — see messages above")
print("OK: load ->", used_model)
"""))

CELLS.append((CODE, """\
# Chat demo — greedy, so the answers are reproducible.
import time

def chat(llm, user, max_tokens=512):
    kwargs = {}
    if used_model.startswith("gpt-oss"):
        kwargs["chat_template_kwargs"] = {"reasoning_effort": "low"}
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.0, **kwargs)
    except (TypeError, ValueError):
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": user}],
            max_tokens=max_tokens, temperature=0.0)
    text = (out["choices"][0]["message"] or {}).get("content") \\
           or out["choices"][0].get("text") or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()

t0 = time.time()
answer = chat(loaded, QUESTION)
print(f"Q: {QUESTION}")
print(f"A ({time.time()-t0:.1f} s): {answer[:600]}")
print("OK: chat")
"""))

CELLS.append((CODE, """\
# OpenAI-compatible API + public URL. Cloudflare quick tunnel: no account.
# The server loads the model in its OWN process, so free this one's copy
# first — two copies of a 9 GB model cannot share 15 GB. Port is allocated
# dynamically: Colab VMs already run services on 8080 (bind fails with E98).
import gc, re, socket, time, urllib.request

BASE_URL = None
if SERVE_API:
    loaded = None
    gc.collect()
    for _ in range(30):  # let VRAM drain before the server claims it
        smi = nvidia_smi()
        if not (HAS_GPU and smi and smi["vram_mb"] > 0):
            break
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.strip()
        if used and float(used) < 1200:
            break
        time.sleep(2)

    _s = socket.socket()
    _s.bind(("0.0.0.0", 0))
    API_PORT = _s.getsockname()[1]
    _s.close()
    print(f"server will use port {API_PORT} (8080 is taken on Colab VMs)")

    server_log = open("server.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server",
         "--model", model_path, "--n_gpu_layers", "-1", "--n_ctx", str(N_CTX),
         "--model_alias", "local-llm", "--host", "0.0.0.0", "--port", str(API_PORT)],
        stdout=server_log, stderr=server_log)

    healthy = False
    for _ in range(80):  # up to ~240 s: big models take ~60 s to load
        if server.poll() is not None:  # died — stop waiting, show why
            break
        # /v1/models, not /health: 0.3.x serves 404 on /health, and the app
        # only accepts requests after the model has finished loading anyway.
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{API_PORT}/v1/models", timeout=4) as r:
                if r.status == 200:
                    healthy = True
                    print("server healthy")
                    break
        except Exception:
            time.sleep(3)
    if not healthy:
        print(open("server.log").read()[-1500:])
        raise SystemExit("server never became healthy — log above")

    if TUNNEL == "ngrok":
        if not NGROK_AUTHTOKEN:
            raise SystemExit("TUNNEL='ngrok' but NGROK_AUTHTOKEN is empty")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "pyngrok"],
                       capture_output=True, text=True)
        from pyngrok import ngrok, conf
        conf.get_default().auth_token = NGROK_AUTHTOKEN
        BASE_URL = ngrok.connect(API_PORT, "http").public_url()
    else:
        r = subprocess.run(
            ["curl", "-fsSL", "-o", "cloudflared",
             "https://github.com/cloudflare/cloudflared/releases/latest/download/"
             "cloudflared-linux-amd64"],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("cloudflared download failed: " + r.stderr[-500:])
        subprocess.run(["chmod", "+x", "cloudflared"])
        tunnel_log = open("tunnel.log", "w")
        subprocess.Popen(["./cloudflared", "tunnel", "--url",
                          f"http://localhost:{API_PORT}", "--no-autoupdate"],
                         stdout=tunnel_log, stderr=tunnel_log)
        for _ in range(20):
            time.sleep(2)
            m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com",
                          open("tunnel.log").read())
            if m:
                BASE_URL = m.group(0)
                break

    if BASE_URL:
        print("API URL:", BASE_URL)
        payload = '{"model":"local-llm","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
        print()
        print("Call it from ANYWHERE:")
        print(f"  curl {BASE_URL}/v1/chat/completions -H 'Content-Type: application/json' -d '{payload}'")
        print()
        print(f"  Python: OpenAI(base_url='{BASE_URL}/v1', api_key='none'), model='local-llm'")
        print("OK: api")
    else:
        print(open("tunnel.log").read()[-800:])
        print("WARN: api — server is healthy but no public URL; see tunnel log above")
else:
    print("SERVE_API is False — skipping the API/tunnel")
"""))

CELLS.append((MD, """\
## Tuning, once it works

Everything below was **measured** on a free T4 (full write-up in
`references/05-tuning.md` of the repo this notebook came from):

- **Quant is the fit lever.** Q4_K_M is the sweet spot. To fit a bigger
  model, drop to Q4_K_S / Q3_K_M — never Q8 on ≥8B at 15 GB.
- **Context costs VRAM**: ~0.2 GB per extra 4k tokens on the 14B. Need long
  documents? Halve KV first: `--cache-type-k q8_0 --cache-type-v q8_0`.
- **Decode speed ≈ 320 GB/s ÷ file size × ~0.6.** Qwen3-8B (5 GB) ≈ 40 tok/s —
  if you measure far below that, you are (probably accidentally) on CPU.
- **gpt-oss-20b** decodes at ~59 tok/s (MoE, ~3.6B active params) but its
  harmony chat template does not auto-apply here — expect `<|channel|>`
  markers in the raw text; strip client-side, keep the `final` channel.
- **Think-mode**: Qwen3 may spend the whole budget inside `<think>`; the
  notebook passes `enable_thinking: False` for it — keep that if you edit.
- **The API is session-bound**: Colab reclaims idle VMs (~90 min without
  browser interaction). Re-run the notebook for a fresh URL — treat every URL
  as ephemeral.

## Teardown

Done in the browser? `Runtime → Manage sessions → Terminate`.
Drove this via the CLI? `colab stop <session-id>`, then `colab sessions`
to confirm nothing is left running.
"""))

CELLS.append((CODE, """\
# Optional teardown — set TEARDOWN = True and run to stop the daemon
# server + tunnel (the chat objects above stay until the VM dies).
TEARDOWN = False
if TEARDOWN and SERVE_API:
    subprocess.run(["pkill", "-f", "llama_cpp.server"])
    subprocess.run(["pkill", "-f", "cloudflared"])
    print("server + tunnel stopped")
else:
    print("nothing torn down (TEARDOWN=False)")
"""))


def build() -> None:
    cells = []
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "metadata": {"id": f"cell-{i:02d}"}, "source": src}
        if kind == CODE:
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    out = Path(__file__).resolve().parent.parent / "colab" / "run-llm-t4.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
