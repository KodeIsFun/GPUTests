#!/usr/bin/env python3
"""W6 probe v2b: rerun ONLY the fp16 rows under --force-fp16 --disable-comfy-compiler.

v2 showed the fp16 crash is in comfy_aimdo's malloc-graph compile
(RuntimeError: aimdo memory compile error) during prefetch of the fp16
forward — the model itself loads fine as torch.float16. This driver disables
the Comfy model compiler (which includes the aimdo/CUDA-graph subfeature) and
reruns rows 2-3 from v2. Weights are already on disk from the v2 run.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from urllib.parse import quote as urlquote

ROOT = "/content"
C = f"{ROOT}/ComfyUI"
OUT = f"{ROOT}/out"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
SEED = 42
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]

T0 = time.time()
TIM = {"run": "qwen-image-2.1-q4km-t4-v2b-fp16-nocompile", "flags": FLAGS,
       "matrix_note": "same rows as v2 phase 2; only change vs v2 fp16 attempt: --disable-comfy-compiler",
       "rows": []}

GGUF_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf"
TE_INT8_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"

PROMPTS = [
    ("p1_cat_raincoat",
     "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones "
     "at night, neon city lights bokeh, 50mm lens"),
    ("p2_freegpulab_sign",
     'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement'),
]


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def save():
    with open(f"{OUT}/timings_v2b.json", "w") as f:
        json.dump(TIM, f, indent=2)


def sh(cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, text=True, timeout=timeout)
    return r.returncode, time.time() - t0


def setup():
    """Fresh-VM safe: install ComfyUI + GGUF nodes and fetch weights if absent."""
    log("=== setup: ComfyUI + weights (skipped if present) ===")
    if not os.path.isdir(C):
        rc, dt = sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {C}")
        log(f"clone ComfyUI rc={rc} {dt:.1f}s")
        rc, dt = sh(f"pip install -q -r {C}/requirements.txt 2>&1 | tail -3", timeout=1200)
        log(f"pip comfy requirements rc={rc} {dt:.1f}s")
    gg = f"{C}/custom_nodes/ComfyUI-GGUF"
    if not os.path.isdir(gg):
        rc, dt = sh(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF {gg}")
        log(f"clone ComfyUI-GGUF rc={rc} {dt:.1f}s")
        rc, dt = sh(f"pip install -q -r {gg}/requirements.txt 2>&1 | tail -3", timeout=1200)
        log(f"pip gguf requirements rc={rc} {dt:.1f}s")
    mdir = f"{C}/models"
    files = [
        (GGUF_URL, f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf"),
        (TE_INT8_URL, f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
        (VAE_URL, f"{mdir}/vae/qwen_image_2.1_vae_bf16.safetensors"),
    ]
    dls = []
    for url, dest in files:
        if os.path.exists(dest) and os.path.getsize(dest) > 1e8:
            log(f"cached: {os.path.basename(dest)}")
            continue
        t0 = time.time()
        rc = subprocess.run(["wget", "-q", "-c", "--tries=3", "--timeout=60", "-O", dest, url]).returncode
        dt = time.time() - t0
        log(f"download {os.path.basename(dest)} rc={rc} in {dt:.1f}s")
        dls.append({"file": os.path.basename(dest), "download_s": round(dt, 1), "rc": rc})
    os.makedirs(f"{mdir}/unet", exist_ok=True)
    shutil.copy(f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf", f"{mdir}/unet/")
    os.makedirs(f"{mdir}/clip", exist_ok=True)
    shutil.copy(f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", f"{mdir}/clip/")
    TIM["downloads"] = dls
    save()


def http_json(path, payload=None, timeout=60):
    if payload is not None:
        req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(BASE + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_server(proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            http_json("/system_stats", timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def build_workflow(pos, size, steps):
    w = h = size
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-2.1-Q4_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": SEED, "steps": steps, "cfg": 2.5,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "v2b_fp16"}},
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    setup()
    log(f"booting server with {FLAGS}")
    comfy_log = open(f"{OUT}/comfyui-v2b.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v2b.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        for name, pos in PROMPTS:
            if proc.poll() is not None:
                log(f"server died before {name}")
                break
            t0 = time.time()
            pid = http_json("/prompt", {"prompt": build_workflow(pos, 768, 12),
                                        "client_id": "w6probe"})["prompt_id"]
            ok, imgs, note = False, [], ""
            while time.time() - t0 < 1500:
                time.sleep(3)
                if proc.poll() is not None:
                    note = "server process died"
                    break
                hist = http_json(f"/history/{pid}", timeout=15)
                if pid not in hist:
                    continue
                entry = hist[pid]
                status = entry.get("status", {})
                imgs = [im["filename"] for o in entry.get("outputs", {}).values()
                        for im in o.get("images", [])]
                if status.get("status_str") == "error":
                    msgs = json.dumps(status.get("messages", []))
                    note = f"comfy error: {msgs[:300]}"
                    break
                if imgs:
                    ok, note = True, "ok"
                    break
            wall = round(time.time() - t0, 1)
            rec = {"key": f"combo_fp16_768_12_{name}", "ok": ok, "wall_s": wall,
                   "note": note, "files": imgs, "flags": FLAGS}
            log(f"{name}: ok={ok} {wall}s ({note})")
            if ok:
                for fn in imgs:
                    data = urllib.request.urlopen(
                        f"{BASE}/view?filename={urlquote(fn)}&subfolder=&type=output", timeout=120).read()
                    with open(f"{OUT}/v2b_{fn}", "wb") as f:
                        f.write(data)
                    rec["saved"] = f"v2b_{fn}"
            TIM["rows"].append(rec)
            save()
        logtxt = open(f"{OUT}/comfyui-v2b.log").read()
        TIM["cast_lines_seen"] = re.findall(r"weight dtype \S+, manual cast: \S+", logtxt)
        TIM["prompt_executed"] = re.findall(r"Prompt executed in ([0-9:.]+)", logtxt)
        it_lines = re.findall(r"(\d+)/12 \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it", logtxt)
        TIM["s_per_step_fp16"] = it_lines[-1][1] if it_lines else None
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        comfy_log.close()
    TIM["total_s"] = round(time.time() - T0, 1)
    TIM["success"] = any(r["ok"] for r in TIM["rows"])
    save()
    log(f"=== DONE success={TIM['success']} total={TIM['total_s']}s ===")
    print("=== TIMINGS_V2B_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
