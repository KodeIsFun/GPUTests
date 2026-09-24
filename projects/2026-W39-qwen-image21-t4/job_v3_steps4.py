#!/usr/bin/env python3
"""W39 probe v3: 4 steps with the 12-step direct pipeline's settings (no Pocket rewriter).

Plan: PLAN-4steps.md. DEVIATION from plan, forced upstream 2026-09-24: the original
non-UC Q4_K_M GGUF 404s — abenzerps/Qwen-Image-2.1-GGUF was restructured into the
Uncensored repo and the file is gone (TE/VAE still resolve via redirect). Baseline
switched from timings_v2b.json (non-UC) to the Pocket probe's DIRECT arm
(../2026-W39-qwen-image21-pocket-t4/results/timings.json): same UC-Q4_K_M weights,
12 steps, cfg 2.5, res_multistep/simple, seed 42 — sign 768x768 87.1 s, cat 864x576
126.1 s cold / bicycle 864x576 75.1 s.

Rows (all 4 steps, seed 42):
  1  p1 cat 864x576 cfg 2.5   vs pocket-probe cat_raincoat_direct (864x576, 12 steps)
  2  p2 sign 768x768 cfg 2.5  vs pocket-probe freegpulab_sign_direct (768x768, 12 steps)
  3  contingency: sign 768x768 cfg 1.0 (only matters if row 2 quality fails)
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
STEPS = 4
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]
GGUF_NAME = "qwen-image-2.1-UC-Q4_K_M.gguf"

T0 = time.time()
TIM = {"run": "qwen-image-2.1-uc-q4km-t4-v3-steps4", "flags": FLAGS,
       "steps": STEPS,
       "matrix_note": ("4 steps vs the Pocket probe's 12-step DIRECT arm (same UC-Q4_K_M weights, "
                       "cfg 2.5, res_multistep/simple, seed 42). Deviation from PLAN-4steps.md: "
                       "non-UC Q4_K_M 404s after repo restructure, so timings_v2b.json is not the "
                       "baseline anymore."),
       "rows": []}

GGUF_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/{GGUF_NAME}"
TE_INT8_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"

PROMPTS = [
    ("p1_cat_raincoat",
     "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones "
     "at night, neon city lights bokeh, 50mm lens", 864, 576),
    ("p2_freegpulab_sign",
     'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement',
     768, 768),
]


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def save():
    with open(f"{OUT}/timings_v3.json", "w") as f:
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
        (GGUF_URL, f"{mdir}/diffusion_models/{GGUF_NAME}"),
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
    shutil.copy(f"{mdir}/diffusion_models/{GGUF_NAME}", f"{mdir}/unet/")
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


def build_workflow(pos, w, h, steps, cfg=2.5, prefix="v3_fp16"):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": SEED, "steps": steps, "cfg": cfg,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def render(name, pos, w, h, cfg, prefix):
    t0 = time.time()
    pid = http_json("/prompt", {"prompt": build_workflow(pos, w, h, STEPS, cfg, prefix),
                                "client_id": "w39v3"})["prompt_id"]
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
    rec = {"key": f"uc_4steps_{w}x{h}_cfg{cfg}_{name}", "ok": ok, "wall_s": wall,
           "note": note, "files": imgs, "flags": FLAGS, "cfg": cfg, "size": f"{w}x{h}"}
    log(f"{name} {w}x{h} cfg={cfg}: ok={ok} {wall}s ({note})")
    if ok:
        for fn in imgs:
            data = urllib.request.urlopen(
                f"{BASE}/view?filename={urlquote(fn)}&subfolder=&type=output", timeout=120).read()
            with open(f"{OUT}/{prefix}_{fn}", "wb") as f:
                f.write(data)
            rec["saved"] = f"{prefix}_{fn}"
    TIM["rows"].append(rec)
    save()


def main():
    global proc
    os.makedirs(OUT, exist_ok=True)
    setup()
    log(f"booting server with {FLAGS}")
    comfy_log = open(f"{OUT}/comfyui-v3.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v3.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        for name, pos, w, h in PROMPTS:
            if proc.poll() is not None:
                log(f"server died before {name}")
                break
            render(name, pos, w, h, 2.5, "v3_fp16")
        # contingency row, front-loaded: only matters if the cfg 2.5 sign quality fails
        if proc.poll() is None:
            _, pos, w, h = PROMPTS[1]
            render("p2_freegpulab_sign_cfg1", pos, w, h, 1.0, "v3_cfg1")
        logtxt = open(f"{OUT}/comfyui-v3.log").read()
        TIM["cast_lines_seen"] = re.findall(r"weight dtype \S+, manual cast: \S+", logtxt)
        TIM["prompt_executed"] = re.findall(r"Prompt executed in ([0-9:.]+)", logtxt)
        it_lines = re.findall(r"(\d+)/(\d+) \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it", logtxt)
        TIM["s_per_step"] = it_lines[-1][2] if it_lines else None
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
    print("=== TIMINGS_V3_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
