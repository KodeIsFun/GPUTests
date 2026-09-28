#!/usr/bin/env python3
"""W40 v7b: warm-repeat pair completing the v7 table — encode-cached pure renders.

v7's 768 warm rows are not apples-to-apples: Z-Image's bf16 TE (8 GB) cannot
stay resident beside its 12.3 GB unet on 15.6 GB, so the FIRST render with a
new prompt pays a ~30 s TE reload (row 2: 45.0 s wall vs 14.6 s of denoise).
This job renders the SAME sign prompt twice per stack with two seeds (distinct
graphs, v6 cache rule): row 1 re-pays the load/encode, row 2 is the pure warm
number a user making variations cares about. Weights are already on disk from
v7 (same VM), so this is minutes.
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

ZB = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"
VIGGLE_BASE = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main"
GGUF_URL = "https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf"
TE_URL = ("https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/"
          "qwen3vl_8b_int8_convrot.safetensors")
VAE_URL = ("https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/"
           "qwen_image_2.1_vae_bf16.safetensors")


def sh(cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, text=True, timeout=timeout)
    return r.returncode, time.time() - t0


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
            log(f"download {os.path.basename(dest)} {size/1e9:.2f} GB in {time.time()-t0:.1f}s (attempt {attempt})")
            return
        log(f"download attempt {attempt} incomplete rc={rc} {size/1e9:.2f}/{want/1e9:.2f} GB — resuming")
        time.sleep(10)
    sys.exit(f"download failed after retries: {url}")


def setup():
    """Fresh-VM safe: the first v7b launch assumed v7's cached weights and died on
    boot (FileNotFoundError /content/ComfyUI) — a fresh VM needs the full setup."""
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
    dst = f"{C}/custom_nodes/viggle_turbo.py"
    if not os.path.exists(dst):
        rc = subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", dst,
                             f"{VIGGLE_BASE}/comfyui/viggle_turbo.py"]).returncode
        log(f"viggle_turbo.py rc={rc}")
    mdir = f"{C}/models"
    for url, dest, minb in [
        (f"{ZB}/diffusion_models/{Z_BF16}", f"{mdir}/diffusion_models/{Z_BF16}", 1e9),
        (f"{ZB}/text_encoders/{Z_TE}", f"{mdir}/text_encoders/{Z_TE}", 1e9),
        (f"{ZB}/vae/{Z_VAE}", f"{mdir}/vae/{Z_VAE}", 1e8),
        (GGUF_URL, f"{mdir}/diffusion_models/{GGUF_NAME}", 1e9),
        (TE_URL, f"{mdir}/text_encoders/{TE_NAME}", 1e9),
        (VAE_URL, f"{mdir}/vae/{VAE_NAME}", 1e8),
        (f"{VIGGLE_BASE}/{LORA_VIGGLE_R256}", f"{mdir}/loras/{LORA_VIGGLE_R256}", 5e8),
    ]:
        wget_or_die(url, dest, minb)
    os.makedirs(f"{mdir}/unet", exist_ok=True)
    if not os.path.exists(f"{mdir}/unet/{GGUF_NAME}"):
        shutil.copy(f"{mdir}/diffusion_models/{GGUF_NAME}", f"{mdir}/unet/")
    os.makedirs(f"{mdir}/clip", exist_ok=True)
    if not os.path.exists(f"{mdir}/clip/{TE_NAME}"):
        shutil.copy(f"{mdir}/text_encoders/{TE_NAME}", f"{mdir}/clip/")

SURFACE = "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "colab"
SURFACE_PROOF = {k: os.environ.get(k) for k in ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_GPU")}
C = "/tmp/ComfyUI" if SURFACE == "kaggle" else "/content/ComfyUI"
OUT = "/kaggle/working/out" if SURFACE == "kaggle" else "/content/out"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]

GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"
TE_NAME = "qwen3vl_8b_int8_convrot.safetensors"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
LORA_VIGGLE_R256 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
SIGMAS_6_VIGGLE = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
Z_BF16 = "z_image_turbo_bf16.safetensors"
Z_TE = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"
SIGN = ('a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement')

T0 = time.time()
TIM = {"run": "zimage-turbo-vs-qwen-t4-v7b-warm", "surface": SURFACE, "surface_proof": SURFACE_PROOF,
       "flags": FLAGS,
       "note": ("warm-repeat pair per stack at 768x768, same sign prompt, seeds 43/44 "
                "(distinct graphs): seed43 pays stack load/encode after boot, seed44 is "
                "the encode-cached pure warm render. Completes timings_v7_zimage.json."),
       "rows": []}


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def save():
    with open(f"{OUT}/timings_v7b_warm.json", "w") as f:
        json.dump(TIM, f, indent=2)


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


def wf_z(pos, seed, prefix):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": Z_BF16, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": Z_TE, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": Z_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 768, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": seed, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def wf_qwen(pos, seed, prefix):
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 768, "batch_size": 1}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "ViggleTurboSigmas", "inputs": {"latent": ["5", 0], "nodes": SIGMAS_6_VIGGLE}},
        "9": {"class_type": "ViggleTurboLora",
              "inputs": {"model": ["1", 0], "lora_name": LORA_VIGGLE_R256, "strength": 1.0}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["4", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["10", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def post_render_stats(rec):
    logtxt = open(f"{OUT}/comfyui-v7b.log").read()
    it_lines = re.findall(r"(\d+)/(\d+) \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it", logtxt)
    rec["s_per_step"] = it_lines[-1][2] if it_lines else None
    ex = re.findall(r"Prompt executed in ([0-9:.]+)", logtxt)
    rec["executed_s"] = ex[-1] if ex else None
    try:
        st = http_json("/system_stats", timeout=10)
        dev = st.get("devices", [{}])[0]
        rec["vram_free_gb"] = round(dev.get("vram_free", 0) / 1e9, 2)
    except Exception:
        pass


def render(name, workflow, prefix, stack, seed):
    t0 = time.time()
    pid = http_json("/prompt", {"prompt": workflow, "client_id": "w40v7b"})["prompt_id"]
    ok, imgs, note = False, [], ""
    while time.time() - t0 < 900:
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
            note = f"comfy error: {json.dumps(status.get('messages', []))[:400]}"
            break
        if imgs:
            ok, note = True, "ok"
            break
    wall = round(time.time() - t0, 1)
    rec = {"key": name, "ok": ok, "wall_s": wall, "note": note, "files": imgs,
           "stack": stack, "seed": seed, "width": 768, "height": 768,
           "flags": FLAGS, "prefix": prefix}
    log(f"{name}: ok={ok} {wall}s ({note[:120]})")
    if ok:
        for fn in imgs:
            data = urllib.request.urlopen(
                f"{BASE}/view?filename={urlquote(fn)}&subfolder=&type=output", timeout=120).read()
            with open(f"{OUT}/{prefix}__{fn}", "wb") as f:
                f.write(data)
            rec["saved"] = f"{prefix}__{fn}"
        post_render_stats(rec)
    TIM["rows"].append(rec)
    save()
    return rec


def main():
    global proc
    os.makedirs(OUT, exist_ok=True)
    setup()
    log(f"booting server with {FLAGS}")
    comfy_log = open(f"{OUT}/comfyui-v7b.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v7b.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        for name, wf, stack, seed, prefix in [
            ("z_bf16_sign_768_seed43_load", wf_z(SIGN, 43, "v7bz43"), "z_image", 43, "v7bz43"),
            ("z_bf16_sign_768_seed44_warm", wf_z(SIGN, 44, "v7bz44"), "z_image", 44, "v7bz44"),
            ("qwen_sign_768_seed43_load", wf_qwen(SIGN, 43, "v7bq43"), "qwen_viggle", 43, "v7bq43"),
            ("qwen_sign_768_seed44_warm", wf_qwen(SIGN, 44, "v7bq44"), "qwen_viggle", 44, "v7bq44"),
        ]:
            if proc.poll() is not None:
                log("server died — stopping row loop")
                break
            render(name, wf, prefix, stack, seed)
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
    print("=== TIMINGS_V7B_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
