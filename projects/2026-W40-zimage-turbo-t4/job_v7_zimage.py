#!/usr/bin/env python3
"""W40 v7: the first free-T4 image-gen table — Z-Image-Turbo vs Qwen-2.1-turbo.

Z-Image-Turbo (Tongyi-MAI, 6B S3-DiT, apache-2.0, ~589k HF downloads) claims
"fits comfortably within 16G VRAM" and 8 NFEs; nobody has published it next to
Qwen-Image-2.1-turbo on the same free card. Both stacks in one session, seed
42, same two canary prompts as v4-v6, in-session cold+warm for BOTH sides so
the table's columns share one VM:

  Z-Image side  (official Comfy-Org split files + template recipe exactly:
                 CLIPLoader type=lumina2 qwen_3_4b bf16, ModelSamplingAuraFlow
                 shift=3, KSampler 8 steps cfg 1 res_multistep/simple,
                 EmptySD3LatentImage; turbo checkpoint is self-contained — the
                 distill_patch_lora ships for the Omni/base path, not t2i)
    1  z bf16  cat 864x576  COLD (z stack load; as-published precision)
    2  z bf16  sign 768x768 warm
    3  z bf16  sign 1024x1024 warm (card's native res)
    4  z int8  sign 768x768 warm (memory-friendly split, same TE/vae)
  Qwen side (v6 recipe verbatim: unsloth GGUF Q4_K_M + viggle r256 runtime
             hook + resolution-shifted 6-step sigmas, cfg off)
    5  qwen cat 864x576  COLD (qwen stack load)
    6  qwen sign 768x768 6s warm   (v6 anchor row: 33.0 s)
    7  qwen sign 1024x1024 6s warm (v6 anchor row: 42.0 s)

VRAM is the open question the table answers: z bf16 unet 12.31 GB + TE 8.04 GB
cannot both stay resident on 15.6 GB, so ComfyUI's memory manager must unload
between stages — an OOM here is a table result, not a failure (the run only
fails if no row renders). Every row is a distinct graph (v6 lesson: repeated
same-seed graphs return execution-cache hits). Surface via env proof (v6
lesson: Colab ships stray /kaggle dirs).
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

SURFACE = "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "colab"
SURFACE_PROOF = {k: os.environ.get(k) for k in ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_GPU")}
if SURFACE == "kaggle":
    C = "/tmp/ComfyUI"
    OUT = "/kaggle/working/out"
else:
    C = "/content/ComfyUI"
    OUT = "/content/out"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
SEED = 42
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]

# --- Qwen-2.1-turbo stack (v6 recipe, unchanged) ---
GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"
GGUF_URL = f"https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/{GGUF_NAME}"
TE_NAME = "qwen3vl_8b_int8_convrot.safetensors"
TE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/{TE_NAME}"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
VAE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/{VAE_NAME}"
VIGGLE_BASE = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main"
LORA_VIGGLE_R256 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
NODES_URL = f"{VIGGLE_BASE}/comfyui/viggle_turbo.py"
SIGMAS_6_VIGGLE = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"

# --- Z-Image-Turbo stack (Comfy-Org split files, template default = bf16 TE) ---
ZB = "https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files"
Z_BF16 = "z_image_turbo_bf16.safetensors"
Z_INT8 = "z_image_turbo_int8_convrot.safetensors"
Z_TE = "qwen_3_4b.safetensors"
Z_VAE = "ae.safetensors"

T0 = time.time()
TIM = {"run": "zimage-turbo-vs-qwen-t4-v7", "surface": SURFACE, "surface_proof": SURFACE_PROOF,
       "flags": FLAGS, "seed": SEED,
       "z_image": {"repo": "Comfy-Org/z_image_turbo (from Tongyi-MAI/Z-Image-Turbo, apache-2.0)",
                   "recipe": "official t2i template: lumina2 clip + AuraFlow shift=3 + 8 steps cfg1 res_multistep/simple",
                   "n_params_b": 6},
       "qwen": {"repo": "unsloth GGUF Q4_K_M + abenzerps TE/VAE + Viggle r256 hook (v6 recipe verbatim)"},
       "matrix_note": ("both stacks in one session; rows 1-4 Z-Image (bf16 cold, 768 warm, 1K, "
                       "int8), rows 5-7 qwen r256 (cat cold, 768/1K warm) = in-session anchors. "
                       "All rows distinct graphs (v6 cache lesson), seed 42, same two canary "
                       "prompts as v4-v6."),
       "rows": []}

PROMPTS = {
    "cat": ("p1_cat_raincoat",
            "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones "
            "at night, neon city lights bokeh, 50mm lens"),
    "sign": ("p2_freegpulab_sign",
             'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement'),
}


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def save():
    with open(f"{OUT}/timings_v7_zimage.json", "w") as f:
        json.dump(TIM, f, indent=2)


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
        return 0, 0.0
    t0 = time.time()
    for attempt in range(1, 9):
        rc = subprocess.run(["wget", "-q", "-c", "--tries=3", "--timeout=120",
                             "--waitretry=10", "-O", dest, url]).returncode
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        if rc == 0 and size >= min_bytes and (not want or size >= want):
            dt = time.time() - t0
            log(f"download {os.path.basename(dest)} {size/1e9:.2f} GB in {dt:.1f}s (attempt {attempt})")
            return rc, dt
        log(f"download attempt {attempt} incomplete rc={rc} {size/1e9:.2f}/{want/1e9:.2f} GB — resuming")
        time.sleep(10)
    sys.exit(f"download failed after retries: {url}")


def setup():
    log("=== setup: ComfyUI + both stacks (skipped if present) ===")
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
        rc = subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", dst, NODES_URL]).returncode
        log(f"viggle_turbo.py rc={rc}")
    mdir = f"{C}/models"
    dls = []
    for url, dest, minb in [
        (f"{ZB}/diffusion_models/{Z_BF16}", f"{mdir}/diffusion_models/{Z_BF16}", 1e9),
        (f"{ZB}/diffusion_models/{Z_INT8}", f"{mdir}/diffusion_models/{Z_INT8}", 5e8),
        (f"{ZB}/text_encoders/{Z_TE}", f"{mdir}/text_encoders/{Z_TE}", 1e9),
        (f"{ZB}/vae/{Z_VAE}", f"{mdir}/vae/{Z_VAE}", 1e8),
        (GGUF_URL, f"{mdir}/diffusion_models/{GGUF_NAME}", 1e9),
        (TE_URL, f"{mdir}/text_encoders/{TE_NAME}", 1e9),
        (VAE_URL, f"{mdir}/vae/{VAE_NAME}", 1e8),
        (f"{VIGGLE_BASE}/{LORA_VIGGLE_R256}", f"{mdir}/loras/{LORA_VIGGLE_R256}", 5e8),
    ]:
        _, dt = wget_or_die(url, dest, minb)
        if dt:
            dls.append({"file": os.path.basename(dest), "download_s": round(dt, 1),
                        "size_gb": round(os.path.getsize(dest) / 1e9, 2)})
    os.makedirs(f"{mdir}/unet", exist_ok=True)
    if not os.path.exists(f"{mdir}/unet/{GGUF_NAME}"):
        shutil.copy(f"{mdir}/diffusion_models/{GGUF_NAME}", f"{mdir}/unet/")
    os.makedirs(f"{mdir}/clip", exist_ok=True)
    if not os.path.exists(f"{mdir}/clip/{TE_NAME}"):
        shutil.copy(f"{mdir}/text_encoders/{TE_NAME}", f"{mdir}/clip/")
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


def wf_z(pos, w, h, model_file, prefix):
    """Official Z-Image t2i template as API graph (template subgraph, verbatim widgets)."""
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": model_file, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": Z_TE, "type": "lumina2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": Z_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": SEED, "steps": 8, "cfg": 1.0,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def wf_qwen(pos, w, h, prefix):
    """v6 recipe verbatim: GGUF + viggle runtime-hook LoRA + resolution-shifted sigmas, cfg off."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": "ViggleTurboSigmas",
              "inputs": {"latent": ["5", 0], "nodes": SIGMAS_6_VIGGLE}},
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
    logtxt = open(f"{OUT}/comfyui-v7.log").read()
    it_lines = re.findall(r"(\d+)/(\d+) \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it", logtxt)
    rec["s_per_step"] = it_lines[-1][2] if it_lines else None
    ex = re.findall(r"Prompt executed in ([0-9:.]+)", logtxt)
    rec["executed_s"] = ex[-1] if ex else None
    try:
        st = http_json("/system_stats", timeout=10)
        dev = st.get("devices", [{}])[0]
        rec["vram_free_gb"] = round(dev.get("vram_free", 0) / 1e9, 2)
        rec["vram_total_gb"] = round(dev.get("vram_total", 0) / 1e9, 2)
    except Exception:
        pass


def render(name, workflow, prefix, stack, model, steps, w, h):
    t0 = time.time()
    pid = http_json("/prompt", {"prompt": workflow, "client_id": "w40v7"})["prompt_id"]
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
            msgs = json.dumps(status.get("messages", []))
            note = f"comfy error: {msgs[:400]}"
            break
        if imgs:
            ok, note = True, "ok"
            break
    wall = round(time.time() - t0, 1)
    rec = {"key": name, "ok": ok, "wall_s": wall, "note": note, "files": imgs,
           "stack": stack, "model": model, "steps": steps, "width": w, "height": h,
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
    comfy_log = open(f"{OUT}/comfyui-v7.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v7.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        _, sign = PROMPTS["sign"]
        _, cat = PROMPTS["cat"]
        rows = [
            ("z_bf16_cat_864x576_cold", "z_image", Z_BF16, 8, 864, 576, cat, "v7zbf16cat"),
            ("z_bf16_sign_768x768_warm", "z_image", Z_BF16, 8, 768, 768, sign, "v7zbf16sign"),
            ("z_bf16_sign_1024x1024_warm", "z_image", Z_BF16, 8, 1024, 1024, sign, "v7zbf16sign1k"),
            ("z_int8_sign_768x768_warm", "z_image", Z_INT8, 8, 768, 768, sign, "v7zint8sign"),
            ("qwen_r256_cat_864x576_cold", "qwen_viggle", LORA_VIGGLE_R256, 6, 864, 576, cat, "v7qwcat"),
            ("qwen_r256_sign_768x768_6s_warm", "qwen_viggle", LORA_VIGGLE_R256, 6, 768, 768, sign, "v7qwsign"),
            ("qwen_r256_sign_1024x1024_6s_warm", "qwen_viggle", LORA_VIGGLE_R256, 6, 1024, 1024, sign, "v7qwsign1k"),
        ]
        sigs = {(r[1], r[2], r[4], r[5], r[6], SEED) for r in rows}
        assert len(sigs) == len(rows), "duplicate graphs in row matrix"
        for name, stack, model, steps, w, h, prompt, prefix in rows:
            if proc.poll() is not None:
                log("server died — stopping row loop")
                break
            wf = wf_z(prompt, w, h, model, prefix) if stack == "z_image" else wf_qwen(prompt, w, h, prefix)
            render(name, wf, prefix, stack, model, steps, w, h)
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
    print("=== TIMINGS_V7_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
