#!/usr/bin/env python3
"""W39 probe v5: Viggle turbo v0.2.1 r256 LoRA + 8-step text schedule — free T4.

Upstream changed under the shipped recipe (run-qwen-image-on-free-colab):
v0.2.1 (2026-09-24) now RECOMMENDS the rank-256 LoRA (r128 is an SVD truncation
keeping ~95% of the update energy) and publishes an 8-step sigma schedule that
"improves small dense text" — the lab's canary is exactly neon-sign text.
v4 used r128; this probe measures what upstream's upgrade buys on the same card.

Rows (seed 42, same prompts as v4):
  1  r256 cat 864x576, 6 steps        (cold — includes model load)
  2  r256 sign 768x768, 6 steps       (warm — the shipped 6-step recipe, r256)
  3  r256 sign 768x768, 8 steps       (warm — card's small-text sigmas)
  4  r128 sign 768x768, 6 steps       (warm — in-session rank anchor: the
      r256-vs-r128 delta must come from the same VM, not cross-session JSONs)
Sigma schedules straight from the v0.2.1 card: 6-step
1.0/0.9375/0.875/0.75/0.5/0.25; 8-step adds 0.625 mid and 0.125 low.
Auto-fallback on hook-node errors: stock LoraLoaderModelOnly merge approximation
(marked fallback_merge), same as v4.

Kernel v2 (2026-09-27): v1 on Kaggle died mid-GGUF at 2.46/4.20 GB — HF egress
stalled past wget's 60 s read timeout and the hard-fail refused the truncated
file (working as designed after the v3 15-byte-404 lesson). Downloads now HEAD
the expected size and resume-loop until it matches. ComfyUI + weights also move
to /tmp on Kaggle so the 16 GB tree stays out of the 19.5 GB /kaggle/working
output cap — kernel output is now only out/. The Colab run of this script
(session w39-v5-r256) completed before this change; its numbers stand.
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

SURFACE = "kaggle" if os.path.isdir("/kaggle") else "colab"
if SURFACE == "kaggle":
    C = "/tmp/ComfyUI"  # ephemeral: keeps 16 GB of weights out of /kaggle/working's output cap
    OUT = "/kaggle/working/out"
else:
    C = "/content/ComfyUI"
    OUT = "/content/out"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
SEED = 42
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]

GGUF_NAME = "qwen-image-2.1-Q4_K_M.gguf"
GGUF_URL = f"https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF/resolve/main/{GGUF_NAME}"
TE_NAME = "qwen3vl_8b_int8_convrot.safetensors"
TE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/{TE_NAME}"
VAE_NAME = "qwen_image_2.1_vae_bf16.safetensors"
VAE_URL = f"https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/{VAE_NAME}"
VIGGLE_BASE = "https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main"
LORA_R256_NAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
LORA_R128_NAME = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors"
NODES_URL = f"{VIGGLE_BASE}/comfyui/viggle_turbo.py"

SIGMAS_6 = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
SIGMAS_8_TEXT = "1.0, 0.9375, 0.875, 0.75, 0.625, 0.5, 0.25, 0.125"

T0 = time.time()
TIM = {"run": "qwen-image-2.1-q4km-t4-v5-turbo-r256", "surface": SURFACE,
       "flags": FLAGS,
       "loras": [LORA_R256_NAME, LORA_R128_NAME],
       "matrix_note": ("v0.2.1 r256 (recommended) vs r128 (shipped recipe) on the same "
                       "unsloth base Q4_K_M weights, plus the card's 8-step small-text "
                       "sigma schedule. Rows 2 vs 4 are the rank delta; row 3 is the "
                       "text-quality contingency. All sign rows warm."),
       "rows": []}

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
    with open(f"{OUT}/timings_v5_r256.json", "w") as f:
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
    """Resume-until-complete: v1 hard-failed on a CDN stall mid-GGUF (correctly —
    a truncated model must never render), so retry policy has to outlive stalls.
    HEAD the expected size, then wget -c in a loop until the file matches."""
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
            log(f"download {os.path.basename(dest)} {size/1e9:.2f} GB in {dt:.1f}s "
                f"(attempt {attempt})")
            return rc, dt
        log(f"download attempt {attempt} incomplete rc={rc} "
            f"{size/1e9:.2f}/{want/1e9:.2f} GB — resuming")
        time.sleep(10)
    sys.exit(f"download failed after retries: {url}")


def setup():
    """Fresh-VM safe: ComfyUI + GGUF nodes + viggle node + weights, skipped if present."""
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
    # author's custom node: runtime-hook LoRA + resolution-shifted sigmas
    dst = f"{C}/custom_nodes/viggle_turbo.py"
    if not os.path.exists(dst):
        rc = subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", dst, NODES_URL]).returncode
        log(f"viggle_turbo.py rc={rc}")
    mdir = f"{C}/models"
    dls = []
    for url, dest, minb in [
        (GGUF_URL, f"{mdir}/diffusion_models/{GGUF_NAME}", 1e9),
        (TE_URL, f"{mdir}/text_encoders/{TE_NAME}", 1e9),
        (VAE_URL, f"{mdir}/vae/{VAE_NAME}", 1e8),
        (f"{VIGGLE_BASE}/{LORA_R256_NAME}", f"{mdir}/loras/{LORA_R256_NAME}", 1e9),
        (f"{VIGGLE_BASE}/{LORA_R128_NAME}", f"{mdir}/loras/{LORA_R128_NAME}", 5e8),
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


def wf_turbo(pos, w, h, sigma_nodes, lora_name, prefix):
    """Author's recipe: runtime-hook LoRA, resolution-shifted sigmas, cfg off."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "ViggleTurboLora", "inputs": {"model": ["1", 0], "lora_name": lora_name,
                                                          "strength": 1.0}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "9": {"class_type": "ViggleTurboSigmas", "inputs": {"latent": ["6", 0], "nodes": sigma_nodes}},
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["5", 0], "conditioning": ["4", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }


def wf_merge_fallback(pos, w, h, lora_name, prefix):
    """If the hook node is broken on GGUF models: stock merge path as approximation."""
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora_name,
                                                              "strength_model": 1.0}},
        "7": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["7", 0], "seed": SEED, "steps": 6, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


def post_render_stats(rec):
    logtxt = open(f"{OUT}/comfyui-v5.log").read()
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


def render(name, workflow, prefix, force_note=None):
    t0 = time.time()
    pid = http_json("/prompt", {"prompt": workflow, "client_id": "w39v5"})["prompt_id"]
    ok, imgs, note = False, [], force_note or ""
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


def render_turbo(name, pos, w, h, sigma_nodes, lora_name, prefix):
    rec = render(name, wf_turbo(pos, w, h, sigma_nodes, lora_name, prefix), prefix)
    rec["lora_name"] = lora_name
    rec["sigmas"] = sigma_nodes
    if not rec["ok"] and proc.poll() is None:
        log(f"hook node failed -> merge fallback for {name}")
        fb = render(name + "_fallback_merge", wf_merge_fallback(pos, w, h, lora_name, prefix + "_fb"),
                    prefix + "_fb", force_note="fallback_merge (hook node path failed)")
        TIM["turbo_hook_failed"] = True
        return fb
    return rec


def main():
    global proc
    os.makedirs(OUT, exist_ok=True)
    setup()
    log(f"booting server with {FLAGS}")
    comfy_log = open(f"{OUT}/comfyui-v5.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v5.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        _, p1, w1, h1 = PROMPTS[0]
        _, p2, w2, h2 = PROMPTS[1]
        render_turbo("r256_cat_864x576_6steps", p1, w1, h1, SIGMAS_6, LORA_R256_NAME, "v5r256_cat")
        render_turbo("r256_sign_768x768_6steps", p2, w2, h2, SIGMAS_6, LORA_R256_NAME, "v5r256_sign6")
        if proc.poll() is None:
            render_turbo("r256_sign_768x768_8steps_text", p2, w2, h2, SIGMAS_8_TEXT,
                         LORA_R256_NAME, "v5r256_sign8text")
        if proc.poll() is None:
            render_turbo("r128_sign_768x768_6steps_anchor", p2, w2, h2, SIGMAS_6,
                         LORA_R128_NAME, "v5r128_sign6")
        TIM["cast_lines_seen"] = re.findall(r"weight dtype \S+, manual cast: \S+",
                                            open(f"{OUT}/comfyui-v5.log").read())
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
    print("=== TIMINGS_V5_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
