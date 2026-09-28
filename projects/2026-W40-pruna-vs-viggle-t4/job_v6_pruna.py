#!/usr/bin/env python3
"""W40 v6: Pruna 5/8-step LoRA vs Viggle r256 6-step — Qwen-Image-2.1 head-to-head on a free T4.

Two competing distillations of the same base model, released 3 days apart
(Pruna 2026-09-24, Viggle v0.2.1 2026-09-24), no published T4 numbers for
either. Fair-comparison design, all rows seed 42, same two canary prompts as
v4/v5, same unsloth GGUF Q4_K_M base weights + abenzerps TE/VAE:

  - Same LoRA loading mechanism for BOTH sides: the viggle author's runtime
    hook node (ViggleTurboLora, "diffusers-format Qwen-Image-2.1 LoRA" per its
    own tooltip) — verified compatible by reading both safetensors headers
    remotely: identical `transformer.` diffusers key layout, Pruna rank 64 vs
    Viggle rank 256, same fused-SwiGLU img_mlp pattern. Hooks keep the update
    exact on quantized weights where stock LoraLoaderModelOnly merge is lossy.
  - Each recipe's own published sigma schedule: Viggle = resolution-shifted
    6-step nodes (their node shifts mu by latent tokens); Pruna = raw sigmas
    exactly as the card's diffusers snippet, "no extra shifting"
    (5-step: 1.0, 0.94, 6/7, 2/3, 0.4 — 8-step adds 14/15, 10/13, 6/11, 2/9;
    terminal 0 appended), carried by a 12-line local node (pruna_sigmas.py,
    written by setup()) mirroring ViggleTurboSigmas minus the shift.
  - cfg off on both sides (both cards prescribe true_cfg_scale=1.0).

Rows (first render is cold — includes one-time model load; every row is a
DISTINCT graph: attempt 1 repeated the headline graph with the same seed and
got a 3.0 s ComfyUI execution-cache hit instead of a render, so warm repeats
of an identical graph are meaningless and now asserted against):
  1  pruna5  cat  864x576   cold
  2  pruna5  sign 768x768   warm  (headline — first execution of this graph)
  3  pruna8  sign 768x768   warm  (card default, quality axis)
  4  pruna5  sign 1024x1024 warm  (card trained at 1K only)
  5  viggle  r256 sign 768x768 6s warm (in-session anchor, v5 measured 24.0 s)
  6  viggle  r256 sign 1024x1024 6s warm (1K-vs-1K pair with row 4)

Fallback on hook-node errors per row: exact-sigma merge path (same sigmas node,
LoraLoaderModelOnly instead of hooks) marked fallback_merge; known lossy on
quantized weights per the hook node's own docstring, so recorded as such.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from urllib.parse import quote as urlquote

# attempt-2/3 lesson: this Colab image ships /kaggle AND /kaggle/working, so no
# path proves Kaggle. KAGGLE_KERNEL_RUN_TYPE is set only inside real kernels;
# COLAB_GPU only on Colab. Record both as proof in the timings JSON.
SURFACE = "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "colab"
SURFACE_PROOF = {k: os.environ.get(k) for k in
                 ("KAGGLE_KERNEL_RUN_TYPE", "COLAB_GPU")}
if SURFACE == "kaggle":
    C = "/tmp/ComfyUI"  # ephemeral: keeps weights out of /kaggle/working output cap
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
LORA_VIGGLE_R256 = "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"
NODES_URL = f"{VIGGLE_BASE}/comfyui/viggle_turbo.py"
PRUNA_BASE = "https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1/resolve/main"
LORA_PRUNA5 = "p_qwen_image_2.1_5step_v0.1.safetensors"
LORA_PRUNA8 = "p_qwen_image_2.1_8step_v0.1.safetensors"

SIGMAS_6_VIGGLE = "1.0, 0.9375, 0.875, 0.75, 0.5, 0.25"
# Pruna card, diffusers snippet, exact fractions — scheduler shift=1.0 identity, terminal 0 appended by node
SIGMAS_5_PRUNA = ",".join(repr(float(x)) for x in [1.0, 0.94, 6 / 7, 2 / 3, 0.4])
SIGMAS_8_PRUNA = ",".join(repr(float(x)) for x in
                          [1.0, 14 / 15, 6 / 7, 10 / 13, 2 / 3, 6 / 11, 0.4, 2 / 9])

PRUNA_SIGMAS_NODE = '''"""Pruna card sigmas for Qwen-Image-2.1 distilled LoRAs, raw (no shift).

Card recipe: FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=False,
shift=1.0) with sigmas passed exactly as given — shift 1.0 is the identity, so
the raw values go straight to the sampler. Terminal 0 appended here (the
diffusers scheduler appends it). Mirrors ViggleTurboSigmas minus the
resolution-dependent exponential shift.
"""

import torch


class PrunaSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "nodes": ("STRING", {"default": "1.0, 0.94, 0.8571428571428571, 0.6666666666666666, 0.4",
                                     "tooltip": "Card sigmas exactly as given, no shift. 5-step: 1.0, 0.94, 6/7, 2/3, 0.4. "
                                                "8-step: 1.0, 14/15, 6/7, 10/13, 2/3, 6/11, 0.4, 2/9. Terminal 0 appended."}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def get_sigmas(self, nodes):
        t = torch.tensor([float(x) for x in nodes.split(",")], dtype=torch.float64)
        return (torch.cat([t, t.new_zeros(1)]).float(),)


NODE_CLASS_MAPPINGS = {"PrunaSigmas": PrunaSigmas}
NODE_DISPLAY_NAME_MAPPINGS = {"PrunaSigmas": "Qwen-Image-2.1 Pruna Sigmas (raw, no shift)"}
'''

T0 = time.time()
TIM = {"run": "pruna-vs-viggle-t4-v6", "surface": SURFACE, "surface_proof": SURFACE_PROOF,
       "flags": FLAGS, "seed": SEED,
       "base_weights": {"gguf": f"unsloth/Qwen-Image-2.1-GGUF {GGUF_NAME}",
                        "te": f"abenzerps/Qwen-Image-2.1-GGUF {TE_NAME}",
                        "vae": f"abenzerps/Qwen-Image-2.1-GGUF {VAE_NAME}"},
       "loras": [LORA_PRUNA5, LORA_PRUNA8, LORA_VIGGLE_R256],
       "recipe_note": ("Both sides load through the same unmerged runtime-hook node "
                       "(ViggleTurboLora, diffusers-format keys verified via remote safetensors "
                       "headers: pruna rank 64, viggle rank 256, same module layout). Each side "
                       "runs its own published sigma schedule; cfg off per both cards. "
                       "kv_cache (Pruna card option) is a diffusers-pipeline feature, not "
                       "exercised on this ComfyUI path — recorded as a recipe difference."),
       "matrix_note": ("rows 1-4 Pruna (cat cold, 768 warm headline, 8-step, 1K); "
                       "rows 5-6 viggle r256 anchors on the same VM (768 warm = v5's 24.0 s row, "
                       "1K pairs with row 4). All sign rows same prompt, seed 42. Attempt 1 of "
                       "this job (results-v6-attempt1/) repeated the headline graph and caught "
                       "a ComfyUI execution-cache hit (3.0 s) — every row is now a distinct "
                       "graph, asserted."),
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
    with open(f"{OUT}/timings_v6_pruna.json", "w") as f:
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
    """Resume-until-complete (v5 lesson: a stalled CDN download must never yield a truncated model)."""
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


def stash_lora_metadata(path):
    """Record the safetensors metadata (hook node reads lora_adapter_metadata for the alpha/r scale)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get("__metadata__", {})
    return meta


def setup():
    """Fresh-VM safe: ComfyUI + GGUF nodes + viggle node + pruna sigmas node + weights."""
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
        rc = subprocess.run(["wget", "-q", "--tries=3", "--timeout=60", "-O", dst, NODES_URL]).returncode
        log(f"viggle_turbo.py rc={rc}")
    own = f"{C}/custom_nodes/pruna_sigmas.py"
    with open(own, "w") as f:
        f.write(PRUNA_SIGMAS_NODE)
    log(f"pruna_sigmas.py written ({len(PRUNA_SIGMAS_NODE)} B)")
    mdir = f"{C}/models"
    dls = []
    for url, dest, minb in [
        (GGUF_URL, f"{mdir}/diffusion_models/{GGUF_NAME}", 1e9),
        (TE_URL, f"{mdir}/text_encoders/{TE_NAME}", 1e9),
        (VAE_URL, f"{mdir}/vae/{VAE_NAME}", 1e8),
        (f"{PRUNA_BASE}/{LORA_PRUNA5}", f"{mdir}/loras/{LORA_PRUNA5}", 1e8),
        (f"{PRUNA_BASE}/{LORA_PRUNA8}", f"{mdir}/loras/{LORA_PRUNA8}", 1e8),
        (f"{VIGGLE_BASE}/{LORA_VIGGLE_R256}", f"{mdir}/loras/{LORA_VIGGLE_R256}", 5e8),
    ]:
        _, dt = wget_or_die(url, dest, minb)
        if dt:
            dls.append({"file": os.path.basename(dest), "download_s": round(dt, 1),
                        "size_gb": round(os.path.getsize(dest) / 1e9, 2)})
    TIM["pruna_lora_metadata_5step"] = stash_lora_metadata(f"{mdir}/loras/{LORA_PRUNA5}")
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


def wf(pos, w, h, sigma_node_class, sigma_nodes, lora_name, prefix, merge=False):
    """One builder for both recipes: hook (or merge) LoRA + that recipe's sigmas, cfg off."""
    wf_ = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": GGUF_NAME}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": TE_NAME, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_NAME}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "7": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "8": {"class_type": sigma_node_class,
              "inputs": ({"latent": ["5", 0], "nodes": sigma_nodes} if sigma_node_class == "ViggleTurboSigmas"
                         else {"nodes": sigma_nodes})},
        "9": ({"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength_model": 1.0}}
              if merge else
              {"class_type": "ViggleTurboLora",
               "inputs": {"model": ["1", 0], "lora_name": lora_name, "strength": 1.0}}),
        "10": {"class_type": "BasicGuider", "inputs": {"model": ["9", 0], "conditioning": ["4", 0]}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["6", 0], "guider": ["10", 0], "sampler": ["7", 0],
                          "sigmas": ["8", 0], "latent_image": ["5", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": prefix}},
    }
    return wf_


def post_render_stats(rec, key_warn_prev):
    logtxt = open(f"{OUT}/comfyui-v6.log").read()
    it_lines = re.findall(r"(\d+)/(\d+) \[[0-9:]+<[0-9:]+, +([0-9.]+)s/it", logtxt)
    rec["s_per_step"] = it_lines[-1][2] if it_lines else None
    ex = re.findall(r"Prompt executed in ([0-9:.]+)", logtxt)
    rec["executed_s"] = ex[-1] if ex else None
    n_warn = len(re.findall(r"lora key not loaded", logtxt))
    rec["lora_key_warnings"] = n_warn - key_warn_prev
    try:
        st = http_json("/system_stats", timeout=10)
        dev = st.get("devices", [{}])[0]
        rec["vram_free_gb"] = round(dev.get("vram_free", 0) / 1e9, 2)
        rec["vram_total_gb"] = round(dev.get("vram_total", 0) / 1e9, 2)
    except Exception:
        pass
    return n_warn


def render(name, workflow, prefix, force_note=None):
    t0 = time.time()
    pid = http_json("/prompt", {"prompt": workflow, "client_id": "w40v6"})["prompt_id"]
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
    TIM["rows"].append(rec)
    save()
    return rec, wall


def render_recipe(name, pos, w, h, sigma_node_class, sigma_nodes, lora_name, prefix, steps):
    """Primary = runtime-hook (exact); fallback = exact-sigma merge, marked lossy."""
    sig_label = sigma_nodes
    rec, wall = render(name, wf(pos, w, h, sigma_node_class, sig_label, lora_name, prefix), prefix)
    rec.update({"lora_name": lora_name, "steps": steps, "width": w, "height": h,
                "loader": "runtime_hook", "sigmas": sig_label,
                "sigmas_node": sigma_node_class})
    if not rec["ok"] and proc.poll() is None:
        log(f"hook path failed -> exact-sigma merge fallback for {name}")
        fb, fbw = render(name + "_fallback_merge",
                         wf(pos, w, h, sigma_node_class, sig_label, lora_name, prefix + "_fb",
                            merge=True),
                         prefix + "_fb", force_note="fallback_merge (hook node path failed)")
        fb.update({"lora_name": lora_name, "steps": steps, "width": w, "height": h,
                   "loader": "merge_lossy_on_quantized", "sigmas": sig_label,
                   "sigmas_node": sigma_node_class, "hook_fallback_of": name})
        TIM["hook_failed_rows"] = TIM.get("hook_failed_rows", []) + [name]
        return fb, fbw
    return rec, wall


def main():
    global proc
    os.makedirs(OUT, exist_ok=True)
    setup()
    log(f"booting server with {FLAGS}")
    comfy_log = open(f"{OUT}/comfyui-v6.log", "w")
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *FLAGS],
                            cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -30 {OUT}/comfyui-v6.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"BOOT FAILED:\n{tail}")
            sys.exit(2)
        _, sign = PROMPTS["sign"]
        _, cat = PROMPTS["cat"]
        rows = [
            ("pruna5_cat_864x576_cold", LORA_PRUNA5, "PrunaSigmas", SIGMAS_5_PRUNA, 864, 576, cat, "v6p5cat", 5),
            ("pruna5_sign_768x768_warm", LORA_PRUNA5, "PrunaSigmas", SIGMAS_5_PRUNA, 768, 768, sign, "v6p5sign", 5),
            ("pruna8_sign_768x768_warm", LORA_PRUNA8, "PrunaSigmas", SIGMAS_8_PRUNA, 768, 768, sign, "v6p8sign", 8),
            ("pruna5_sign_1024x1024_warm", LORA_PRUNA5, "PrunaSigmas", SIGMAS_5_PRUNA, 1024, 1024, sign, "v6p5sign1k", 5),
            ("viggle_r256_sign_768x768_6s_warm", LORA_VIGGLE_R256, "ViggleTurboSigmas", SIGMAS_6_VIGGLE, 768, 768, sign, "v6vg256sign", 6),
            ("viggle_r256_sign_1024x1024_6s_warm", LORA_VIGGLE_R256, "ViggleTurboSigmas", SIGMAS_6_VIGGLE, 1024, 1024, sign, "v6vg256sign1k", 6),
        ]
        # attempt-1 lesson: a repeated graph (same seed/inputs) returns the cached
        # image instead of rendering — warm rows must be distinct graphs
        sigs = {(r[1], r[2], r[3], r[4], r[5], r[6], SEED) for r in rows}
        assert len(sigs) == len(rows), f"duplicate graphs in row matrix: {len(rows)-len(sigs)}"
        key_warn = 0
        for name, lora, sig_class, sig, w, h, prompt, prefix, steps in rows:
            if proc.poll() is not None:
                log("server died — stopping row loop")
                break
            rec, _ = render_recipe(name, prompt, w, h, sig_class, sig, lora, prefix, steps)
            key_warn = post_render_stats(rec, key_warn)
        TIM["cast_lines_seen_n"] = len(re.findall(r"weight dtype \S+, manual cast: \S+",
                                                  open(f"{OUT}/comfyui-v6.log").read()))
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
    print("=== TIMINGS_V6_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
