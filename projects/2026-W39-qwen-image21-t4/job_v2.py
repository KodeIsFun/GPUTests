#!/usr/bin/env python3
"""W6 probe v2: cut generation time — fp16 cast + 12 steps + 768x768.

Row 1 is a control (default fp32 cast, 768/12) so the fp16 gain is measured,
not guessed: rows 2-3 differ from row 1 only in dtype. Same seed (42) and
prompts as v1. Server boots once; dtype is per-node via UnetLoaderGGUFAdvanced
weight_dtype=fp16, with a --force-fp16 server relaunch as fallback if that
node/enum is missing. Everything of value goes to stdout and /content/out/.

v1 baseline for the same prompts: 765.6 s warm @ 1024/20/fp32 (38.0 s/step).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from urllib.parse import quote as urlquote

ROOT = "/content"
C = f"{ROOT}/ComfyUI"
OUT = f"{ROOT}/out"
GGUF_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/qwen-image-2.1-Q4_K_M.gguf"
TE_INT8_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
SEED = 42

T0 = time.time()
TIM = {
    "run": "qwen-image-2.1-q4km-t4-v2-fp16-combo",
    "platform": "colab-t4",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "matrix_note": "row1 = fp32 control (768/12); rows 2-3 = fp16 combo, only dtype differs; "
                   "v1 warm baseline same prompts: 765.6 s @ 1024/20/fp32",
    "rows": [],
}

PROMPTS = [
    ("p1_cat_raincoat",
     "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones "
     "at night, neon city lights bokeh, 50mm lens"),
    ("p2_freegpulab_sign",
     'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement'),
]

ROWS = [
    {"key": "ctl_fp32_768_12_p1", "prompt": PROMPTS[0], "steps": 12, "size": 768, "fp16": False},
    {"key": "combo_fp16_768_12_p1", "prompt": PROMPTS[0], "steps": 12, "size": 768, "fp16": True},
    {"key": "combo_fp16_768_12_p2", "prompt": PROMPTS[1], "steps": 12, "size": 768, "fp16": True},
]


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def save():
    with open(f"{OUT}/timings.json", "w") as f:
        json.dump(TIM, f, indent=2)


def sh(cmd, timeout=1800):
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, text=True, timeout=timeout)
    return r.returncode, time.time() - t0


class Pollers:
    def __init__(self):
        self.vram_max_mb = 0
        self.vram_samples = []
        self.ram_min_avail_mb = None
        self.ram_total_mb = None
        self._stop = False

    def start(self):
        self._stop = False
        for target in (self._vram, self._ram):
            threading.Thread(target=target, daemon=True).start()

    def _vram(self):
        try:
            p = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-l", "5"],
                stdout=subprocess.PIPE, text=True)
            for line in p.stdout:
                if self._stop:
                    p.terminate()
                    break
                mb = int(line.strip())
                self.vram_samples.append([round(time.time() - T0, 1), mb])
                self.vram_max_mb = max(self.vram_max_mb, mb)
        except Exception as e:
            log(f"vram poller stopped: {e}")

    def _ram(self):
        while not self._stop:
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemTotal:"):
                        self.ram_total_mb = int(line.split()[1]) // 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) // 1024
                        if self.ram_min_avail_mb is None or avail < self.ram_min_avail_mb:
                            self.ram_min_avail_mb = avail
                time.sleep(5)
            except Exception:
                time.sleep(5)

    def stop(self):
        self._stop = True
        ram_used_peak = None
        if self.ram_total_mb and self.ram_min_avail_mb is not None:
            ram_used_peak = self.ram_total_mb - self.ram_min_avail_mb
        return {"vram_max_mb": self.vram_max_mb, "ram_total_mb": self.ram_total_mb,
                "ram_used_peak_mb": ram_used_peak, "vram_samples_n": len(self.vram_samples)}


def http_json(path, payload=None, timeout=60):
    url = BASE + path
    if payload is not None:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    else:
        req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_bytes(path, timeout=120):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read()


def env_report():
    log("=== PHASE 0: env report ===")
    subprocess.run(["nvidia-smi"])
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal:"):
            log("meminfo: " + line.strip())
    import torch
    TIM["env"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "python": sys.version.split()[0]}
    log(f"env: {TIM['env']}")
    save()


def install():
    log("=== PHASE 1: install ComfyUI + GGUF nodes ===")
    if not os.path.isdir(C):
        rc, dt = sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {C}")
        log(f"clone ComfyUI rc={rc} {dt:.1f}s")
    gg = f"{C}/custom_nodes/ComfyUI-GGUF"
    if not os.path.isdir(gg):
        rc, dt = sh(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF {gg}")
        log(f"clone ComfyUI-GGUF (leejet) rc={rc} {dt:.1f}s")
    rc, dt = sh(f"pip install -q -r {C}/requirements.txt 2>&1 | tail -3", timeout=1200)
    log(f"pip comfy requirements rc={rc} {dt:.1f}s")
    rc, dt = sh(f"pip install -q -r {gg}/requirements.txt 2>&1 | tail -3", timeout=1200)
    log(f"pip gguf requirements rc={rc} {dt:.1f}s")
    save()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = time.time()
    rc = subprocess.run(["wget", "-q", "-c", "--tries=3", "--timeout=60", "-O", dest, url]).returncode
    dt = time.time() - t0
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    log(f"download {os.path.basename(dest)} rc={rc} {size/1e9:.2f}GB in {dt:.1f}s")
    return {"url": url.rsplit("/", 1)[-1], "file_gb": round(size / 1e9, 2),
            "download_s": round(dt, 1), "rc": rc}


def fetch_weights():
    log("=== PHASE 2: download weights ===")
    mdir = f"{C}/models"
    recs = [
        download(GGUF_URL, f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf"),
        download(TE_INT8_URL, f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
        download(VAE_URL, f"{mdir}/vae/qwen_image_2.1_vae_bf16.safetensors"),
    ]
    os.makedirs(f"{mdir}/unet", exist_ok=True)
    shutil.copy(f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf", f"{mdir}/unet/")
    os.makedirs(f"{mdir}/clip", exist_ok=True)
    shutil.copy(f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", f"{mdir}/clip/")
    TIM["downloads"] = recs
    save()


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


def object_info_check():
    oi = http_json("/object_info")
    checks = {}
    gguf_loaders = [k for k in oi if "UnetLoaderGGUF" in k]
    checks["gguf_loader_classes"] = gguf_loaders
    if not gguf_loaders:
        raise RuntimeError("no UnetLoaderGGUF* node found")
    checks["gguf_loader"] = "UnetLoaderGGUF" if "UnetLoaderGGUF" in oi else gguf_loaders[0]
    checks["has_advanced_loader"] = "UnetLoaderGGUFAdvanced" in oi
    checks["advanced_weight_dtype_enum"] = None
    if checks["has_advanced_loader"]:
        spec = oi["UnetLoaderGGUFAdvanced"]["input"]
        wd = (spec.get("required", {}).get("weight_dtype") or
              spec.get("optional", {}).get("weight_dtype"))
        checks["advanced_weight_dtype_enum"] = wd[0] if wd else None
    spec = oi["CLIPLoader"]["input"]
    checks["has_qwen_image_type"] = "qwen_image" in (
        spec.get("required", {}).get("type") or spec.get("optional", {}).get("type"))[0]
    checks["has_EmptySD3LatentImage"] = "EmptySD3LatentImage" in oi
    log(f"object_info checks: {json.dumps(checks)}")
    if not (checks["has_qwen_image_type"] and checks["has_EmptySD3LatentImage"]):
        raise RuntimeError(f"node graph unsupported: {checks}")
    return checks


def build_workflow(checks, row):
    _, pos = row["prompt"]
    adv_enum = checks.get("advanced_weight_dtype_enum") or []
    if row["fp16"] and checks.get("has_advanced_loader") and "fp16" in adv_enum:
        loader, loader_inputs = "UnetLoaderGGUFAdvanced", {
            "unet_name": "qwen-image-2.1-Q4_K_M.gguf", "weight_dtype": "fp16"}
    else:
        loader, loader_inputs = checks["gguf_loader"], {
            "unet_name": "qwen-image-2.1-Q4_K_M.gguf"}
    w = h = row["size"]
    return {
        "1": {"class_type": loader, "inputs": loader_inputs},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": SEED, "steps": row["steps"], "cfg": 2.5,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"v2_{row['key']}"}},
    }


def run_prompt(wf_dict, proc, timeout_s=1500):
    t0 = time.time()
    resp = http_json("/prompt", {"prompt": wf_dict, "client_id": "w6probe"})
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        if proc.poll() is not None:
            return False, round(time.time() - t0, 1), [], "server process died"
        hist = http_json(f"/history/{pid}", timeout=15)
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        wall = round(time.time() - t0, 1)
        imgs = [im["filename"] for node_out in entry.get("outputs", {}).values()
                for im in node_out.get("images", [])]
        if status.get("status_str") == "error":
            return False, wall, imgs, f"comfy error: {json.dumps(status.get('messages', []))[:500]}"
        if imgs:
            return True, wall, imgs, "ok"
    return False, round(time.time() - t0, 1), [], "timeout waiting for history"


def run_rows(pollers, server_flags, rows, boot_no):
    """Boot one server and run rows in order. fp16 rows need the Advanced
    loader's fp16 enum; if it is missing and --force-fp16 is not set, the row
    is skipped with a marker so the caller can relaunch with the flag."""
    comfy_log = open(f"{OUT}/comfyui-v2-{boot_no}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *server_flags],
        cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    results = []
    try:
        if not wait_server(proc):
            tail = subprocess.run(f"tail -40 {OUT}/comfyui-v2-{boot_no}.log", shell=True,
                                  capture_output=True, text=True).stdout
            log(f"server boot FAILED (flags={server_flags}):\n{tail}")
            return [{"error": "boot failed", "flags": server_flags}]
        checks = object_info_check()
        for row in rows:
            adv_enum = checks.get("advanced_weight_dtype_enum") or []
            if row["fp16"] and "fp16" not in adv_enum and "--force-fp16" not in server_flags:
                log(f"{row['key']}: skipping — Advanced loader enum lacks fp16 {adv_enum}")
                results.append({"key": row["key"], "ok": False, "steps": row["steps"],
                                "size": row["size"], "fp16": row["fp16"],
                                "note": "fp16 path unavailable, needs --force-fp16 relaunch"})
                continue
            wf_dict = build_workflow(checks, row)
            ok, wall, imgs, note = run_prompt(wf_dict, proc)
            rec = {"key": row["key"], "steps": row["steps"], "size": row["size"], "fp16": row["fp16"],
                   "flags": server_flags, "ok": ok, "wall_s": wall, "files": imgs, "note": note,
                   "vram_max_mb_so_far": pollers.vram_max_mb}
            log(f"{row['key']}: ok={ok} {wall}s ({note})")
            if ok:
                for fn in imgs:
                    data = http_bytes(f"/view?filename={urlquote(fn)}&subfolder=&type=output")
                    with open(f"{OUT}/v2_{fn}", "wb") as f:
                        f.write(data)
                    rec["saved"] = f"v2_{fn}"
            results.append(rec)
            save()
        try:
            logtxt = open(f"{OUT}/comfyui-v2-{boot_no}.log").read()
            casts = re.findall(r"weight dtype \S+, manual cast: \S+", logtxt)
            for r in results:
                if isinstance(r, dict) and "wall_s" in r:
                    r["cast_lines_seen"] = casts
            results.append({"server_prompt_executed": re.findall(r"Prompt executed in ([0-9:.]+)", logtxt),
                            "server_flags": server_flags})
        except Exception:
            pass
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        comfy_log.close()
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    env_report()
    install()
    fetch_weights()
    pollers = Pollers()
    pollers.start()

    # phase 1: one server for the fp32 control row + the Advanced-loader fp16 rows
    TIM["rows"].extend(run_rows(pollers, [], ROWS, 0))
    save()

    # phase 2: if any fp16 row could not run (no fp16 enum), retry under --force-fp16
    pending = [r for r in ROWS if r["fp16"] and not any(
        isinstance(t, dict) and t.get("key") == r["key"] and t.get("ok") for t in TIM["rows"])]
    if pending:
        log(f"Advanced-loader fp16 unavailable for {[r['key'] for r in pending]}; "
            "relaunching with --force-fp16")
        res = run_rows(pollers, ["--force-fp16"], pending, 1)
        for r in res:
            if isinstance(r, dict):
                r["method"] = "force_fp16_flag"
        TIM["rows"].extend(res)
        save()

    final = pollers.stop()
    TIM["final_pollers"] = final
    TIM["total_s"] = round(time.time() - T0, 1)
    TIM["success"] = any(isinstance(r, dict) and r.get("ok") and r.get("fp16") for r in TIM["rows"])
    save()
    log(f"=== DONE success={TIM['success']} total={TIM['total_s']}s ===")
    print("=== TIMINGS_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if TIM["success"] else 1)


if __name__ == "__main__":
    main()
