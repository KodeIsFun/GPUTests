#!/usr/bin/env python3
"""W6 probe: Qwen-Image-2.1 GGUF Q4_K_M on a free Colab T4.

Runs ComfyUI headless with the leejet/ComfyUI-GGUF fork (per the model card),
generates 2 fixed prompts, and records honest timings. Designed for
`colab run`/`colab exec`: everything of value goes to stdout (flushed) and
/content/out/. OOM is a result, not a failure — a 3-attempt ladder tries the
card-recommended setup first, then --lowvram, then a smaller Comfy-Org
text-encoder quant picked at runtime.

Settings note (recorded in TIM["settings_assumptions"]): official card uses
40 steps @ 2048x2048; we use 20 steps @ 1024x1024 for free-GPU feasibility,
cfg 2.5 / res_multistep / simple carried over from the Qwen-Image 1.0
Comfy-Org template because the 2.1 card publishes no sampler settings.
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

T0 = time.time()
TIM = {
    "run": "qwen-image-2.1-q4km-t4",
    "platform": "colab-t4",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "settings_assumptions": {
        "official": "40 steps @ 2048x2048 (diffusers card example), sampler unpublished",
        "used": "20 steps @ 1024x1024, cfg 2.5, res_multistep/simple, seed 42",
        "why": "free T4 budget; cfg/sampler carried from Qwen-Image 1.0 Comfy-Org template",
    },
    "attempts": [],
}


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
        self._threads = []

    def start(self):
        self._stop = False
        self._threads = [
            threading.Thread(target=self._vram, daemon=True),
            threading.Thread(target=self._ram, daemon=True),
        ]
        for t in self._threads:
            t.start()

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
    data = None
    req = urllib.request.Request(url, method="GET")
    if payload is not None:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
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
    sh("df -h /content")
    import torch
    TIM["env"] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": sys.version.split()[0],
    }
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
    TIM["install_done_s"] = round(time.time() - T0, 1)
    save()


def download(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = time.time()
    rc = subprocess.run(
        ["wget", "-q", "-c", "--tries=3", "--timeout=60", "-O", dest, url]).returncode
    dt = time.time() - t0
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    log(f"download {os.path.basename(dest)} rc={rc} {size/1e9:.2f}GB in {dt:.1f}s "
        f"({size/dt/1e6:.0f} MB/s)" if dt > 0 and size else
        f"download {os.path.basename(dest)} rc={rc} FAILED")
    return {"url": url.rsplit("/", 1)[-1], "file_gb": round(size / 1e9, 2),
            "download_s": round(dt, 1), "rc": rc}


def fetch_weights():
    log("=== PHASE 2: download weights ===")
    mdir = f"{C}/models"
    recs = []
    recs.append(download(GGUF_URL, f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf"))
    # loader insurance: some node versions scan models/unet instead of diffusion_models
    os.makedirs(f"{mdir}/unet", exist_ok=True)
    shutil.copy(f"{mdir}/diffusion_models/qwen-image-2.1-Q4_K_M.gguf", f"{mdir}/unet/")
    recs.append(download(TE_INT8_URL, f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"))
    os.makedirs(f"{mdir}/clip", exist_ok=True)
    shutil.copy(f"{mdir}/text_encoders/qwen3vl_8b_int8_convrot.safetensors", f"{mdir}/clip/")
    recs.append(download(VAE_URL, f"{mdir}/vae/qwen_image_2.1_vae_bf16.safetensors"))
    TIM["downloads"] = recs
    TIM["download_total_s"] = round(sum(r["download_s"] for r in recs), 1)
    save()


def pick_small_te():
    """Find the smallest Comfy-Org text-encoder quant for Qwen-Image-2.1."""
    try:
        with urllib.request.urlopen(
                "https://huggingface.co/api/models/Comfy-Org/Qwen-Image-2.1", timeout=60) as r:
            sib = [s["rfilename"] for s in json.loads(r.read())["siblings"]]
        cands = [s for s in sib if s.startswith("text_encoders/") and
                 re.search(r"(int4|w4a8|nvfp4|fp8|int8)", s)]
        prio = {"int4": 0, "w4a8": 1, "nvfp4": 2, "fp8": 3, "int8": 4}
        cands.sort(key=lambda s: (next((p for k, p in prio.items() if k in s), 9), len(s)))
        log(f"Comfy-Org TE candidates: {cands}")
        return cands[0] if cands else None
    except Exception as e:
        log(f"pick_small_te failed: {e}")
        return None


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
        raise RuntimeError(f"no UnetLoaderGGUF* node found. gguf-ish nodes: "
                           f"{[k for k in oi if 'gguf' in k.lower()][:20]}")
    checks["gguf_loader"] = "UnetLoaderGGUF" if "UnetLoaderGGUF" in oi else gguf_loaders[0]
    spec = oi["CLIPLoader"]["input"]
    te_type = (spec.get("required", {}).get("type") or spec.get("optional", {}).get("type"))[0]
    checks["cliploader_types"] = te_type
    checks["has_qwen_image_type"] = "qwen_image" in te_type
    checks["has_EmptySD3LatentImage"] = "EmptySD3LatentImage" in oi
    kspec = oi["KSampler"]["input"]
    samplers = (kspec.get("required", {}).get("sampler_name") or
                kspec.get("optional", {}).get("sampler_name"))[0]
    checks["has_res_multistep"] = "res_multistep" in samplers
    log(f"object_info checks: {json.dumps(checks)}")
    if not (checks["has_qwen_image_type"] and checks["has_EmptySD3LatentImage"]):
        raise RuntimeError(f"node graph unsupported: {checks}")
    return checks


def build_workflow(unet_loader, te_name, pos, prefix, seed=42, steps=20, cfg=2.5, w=1024, h=1024):
    return {
        "1": {"class_type": unet_loader, "inputs": {"unet_name": "qwen-image-2.1-Q4_K_M.gguf"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te_name, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": pos}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0], "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": "res_multistep", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def run_prompt(wf_dict, proc, timeout_s=2400):
    """POST workflow, wait for completion. Returns (ok, wall_s, images, note).

    Aborts immediately if the ComfyUI server process dies (RAM-kill), instead
    of burning the full timeout.
    """
    t0 = time.time()
    resp = http_json("/prompt", {"prompt": wf_dict, "client_id": "w6probe"})
    pid = resp["prompt_id"]
    while time.time() - t0 < timeout_s:
        time.sleep(3)
        if proc.poll() is not None:
            return False, round(time.time() - t0, 1), [], "server process died (RAM kill?)"
        hist = http_json(f"/history/{pid}", timeout=15)
        if pid not in hist:
            continue
        entry = hist[pid]
        status = entry.get("status", {})
        wall = round(time.time() - t0, 1)
        imgs = []
        for node_out in entry.get("outputs", {}).values():
            for im in node_out.get("images", []):
                imgs.append(im["filename"])
        if status.get("status_str") == "error":
            return False, wall, imgs, f"comfy error: {json.dumps(status.get('messages', []))[:500]}"
        if imgs:
            return True, wall, imgs, "ok"
    return False, round(time.time() - t0, 1), [], "timeout waiting for history"


PROMPTS = [
    ("p1_cat_raincoat",
     "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones "
     "at night, neon city lights bokeh, 50mm lens"),
    ("p2_freegpulab_sign",
     'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement'),
]


def attempt(n, te_name, flags, pollers):
    log(f"=== ATTEMPT {n}: te={te_name} flags={flags} ===")
    rec = {"attempt": n, "te": te_name, "flags": flags, "images": [], "errors": []}
    comfy_log = open(f"{OUT}/comfyui-a{n}.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), *flags],
        cwd=C, stdout=comfy_log, stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            rec["errors"].append("server failed to boot")
            tail = subprocess.run(f"tail -40 {OUT}/comfyui-a{n}.log", shell=True,
                                  capture_output=True, text=True).stdout
            log("server boot FAILED, log tail:\n" + tail)
            return rec
        rec["boot_s"] = round(time.time() - T0, 1)
        checks = object_info_check()
        rec["node_checks"] = checks
        for name, pos in PROMPTS:
            if proc.poll() is not None:
                rec["errors"].append(f"server died before {name}")
                break
            wf_dict = build_workflow(checks["gguf_loader"], te_name, pos, f"w6t4_{name}")
            ok, wall, imgs, note = run_prompt(wf_dict, proc)
            rec["images"].append({"name": name, "ok": ok, "wall_s": wall, "files": imgs, "note": note})
            log(f"{name}: ok={ok} {wall}s ({note})")
            if ok:
                for fn in imgs:
                    data = http_bytes(f"/view?filename={urlquote(fn)}&subfolder=&type=output")
                    with open(f"{OUT}/a{n}_{fn}", "wb") as f:
                        f.write(data)
                    rec["images"][-1]["saved"] = f"a{n}_{fn}"
            save()
        # pull per-image "Prompt executed" lines out of the server log
        try:
            logtxt = open(f"{OUT}/comfyui-a{n}.log").read()
            rec["prompt_executed_lines"] = re.findall(r"Prompt executed in ([0-9.]+) seconds", logtxt)
        except Exception:
            pass
        rec["pollers_end"] = pollers.stop()
        pollers.start()
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        comfy_log.close()
        if rec.get("errors") or not all(i["ok"] for i in rec["images"]):
            tail = subprocess.run(f"tail -60 {OUT}/comfyui-a{n}.log", shell=True,
                                  capture_output=True, text=True).stdout
            rec["comfy_log_tail"] = tail[-4000:]
            log(f"attempt {n} log tail:\n{tail[-2000:]}")
    return rec


def main():
    os.makedirs(OUT, exist_ok=True)
    env_report()
    install()
    fetch_weights()
    TIM["node_checks_note"] = "verified at runtime via /object_info per attempt"
    pollers = Pollers()
    pollers.start()

    ladder = [
        ("qwen3vl_8b_int8_convrot.safetensors", []),                # card-recommended
        ("qwen3vl_8b_int8_convrot.safetensors", ["--lowvram"]),     # card's own fallback
    ]
    small = pick_small_te()
    if small:
        fn = small.rsplit("/", 1)[-1]
        dest = f"{C}/models/text_encoders/{fn}"
        if not os.path.exists(dest):
            TIM["small_te_download"] = download(
                f"https://huggingface.co/Comfy-Org/Qwen-Image-2.1/resolve/main/{small}", dest)
            shutil.copy(dest, f"{C}/models/clip/{fn}")
        ladder.append((fn, []))
    else:
        log("no smaller Comfy-Org TE found; ladder stays at 2 attempts")
    TIM["ladder"] = [{"te": t, "flags": f} for t, f in ladder]
    save()

    success = False
    for n, (te, flags) in enumerate(ladder, start=1):
        rec = attempt(n, te, flags, pollers)
        TIM["attempts"].append(rec)
        save()
        if rec.get("images") and all(i["ok"] for i in rec["images"]):
            success = True
            rec["success"] = True
            TIM["final_attempt"] = n
            save()
            break
        log(f"attempt {n} did not fully succeed; moving down the ladder")

    final = pollers.stop()
    TIM["final_pollers"] = final
    TIM["total_s"] = round(time.time() - T0, 1)
    TIM["success"] = success
    save()
    log(f"=== DONE success={success} total={TIM['total_s']}s ===")
    print("=== TIMINGS_JSON ===", flush=True)
    print(json.dumps(TIM, indent=2), flush=True)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
