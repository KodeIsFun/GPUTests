#!/usr/bin/env python3
"""Paired Colab T4 test: Pocket 0.8B prompt rewrite + Qwen-Image-2.1 GGUF renders."""
import base64, json, os, re, shutil, subprocess, sys, time, urllib.request, zipfile
from urllib.parse import quote as urlquote

ROOT = "/content"
C = f"{ROOT}/ComfyUI"
OUT = f"{ROOT}/out"
PORT = 8188
BASE = f"http://127.0.0.1:{PORT}"
POCKET = "ML-Intern-lab/Qwen-Image-2.1-PE-T2I-Pocket-0.8B"
SEED = 42
SIZE = 768
STEPS = 12
FLAGS = ["--force-fp16", "--disable-comfy-compiler"]
GGUF_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF/resolve/main/qwen-image-2.1-UC-Q4_K_M.gguf"
TE_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"
VAE_URL = "https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"
PROMPTS = [
    ("cat_raincoat", "a photo of a siamese cat wearing a tiny yellow raincoat, sitting on wet cobblestones at night, neon city lights bokeh, 50mm lens"),
    ("freegpulab_sign", 'a neon shop sign that reads "FREE GPU LAB", rainy night, reflections on wet pavement'),
    ("bakery_bicycle", "a photo of a red bicycle leaning against a bakery door"),
]
RATIO_SIZES = {
    "1:1": (768, 768), "3:2": (576, 864), "2:3": (864, 576),
    "16:9": (512, 896), "9:16": (896, 512), "4:3": (832, 640),
    "3:4": (640, 832), "2:1": (480, 960), "1:2": (960, 480),
    "21:9": (448, 1024), "9:21": (1024, 448), "4:5": (768, 608),
    "5:4": (608, 768), "3:1": (352, 1024), "1:3": (1024, 384),
}
T0 = time.time()
TIM = {
    "run": "qwen-image-2.1-pocket-0.8b-paired-t4",
    "platform": "colab-t4",
    "pocket_model": POCKET,
    "image_model": "Qwen-Image-2.1 GGUF Q4_K_M via leejet/ComfyUI-GGUF",
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "settings": {"size_reference": "768x768 target; pocket-selected aspect ratio mapped to ~0.6MP", "steps": STEPS, "cfg": 2.5, "sampler": "res_multistep/simple", "seed": SEED, "image_flags": FLAGS},
    "rows": [],
}

def log(m): print(f"[{time.time()-T0:8.1f}s] {m}", flush=True)
def save():
    with open(f"{OUT}/timings.json", "w") as f: json.dump(TIM, f, indent=2)
def sh(args, timeout=1800):
    t=time.time(); r=subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode, time.time()-t, r.stdout[-2500:]
def download(url,dest):
    os.makedirs(os.path.dirname(dest),exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest)>100_000_000:
        log(f"cached {os.path.basename(dest)} ({os.path.getsize(dest)/1e9:.2f} GB)"); return 0,0.0
    t=time.time(); rc=subprocess.run(["wget","-q","-c","--tries=3","--timeout=60","-O",dest,url]).returncode
    dt=time.time()-t; size=os.path.getsize(dest) if os.path.exists(dest) else 0
    log(f"download {os.path.basename(dest)} rc={rc} {size/1e9:.2f} GB {dt:.1f}s")
    return rc,dt

def setup():
    log("=== install dependencies and fetch image weights ===")
    if not os.path.isdir(C):
        rc,dt,out=sh(["git","clone","--depth","1","https://github.com/comfyanonymous/ComfyUI",C]); log(f"clone ComfyUI rc={rc} {dt:.1f}s {out[-500:]}")
        rc,dt,out=sh([sys.executable,"-m","pip","install","-q","-r",f"{C}/requirements.txt"],timeout=1200); log(f"ComfyUI deps rc={rc} {dt:.1f}s {out[-500:]}")
    gg=f"{C}/custom_nodes/ComfyUI-GGUF"
    if not os.path.isdir(gg):
        rc,dt,out=sh(["git","clone","--depth","1","https://github.com/leejet/ComfyUI-GGUF",gg]); log(f"clone GGUF nodes rc={rc} {dt:.1f}s {out[-500:]}")
        rc,dt,out=sh([sys.executable,"-m","pip","install","-q","-r",f"{gg}/requirements.txt"],timeout=1200); log(f"GGUF deps rc={rc} {dt:.1f}s {out[-500:]}")
    rc,dt,out=sh([sys.executable,"-m","pip","install","-q","transformers>=4.57.0","accelerate>=1.2.0","safetensors>=0.4.5"],timeout=1200)
    log(f"pocket dependencies rc={rc} {dt:.1f}s {out[-700:]}")
    if rc: raise RuntimeError("Pocket model dependencies failed")
    m=f"{C}/models"
    dls=[]
    for url,dest in [(GGUF_URL,f"{m}/diffusion_models/qwen-image-2.1-UC-Q4_K_M.gguf"),(TE_URL,f"{m}/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),(VAE_URL,f"{m}/vae/qwen_image_2.1_vae_bf16.safetensors")]:
        rc,dt=download(url,dest); dls.append({"file":os.path.basename(dest),"rc":rc,"download_s":round(dt,1)})
        if rc: raise RuntimeError(f"weight download failed: {dest}")
    os.makedirs(f"{m}/unet",exist_ok=True); shutil.copy2(f"{m}/diffusion_models/qwen-image-2.1-UC-Q4_K_M.gguf",f"{m}/unet/")
    os.makedirs(f"{m}/clip",exist_ok=True); shutil.copy2(f"{m}/text_encoders/qwen3vl_8b_int8_convrot.safetensors",f"{m}/clip/")
    TIM["downloads"]=dls; save()

def rewrite_prompts():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is unavailable")
    TIM["env"]={"gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda":torch.version.cuda,"python":sys.version.split()[0]}
    log(f"Pocket load on {TIM['env']['gpu']}")
    t=time.time()
    tok=AutoTokenizer.from_pretrained(POCKET)
    model=AutoModelForCausalLM.from_pretrained(POCKET,torch_dtype=torch.float16,device_map="cuda")
    TIM["pocket_load_s"]=round(time.time()-t,1)
    out=[]
    for key,request in PROMPTS:
        prompt=tok.apply_chat_template([{"role":"user","content":request}],add_generation_prompt=True,tokenize=False)
        ids=tok(prompt,return_tensors="pt").to(model.device)
        torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        t=time.time()
        with torch.inference_mode():
            gen=model.generate(**ids,max_new_tokens=1024,do_sample=True,temperature=1.0,top_p=0.95,top_k=20,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        latency=time.time()-t
        raw=tok.decode(gen[0][ids["input_ids"].shape[1]:],skip_special_tokens=True)
        try:
            result=json.loads(raw)
            assert isinstance(result.get("rewritten_prompt"),str) and result["rewritten_prompt"].strip()
            ratio=result.get("wh_ratio","3:2")
            if ratio not in RATIO_SIZES: ratio="3:2"
            parse_note="ok"
        except Exception as e:
            result={"rewritten_prompt":request,"wh_ratio":"3:2"}; ratio="3:2"; parse_note=f"fallback: {type(e).__name__}: {str(e)[:120]}"
        h,w=RATIO_SIZES[ratio]
        rec={"key":key,"request":request,"rewritten_prompt":result["rewritten_prompt"],"wh_ratio":ratio,"pocket_latency_s":round(latency,3),"pocket_output_tokens":int(gen.shape[1]-ids["input_ids"].shape[1]),"parse":parse_note,"height":h,"width":w}
        out.append(rec); log(f"rewrite {key}: {latency:.2f}s ratio={ratio} tokens={rec['pocket_output_tokens']} parse={parse_note}")
        TIM.setdefault("rewrites",[]).append(rec); save()
    del model,tok
    torch.cuda.empty_cache()
    return out

def http_json(path,payload=None,timeout=60):
    req=urllib.request.Request(BASE+path,method="GET")
    if payload is not None: req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode(),method="POST",headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as r: return json.loads(r.read())
def wait_server(proc,timeout=300):
    t=time.time()
    while time.time()-t<timeout:
        if proc.poll() is not None: return False
        try: http_json("/system_stats",timeout=5); return True
        except Exception: time.sleep(3)
    return False

def workflow(text,h,w,prefix):
    return {
      "1":{"class_type":"UnetLoaderGGUF","inputs":{"unet_name":"qwen-image-2.1-UC-Q4_K_M.gguf"}},
      "2":{"class_type":"CLIPLoader","inputs":{"clip_name":"qwen3vl_8b_int8_convrot.safetensors","type":"qwen_image"}},
      "3":{"class_type":"VAELoader","inputs":{"vae_name":"qwen_image_2.1_vae_bf16.safetensors"}},
      "4":{"class_type":"CLIPTextEncode","inputs":{"clip":["2",0],"text":text}},
      "5":{"class_type":"CLIPTextEncode","inputs":{"clip":["2",0],"text":""}},
      "6":{"class_type":"EmptySD3LatentImage","inputs":{"width":w,"height":h,"batch_size":1}},
      "7":{"class_type":"KSampler","inputs":{"model":["1",0],"positive":["4",0],"negative":["5",0],"latent_image":["6",0],"seed":SEED,"steps":STEPS,"cfg":2.5,"sampler_name":"res_multistep","scheduler":"simple","denoise":1.0}},
      "8":{"class_type":"VAEDecode","inputs":{"samples":["7",0],"vae":["3",0]}},
      "9":{"class_type":"SaveImage","inputs":{"images":["8",0],"filename_prefix":prefix}},
    }

def render(outrow,arm,text,h,w,proc):
    key=outrow["key"]+"_"+arm
    t=time.time(); pid=http_json("/prompt",{"prompt":workflow(text,h,w,key),"client_id":"pocket-test"})["prompt_id"]
    note="timeout"; files=[]; ok=False
    while time.time()-t<1500:
        time.sleep(3)
        if proc.poll() is not None: note="ComfyUI exited"; break
        hist=http_json(f"/history/{pid}",timeout=15)
        if pid not in hist: continue
        entry=hist[pid]; status=entry.get("status",{})
        files=[im["filename"] for o in entry.get("outputs",{}).values() for im in o.get("images",[])]
        if status.get("status_str")=="error": note=json.dumps(status.get("messages",[]))[:600]; break
        if files: ok=True; note="ok"; break
    wall=round(time.time()-t,1); row={"key":key,"arm":arm,"ok":ok,"wall_s":wall,"files":files,"note":note}
    if ok:
        for fn in files:
            data=urllib.request.urlopen(f"{BASE}/view?filename={urlquote(fn)}&subfolder=&type=output",timeout=120).read()
            local=f"{key}.png"
            with open(f"{OUT}/{local}","wb") as f: f.write(data)
            row["saved"]=local
    log(f"render {key}: ok={ok} {wall:.1f}s {note}")
    TIM["rows"].append(row); save()

def main():
    os.makedirs(OUT,exist_ok=True)
    setup()
    rewrites=rewrite_prompts()
    log(f"starting ComfyUI {FLAGS}")
    clog=open(f"{OUT}/comfyui.log","w")
    proc=subprocess.Popen([sys.executable,"main.py","--listen","127.0.0.1","--port",str(PORT),*FLAGS],cwd=C,stdout=clog,stderr=subprocess.STDOUT)
    try:
        if not wait_server(proc):
            clog.flush(); tail=open(f"{OUT}/comfyui.log").read()[-5000:]; raise RuntimeError("ComfyUI boot failed: "+tail)
        for item in rewrites:
            # Keep both arms at the pocket model's selected aspect ratio for a controlled visual comparison.
            h,w=item["height"],item["width"]
            render(item,"direct",item["request"],h,w,proc)
            render(item,"pocket",item["rewritten_prompt"],h,w,proc)
    finally:
        proc.terminate()
        try: proc.wait(10)
        except subprocess.TimeoutExpired: proc.kill()
        clog.close()
    TIM["total_s"]=round(time.time()-T0,1)
    TIM["success"]=sum(r["ok"] for r in TIM["rows"])==len(PROMPTS)*2
    save()
    log(f"=== DONE success={TIM['success']} total={TIM['total_s']:.1f}s ===")
    print("=== POCKET_TIMINGS_JSON ===",flush=True); print(json.dumps(TIM,indent=2),flush=True)
    # Colab's ephemeral runner releases /content as soon as this script exits.
    # Return the generated images in stdout so the caller can save the artifacts.
    archive=f"{OUT}/pocket-test-images.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for row in TIM["rows"]:
            if row.get("saved"):
                z.write(f"{OUT}/{row['saved']}",arcname=row["saved"])
    payload=base64.b64encode(open(archive,"rb").read()).decode("ascii")
    print("=== POCKET_IMAGES_ZIP_BASE64_BEGIN ===",flush=True)
    print(payload,flush=True)
    print("=== POCKET_IMAGES_ZIP_BASE64_END ===",flush=True)
    sys.exit(0 if TIM["success"] else 1)
if __name__=="__main__": main()
