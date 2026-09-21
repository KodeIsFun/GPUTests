# 04 — The API: OpenAI-compatible, reachable from anywhere

The notebook's `SERVE_API = True` path does four things; this reference is
the mechanics, so an agent can reproduce it outside the notebook too.

## 1. The server

llama-cpp-python ships an OpenAI-compatible server as a module extra:

```bash
pip install "llama-cpp-python[server]" \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

```bash
python -m llama_cpp.server \
  --model /path/to/model-Q4_K_M.gguf \
  --n_gpu_layers -1 \
  --n_ctx 4096 \
  --model_alias local-llm \
  --host 0.0.0.0 --port 8080 &
```

- `--model_alias local-llm` is what clients pass as `"model"` — pick one
  alias and keep it stable in your examples.
- Wait for readiness on `GET /v1/models` (200 once serving). The app only
  accepts requests after the model is loaded, so the first 200 means ready.
  (Don't probe `/health` — 0.3.x of the server returns 404 on it.) Big models
  take ~60 s to load; poll, don't sleep blind, and check `poll()` so a crashed
  server fails fast.
- Use `Qwen3-*` for serving. gpt-oss-20b's harmony chat template does not
  auto-apply in this server (see troubleshooting §8) — the endpoint works but
  answers arrive wrapped in `<|channel|>` markers unless you fight the
  template.

## 2. The tunnel (consumable from anywhere)

Colab VMs have no inbound ports. Tunnel out:

**Default — cloudflared quick tunnel (no account, no signup):**

```bash
curl -fsSL -o cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared
./cloudflared tunnel --url http://localhost:8080 --no-autoupdate > tunnel.log 2>&1 &
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1
```

The URL is random per run — treat it as ephemeral infrastructure, print it,
hand it to the user.

**Optional — ngrok (stable hostname, needs free authtoken):** set
`NGROK_AUTHTOKEN`, or outside the notebook:

```bash
pip install pyngrok && python -m pyngrok ngrok --authtoken "$NGROK_AUTHTOKEN" \
  && python -m pyngrok http 8080
```

## 3. Consuming it (from anywhere)

Everything that speaks OpenAI works. `API_KEY` is ignored; pass any string.

```bash
curl -s "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"local-llm","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

```python
from openai import OpenAI
client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="none")
reply = client.chat.completions.create(
    model="local-llm",
    messages=[{"role": "user", "content": "Name three planets."}],
    max_tokens=256,
)
print(reply.choices[0].message.content)
```

## 4. Expectations (measured, not aspirational)

Decode speed through the API ≈ the raw numbers in
[05-tuning.md](05-tuning.md) (8B ≈ 39.55 tok/s on the T4) plus tunnel latency
(trycloudflare adds ~50–150 ms per request; the stream itself runs at full
speed). Sessions die on Colab idle limits (~90 min without browser
interaction is typical) — the API is for a working session, not a deployment.
When the session dies, everything is gone; re-run the notebook.

## 5. Teardown

Kill in this order: tunnel process → server process → `colab stop` (CLI) or
Runtime → Disconnect (browser). A tunnel left pointing at a dead VM just
returns connection refused — harmless, but clean up anyway.
