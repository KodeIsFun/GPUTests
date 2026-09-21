# PROGRESS — `run-llm-on-free-gpu` — COMPLETED 2026-09-20 (paused once on owner order, resumed same night)

**FINAL STATE: shipped and verified.** Repo: <https://github.com/KodeIsFun/run-llm-on-free-gpu> (public, default branch `main`). The verification lap that was pending at the pause is GREEN: all cells OK on a fresh free T4, `server healthy`, `API URL: https://...trycloudflare.com` printed, and a real completion served through the public tunnel to a machine outside Colab (model `local-llm`, usage accounting intact). Session stopped after; `colab sessions` empty. What changed since the snapshot below: `drain_vram()` moved to the top of the load cell, load-failure messages carry real error text, api-server reference documents Qwen3 thinking through the server. The snapshot/ folder and the notes below are kept as the historical record of the pause; the live repo supersedes them.

The standalone repo at `/Users/tuhin/RND/run-llm-on-free-gpu` was **deleted on
the owner's order** after this snapshot was committed here. `snapshot/` is the
complete working copy (18 files) — restore it with:

```bash
mkdir -p /Users/tuhin/RND/run-llm-on-free-gpu
cp -r projects/2026-W38-free-gpu-skill/snapshot/. /Users/tuhin/RND/run-llm-on-free-gpu/
cd /Users/tuhin/RND/run-llm-on-free-gpu && git init -q && git add -A
# git history did not exist yet (first commit was never made) — no history to lose
```

## What this repo is (the goal, unchanged)

An agent-consumable skill + playbook, A-to-Z: install + authenticate the
Colab CLI → run open LLMs on free T4s via llama.cpp → expose an
OpenAI-compatible API consumable from anywhere → tuning. Self-sustainable
(no GPUTests/MCP deps), holds the user's hand through manual steps, and
centers on a Colab notebook verified to actually open and run.

## State: ~85% done, verification loop 3 of 4 stages green

### Done and VERIFIED on a real free T4 (Colab session `verify-t4`, 2026-09-20)

- Notebook parses and executes via `colab new --gpu T4 -s <name>` →
  `colab exec -s <name> -f colab/run-llm-t4.ipynb --timeout 900`:
  - config cell ✓, GPU-detect cell ✓ (`Tesla T4, 15360 MB`),
  - install cell ✓ (prebuilt cu124 wheel, `GPU offload supported`),
  - download cell ✓ (Qwen3-8B, ungated, cached),
  - load cell ✓ — **offload proof printed**: `offloaded 37/37 layers to GPU`,
  - chat cell ✓ — coherent answer, llama.cpp perf line **35.11 tok/s** decode,
  - all code cells compile clean; every cell terminates (headless-safe).
- Full repo authored: SKILL.md (agent routing + 7 invariants), references
  01–06 (manual steps, colab CLI lane, kaggle lane, API server, tuning,
  14 troubleshooting entries), scripts/verify_env.py, templates/kaggle-job.py
  (proven W5 v3 job, generalized), tools/build_notebook.py, README.
- Cross-links checked (none broken). Colab VM stopped, `colab sessions` empty.

### The verification loop's remaining failure: API cell (cell 7)

Three real bugs found and fixed through `colab exec` runs (logs in this
folder: `colab-exec2.log`, `-3.log`, `-4.log`):

1. **Port 8080 is taken on Colab VMs** (`[Errno 98] address already in use`)
   → fixed: runtime free-port allocation (`socket.bind(("0.0.0.0", 0))`).
2. **`/health` returns 404** on llama-cpp-python 0.3.x
   → fixed: readiness probe now `GET /v1/models` (first 200 = model loaded).
3. **`colab run` is for .py scripts, not notebooks** — it executed raw ipynb
   JSON as Python and died on `true`. Docs now teach
   `new` → `exec -f` → `stop`. (`--timeout` on both is SECONDS, default 30.)

**Not yet seen passing:** the `API URL:` line (cloudflared tunnel) — exec4
was aborted before reaching it. That is the ONE thing left to verify.

### Open bug found in exec4 (likely multi-exec artifact, still fix before publishing)

exec4 re-ran on the SAME session: the previous exec's 8B model was still
resident in VRAM when cell 5 tried to load, so the load fell back
14B → gpt-oss-20b. Fresh sessions don't have this (browser users start
clean), but the fix is one move: put the `gc.collect()` + VRAM-drain loop at
the TOP of cell 5 (before the load loop), not only in cell 7. Also worth
surfacing the actual load error text in the fallback message (exec4's
`ValueError` detail was swallowed).

### Next steps on resume (in order)

1. Restore the snapshot (commands above).
2. Move gc/VRAM-drain to the top of cell 5 in `tools/build_notebook.py`;
   rebuild (`python tools/build_notebook.py`).
3. `colab new --gpu T4 -s verify-t4` → `colab exec -s verify-t4 -f
   colab/run-llm-t4.ipynb --timeout 900` → grep for `OK: api` + `API URL:`.
4. From the Mac, `curl <API URL>/v1/chat/completions -d '{"model":"local-llm",...}'`
   — the "consumable from anywhere" proof.
5. `colab stop -s verify-t4`; `colab sessions` must be empty.
6. `git init` + first commit; `gh repo create KodeIsFun/run-llm-on-free-gpu
   --public --source . --push` (owner previously intended public).
7. Consider: pip-install instruction also on CPU-only runtimes works (wheel
   is py3-none) — verified implicitly by Colab py3.13.

### Environment facts that will matter on resume

- Colab image is **Python 3.13** now (`/usr/local/lib/python3.13/...`);
  py3-none wheel works. Colab VM has something on port 8080.
- colab CLI (0.6.0) at `~/.local/bin/colab`, oauth2 authed.
  `colab stop -s <name>` (flag, not positional).
- Kaggle quota untouched by this work; proxy spend $0.0435 lifetime.
