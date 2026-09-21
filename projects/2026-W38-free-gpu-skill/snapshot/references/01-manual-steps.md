# 01 — What the human must do manually

Everything else is agent-executable. This file is the complete list of steps
that need a human with a browser. Walk the user through only the section(s)
their chosen path needs, verbatim or paraphrased.

---

## Path A — Notebook only (most common, ~2 minutes)

**Needed:** any Google account. Nothing else.

1. Tell the user: open <https://colab.research.google.com> and sign in.
   First visit may ask them to accept Colab's terms — accept.
2. Give them the notebook: `colab/run-llm-t4.ipynb`
   (upload via File → Upload notebook, or open from the repo/GitHub).
3. Tell them: **Runtime → Change runtime type → Hardware accelerator: T4 GPU
   → Save.** The notebook's first cell also checks this and tells them.
4. **Runtime → Run all.** Expected wall time: ~3 minutes with the default
   Qwen3-4B, ~5 minutes with Qwen3-8B. The last cells print a chat demo and,
   if `SERVE_API = True`, a public API URL.

No other manual step exists on this path.

---

## Path B — Agent-driven via the Colab CLI (~5 minutes, one browser sitting)

The agent does everything except one OAuth consent:

1. Agent installs the CLI (no human):
   `uv tool install google-colab-cli` (or `pipx install google-colab-cli`).
2. Agent runs `colab --auth=oauth2`. The CLI prints a URL + device code.
   **Human:** open the URL in a browser, sign in with the Google account,
   approve the consent screen. (No gcloud, no service accounts — the CLI is
   OAuth2 only. Documentation claiming ADC is stale.)
3. Human can close the browser. Everything after this is headless.

## Path C — Unattended Kaggle lane (~10 minutes, one sitting)

For repeatable headless benchmarks. Needs a Kaggle account:

1. **Human:** create/sign in at <https://www.kaggle.com>.
2. **Human:** verify the account by phone (Account settings → Phone
   verification). GPU access is refused without it; there is no API
   workaround.
3. **Human:** <https://www.kaggle.com/settings/api> → **Generate New Token**
   → copy the token string and give it to the agent (or paste into
   `~/.kaggle/access_token`). Note: this page has no JSON download — the
   "Create Legacy API Key" button is the only JSON source and it is not
   needed; CLI ≥ 2.2 accepts the access token.

## Optional — stable API URL via ngrok (~3 minutes)

The default tunnel (cloudflared quick tunnel) needs no account but produces a
new random URL each run. If the user wants a *stable* hostname while a
session lives:

1. **Human:** create a free account at <https://dashboard.ngrok.com>, copy
   the authtoken from the Getting Started page.
2. Hand the authtoken to the agent (notebook cell `NGROK_AUTHTOKEN`), or set
   it as the `NGROK_AUTHTOKEN` env var.

---

## What the human should NOT be asked to do

- Install gcloud, create service accounts, or configure ADC for Colab.
- Download any `kaggle.json` legacy file.
- Accept Hugging Face license walls (use the ungated catalog instead).
- Pay anyone anything. If a step asks for a credit card, it is the wrong step.
