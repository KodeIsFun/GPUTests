# 02 — Colab CLI lane (agent-driven)

Install once, auth once (see 01-manual-steps.md Path B), then everything here
is headless. Binary is `colab` (google-colab-cli).

## Install

```bash
uv tool install google-colab-cli        # or: pipx install google-colab-cli
~/.local/bin/colab --version
```

## Auth (needs the human once)

```bash
colab --auth=oauth2      # prints a URL + code; human approves in browser
colab sessions           # must list sessions without error = authed
```

Stale docs mention `gcloud` / ADC — ignore them. OAuth2 is the only supported
flow; a refresh token lands in `~/.config/colab-cli/token.json`.

## Run the notebook headlessly

`colab run` executes a **.py script** on a fresh VM. Notebooks go through
`colab new` + `colab exec -f` (exec's `--timeout` is **seconds**, default 30 —
always set it):

```bash
colab new --gpu T4 -s llm-notebook       # creates the session; does NOT self-clean
colab exec -s llm-notebook -f colab/run-llm-t4.ipynb --timeout 900
colab stop llm-notebook                  # ALWAYS; then verify with colab sessions
colab sessions
```

- Cell output streams to the console — grep for the notebook's success
  markers: `OK: install`, `OK: download`, `OK: load`, `OK: chat`, `OK: api`,
  `API URL:`.
- If `colab new` 400s on the accelerator, the free tier has no T4 entitlement
  *right now*: retry later, or `colab new -s llm-notebook` (CPU works, ~3
  tok/s on 8B — fine for a smoke test, useless for real serving).
- A plain script (not the notebook) can skip `new` entirely:
  `colab run --gpu T4 --timeout 1200 myscript.py` creates, runs, and
  self-cleans the VM. (`run`'s `--timeout` is also seconds; its default 30 s
  is shorter than the pip install.)

## Never run (interactive / stateful subcommands)

```text
colab auth --interactive, colab repl, anything that mounts Drive or opens an
editor session. They leave VMs running that nothing will stop for you.
```

`colab run` (script) is the exception that self-cleans; `colab new` never
does — pair every `colab new` with a `colab stop`.

## Known breakage (fixed once, will recur on upgrades)

A `uv` upgrade once pulled `jupyter_kernel_client` 1.0.2, which renamed
`KernelClient` and crashed `colab` **before** allocating any VM:

```bash
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

Fix: pin the dependency inside the tool's venv:

```bash
uv pip install --python ~/.local/share/uv/tools/google-colab-cli/bin/python \
  'jupyter_kernel_client<1'
```

If `colab` ever dies with an AttributeError right after upgrade, check this
pin first.
