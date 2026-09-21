#!/usr/bin/env python3
"""Agent preflight: what is ready, what needs the human, with exact commands.

Run this FIRST in any new environment. It never installs, never authenticates,
never spends anything — it only checks and prints a report. Exit code is 0
when the requested lane is ready, 1 when a manual step is missing (the report
says which).

Usage:
    python verify_env.py            # check everything, print full report
    python verify_env.py --lane notebook   # only what the notebook path needs
    python verify_env.py --lane colab-cli  # CLI lane (colab binary + auth)
    python verify_env.py --lane kaggle     # Kaggle lane (kaggle>=2.2 + auth)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/cu124"
FIXES: list[tuple[str, str]] = []  # (check name, exact fix command)


def run(cmd: list[str], timeout: int = 60) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, f"{type(e).__name__}: {e}"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def check(name: str, ok: bool, detail: str = "", fix: str | None = None) -> bool:
    mark = "READY" if ok else "MISSING"
    print(f"  [{mark:7s}] {name}" + (f" — {detail}" if detail else ""))
    if not ok and fix:
        FIXES.append((name, fix))
    return ok


def check_gpu() -> bool:
    code, out = run(["nvidia-smi", "--query-gpu=name,memory.total",
                     "--format=csv,noheader,nounits"])
    ok = code == 0 and bool(out.strip())
    if ok:
        name, total = [p.strip() for p in out.splitlines()[0].split(",")]
        check(f"local GPU ({name}, {total} MB)", True)
    else:
        check("local GPU", False,
              fix="none needed if you will run on Colab — this machine just can't host the model itself")
    return ok


def check_colab_cli() -> bool:
    binary = shutil.which("colab") or (str(Path.home() / ".local/bin/colab")
                                       if (Path.home() / ".local/bin/colab").exists()
                                       else None)
    installed = check("colab CLI installed", binary is not None,
                      detail=binary or "",
                      fix="uv tool install google-colab-cli  # or: pipx install google-colab-cli")
    if not installed:
        return False
    # `colab sessions` is the readiness probe — the CLI has no --version flag,
    # and exit 0 here proves both the install and the oauth2 token.
    code, out = run([binary, "sessions"], timeout=60)
    authed = code == 0
    check("colab authed (oauth2)", authed,
          detail="sessions listable" if authed else out.strip()[:80],
          fix="colab --auth=oauth2   # HUMAN STEP: open the printed URL and approve")
    return authed


def check_kaggle() -> bool:
    binary = None
    for cand in (Path(".venv/bin/kaggle"), shutil.which("kaggle")):
        if cand and Path(cand).exists():
            binary = str(cand)
            break
    code, out = run([binary, "--version"], timeout=30) if binary else (1, "no binary")
    version = re.search(r"(\d+\.\d+)", out or "")
    major_ok = bool(version) and float(version.group(1)) >= 2.2
    if not check("kaggle CLI installed (>= 2.2)", code == 0 and major_ok,
                 detail=(out.strip().splitlines() or ["?"])[0][:60],
                 fix="python3.11 -m venv .venv && .venv/bin/pip install -U 'kaggle>=2.2' "
                     "   # 1.7.x cannot read access_token"):
        return False
    code, out = run([binary, "kernels", "list", "-m", "--page-size", "1"], timeout=90)
    authed = code == 0 and "could not find kaggle.json" not in out.lower()
    check("kaggle authed (access_token)", authed,
          fix="HUMAN STEP: kaggle.com → Settings → API → Generate New Token → "
              "paste into ~/.kaggle/access_token (chmod 600)")
    return authed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["notebook", "colab-cli", "kaggle", "all"],
                        default="all")
    args = parser.parse_args()

    print(f"verify_env — lane: {args.lane}")
    ok = True
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    check(f"python (this interpreter {py})", sys.version_info >= (3, 9),
          fix="use python3.11+ for the kaggle venv; any 3.9+ works elsewhere")

    if args.lane in ("notebook", "all"):
        print("— notebook path (nothing to install locally) —")
        check_gpu()
    if args.lane in ("colab-cli", "all"):
        print("— colab CLI lane —")
        ok &= check_colab_cli()
    if args.lane in ("kaggle", "all"):
        print("— kaggle lane —")
        ok &= check_kaggle()

    if FIXES:
        print("\nTo finish setup, in order:")
        for name, fix in FIXES:
            print(f"  • {name}:")
            print(f"      {fix}")
        print("\n'HUMAN STEP' items need the person, not the agent — see "
              "references/01-manual-steps.md for click-by-click text.")
        return 1
    print("\nAll checks passed for this lane. Nothing manual is missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
