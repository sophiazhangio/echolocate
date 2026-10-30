"""Run job-radar's command line from echolocate (fetch postings, create a career profile).

Every job-radar command used here ends by printing one JSON line, e.g.
{"ok": true, "profile": "it", "file": "...", "postings": 812}; we return that line as a dict.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess

from . import config

NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


def _run(args: list[str], timeout: int) -> dict:
    npm = shutil.which("npm")
    if not npm:
        return {"ok": False, "error": "npm was not found. Install Node.js (https://nodejs.org), then reopen the app."}
    if not (config.JOB_RADAR / "package.json").exists():
        return {"ok": False, "error": f"job-radar was not found at {config.JOB_RADAR}. Put it next to echolocate "
                                      "or set JOB_RADAR_DIR."}
    try:
        p = subprocess.run([npm, "run", "--silent", *args], cwd=config.JOB_RADAR, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"job-radar took longer than {timeout // 60} minutes and was stopped."}
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
    return {"ok": False, "error": "job-radar did not finish: " + (" ".join(tail) or f"exit code {p.returncode}")}


def dump(profile: str) -> dict:
    """Fetch today's postings for a profile into job-radar's out/all/[<profile>/]<date>.jsonl."""
    return _run(["dump", "--", "--profile", profile], timeout=15 * 60)


def new_profile(name: str, description: str, model: str | None = None) -> dict:
    """Draft profiles/<name>/ (skills and companies) with the local Ollama model."""
    if not NAME.fullmatch(name):
        return {"ok": False, "error": "Use lowercase letters, digits and dashes for the name, e.g. nursing or ux-design."}
    args = ["new-profile", "--", name, description] + (["--model", model] if model else [])
    return _run(args, timeout=15 * 60)


def add_company(profile: str, url: str, name: str | None = None) -> dict:
    """Check a careers page and add it to profiles/<profile>/companies.json."""
    return _run(["add-company", "--", profile, url] + ([name] if name else []), timeout=3 * 60)
