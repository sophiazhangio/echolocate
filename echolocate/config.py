"""Paths and settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOB_RADAR = Path(os.environ.get("JOB_RADAR_DIR", ROOT.parent / "job-radar")).expanduser()
DATA_ROOT = Path(os.environ.get("ECHO_DATA", ROOT / "data")).expanduser()
# DUMPS / DATA / DB belong to the active profile; use_profile() below sets them.
DUMPS = DATA = DB = Path()

# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast on a laptop CPU or Apple GPU.
MODEL = os.environ.get("ECHO_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CPU by default: MiniLM is small enough, and the PyTorch Apple-GPU (mps) backend crashes on
# older macOS versions. Set ECHO_DEVICE=mps or cuda to use a GPU.
DEVICE = os.environ.get("ECHO_DEVICE", "cpu")
# Hosted demo: fixed snapshot, no local LLM, no default resume. On by ECHO_DEMO=1, and
# automatically on Streamlit Community Cloud (which checks repos out under /mount/src).
DEMO = os.environ.get("ECHO_DEMO", "") not in ("", "0", "false") or str(ROOT).startswith("/mount/src/")
# Where the demo downloads its prebuilt index and map cache (a GitHub Release of this repo).
DEMO_ASSETS = os.environ.get(
    "ECHO_DEMO_ASSETS", "https://github.com/sophiazhangio/echolocate/releases/download/demo-2026-10-29")
DEMO_FILES = ["index.sqlite", "map-47021c78941c.pkl", "map-88aebda555ca.pkl"]
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def _job_radar_config() -> dict:
    """job-radar's settings: config.local.json if present, else config.json."""
    for name in ("config.local.json", "config.json"):
        cfg = JOB_RADAR / name
        if cfg.exists():
            try:
                return json.loads(cfg.read_text())
            except ValueError:
                return {}
    return {}


def profile() -> str:
    """Which career profile this install uses: job-radar's profiles/<name>/ (default "software").

    ECHO_PROFILE wins, then job-radar's "profile" setting. The older "field" setting and
    ECHO_FIELD still work. The hosted demo is always "software".
    """
    if DEMO:
        return "software"
    cfg = _job_radar_config()
    name = (os.environ.get("ECHO_PROFILE") or os.environ.get("ECHO_FIELD") or cfg.get("profile")
            or cfg.get("field") or "software")
    name = str(name).strip().lower()
    return name if re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) else "software"


def dumps_dir(name: str) -> Path:
    """Where job-radar writes a profile's daily dumps: out/all/ for software, out/all/<name>/ otherwise."""
    if os.environ.get("ECHO_DUMPS"):
        return Path(os.environ["ECHO_DUMPS"]).expanduser()
    base = JOB_RADAR / "out" / "all"
    return base if name == "software" else base / name


def data_dir(name: str) -> Path:
    """echolocate's index and caches for a profile: data/ for software, data/<name>/ otherwise."""
    return DATA_ROOT if name == "software" else DATA_ROOT / name


def use_profile(name: str) -> None:
    """Switch the active profile (the app's career-type picker calls this)."""
    global PROFILE, DUMPS, DATA, DB
    PROFILE, DUMPS, DATA = name, dumps_dir(name), data_dir(name)
    DB = DATA / "index.sqlite"


PROFILE = DEFAULT_PROFILE = profile()  # DEFAULT_PROFILE: what job-radar's config says
use_profile(PROFILE)


def resume_path() -> Path | None:
    """The resume PDF that job-radar is configured to use (config.local.json, else config.json)."""
    if DEMO:
        return None
    if p := os.environ.get("ECHO_RESUME"):
        return Path(p).expanduser()
    resume = _job_radar_config().get("resume")
    return (JOB_RADAR / resume).resolve() if resume else None
