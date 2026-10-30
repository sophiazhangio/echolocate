"""Paths and settings. Everything can be overridden with environment variables."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JOB_RADAR = Path(os.environ.get("JOB_RADAR_DIR", ROOT.parent / "job-radar")).expanduser()
DUMPS = Path(os.environ.get("ECHO_DUMPS", JOB_RADAR / "out" / "all")).expanduser()
DATA = Path(os.environ.get("ECHO_DATA", ROOT / "data")).expanduser()
DB = DATA / "index.sqlite"

# all-MiniLM-L6-v2: 384-dim, ~80 MB, fast on a laptop CPU or Apple GPU.
MODEL = os.environ.get("ECHO_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# CPU by default: MiniLM is small enough, and the PyTorch Apple-GPU (mps) backend crashes on
# older macOS versions. Set ECHO_DEVICE=mps or cuda to use a GPU.
DEVICE = os.environ.get("ECHO_DEVICE", "cpu")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")


def resume_path() -> Path | None:
    """The resume PDF that job-radar is configured to use (config.local.json, else config.json)."""
    if p := os.environ.get("ECHO_RESUME"):
        return Path(p).expanduser()
    for name in ("config.local.json", "config.json"):
        cfg = JOB_RADAR / name
        if cfg.exists():
            resume = json.loads(cfg.read_text()).get("resume")
            if resume:
                return (JOB_RADAR / resume).resolve()
    return None
