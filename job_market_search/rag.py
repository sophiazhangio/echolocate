"""Question answering over postings with a local Ollama model (retrieval-augmented generation).

Retrieve the most relevant chunks (at most two per posting), number them as sources,
and ask the model to answer only from those sources and cite them as [1], [2], ...
"""

from __future__ import annotations

import json
from typing import Callable

import requests

from . import config
from .search import Filters, Searcher, parse_query

SYSTEM = """You answer questions about job postings using ONLY the numbered sources given.
Cite sources inline like [1] or [2][5]. If the sources do not contain the answer, say so plainly.
Be concise: short paragraphs or bullet points. Do not invent companies, salaries or requirements."""


def ollama_status(url: str = config.OLLAMA_URL, model: str = config.OLLAMA_MODEL) -> tuple[bool, str]:
    """(ready, message). Ready means the server is up and the model is pulled."""
    try:
        tags = requests.get(f"{url}/api/tags", timeout=2).json()
    except requests.RequestException:
        return False, f"Ollama is not running at {url}. Install it (brew install ollama), then run `ollama serve`."
    names = {m["name"] for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return False, f"Ollama is running but `{model}` is not pulled. Run `ollama pull {model}`."
    return True, f"{model} via Ollama"


def build_prompt(question: str, sources: list[dict]) -> str:
    blocks = []
    for n, s in enumerate(sources, 1):
        level = {"entry": "entry level / new grad", "intern": "internship", "senior+": "senior or above",
                 "unspecified": "level not stated"}.get(s.get("level", ""), "")
        meta = ", ".join(x for x in [s["location"] or "", "remote" if s["remote"] else "", level, s["pay"] or "",
                                      f"posted {s['posted']}" if s["posted"] else ""] if x)
        blocks.append(f"[{n}] {s['title']} at {s['company']} ({meta})\n{s['text']}")
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\nAnswer with citations:"


def gather(searcher: Searcher, question: str, filters: Filters | None = None, k: int = 8) -> list[dict]:
    """Retrieved chunks merged per posting, in rank order.

    Words like "remote" or "new grad" in the question become hard filters first; otherwise
    retrieval happily returns on-site roles and the model may call them remote.
    """
    df = searcher.ix.postings
    _, filters, _ = parse_query(question, filters or Filters())
    merged: dict[int, dict] = {}
    for row, text, score in searcher.retrieve_chunks(question, filters, k=k):
        body = text.split(":", 1)[1].strip() if ":" in text else text
        if row in merged:
            merged[row]["text"] += " ... " + body
            continue
        r = df.iloc[row]
        merged[row] = {"id": r["id"], "company": r["company"], "title": r["title"], "location": r["location"],
                       "remote": bool(r["remote"]), "level": r["level"], "pay": r["pay"], "posted": r["posted"], "url": r["url"],
                       "score": score, "text": body}
    return list(merged.values())


def ask(searcher: Searcher, question: str, filters: Filters | None = None, k: int = 8,
        stream: bool = False, on_token: Callable[[str], None] | None = None,
        url: str = config.OLLAMA_URL, model: str = config.OLLAMA_MODEL) -> tuple[str, list[dict]]:
    ok, msg = ollama_status(url, model)
    if not ok:
        raise RuntimeError(msg)
    sources = gather(searcher, question, filters, k)
    if not sources:
        return "No postings match these filters.", []
    body = {"model": model, "system": SYSTEM, "prompt": build_prompt(question, sources),
            "stream": stream, "options": {"temperature": 0.1, "num_ctx": 8192}}
    r = requests.post(f"{url}/api/generate", json=body, stream=stream, timeout=300)
    r.raise_for_status()
    if not stream:
        return r.json()["response"], sources
    parts = []
    for line in r.iter_lines():
        if not line:
            continue
        piece = json.loads(line).get("response", "")
        parts.append(piece)
        if on_token:
            on_token(piece)
    return "".join(parts), sources
