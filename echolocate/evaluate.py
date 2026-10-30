"""Small retrieval benchmark: dense vs keyword vs hybrid, precision@10.

There are no human relevance labels, so each query comes with a rule that decides whether
a posting is relevant (title pattern + description pattern). The rules are keyword-based,
which favours the keyword retriever a little; treat the numbers as a sanity check that
shows the trade-offs, not as a leaderboard.

    python -m echolocate.evaluate
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .search import Filters, Searcher
from .store import Index, SentenceEmbedder

# (query, title pattern, description pattern) — both must match. "" = no constraint.
QUERIES = [
    ("frontend engineer building React and TypeScript user interfaces", r"front.?end|\bui\b|web", r"\breact"),
    ("backend engineer writing services in Go", r"back.?end|server|platform|software|infrastructure", r"\bgolang\b|\bgo\b"),
    ("machine learning engineer training PyTorch models", r"machine learning|\bml\b|\bai\b|research", r"pytorch"),
    ("iOS developer writing Swift", r"\bios\b|mobile|apple", r"\bswift\b"),
    ("Android engineer using Kotlin", r"android|mobile", r"kotlin"),
    ("data engineer building pipelines with Spark and Airflow", r"data", r"spark|airflow"),
    ("site reliability engineer running Kubernetes in production", r"reliability|\bsre\b|devops|infrastructure|platform|cloud", r"kubernetes|\bk8s\b"),
    ("application security engineer", r"security", ""),
    ("new grad software engineer, recent graduates welcome", r"new grad|graduate|entry|early career|university|junior", ""),
    ("engineer building LLM agents and retrieval-augmented generation", "", r"(llm|agent).*(\brag\b|retrieval|embedding)|(\brag\b|retrieval|embedding).*(llm|agent)"),
    ("full stack engineer using Next.js", "", r"next\.?js"),
    ("embedded firmware engineer", r"embedded|firmware", ""),
    ("engineering manager leading a software team", r"manager", ""),
    ("QA automation engineer writing end-to-end tests", r"\bqa\b|quality|test|sdet", ""),
    ("developer advocate, developer relations", r"advocate|devrel|developer relations|evangelist", ""),
]


def relevant(df: pd.DataFrame, title_pat: str, desc_pat: str) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    if title_pat:
        ok &= df["title"].str.lower().str.contains(title_pat, regex=True).to_numpy()
    if desc_pat:
        rx = re.compile(desc_pat, re.S)
        ok &= df["description"].fillna("").str.lower().map(lambda t: bool(rx.search(t))).to_numpy()
    return ok


def run(k: int = 10) -> pd.DataFrame:
    ix = Index.load()
    s = Searcher(ix, SentenceEmbedder(ix.model))
    f = Filters(software_only=False)
    rows = []
    for q, tp, dp in QUERIES:
        rel = relevant(ix.postings, tp, dp)
        ids = set(ix.postings["id"][rel])
        dense = s.search(q, f, k, hybrid=False)["id"]
        sparse_scores = s.sparse_scores(q)
        sparse = ix.postings["id"].iloc[np.argsort(-sparse_scores)[:k]]
        hybrid = s.search(q, f, k, hybrid=True)["id"]
        rows.append({"query": q, "relevant in corpus": int(rel.sum()),
                     "dense": np.mean([i in ids for i in dense]),
                     "keyword": np.mean([i in ids for i in sparse]),
                     "hybrid": np.mean([i in ids for i in hybrid])})
    out = pd.DataFrame(rows)
    mean = out[["dense", "keyword", "hybrid"]].mean()
    out = pd.concat([out.astype({"relevant in corpus": object}), pd.DataFrame([{"query": "MEAN", "relevant in corpus": "", **mean.to_dict()}])], ignore_index=True)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(run().round(2).to_string(index=False))
