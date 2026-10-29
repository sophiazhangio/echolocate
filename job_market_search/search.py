"""Semantic and hybrid search over the in-memory index.

Ranking for a text query:
  1. dense: cosine similarity between the query and every chunk; a posting scores its best chunk
  2. sparse: TF-IDF cosine over title + description (catches exact names like "Next.js")
  3. the two rankings are merged with reciprocal rank fusion (RRF), which needs no score tuning
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .store import Embedder, Index
from .text import chunk, read_resume

RRF_K = 60


@dataclass
class Filters:
    software_only: bool = True
    remote_only: bool = False
    levels: list[str] = field(default_factory=list)  # empty = any level
    location: str = ""      # substring, case-insensitive; "remote" also matches remote postings
    company: str = ""
    posted_after: str = ""  # YYYY-MM-DD
    must_include: list[str] = field(default_factory=list)  # every term must appear in the text

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        if self.software_only:
            m &= df["software"].to_numpy()
        if self.remote_only:
            m &= df["remote"].to_numpy()
        if self.levels:
            m &= df["level"].isin(self.levels).to_numpy()
        if self.location:
            locs = [x.strip().lower() for x in self.location.split(",") if x.strip()]
            loc = df["location"].fillna("").str.lower()
            hit = np.zeros(len(df), dtype=bool)
            for x in locs:
                hit |= loc.str.contains(x, regex=False).to_numpy()
                if x == "remote":
                    hit |= df["remote"].to_numpy()
            m &= hit
        if self.company:
            m &= df["company"].str.lower().str.contains(self.company.lower(), regex=False).to_numpy()
        if self.posted_after:
            m &= (df["posted"].fillna("") >= self.posted_after).to_numpy()
        for term in self.must_include:
            text = (df["title"] + "\n" + df["description"].fillna("")).str.lower()
            m &= text.str.contains(term.lower(), regex=False).to_numpy()
        return m


_INTENTS = [
    # (pattern in the query, what it sets); the matched words are removed from the semantic query
    (r"\b(fully )?remote\b", "remote"),
    (r"\b(new[- ]?grads?|entry[- ]level|junior|early[- ]career|recent grad(uate)?s?)\b", "entry"),
    (r"\b(internships?|interns?)\b", "intern"),
]


def parse_query(query: str, filters: Filters) -> tuple[str, Filters, list[str]]:
    """Turn words like "remote" or "new grad" into filters instead of leaving them to the embedding.

    Embeddings are bad at hard constraints: "remote new grad" still ranks an on-site senior
    role highly if the rest matches. Returns (query for ranking, filters, notes for the UI).
    """
    notes: list[str] = []
    q = query
    for pat, what in _INTENTS:
        if not re.search(pat, q, re.I):
            continue
        if what == "remote" and not filters.remote_only:
            filters = replace(filters, remote_only=True)
            notes.append("remote only")
        elif what in ("entry", "intern") and filters.levels != [what]:
            filters = replace(filters, levels=[what, "unspecified"] if what == "entry" else [what])
            notes.append("entry level (or no level in title)" if what == "entry" else "internships")
        if what == "remote":
            q = re.sub(pat, " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", re.sub(r"\b(roles?|jobs?|positions?|postings?)\b", " ", q, flags=re.I)).strip()
    return q or query, filters, notes


class Searcher:
    def __init__(self, index: Index, embedder: Embedder):
        self.ix = index
        self.embedder = embedder
        self._tfidf = None
        self._tfidf_matrix = None

    # --- building blocks -------------------------------------------------------------

    def _sparse(self):
        if self._tfidf is None:
            df = self.ix.postings
            docs = (df["title"] + " " + df["title"] + " " + df["description"].fillna("")).tolist()
            # token_pattern keeps "next.js", "c++", "c#" and "ci/cd" as single tokens.
            self._tfidf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.5,
                                          token_pattern=r"(?u)\b\w[\w.+#/-]*[\w+#]|\b\w\b")
            self._tfidf_matrix = self._tfidf.fit_transform(docs)
        return self._tfidf, self._tfidf_matrix

    def dense_scores(self, query: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-posting best-chunk cosine, and the index of that best chunk."""
        q = self.embedder.encode([query])[0]
        sims = self.ix.chunk_vecs @ q
        n = len(self.ix.postings)
        best = np.full(n, -1.0, dtype=np.float32)
        np.maximum.at(best, self.ix.chunk_owner, sims)
        # index of the best chunk per posting (for the snippet)
        order = np.lexsort((-sims, self.ix.chunk_owner))
        first = np.ones(len(order), dtype=bool)
        first[1:] = self.ix.chunk_owner[order][1:] != self.ix.chunk_owner[order][:-1]
        best_chunk = np.full(n, -1, dtype=np.int64)
        best_chunk[self.ix.chunk_owner[order][first]] = order[first]
        return best, best_chunk

    def sparse_scores(self, query: str) -> np.ndarray:
        vec, mat = self._sparse()
        return (mat @ vec.transform([query]).T).toarray().ravel()

    # --- public API ------------------------------------------------------------------

    def search(self, query: str, filters: Filters | None = None, k: int = 20, hybrid: bool = True,
               understand: bool = True) -> pd.DataFrame:
        """Hybrid search. With `understand`, words like "remote" / "new grad" become filters;
        the notes end up in `result.attrs["understood"]`."""
        filters = filters or Filters()
        notes: list[str] = []
        if understand:
            query, filters, notes = parse_query(query, filters)
        mask = filters.mask(self.ix.postings)
        dense, best_chunk = self.dense_scores(query)
        if hybrid:
            sparse = self.sparse_scores(query)
            score = _rrf([np.where(mask, dense, -np.inf), np.where(mask & (sparse > 0), sparse, -np.inf)])
        else:
            score = dense.copy()
        score[~mask] = -np.inf
        top, more = self._dedupe(score, k)
        out = self._rows(top, score, dense, more)
        out["snippet"] = [_snippet(self.ix.chunk_text[best_chunk[i]]) for i in top]
        out.attrs["understood"] = notes
        out.attrs["query"] = query
        return out

    def similar_to_posting(self, posting_id: str, filters: Filters | None = None, k: int = 20) -> pd.DataFrame:
        df = self.ix.postings
        hits = np.flatnonzero(df["id"].to_numpy() == posting_id)
        if not len(hits):
            raise KeyError(posting_id)
        i = hits[0]
        score = self.ix.posting_vecs @ self.ix.posting_vecs[i]
        mask = (filters or Filters(software_only=False)).mask(df)
        mask[i] = False
        # The same role re-posted for several cities is not an interesting "similar" result.
        mask &= ~((df["company"] == df.at[i, "company"]) & (df["title"] == df.at[i, "title"])).to_numpy()
        score[~mask] = -np.inf
        top, more = self._dedupe(score, k)
        return self._rows(top, score, score, more)

    def similar_to_text(self, text: str, filters: Filters | None = None, k: int = 20) -> pd.DataFrame:
        """Rank postings against a long document (a resume): mean of its chunk vectors vs posting vectors."""
        chunks = chunk("Resume", "candidate", text)
        v = self.embedder.encode(chunks).mean(axis=0)
        v /= max(np.linalg.norm(v), 1e-9)
        score = self.ix.posting_vecs @ v
        score[~(filters or Filters()).mask(self.ix.postings)] = -np.inf
        top, more = self._dedupe(score, k)
        return self._rows(top, score, score, more)

    def similar_to_resume(self, path, filters: Filters | None = None, k: int = 20) -> pd.DataFrame:
        return self.similar_to_text(read_resume(path), filters, k)

    def retrieve_chunks(self, query: str, filters: Filters | None = None, k: int = 8,
                        per_posting: int = 2) -> list[tuple[int, str, float]]:
        """Top chunks for RAG: (posting row, chunk text, score), at most `per_posting` per posting."""
        q = self.embedder.encode([query])[0]
        sims = self.ix.chunk_vecs @ q
        mask = (filters or Filters()).mask(self.ix.postings)
        sims[~mask[self.ix.chunk_owner]] = -np.inf
        df = self.ix.postings
        keys = (df["company"] + "\x1f" + df["title"].str.lower()).to_numpy()
        out, count, owner_of = [], {}, {}
        for c in np.argsort(-sims):
            if not np.isfinite(sims[c]) or len(out) >= k:
                break
            p = int(self.ix.chunk_owner[c])
            # the same role posted in several cities counts as one posting
            if owner_of.setdefault(keys[p], p) != p or count.get(p, 0) >= per_posting:
                continue
            count[p] = count.get(p, 0) + 1
            out.append((p, self.ix.chunk_text[c], float(sims[c])))
        return out

    def _dedupe(self, score: np.ndarray, k: int) -> tuple[np.ndarray, list[int]]:
        """Top k with one row per (company, title); counts the hidden copies (other cities)."""
        df = self.ix.postings
        keys = (df["company"] + "\x1f" + df["title"].str.lower()).to_numpy()
        ranked = _top(score, k * 6)
        keep, more, where = [], [], {}
        for i in ranked:
            if keys[i] in where:
                more[where[keys[i]]] += 1
                continue
            if len(keep) == k:
                continue
            where[keys[i]] = len(keep)
            keep.append(i)
            more.append(0)
        return np.array(keep, dtype=np.int64), more

    def _rows(self, top: np.ndarray, score: np.ndarray, similarity: np.ndarray,
              more: list[int] | None = None) -> pd.DataFrame:
        cols = ["id", "company", "title", "location", "remote", "level", "posted", "pay", "url", "skills"]
        out = self.ix.postings.iloc[top][cols].reset_index(drop=True)
        out.insert(0, "score", np.round(score[top], 4))
        out.insert(1, "similarity", np.round(similarity[top], 3))
        out["more_locations"] = more if more is not None else [0] * len(out)
        return out


def _top(score: np.ndarray, k: int) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(score))
    if len(finite) <= k:
        return finite[np.argsort(-score[finite])]
    part = finite[np.argpartition(-score[finite], k)[:k]]
    return part[np.argsort(-score[part])]


def _rrf(scores: list[np.ndarray]) -> np.ndarray:
    """Reciprocal rank fusion: sum of 1 / (60 + rank) over the rankings a posting appears in."""
    total = np.zeros(len(scores[0]))
    for s in scores:
        ok = np.isfinite(s)
        ranks = np.empty(len(s))
        order = np.argsort(-np.where(ok, s, -1e18), kind="stable")
        ranks[order] = np.arange(1, len(s) + 1)
        total += np.where(ok, 1.0 / (RRF_K + ranks), 0.0)
    return total


def _snippet(text: str, n: int = 260) -> str:
    body = text.split(":", 1)[1].strip() if ":" in text else text
    return body if len(body) <= n else body[:n].rsplit(" ", 1)[0] + " ..."
