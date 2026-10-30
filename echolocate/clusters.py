"""Market map: UMAP projection of posting embeddings, clustered and labelled.

Raw posting vectors cluster by employer: every Stripe posting shares Stripe's vocabulary
and tone. Subtracting each company's mean vector ("company centering") removes that shared
part, so what is left groups by kind of role. On the 2026-10-29 snapshot this cut the
average top-company share per cluster from 81% to 28%.

HDBSCAN runs on a 10-dim UMAP projection (density clustering works poorly in 384 dims);
the 2-dim projection is only for the picture. Each cluster is named by the title words
that are most distinctive for it (class-based TF-IDF over job titles).
"""

from __future__ import annotations

import hashlib
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, profiles
from .store import Index

_TITLE_NOISE = (r"\b(i{1,3}|iv|\d+|remote|us|usa|new york|san francisco|hybrid|nyc|sf|ny|ca|onsite|"
                r"senior|sr|staff|principal|lead|engineer|engineers|engineering|software|developer|"
                r"member|technical|team|manager)\b")
def _profile_key() -> str:
    """Empty for software, so existing map caches (and the demo's) keep their names."""
    p = profiles.get()
    return "" if p.name == "software" else f"{p.name}{p.version}"


def center_by_company(X: np.ndarray, companies: pd.Series, min_postings: int = 3) -> np.ndarray:
    """Subtract each company's mean vector (companies with few postings are left as is)."""
    out = X.copy()
    for _, idx in companies.reset_index(drop=True).groupby(companies.reset_index(drop=True)).indices.items():
        if len(idx) >= min_postings:
            out[idx] -= X[idx].mean(axis=0)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


MIN_POSTINGS = 20


def cluster_map(ix: Index, mask: np.ndarray | None = None, method: str = "hdbscan",
                n_clusters: int = 20, seed: int = 42, cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (points, clusters).

    points: one row per posting in the mask with x, y, cluster, label (cluster -1 = noise).
    clusters: cluster, label, postings, top companies, top skills.
    """
    mask = np.ones(len(ix.postings), dtype=bool) if mask is None else mask
    key = hashlib.sha1(np.packbits(mask).tobytes() + f"{ix.latest_date}{ix.model}{method}{n_clusters}{seed}v4{_profile_key()}".encode()).hexdigest()[:12]
    path = config.DATA / f"map-{key}.pkl"
    if cache and path.exists():
        return pickle.loads(path.read_bytes())

    import umap
    df = ix.postings[mask].reset_index(drop=True)
    X = center_by_company(ix.posting_vecs[mask], df["company"])
    n = len(df)
    if n < MIN_POSTINGS:
        raise ValueError(f"need at least {MIN_POSTINGS} postings to draw a map, got {n}")
    xy = umap.UMAP(n_components=2, n_neighbors=min(15, n - 1), min_dist=0.1, metric="cosine",
                   random_state=seed).fit_transform(X)
    labels = None
    if method == "hdbscan":
        from sklearn.cluster import HDBSCAN
        z = umap.UMAP(n_components=min(10, n - 2), n_neighbors=min(30, n - 1), min_dist=0.0, metric="cosine",
                      random_state=seed).fit_transform(X)
        # small selections (one level, remote only) need smaller clusters to find any
        size = max(5, min(25, n // 12))
        labels = HDBSCAN(min_cluster_size=size, min_samples=min(10, size)).fit_predict(z)
        if len(set(labels) - {-1}) < 2:
            labels = None  # nothing found; fall back to KMeans below
    if labels is None:
        from sklearn.cluster import KMeans
        k = max(2, min(n_clusters, n // 10))
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(X)

    names = _label_clusters(df["title"], labels)
    points = df[["id", "company", "title", "location", "level", "remote", "url"]].copy()
    points["x"], points["y"], points["cluster"] = xy[:, 0], xy[:, 1], labels
    points["label"] = points["cluster"].map(names).fillna("(unclustered)")

    rows = []
    for c in sorted(set(labels) - {-1}):
        sub = df[labels == c]
        rows.append({"cluster": c, "label": names[c], "postings": len(sub),
                     "top companies": ", ".join(sub["company"].value_counts().head(4).index),
                     "top skills": ", ".join(sub["skills"].explode().value_counts().head(6).index)})
    clusters = pd.DataFrame(rows, columns=["cluster", "label", "postings", "top companies", "top skills"])
    clusters = clusters.sort_values("postings", ascending=False).reset_index(drop=True)
    if cache:
        try:
            config.DATA.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pickle.dumps((points, clusters)))
        except OSError:
            pass  # read-only deployments just recompute
    return points, clusters


def _label_clusters(titles: pd.Series, labels: np.ndarray, top: int = 3) -> dict[int, str]:
    docs, ids = [], []
    for c in sorted(set(labels) - {-1}):
        t = titles[labels == c].str.lower().str.replace(_TITLE_NOISE, " ", regex=True)
        if noise := profiles.get().title_noise:
            t = t.str.replace(noise, " ", regex=True)
        docs.append(" . ".join(t))
        ids.append(c)
    if not docs:
        return {}
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-z][a-z.+#/-]*[a-z+#]\b")
    m = vec.fit_transform(docs)
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for row, c in enumerate(ids):
        scores = m[row].toarray().ravel()
        picked: list[str] = []
        used: set[str] = set()
        for term in vocab[np.argsort(-scores)]:
            words = term.split()
            # skip "intern intern", and terms whose words are all already shown
            if len(set(words)) < len(words) or set(words) <= used:
                continue
            picked.append(term)
            used |= set(words)
            if len(picked) == top:
                break
        out[c] = " / ".join(picked)
    return out
