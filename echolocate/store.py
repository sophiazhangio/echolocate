"""SQLite store for postings, daily snapshots and chunk embeddings.

`build()` reads job-radar dumps (out/all/<date>.jsonl), upserts postings, records which
postings were live on which day, and embeds only postings whose text is new or changed.
`Index.load()` pulls everything into numpy for fast in-memory search.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
import pandas as pd

from . import config, skills
from .text import boilerplate, chunk, in_field, level, strip_boilerplate, text_hash

SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
  id TEXT PRIMARY KEY, company TEXT, title TEXT, location TEXT, remote INTEGER, url TEXT,
  posted TEXT, pay TEXT, description TEXT, level TEXT, software INTEGER, skills TEXT,
  hash TEXT, embedded_hash TEXT, first_seen TEXT, last_seen TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (date TEXT, id TEXT, PRIMARY KEY (date, id));
CREATE TABLE IF NOT EXISTS chunks (id TEXT, idx INTEGER, text TEXT, vec BLOB, PRIMARY KEY (id, idx));
CREATE TABLE IF NOT EXISTS dumps (date TEXT PRIMARY KEY, file TEXT, mtime REAL, n INTEGER);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class Embedder(Protocol):
    name: str
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceEmbedder:
    """sentence-transformers model, run locally. Vectors are L2-normalised float32."""

    def __init__(self, name: str = config.MODEL):
        self.name = name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.name, device=config.DEVICE)
        return self._model

    def encode(self, texts: list[str], progress: bool = False) -> np.ndarray:
        return self.model.encode(texts, batch_size=128, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=progress).astype(np.float32)


def connect(db: Path = config.DB) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.executescript(SCHEMA)
    return con


def dump_files(folder: Path = config.DUMPS) -> list[tuple[str, Path]]:
    out = []
    for f in sorted(folder.glob("*.jsonl")):
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", f.stem):
            out.append((f.stem, f))
    return out


def _meta(con, key: str) -> str | None:
    row = con.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(con, key: str, value: str):
    con.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))


def ingest(con, date: str, rows: Iterable[dict]) -> int:
    """Upsert one day's postings and record the snapshot. Returns the number of postings."""
    n = 0
    for r in rows:
        desc = r.get("description") or ""
        h = text_hash(r["title"], r["company"], desc)
        con.execute(
            """INSERT INTO postings (id, company, title, location, remote, url, posted, pay, description,
                                     level, software, skills, hash, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 company=excluded.company, title=excluded.title, location=excluded.location,
                 remote=excluded.remote, url=excluded.url, posted=excluded.posted, pay=excluded.pay,
                 description=excluded.description, level=excluded.level, software=excluded.software,
                 skills=CASE WHEN postings.hash = excluded.hash THEN postings.skills END, hash=excluded.hash,
                 first_seen=min(postings.first_seen, excluded.first_seen),
                 last_seen=max(postings.last_seen, excluded.last_seen)""",
            (r["id"], r["company"], r["title"], r.get("location", ""), int(bool(r.get("remote"))),
             r.get("url", ""), r.get("posted", ""), r.get("pay", ""), desc, level(r["title"]),
             int(in_field(r["title"], config.FIELD)), None, h, date, date))
        con.execute("INSERT OR IGNORE INTO snapshots VALUES (?, ?)", (date, r["id"]))
        n += 1
    return n


def embed_pending(con, embedder: Embedder, log: Callable[[str], None] = print, batch: int = 2048) -> int:
    """Embed every posting whose current text has not been embedded with this model yet."""
    if _meta(con, "model") != embedder.name:
        con.execute("DELETE FROM chunks")
        con.execute("UPDATE postings SET embedded_hash = NULL")
        _set_meta(con, "model", embedder.name)
    # Embed the description without company boilerplate; the full text stays in `postings`
    # for keyword search and skill counts. embedded_hash tracks the stripped text.
    # Skill tags also use the stripped text, so "We are the leading platform for AI agents"
    # in a company blurb does not count as every posting asking for AI agents.
    # The role flag ("software" column; for IT installs it means "IT role") and skill tags
    # follow config.FIELD, so changing the field takes effect on the next run.
    rows = con.execute("SELECT id, title, company, description, embedded_hash, skills, software "
                       "FROM postings").fetchall()
    common = boilerplate((r[2], r[3]) for r in rows)
    todo, retag, reflag = [], [], []
    for pid, title, company, desc, done, tags, flag in rows:
        body = strip_boilerplate(company, desc, common)
        new_tags = ",".join(skills.find(f"{title}\n{body}", config.FIELD))
        if new_tags != tags:
            retag.append((new_tags, pid))
        if int(in_field(title, config.FIELD)) != flag:
            reflag.append((int(in_field(title, config.FIELD)), pid))
        h = text_hash(title, company, body)
        if h != done:
            todo.append((pid, title, company, body, h))
    con.executemany("UPDATE postings SET skills = ? WHERE id = ?", retag)
    con.executemany("UPDATE postings SET software = ? WHERE id = ?", reflag)
    _set_meta(con, "field", config.FIELD)
    con.commit()
    if not todo:
        return 0
    log(f"Embedding {len(todo):,} postings with {embedder.name} ...")
    pending: list[tuple[str, int, str]] = []
    done_ids: list[tuple[str, str]] = []

    def flush():
        if not pending:
            return
        vecs = embedder.encode([t for _, _, t in pending])
        con.executemany("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?)",
                        [(pid, i, t, v.astype(np.float16).tobytes()) for (pid, i, t), v in zip(pending, vecs)])
        con.executemany("UPDATE postings SET embedded_hash = ? WHERE id = ?", done_ids)
        con.commit()
        pending.clear()
        done_ids.clear()

    for k, (pid, title, company, desc, h) in enumerate(todo, 1):
        con.execute("DELETE FROM chunks WHERE id = ?", (pid,))
        pending.extend((pid, i, c) for i, c in enumerate(chunk(title, company, desc)))
        done_ids.append((h, pid))
        if len(pending) >= batch:
            flush()
            log(f"  {k:,}/{len(todo):,} postings")
    flush()
    return len(todo)


def build(dumps: Path = config.DUMPS, db: Path = config.DB, embedder: Embedder | None = None,
          log: Callable[[str], None] = print) -> dict:
    """Ingest new or changed dump files, then embed whatever is pending."""
    con = connect(db)
    files = dump_files(dumps)
    if not files:
        raise SystemExit(f"No dumps found in {dumps}. Run `npm run scan -- --dump` in job-radar first.")
    ingested = 0
    for date, f in files:
        mtime = f.stat().st_mtime
        row = con.execute("SELECT mtime FROM dumps WHERE date = ?", (date,)).fetchone()
        if row and row[0] == mtime:
            continue
        with f.open() as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        con.execute("DELETE FROM snapshots WHERE date = ?", (date,))
        n = ingest(con, date, rows)
        con.execute("INSERT OR REPLACE INTO dumps VALUES (?,?,?,?)", (date, str(f), mtime, n))
        con.commit()
        log(f"Read {f.name}: {n:,} postings")
        ingested += 1
    embedded = embed_pending(con, embedder or SentenceEmbedder(), log)
    stats = {"dumps": len(files), "new_dumps": ingested, "embedded": embedded,
             "postings": con.execute("SELECT count(*) FROM postings").fetchone()[0],
             "chunks": con.execute("SELECT count(*) FROM chunks").fetchone()[0]}
    con.close()
    return stats


@dataclass
class Index:
    """Everything search needs, in memory.

    postings: one row per posting (DataFrame, positional index 0..n-1)
    chunk_vecs: (n_chunks, dim) normalised vectors; chunk_owner maps each chunk to a posting row
    posting_vecs: (n_postings, dim) normalised mean of a posting's chunks
    """
    postings: pd.DataFrame
    chunk_vecs: np.ndarray
    chunk_owner: np.ndarray
    chunk_text: list[str]
    posting_vecs: np.ndarray
    snapshots: pd.DataFrame
    model: str

    @classmethod
    def load(cls, db: Path = config.DB, latest_only: bool = True) -> "Index":
        con = connect(db)
        model = _meta(con, "model") or config.MODEL
        snaps = pd.read_sql("SELECT date, id FROM snapshots", con)
        latest = snaps["date"].max() if len(snaps) else None
        where = "WHERE id IN (SELECT id FROM snapshots WHERE date = ?)" if latest_only and latest else ""
        params = (latest,) if where else ()
        df = pd.read_sql(f"SELECT * FROM postings {where} ORDER BY id", con, params=params)
        df["remote"] = df["remote"].astype(bool)
        df["software"] = df["software"].astype(bool)
        df["skills"] = df["skills"].fillna("").map(lambda s: [x for x in s.split(",") if x])
        pos = {pid: i for i, pid in enumerate(df["id"])}
        rows = con.execute(f"SELECT id, text, vec FROM chunks {where} ORDER BY id, idx", params).fetchall()
        con.close()
        rows = [r for r in rows if r[0] in pos]
        if not rows:
            raise SystemExit("The index is empty. Run `python -m echolocate index` first.")
        owner = np.fromiter((pos[r[0]] for r in rows), dtype=np.int64, count=len(rows))
        vecs = np.stack([np.frombuffer(r[2], dtype=np.float16) for r in rows]).astype(np.float32)
        pv = np.zeros((len(df), vecs.shape[1]), dtype=np.float32)
        np.add.at(pv, owner, vecs)
        pv /= np.maximum(np.linalg.norm(pv, axis=1, keepdims=True), 1e-9)
        return cls(df, vecs, owner, [r[1] for r in rows], pv, snaps, model)

    @property
    def latest_date(self) -> str:
        return str(self.snapshots["date"].max())
