"""Skill demand: counts in one snapshot, and change across daily snapshots."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import config, skills


def demand(postings: pd.DataFrame) -> pd.DataFrame:
    """How many postings mention each skill. `postings` needs a `skills` list column."""
    n = max(len(postings), 1)
    counts = postings["skills"].explode().value_counts()
    out = pd.DataFrame({"skill": counts.index, "postings": counts.to_numpy()})
    out["share"] = (100 * out["postings"] / n).round(1)
    out["group"] = out["skill"].map(skills.group)
    return out.reset_index(drop=True)


def over_time(db: Path | None = None, software_only: bool = True, levels: list[str] | None = None) -> pd.DataFrame:
    """Share (%) of live postings that mention each skill, for every snapshot date.

    Long format: date, skill, postings, total, share.
    """
    con = sqlite3.connect(db or config.DB)
    q = """SELECT s.date, p.skills FROM snapshots s JOIN postings p ON p.id = s.id WHERE 1=1"""
    params: list = []
    if software_only:
        q += " AND p.software = 1"
    if levels:
        q += f" AND p.level IN ({','.join('?' * len(levels))})"
        params += levels
    df = pd.read_sql(q, con, params=params)
    con.close()
    totals = df.groupby("date").size().rename("total")
    df["skill"] = df["skills"].fillna("").str.split(",")
    long = df.explode("skill")
    long = long[long["skill"] != ""]
    out = long.groupby(["date", "skill"]).size().rename("postings").reset_index().join(totals, on="date")
    out["share"] = 100 * out["postings"] / out["total"]
    return out


def by_posted_month(postings: pd.DataFrame, min_postings: int = 30) -> pd.DataFrame:
    """Skill share by the month a posting was first published, from one snapshot.

    Works with a single dump, but it only sees postings that are still open, so older
    months are biased toward hard-to-fill roles. Months with few postings are dropped.
    """
    df = postings[postings["posted"].fillna("").str.len() >= 7].copy()
    df["month"] = df["posted"].str[:7]
    totals = df.groupby("month").size().rename("total")
    totals = totals[totals >= min_postings]
    long = df[df["month"].isin(totals.index)].explode("skills").dropna(subset=["skills"])
    out = long.groupby(["month", "skills"]).size().rename("postings").reset_index().join(totals, on="month")
    out = out.rename(columns={"skills": "skill"})
    out["share"] = 100 * out["postings"] / out["total"]
    return out


def rising(over: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Skills whose share moved the most between the first and last snapshot."""
    dates = sorted(over["date"].unique())
    if len(dates) < 2:
        return pd.DataFrame(columns=["skill", "first", "last", "change"])
    wide = over.pivot(index="skill", columns="date", values="share").fillna(0)
    out = pd.DataFrame({"first": wide[dates[0]], "last": wide[dates[-1]]})
    out["change"] = out["last"] - out["first"]
    return out.reindex(out["change"].abs().sort_values(ascending=False).index).head(top).round(2).reset_index()
