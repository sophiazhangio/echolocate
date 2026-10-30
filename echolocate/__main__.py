"""Command line: python -m echolocate <command>

  index                          read new job-radar dumps and embed new/changed postings
  search "query" [filters]       hybrid semantic + keyword search
  similar --resume [PATH]        postings closest to your resume (default: job-radar's resume)
  similar --id gh:quora:123      postings closest to one posting
  trends [--top 15]              skill demand: latest snapshot, and change across snapshots
  map                            cluster the market and print the clusters
  ask "question"                 RAG answer from a local Ollama model, with cited postings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config


def _filters(a):
    from .search import Filters
    return Filters(software_only=not a.any_role, remote_only=a.remote, levels=a.level or [],
                   location=a.location or "", company=a.company or "", posted_after=a.since or "",
                   must_include=a.include or [])


def _add_filters(p):
    p.add_argument("--remote", action="store_true", help="remote postings only")
    p.add_argument("--level", action="append", choices=["entry", "unspecified", "intern", "senior+"],
                   help="seniority from the title; repeatable")
    p.add_argument("--location", help='substring(s), comma separated, e.g. "los angeles,irvine,remote"')
    p.add_argument("--company")
    p.add_argument("--since", help="posted on or after YYYY-MM-DD")
    p.add_argument("--include", action="append", help="term that must appear; repeatable")
    p.add_argument("--any-role", action="store_true", help="include roles outside the configured field (software or IT)")
    p.add_argument("-k", type=int, default=15)


def _print(df: pd.DataFrame):
    if df.attrs.get("understood"):
        print(f"Understood: {', '.join(df.attrs['understood'])} · ranking on \"{df.attrs['query']}\"\n")
    for _, r in df.iterrows():
        where = "Remote" if r["remote"] else (r["location"] or "?")[:50]
        extra = f" (+{r['more_locations']} more locations)" if r.get("more_locations") else ""
        print(f"{r['similarity']:.3f}  {r['company']} · {r['title']}  [{r['level']}] {where}{extra}  {r['posted']}")
        print(f"       {r['url']}")
        if "snippet" in r and r["snippet"]:
            print(f"       {r['snippet'][:180]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="echolocate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index")
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--dense-only", action="store_true"); _add_filters(p)
    p = sub.add_parser("similar"); p.add_argument("--resume", nargs="?", const="", default=None); p.add_argument("--id"); _add_filters(p)
    p = sub.add_parser("trends"); p.add_argument("--top", type=int, default=15); _add_filters(p)
    p = sub.add_parser("map"); p.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan"); _add_filters(p)
    p = sub.add_parser("ask"); p.add_argument("question"); _add_filters(p)
    a = ap.parse_args(argv)

    if a.cmd == "index":
        from .store import build
        print(build())
        return

    from .store import Index, SentenceEmbedder
    ix = Index.load()
    emb = SentenceEmbedder(ix.model)

    if a.cmd == "search":
        from .search import Searcher
        _print(Searcher(ix, emb).search(a.query, _filters(a), a.k, hybrid=not a.dense_only))
    elif a.cmd == "similar":
        from .search import Searcher
        s = Searcher(ix, emb)
        if a.id:
            _print(s.similar_to_posting(a.id, _filters(a), a.k))
        else:
            path = Path(a.resume) if a.resume else config.resume_path()
            if not path or not path.exists():
                sys.exit(f"Resume not found: {path}. Pass --resume PATH.")
            print(f"Resume: {path}\n")
            from echolocate.text import ResumeError
            try:
                _print(s.similar_to_resume(path, _filters(a), a.k))
            except ResumeError as e:
                sys.exit(str(e))
    elif a.cmd == "trends":
        from .trends import demand, over_time
        mask = _filters(a).mask(ix.postings)
        d = demand(ix.postings[mask]).head(a.top)
        print(f"Skill demand, {ix.latest_date} snapshot, {mask.sum():,} postings\n")
        print(d.to_string(index=False))
        t = over_time(config.DB, software_only=not a.any_role)
        if t["date"].nunique() > 1:
            print("\nShare of postings by snapshot date (%)\n")
            print(t.pivot(index="date", columns="skill", values="share")[d["skill"].head(8)].round(1).to_string())
    elif a.cmd == "map":
        from .clusters import cluster_map
        pts, labels = cluster_map(ix, _filters(a).mask(ix.postings), method=a.method)
        print(labels.to_string(index=False))
    elif a.cmd == "ask":
        from .rag import ask
        from .search import Searcher
        answer, sources = ask(Searcher(ix, emb), a.question, _filters(a), stream=True,
                              on_token=lambda t: print(t, end="", flush=True))
        print("\n\nSources:")
        for n, s in enumerate(sources, 1):
            print(f"  [{n}] {s['company']} · {s['title']}  {s['url']}")


if __name__ == "__main__":
    main()
