"""Streamlit UI: streamlit run app.py"""

from __future__ import annotations

import re

import pandas as pd
import plotly.express as px
import streamlit as st

from echolocate import config, skills
from echolocate.search import Filters, Searcher
from echolocate.store import Index, SentenceEmbedder
from echolocate.text import LEVELS, read_resume

st.set_page_config(page_title="echolocate", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model ...")
def load(db_mtime: float):
    ix = Index.load()
    s = Searcher(ix, SentenceEmbedder(ix.model))
    s.embedder.encode(["warm up"])
    s._sparse()
    return s


if not config.DB.exists():
    st.error("No index yet. Run `python -m echolocate index` first.")
    st.stop()
searcher = load(config.DB.stat().st_mtime)
ix = searcher.ix
df = ix.postings

# --- sidebar filters ----------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    software_only = st.checkbox("Software roles only", True)
    remote_only = st.checkbox("Remote only", False)
    levels = st.multiselect("Level (from title)", LEVELS, default=["entry", "unspecified"],
                            help="'unspecified' = the title names no level; many entry roles look like that.")
    location = st.text_input("Location contains", "", placeholder="los angeles, irvine, remote")
    company = st.text_input("Company contains", "")
    since = st.text_input("Posted on or after", "", placeholder="2026-09-01")
    include = st.text_input("Must mention (comma separated)", "", placeholder="python, react")
    k = st.slider("Results", 5, 50, 15)
    f = Filters(software_only=software_only, remote_only=remote_only, levels=levels, location=location,
                company=company, posted_after=since.strip(),
                must_include=[x.strip() for x in include.split(",") if x.strip()])
    n_match = int(f.mask(df).sum())
    st.caption(f"{n_match:,} of {len(df):,} postings match · snapshot {ix.latest_date} · {ix.model.split('/')[-1]}")


def md(text) -> str:
    """Escape text for st.markdown: "$" would start LaTeX, brackets/asterisks would format."""
    return re.sub(r"([\\`*_\[\]$<>#|])", r"\\\1", str(text or ""))


def show_results(res: pd.DataFrame, snippet: bool = False):
    if res.empty:
        st.info("Nothing matches these filters.")
        return
    if res.attrs.get("understood"):
        st.caption(f"Understood: **{', '.join(res.attrs['understood'])}** · ranking on “{res.attrs['query']}”")
    for _, r in res.iterrows():
        where = "Remote" if r["remote"] else (r["location"] or "location n/a")
        if r.get("more_locations"):
            where += f" (+{r['more_locations']} more locations)"
        with st.container(border=True):
            st.markdown(f"**[{md(r['title'])}]({r['url']})** · {md(r['company'])}")
            st.caption(md(f"{where} · {r['level']} · posted {r['posted'] or 'n/a'}"
                          f"{' · ' + r['pay'] if r['pay'] else ''} · similarity {r['similarity']:.2f}"))
            if snippet and r.get("snippet"):
                st.markdown(md(r["snippet"]))
            if r["skills"]:
                st.caption("Skills: " + ", ".join(r["skills"][:12]))
            if st.button("More like this", key=f"sim-{r['id']}-{st.session_state.get('tab', '')}"):
                st.session_state["similar_id"] = r["id"]
                st.session_state["similar_title"] = f"{r['title']} · {r['company']}"


st.title("echolocate")
st.markdown("*Find roles by meaning, not keywords.*")
st.caption(f"Semantic search, resume matching and skill trends over {len(df):,} live postings from "
           f"{df['company'].nunique():,} companies (Greenhouse, Ashby, Lever). Everything runs locally.")

tabs = st.tabs(["Search", "Match my resume", "Skill trends", "Market map", "Ask (RAG)"])

# --- Search ---------------------------------------------------------------------------
with tabs[0]:
    q = st.text_input("Describe the job you want", "remote new grad roles using Next.js and LLM agents")
    hybrid = st.toggle("Hybrid (semantic + keyword)", True,
                       help="Off = embeddings only. Hybrid also rewards exact names like 'Next.js'.")
    if sim := st.session_state.get("similar_id"):
        st.subheader(f"Similar to: {st.session_state['similar_title']}")
        if st.button("Clear"):
            del st.session_state["similar_id"]
            st.rerun()
        st.session_state["tab"] = "sim"
        show_results(searcher.similar_to_posting(sim, f, k))
    elif q.strip():
        st.session_state["tab"] = "search"
        show_results(searcher.search(q, f, k, hybrid=hybrid), snippet=True)

# --- Resume -----------------------------------------------------------------------------
with tabs[1]:
    default = config.resume_path()
    up = st.file_uploader("Resume (PDF or text). Defaults to the one in job-radar's config.", type=["pdf", "txt", "md"])
    text = ""
    if up is not None:
        tmp = config.DATA / f"upload-{up.name}"
        tmp.write_bytes(up.getvalue())
        text = read_resume(tmp)
        tmp.unlink()
        st.caption(f"Using uploaded {up.name}")
    elif default and default.exists():
        text = read_resume(default)
        st.caption("Using the resume set in job-radar's config")
    else:
        st.warning("No resume found. Upload one, or set ECHO_RESUME.")
    if text:
        st.session_state["tab"] = "resume"
        res = searcher.similar_to_text(text, f, k)
        have = set(skills.find(text))
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Closest postings")
            show_results(res)
        with c2:
            st.subheader("Skills these postings ask for")
            from echolocate.trends import demand
            top = demand(df[f.mask(df)]).head(25)
            top["on resume"] = top["skill"].isin(have).map({True: "yes", False: "-"})
            st.dataframe(top[["skill", "share", "on resume"]].rename(columns={"share": "% of postings"}),
                         hide_index=True, height=600)

# --- Trends -----------------------------------------------------------------------------
with tabs[2]:
    from echolocate.trends import by_posted_month, demand, over_time, rising
    sub = df[f.mask(df)]
    d = demand(sub)
    st.subheader(f"Skill demand in the {ix.latest_date} snapshot ({len(sub):,} postings)")
    grp = st.multiselect("Groups", sorted({g for g, _ in skills.SKILLS.values()}),
                         default=["Languages", "Web", "Cloud", "AI", "Data"])
    d = d[d["group"].isin(grp)].head(30)
    fig = px.bar(d.sort_values("share"), x="share", y="skill", color="group", orientation="h",
                 labels={"share": "% of postings that mention it", "skill": ""}, height=720)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig)

    over = over_time(config.DB, software_only=software_only, levels=levels or None)
    picks = st.multiselect("Skills to plot over time", list(skills.SKILLS),
                           default=["TypeScript", "Python", "LLMs", "AI agents", "RAG / embeddings", "AWS"])
    if over["date"].nunique() > 1:
        st.subheader("Share of live postings, by daily snapshot")
        fig = px.line(over[over["skill"].isin(picks)], x="date", y="share", color="skill", markers=True,
                      labels={"share": "% of postings"})
        st.plotly_chart(fig)
        st.dataframe(rising(over), hide_index=True)
    else:
        st.info("Only one snapshot so far. Daily trends appear once job-radar has dumped a second day "
                "(`npm run scan -- --dump`). Meanwhile, the chart below uses each posting's publish month.")
    pm = by_posted_month(sub)
    if not pm.empty:
        st.subheader("Share by month the posting was published (still-open postings only)")
        fig = px.line(pm[pm["skill"].isin(picks)], x="month", y="share", color="skill", markers=True,
                      labels={"share": "% of postings published that month"})
        st.plotly_chart(fig)
        st.caption("Older months only include postings that are still open, which skews toward hard-to-fill roles.")

# --- Map --------------------------------------------------------------------------------
with tabs[3]:
    from echolocate.clusters import cluster_map
    c1, c2 = st.columns([1, 3])
    method = c1.radio("Clustering", ["hdbscan", "kmeans"], help="HDBSCAN finds its own number of clusters "
                      "and leaves outliers unclustered; KMeans assigns everything to 20 clusters.")
    map_mask = Filters(software_only=software_only, remote_only=remote_only, levels=levels).mask(df)
    c2.caption(f"{map_mask.sum():,} postings (map uses the role / remote / level filters). "
               "The first run takes about a minute; results are cached.")
    with st.spinner("Projecting with UMAP and clustering ..."):
        pts, clusters = cluster_map(ix, map_mask, method=method)
    pts["hover"] = pts["title"] + " · " + pts["company"]
    shown = pts[pts["cluster"] >= 0]
    fig = px.scatter(shown, x="x", y="y", color="label", hover_name="hover", height=720,
                     labels={"label": "cluster"}, render_mode="webgl")
    noise = pts[pts["cluster"] < 0]
    if len(noise):
        fig.add_scatter(x=noise["x"], y=noise["y"], mode="markers", name="(unclustered)",
                        marker=dict(color="lightgray", size=3), hovertext=noise["hover"], hoverinfo="text")
    fig.update_traces(marker=dict(size=4, opacity=0.75), selector=dict(mode="markers"))
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, legend=dict(font=dict(size=10)))
    st.plotly_chart(fig)
    st.dataframe(clusters, hide_index=True)

# --- RAG --------------------------------------------------------------------------------
with tabs[4]:
    from echolocate.rag import ask, gather, ollama_status
    ok, msg = ollama_status()
    st.caption(("✅ " if ok else "⚠️ ") + msg)
    question = st.text_input("Ask about the postings (uses the sidebar filters)",
                             "Which remote new grad roles mention LLM agents, and what stack do they use?")
    if st.button("Ask", type="primary", disabled=not ok) and question.strip():
        out = st.empty()
        acc: list[str] = []

        def on_token(t):
            acc.append(t)
            out.markdown("".join(acc).replace("$", "\\$"))

        with st.spinner("Retrieving and generating ..."):
            answer, sources = ask(searcher, question, f, stream=True, on_token=on_token)
        out.markdown(answer.replace("$", "\\$"))
        st.subheader("Sources")
        for n, s in enumerate(sources, 1):
            st.markdown(f"**[{n}]** [{md(s['title'])}]({s['url']}) · {md(s['company'])} · "
                        f"{'Remote' if s['remote'] else md(s['location'])}")
    elif not ok and question.strip():
        st.subheader("Retrieved postings (no model needed)")
        for n, s in enumerate(gather(searcher, question, f), 1):
            st.markdown(f"**[{n}]** [{md(s['title'])}]({s['url']}) · {md(s['company'])}")
