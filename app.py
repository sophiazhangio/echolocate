"""Streamlit UI: streamlit run app.py"""

from __future__ import annotations

import re
import tempfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from echolocate import config, profiles, skills
from echolocate.search import Filters, Searcher
from echolocate.store import Index, SentenceEmbedder
from echolocate.text import LEVELS, ResumeError, read_resume

st.set_page_config(page_title="echolocate", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading index and embedding model ...")
def load(db_mtime: float):
    ix = Index.load()
    s = Searcher(ix, SentenceEmbedder(ix.model))
    s.embedder.encode(["warm up"])
    s._sparse()
    return s


def download_demo_data():
    """The hosted demo has no job-radar dumps; fetch the prebuilt index once."""
    import requests
    config.DATA.mkdir(parents=True, exist_ok=True)
    bar = st.progress(0.0, "Downloading the demo index (about 250 MB, first start only) ...")
    for i, name in enumerate(config.DEMO_FILES):
        dest = config.DATA / name
        if dest.exists():
            continue
        part = dest.with_suffix(dest.suffix + ".part")
        with requests.get(f"{config.DEMO_ASSETS}/{name}", stream=True, timeout=60) as r:
            r.raise_for_status()
            total, done = int(r.headers.get("content-length", 0)), 0
            with open(part, "wb") as fh:
                for block in r.iter_content(1 << 20):
                    fh.write(block)
                    done += len(block)
                    if total:
                        bar.progress((i + done / total) / len(config.DEMO_FILES),
                                     f"Downloading {name} ({done >> 20} / {total >> 20} MB) ...")
        part.rename(dest)
    bar.empty()


if config.DEMO and not config.DB.exists():
    download_demo_data()
if not config.DB.exists():
    st.error("No index yet. Run `python -m echolocate index` first.")
    st.stop()
searcher = load(config.DB.stat().st_mtime)
ix = searcher.ix
df = ix.postings

# Texts and defaults that depend on the career profile (job-radar's "profile" setting).
PROFILE = profiles.get()
UI = PROFILE.ui
SKILL_TABLE = skills.for_field()

POSTED_WINDOWS = {"Any time": 0, "Past week": 7, "Past month": 30, "Past 3 months": 90}

# --- sidebar filters ----------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    software_only = st.checkbox(UI["role_filter"], True)
    remote_only = st.checkbox("Remote only", False)
    levels = st.multiselect("Level (from title)", LEVELS, default=["entry", "unspecified"],
                            help="'unspecified' = the title names no level; many entry roles look like that.")
    location = st.text_input("Location contains", "", placeholder="los angeles, irvine, remote")
    company = st.text_input("Company contains", "")
    snap = date.fromisoformat(ix.latest_date)
    posted_range = st.selectbox("Posted", list(POSTED_WINDOWS) + ["Pick a date…"],
                                help=f"Counted back from the snapshot date ({snap:%b %-d, %Y}).")
    if posted_range == "Pick a date…":
        first = pd.to_datetime(df["posted"], errors="coerce").min()
        lo = first.date() if pd.notna(first) else snap - timedelta(days=365)
        picked = st.date_input("Posted on or after", snap - timedelta(days=30), min_value=lo, max_value=snap,
                               format="MM/DD/YYYY")
        since = picked.isoformat() if picked else ""
    else:
        days = POSTED_WINDOWS[posted_range]
        since = (snap - timedelta(days=days)).isoformat() if days else ""
    include = st.text_input("Must mention (comma separated)", "", placeholder="python, react")
    k = st.slider("Results", 5, 50, 15)
    f = Filters(software_only=software_only, remote_only=remote_only, levels=levels, location=location,
                company=company, posted_after=since,
                must_include=[x.strip() for x in include.split(",") if x.strip()])
    n_match = int(f.mask(df).sum())
    st.caption(f"{n_match:,} of {len(df):,} postings match · snapshot {ix.latest_date} · {ix.model.split('/')[-1]}")


def md(text) -> str:
    """Escape text for st.markdown: "$" would start LaTeX, brackets/asterisks would format."""
    return re.sub(r"([\\`*_\[\]$<>#|])", r"\\\1", str(text or ""))


_GENERIC_TITLES = {"Software Engineer", "Engineer", "Software Developer", "Member Of Technical Staff",
                   "Full Stack Engineer", "Software Engineer, New Grad"}


def _norm_title(t: str) -> str:
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", t)
    t = re.split(r"\s[-–—|:]\s|,", t)[0]
    t = re.sub(r"\b(senior|sr\.?|staff|principal|lead|ii|iii|iv|i|\d+|phd|new grad(uate)?|intern(ship)?)\b",
               " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip().title()
    return re.sub(r"\bAi\b", "AI", re.sub(r"\bMl\b", "ML", re.sub(r"\bIos\b", "iOS", t)))


def area_names(pts: pd.DataFrame) -> dict[str, str]:
    """A readable name per map area: its most common specific job title, else its keywords."""
    out = {}
    for lab, g in pts[pts["cluster"] >= 0].groupby("label"):
        vc = g["title"].map(_norm_title).value_counts()
        specific = [t for t, n in vc.items() if t and t not in _GENERIC_TITLES | UI["generic_titles"] and len(t.split()) > 1 and n >= 3]
        out[lab] = specific[0] if specific else " / ".join(lab.split(" / ")[:2]).title()
    return out


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
           f"{df['company'].nunique():,} companies"
           + (" (Greenhouse, Ashby, Lever). " if PROFILE.name == "software" else "' public job boards. ")
           + (f"Live demo on a fixed snapshot from {ix.latest_date}; "
              "[source and setup on GitHub](https://github.com/sophiazhangio/echolocate)."
              if config.DEMO else "Everything runs locally."))
if PROFILE.name != "software":
    st.caption(f"Career profile: **{md(UI['label'])}**" + (f" · {md(UI['description'])}" if UI.get("description") else "")
               + f" · from `{'job-radar profiles/' + PROFILE.name if PROFILE.source != 'built-in' else 'built-in list'}`")

tabs = st.tabs(["Search", "Match my resume", "Where the jobs are", "Skill trends", "Role map", "Ask (RAG)"])

# --- Search ---------------------------------------------------------------------------
with tabs[0]:
    q = st.text_input("Describe the job you want", UI["query"])
    hybrid = st.toggle("Hybrid (semantic + keyword)", True,
                       help=UI["hybrid_help"])
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
    label = ("Upload a resume (PDF or text). It is read once to compute the match and is not saved."
             if config.DEMO else "Resume (PDF or text). Defaults to the one in job-radar's config.")
    up = st.file_uploader(label, type=["pdf", "txt", "md"])
    text = ""
    try:
        if up is not None:
            with tempfile.TemporaryDirectory() as d:
                tmp = Path(d) / f"resume{Path(up.name).suffix.lower()}"
                tmp.write_bytes(up.getvalue())
                text = read_resume(tmp)
            st.caption(f"Using uploaded {up.name}")
        elif default and default.exists():
            text = read_resume(default)
            st.caption("Using the resume set in job-radar's config")
        elif config.DEMO:
            st.info("Upload a resume to see the postings closest to it and which in-demand skills it shows.")
        else:
            st.warning("No resume found. Upload one, or set ECHO_RESUME.")
    except ResumeError as e:
        st.error(str(e))
    except Exception:
        st.error("Something went wrong while reading this file. Try a PDF exported from Word or Google Docs, "
                 "or a .txt copy of your resume.")
    if text:
        st.session_state["tab"] = "resume"
        try:
            res = searcher.similar_to_text(text, f, k)
        except Exception:
            st.error("The resume was read, but matching it failed. Reload the page and try again.")
            text = ""
    if text:
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

# --- Where ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Placing postings on the map ...")
def all_cities(_date: str) -> pd.DataFrame:
    from echolocate.geo import city_table
    return city_table(df)


REGIONS = {"World": "world", "North America": "north america", "USA": "usa", "Europe": "europe", "Asia": "asia"}

with tabs[2]:
    try:
        sub = df[f.mask(df)]
        ct = all_cities(ix.latest_date)
        here = ct[ct["row"].isin(sub.index)]
        st.markdown("Each circle is a city. Bigger circles have more open postings that match the sidebar "
                    "filters; a posting listed in several cities counts in each. Hover a circle for its top "
                    "companies, or pick a city below to list its jobs.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Postings with a city", f"{here['row'].nunique():,}")
        m2.metric("Remote postings", f"{int(sub['remote'].sum()):,}")
        m3.metric("Cities", f"{here.groupby(['city', 'country']).ngroups:,}")
        region = st.radio("Region", list(REGIONS), horizontal=True, label_visibility="collapsed")
        if here.empty:
            st.info("No postings with a city match these filters.")
        else:
            agg = (here.groupby(["city", "country", "lat", "lon"])["row"]
                   .agg(postings="nunique", companies=lambda r: ", ".join(df.loc[r, "company"].value_counts()
                                                                           .head(4).index))
                   .reset_index().sort_values("postings", ascending=False))
            agg["place"] = agg["city"] + ", " + agg["country"]
            fig = px.scatter_geo(agg, lat="lat", lon="lon", size="postings", hover_name="place",
                                 hover_data={"postings": True, "companies": True, "lat": False, "lon": False},
                                 size_max=45, scope=REGIONS[region],
                                 **({} if region == "USA" else {"projection": "natural earth"}))
            fig.update_traces(marker=dict(color="#ff4b4b", opacity=0.65, line=dict(width=0.5, color="white")))
            # name the biggest cities, skipping any that sit right next to a bigger named one
            near = {"World": 6.0, "USA": 1.5}.get(region, 3.0)
            named = []
            for r in agg.itertuples():
                if len(named) == 15:
                    break
                if all(abs(r.lat - n.lat) > near or abs(r.lon - n.lon) > near for n in named):
                    named.append(r)
            top = pd.DataFrame(named)
            fig.add_scattergeo(lat=top["lat"], lon=top["lon"], text=top["city"], mode="text",
                               textposition="top center", textfont=dict(size=11), hoverinfo="skip",
                               showlegend=False)
            fig.update_geos(showcountries=True, countrycolor="rgba(140,140,140,0.7)", showcoastlines=True,
                            coastlinecolor="rgba(140,140,140,0.7)", showland=True,
                            landcolor="rgba(140,140,140,0.15)", showocean=False, showlakes=False,
                            showsubunits=True, subunitcolor="rgba(140,140,140,0.35)", bgcolor="rgba(0,0,0,0)",
                            showframe=False)
            fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig)
            choices = [f"{r.place} ({r.postings})" for r in agg.itertuples()]
            pick = st.selectbox("Jobs in a city", choices)
            r = agg.iloc[choices.index(pick)]
            rows = here[(here["city"] == r["city"]) & (here["country"] == r["country"])]["row"].unique()
            jobs = df.loc[rows, ["title", "company", "level", "posted", "pay", "url"]].sort_values("posted",
                                                                                                 ascending=False)
            st.dataframe(jobs, hide_index=True, column_config={
                "url": st.column_config.LinkColumn("link", display_text="open")})
    except Exception:
        st.error("The location map couldn't be drawn for these filters. Try widening them in the sidebar.")

# --- Trends -----------------------------------------------------------------------------
with tabs[3]:
    from echolocate.trends import by_posted_month, demand, over_time, rising
    sub = df[f.mask(df)]
    d = demand(sub)
    st.subheader(f"Skill demand in the {ix.latest_date} snapshot ({len(sub):,} postings)")
    grp = st.multiselect("Groups", sorted({g for g, _ in SKILL_TABLE.values()}),
                         default=UI["groups"])
    d = d[d["group"].isin(grp)].head(30)
    fig = px.bar(d.sort_values("share"), x="share", y="skill", color="group", orientation="h",
                 labels={"share": "% of postings that mention it", "skill": ""}, height=720)
    fig.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig)

    over = over_time(config.DB, software_only=software_only, levels=levels or None)
    picks = st.multiselect("Skills to plot over time", list(SKILL_TABLE),
                           default=UI["over_time"])
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

# --- Role map ---------------------------------------------------------------------------
def role_map_tab():
    from echolocate.clusters import MIN_POSTINGS, cluster_map
    c1, c2 = st.columns([1, 3])
    method = c1.radio("Clustering", ["hdbscan", "kmeans"], help="HDBSCAN finds its own number of clusters "
                      "and leaves outliers unclustered; KMeans assigns everything to 20 clusters.")
    map_mask = Filters(software_only=software_only, remote_only=remote_only, levels=levels).mask(df)
    c2.caption(f"{map_mask.sum():,} postings (map uses the role / remote / level filters). "
               "The first run takes about a minute; results are cached.")
    if map_mask.sum() < MIN_POSTINGS:
        st.info(f"Only {map_mask.sum()} postings match the role / remote / level filters; the role map needs "
                f"at least {MIN_POSTINGS}. Add a level or turn off 'Remote only' in the sidebar.")
        return
    with st.spinner("Projecting with UMAP and clustering ..."):
        pts, clusters = cluster_map(ix, map_mask, method=method)
    st.markdown("Each dot is one posting. Dots close together have similar job descriptions, so each "
                "colored area is a kind of role; its name is written in the middle of it. The axes have no "
                "meaning. Hover a dot to see the job, or pick an area below to highlight it.")
    pts["hover"] = pts["title"] + " · " + pts["company"]
    shown = pts[pts["cluster"] >= 0]
    order = clusters["label"].tolist()  # biggest area first
    names = area_names(pts)
    choices = [f"{names.get(r.label, r.label)} · {r.label} ({r.postings})" for r in clusters.itertuples()]
    focus = st.selectbox("Highlight an area", ["(all)"] + choices)
    focus_label = None if focus == "(all)" else order[choices.index(focus)]
    palette = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24
    colors = {lab: (palette[i % len(palette)] if focus_label in (None, lab) else "rgba(150,150,150,0.25)")
              for i, lab in enumerate(order)}
    fig = px.scatter(shown, x="x", y="y", color="label", hover_name="hover", height=720,
                     category_orders={"label": order}, color_discrete_map=colors,
                     hover_data={"x": False, "y": False, "label": True},
                     labels={"label": "area"}, render_mode="webgl")
    noise = pts[pts["cluster"] < 0]
    if len(noise) and focus_label is None:
        fig.add_scatter(x=noise["x"], y=noise["y"], mode="markers", name="(unclustered)",
                        marker=dict(color="rgba(150,150,150,0.35)", size=3), hovertext=noise["hover"],
                        hoverinfo="text")
    fig.update_traces(marker=dict(size=5, opacity=0.8), selector=dict(mode="markers"))
    # write each area's name at its center so the map reads without the legend
    for lab, g in shown.groupby("label"):
        if focus_label not in (None, lab):
            continue
        keys = " · ".join(lab.split(" / ")[:2])
        fig.add_annotation(x=g["x"].median(), y=g["y"].median(), showarrow=False,
                           text=f"<b>{names[lab]}</b>", hovertext=keys,
                           font=dict(size=11 if focus_label is None else 15, color="white"),
                           bgcolor="rgba(20,20,20,0.7)", borderpad=3)
    # zoom to where the dots are; a few far-off outliers would otherwise squeeze the map
    def span(v):
        lo, hi = v.quantile(0.01), v.quantile(0.99)
        pad = (hi - lo) * 0.08
        return [lo - pad, hi + pad]
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, showlegend=False,
                      xaxis_range=span(shown["x"]), yaxis_range=span(shown["y"]),
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)
    st.markdown("**What is in each area**")
    table = clusters.assign(area=clusters["label"].map(names)).rename(columns={"label": "keywords"})
    st.dataframe(table[["area", "keywords", "postings", "top companies", "top skills"]], hide_index=True)


with tabs[4]:
    st.caption("A map of *kinds of roles*, not of places: postings with similar descriptions sit close "
               "together. For cities, see **Where the jobs are**.")
    try:
        role_map_tab()
    except Exception:
        st.error("The role map couldn't be drawn for these filters. Try widening them in the sidebar.")

# --- RAG --------------------------------------------------------------------------------
with tabs[5]:
    from echolocate.rag import ask, gather, ollama_status
    ok, msg = (False, "") if config.DEMO else ollama_status()
    if config.DEMO:
        st.info("This free demo server has no GPU, so it shows the retrieval half of RAG: the passages a "
                "local Llama 3.2 model would answer from, with citations. Generated answers need Ollama "
                "on your own machine ([setup](https://github.com/sophiazhangio/echolocate#setup)); "
                "the screenshot below is a real local answer.")
        st.image("docs/rag.png", caption="Local run: Llama 3.2 3B via Ollama")
    else:
        st.caption(("✅ " if ok else "⚠️ ") + msg)
    question = st.text_input("Ask about the postings (uses the sidebar filters)",
                             UI["question"])
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
        st.subheader("Retrieved sources (what the model would read)")
        for n, s in enumerate(gather(searcher, question, f), 1):
            with st.container(border=True):
                st.markdown(f"**[{n}]** [{md(s['title'])}]({s['url']}) · {md(s['company'])} · "
                            f"{'Remote' if s['remote'] else md(s['location'])}")
                st.caption(md(s["text"][:500] + (" ..." if len(s["text"]) > 500 else "")))
