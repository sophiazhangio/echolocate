import json
import zlib

import numpy as np
import pytest

from echolocate import skills, store
from echolocate.search import Filters, Searcher, _rrf
from echolocate.text import chunk, is_software, level
from echolocate.trends import demand, over_time, rising


class FakeEmbedder:
    """Bag-of-words hashed into 64 dims: deterministic and close enough to test ranking logic."""
    name = "fake"

    def encode(self, texts, progress=False):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace(".", " ").replace(",", " ").split():
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


POSTINGS = [
    dict(id="a", company="Acme", title="Software Engineer, New Grad", location="Remote - US", remote=True,
         url="u/a", posted="2026-10-01", pay="", description="Build Next.js apps with TypeScript and React. LLM agents."),
    dict(id="b", company="Beta", title="Senior Backend Engineer", location="Irvine, CA", remote=False,
         url="u/b", posted="2026-09-01", pay="$200K", description="Go and Kafka services on AWS with Kubernetes."),
    dict(id="c", company="Gamma", title="Account Executive", location="Los Angeles", remote=False,
         url="u/c", posted="2026-10-02", pay="", description="Sell software to enterprises."),
    dict(id="d", company="Delta", title="Frontend Engineer I", location="Remote", remote=True,
         url="u/d", posted="2026-10-03", pay="", description="React and TypeScript frontend, design systems, Next.js."),
]


@pytest.fixture
def built(tmp_path):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    (dumps / "2026-10-28.jsonl").write_text("\n".join(json.dumps(p) for p in POSTINGS[:3]))
    (dumps / "2026-10-29.jsonl").write_text("\n".join(json.dumps(p) for p in POSTINGS))
    db = tmp_path / "index.sqlite"
    stats = store.build(dumps, db, FakeEmbedder(), log=lambda *_: None)
    return db, stats


def test_chunk_prefixes_title_and_covers_all_words():
    desc = " ".join(f"w{i}." if i % 30 == 29 else f"w{i}" for i in range(250))
    parts = chunk("SWE", "Acme", desc)
    assert all(p.startswith("SWE at Acme:") for p in parts)
    seen = set(" ".join(parts).split())
    assert {f"w{i}" for i in range(250) if i % 30 != 29} <= seen
    assert len(parts) >= 3


def test_level_and_family():
    assert level("Software Engineer, New Grad") == "entry"
    assert level("Software Engineer I") == "entry"
    assert level("Senior Software Engineer") == "senior+"
    assert level("Software Engineering Intern") == "intern"
    assert level("Software Engineer II") == "unspecified"
    assert is_software("Frontend Engineer")
    assert is_software("Sales Engineer, Software") and not is_software("Sales Engineer")
    assert not is_software("Account Executive")


def test_skill_patterns_are_strict():
    assert "Java" not in skills.find("JavaScript developer")
    assert "Go" not in skills.find("go to market and get things going")
    assert "Go" in skills.find("services in Go, Python")
    assert "Next.js" in skills.find("NextJS and React")
    assert "RAG / embeddings" in skills.find("vector database for retrieval-augmented generation")


def test_build_is_incremental(built, tmp_path):
    db, stats = built
    assert stats["postings"] == 4 and stats["embedded"] == 4
    again = store.build(tmp_path / "dumps", db, FakeEmbedder(), log=lambda *_: None)
    assert again["new_dumps"] == 0 and again["embedded"] == 0


def test_first_and_last_seen(built):
    db, _ = built
    con = store.connect(db)
    rows = dict((r[0], r[1:]) for r in con.execute("SELECT id, first_seen, last_seen FROM postings"))
    assert rows["d"] == ("2026-10-29", "2026-10-29")
    assert rows["b"] == ("2026-10-28", "2026-10-29")


def test_search_filters_and_ranking(built):
    db, _ = built
    s = Searcher(store.Index.load(db), FakeEmbedder())
    res = s.search("remote Next.js TypeScript", Filters(), k=5)
    assert set(res["id"]) <= {"a", "b", "d"}          # software_only drops the account executive
    assert res.iloc[0]["id"] in {"a", "d"}
    remote = s.search("engineer", Filters(remote_only=True), k=5)
    assert set(remote["id"]) <= {"a", "d"}
    entry = s.search("engineer", Filters(levels=["entry"]), k=5)
    assert set(entry["id"]) == {"a", "d"}
    oc = s.search("engineer", Filters(location="irvine"), k=5)
    assert list(oc["id"]) == ["b"]


def test_similar_to_posting_excludes_itself(built):
    db, _ = built
    s = Searcher(store.Index.load(db), FakeEmbedder())
    res = s.similar_to_posting("a", k=3)
    assert "a" not in set(res["id"])
    assert res.iloc[0]["id"] == "d"


def test_rrf_prefers_items_ranked_well_in_both():
    dense = np.array([0.9, 0.8, 0.1])
    sparse = np.array([0.0, 0.7, 0.6])
    fused = _rrf([dense, np.where(sparse > 0, sparse, -np.inf)])
    assert fused.argmax() == 1


def test_trends(built):
    db, _ = built
    ix = store.Index.load(db)
    d = demand(ix.postings[ix.postings["software"]])
    assert d.set_index("skill").loc["TypeScript", "postings"] == 2
    t = over_time(db)
    ts = t[t["skill"] == "TypeScript"].set_index("date")["share"]
    assert ts["2026-10-28"] == pytest.approx(50.0) and ts["2026-10-29"] == pytest.approx(200 / 3)
    r = rising(t)
    assert "TypeScript" in set(r["skill"])


def test_parse_query_turns_words_into_filters():
    from echolocate.search import parse_query
    q, f, notes = parse_query("remote new grad roles using Next.js", Filters())
    assert f.remote_only and f.levels == ["entry", "unspecified"]
    assert "remote" not in q.lower() and "roles" not in q.lower() and "Next.js" in q
    assert len(notes) == 2
    q2, f2, notes2 = parse_query("backend engineer Go", Filters())
    assert q2 == "backend engineer Go" and not notes2 and not f2.remote_only


def test_duplicate_roles_collapse(tmp_path):
    dumps = tmp_path / "dumps"
    dumps.mkdir()
    rows = POSTINGS + [dict(POSTINGS[3], id="d2", location="Austin"), dict(POSTINGS[3], id="d3", location="NYC")]
    (dumps / "2026-10-29.jsonl").write_text("\n".join(json.dumps(p) for p in rows))
    db = tmp_path / "x.sqlite"
    store.build(dumps, db, FakeEmbedder(), log=lambda *_: None)
    s = Searcher(store.Index.load(db), FakeEmbedder())
    res = s.search("React TypeScript frontend", Filters(), k=5)
    assert (res["title"] == "Frontend Engineer I").sum() == 1
    assert res.loc[res["title"] == "Frontend Engineer I", "more_locations"].item() == 2


def test_rag_gather_applies_question_filters(built):
    from echolocate.rag import gather
    db, _ = built
    s = Searcher(store.Index.load(db), FakeEmbedder())
    got = gather(s, "which remote roles use React?", Filters())
    assert got and all(x["remote"] for x in got)


def test_or_remote_is_not_a_filter():
    from echolocate.search import parse_query
    q, f, notes = parse_query("frontend React in Irvine or remote", Filters())
    assert not f.remote_only and not notes and "remote" in q


def test_read_resume_errors_are_readable(tmp_path):
    import pytest
    from echolocate.text import ResumeError, read_resume
    bad = tmp_path / "r.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ResumeError, match="couldn't be opened"):
        read_resume(bad)
    empty = tmp_path / "r.txt"
    empty.write_text("")
    with pytest.raises(ResumeError, match="No readable text"):
        read_resume(empty)
    latin = tmp_path / "l.txt"
    latin.write_bytes(("Python developer caf\xe9 " * 10).encode("latin-1"))
    assert "Python" in read_resume(latin)


def test_geo_parse_locations():
    from echolocate.geo import parse
    city = lambda s: [(c["city"], c["country"]) for c in parse(s)]
    assert city("NY or SF; US") == [("New York City", "United States"), ("San Francisco", "United States")]
    assert city("Portland, OR")[0] == ("Portland", "United States")
    assert city("Vancouver, BC")[0][1] == "Canada"
    assert city("Cambridge, UK")[0][1] == "United Kingdom"
    assert city("ES - Barcelona, Spain; Barcelona") == [("Barcelona", "Spain")]
    assert city("Remote - US") == [] and city("Phillipsburg Warehouse") == []


def test_cluster_map_small_selection():
    import numpy as np
    import pytest
    from echolocate.clusters import cluster_map
    ix = _index(30)  # helper below builds a fake index
    pts, clusters = cluster_map(ix, np.ones(30, bool), cache=False)
    assert len(clusters) >= 2 and len(pts) == 30
    with pytest.raises(ValueError):
        cluster_map(ix, np.arange(30) < 10, cache=False)


def _index(n):
    import numpy as np
    import pandas as pd
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 16))
    vecs = np.vstack([centers[i % 3] + 0.05 * rng.normal(size=16) for i in range(n)]).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    df = pd.DataFrame({"id": [str(i) for i in range(n)], "company": [f"c{i % 5}" for i in range(n)],
                       "title": [["Backend Engineer", "iOS Engineer", "Data Engineer"][i % 3] for i in range(n)],
                       "location": "SF", "level": "entry", "remote": False, "url": "",
                       "skills": [["Python"]] * n})
    return SimpleNamespace(postings=df, posting_vecs=vecs, latest_date="2026-10-29", model="fake")


def test_it_field(tmp_path, monkeypatch):
    import json
    from echolocate import config, skills
    from echolocate.text import in_field, is_it, level
    assert is_it("Help Desk Technician - Tier 1") and is_it("Systems Administrator") and is_it("Director of IT")
    assert not is_it("Software Engineer") and not is_it("IT Sales Account Executive")
    assert in_field("Software Engineer", "software") and not in_field("Software Engineer", "it")
    assert {"Active Directory", "Intune", "CompTIA A+"} <= set(skills.find("AD, Intune and CompTIA A+ required "
                                                                           "(active directory)", "it"))
    assert skills.find("Python and React", "software") == ["Python", "React"]
    # the field comes from job-radar's config.local.json; software behaviour is the default
    (tmp_path / "config.local.json").write_text(json.dumps({"field": "IT"}))
    monkeypatch.setattr(config, "JOB_RADAR", tmp_path)
    monkeypatch.setattr(config, "DEMO", False)
    monkeypatch.delenv("ECHO_FIELD", raising=False)
    assert config.field() == "it"
    monkeypatch.setattr(config, "DEMO", True)
    assert config.field() == "software"  # the hosted demo never changes
    monkeypatch.setattr(config, "FIELD", "software")
    assert level("Support Specialist I (Merchant Care)") == "unspecified"
    monkeypatch.setattr(config, "FIELD", "it")
    assert level("Tier 1 Service Desk Analyst") == "entry"
