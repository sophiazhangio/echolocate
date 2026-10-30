"""Career profiles: which titles count as "this kind of job", which skills to count, and the
example texts in the app.

job-radar keeps one folder per career path, profiles/<name>/profile.json, and its config picks
one with "profile": "<name>". echolocate reads the same file so one setting drives both tools.

Where each profile comes from:
- "software" always uses the list built into echolocate (grouped and tuned on real postings),
  so the live demo and existing results never change when job-radar's file is edited.
- any other name uses job-radar's profiles/<name>/profile.json when it exists, else the
  built-in profile of that name ("it"), else falls back to "software".

profile.json keys used here (all optional except skills):
  skills        [{"name", "pattern", "core", "group"?}]  pattern: regex source run on lowercase text
  description   one line, shown in the app
  searchTerms   ["help desk", ...]; the first one seeds the example search
  entryTitle    regex; titles that match count as entry level
  defaults.includeTitleRe / includeTitle   regexes / phrases that mark a title as this career
  defaults.excludeTitle / excludeTitleRe   phrases / regexes that rule a title out (level words are ignored:
                                           the app has its own level filter)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

from . import config
from .skills import IT_SKILLS, SKILLS
from .text import _IT_ENTRY, is_it, is_software

# Level words in job-radar's excludeTitle lists; echolocate filters levels separately.
_LEVEL_WORDS = {"senior", "sr.", "sr", "staff", "principal", "lead", "manager", "director", "head of", "vp",
                "vice president", "architect", "intern", "internship", "co-op", "coop", "mid level", "mid-level",
                "junior", "jr", "jr.", "new grad", "entry level", "ii", "iii", "iv", "2", "3"}


@dataclass
class Profile:
    name: str
    source: str                                   # "built-in" or the profile.json path
    skills: dict[str, tuple[str, str]]            # name -> (group, pattern)
    role: Callable[[str], bool]                   # is this title in the career?
    entry: re.Pattern | None = None               # extra entry-level title pattern
    title_noise: str | None = None                # extra generic title words for map labels
    ui: dict = field(default_factory=dict)
    compiled: dict[str, re.Pattern] = field(default_factory=dict)
    version: str = ""                             # changes when the profile file changes

    def __post_init__(self):
        self.compiled = {}
        for name, (_, pat) in self.skills.items():
            rx = _compile(pat)
            if rx is not None:
                self.compiled[name] = rx


_UI_SOFTWARE = dict(
    label="Software",
    role_filter="Software roles only",
    query="remote new grad roles using Next.js and LLM agents",
    hybrid_help="Off = embeddings only. Hybrid also rewards exact names like 'Next.js'.",
    question="Which remote new grad roles mention LLM agents, and what stack do they use?",
    groups=["Languages", "Web", "Cloud", "AI", "Data"],
    over_time=["TypeScript", "Python", "LLMs", "AI agents", "RAG / embeddings", "AWS"],
    generic_titles=set())

_UI_IT = dict(
    label="IT",
    role_filter="IT roles only",
    query="entry level help desk or IT support role with Microsoft 365 and Active Directory",
    hybrid_help="Off = embeddings only. Hybrid also rewards exact names like 'Intune' or 'Hyper-V'.",
    question="Which entry level IT support roles are hiring, and which tools and certifications do they ask for?",
    groups=["Microsoft", "Networking", "Infrastructure", "Service desk", "Certifications"],
    over_time=["Microsoft 365", "Active Directory", "Intune", "Azure", "PowerShell", "CompTIA A+"],
    generic_titles={"It Specialist", "It Support Specialist", "It Technician", "It Support", "It Analyst",
                    "It Administrator", "Help Desk Technician"})

_IT_NOISE = r"\b(it|specialist|technician|administrator|admin|analyst|support|tier|level|services?)\b"


def _builtin(name: str) -> Profile | None:
    if name == "software":
        return Profile("software", "built-in", SKILLS, is_software, ui=_UI_SOFTWARE)
    if name == "it":
        return Profile("it", "built-in", IT_SKILLS, is_it, entry=_IT_ENTRY, title_noise=_IT_NOISE, ui=_UI_IT)
    return None


def _compile(js: str) -> re.Pattern | None:
    """A JavaScript regex source as a Python pattern (named groups differ); None if it won't compile."""
    try:
        return re.compile(re.sub(r"\(\?<(?=[A-Za-z])", "(?P<", js))
    except re.error as e:
        print(f"echolocate: skipping skill pattern {js!r}: {e}", file=sys.stderr)
        return None


def _any(patterns: list[str]) -> re.Pattern | None:
    good = [rx.pattern for rx in map(_compile, patterns) if rx is not None]
    return re.compile("|".join(f"(?:{p})" for p in good)) if good else None


def _is_level_pattern(pat: str) -> bool:
    """True for patterns that only name levels, like \\b(senior|sr\\.?|staff)\\b."""
    words = re.sub(r"\\b|\(\?:|[()^$?]", "", pat).replace("\\.", ".").replace("\\s", " ").split("|")
    return all(w.strip().lower() in _LEVEL_WORDS for w in words if w.strip())


def _first6(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))[:6]


def _from_json(name: str, path: Path) -> Profile:
    data = json.loads(path.read_text())
    base = _builtin(name)
    d = data.get("defaults") or {}
    skills = {}
    for s in data.get("skills") or []:
        if s.get("name") and s.get("pattern") and _compile(s["pattern"]) is not None:
            # tags are stored comma-separated, so a comma can't be part of a name
            skills[re.sub(r"\s*,\s*", " / ", s["name"].strip())] = (s.get("group") or ("Core skills" if s.get("core") else "Other skills"), s["pattern"])
    if not skills and base:
        skills = base.skills

    include = _any(list(d.get("includeTitleRe") or []) + [re.escape(p.lower()) for p in d.get("includeTitle") or []])
    exclude = _any([re.escape(p.strip().lower()) for p in d.get("excludeTitle") or []
                    if p.strip().lower() not in _LEVEL_WORDS]
                   + [p for p in d.get("excludeTitleRe") or [] if not _is_level_pattern(p)])
    if include is not None:
        def role(title: str, _inc=include, _exc=exclude) -> bool:
            t = title.lower()
            return bool(_inc.search(t)) and not (_exc and _exc.search(t))
    elif base:
        role = base.role
    else:
        def role(title: str) -> bool:  # no title rules: every posting counts
            return True

    entry = _compile(data["entryTitle"]) if data.get("entryTitle") else (base.entry if base else None)
    terms = data.get("searchTerms") or []
    label = (base.ui.get("label") if base else None) or data.get("label") or name.replace("-", " ").title()
    core = [n for n, (g, _) in skills.items() if g == "Core skills"]
    ui = dict(base.ui) if base else {}
    groups = sorted({g for g, _ in skills.values()})
    ui.update(
        label=label,
        description=data.get("description", ""),
        role_filter=ui.get("role_filter") or f"{label} roles only",
        query=ui.get("query") or (f"entry level {terms[0]} role" if terms else f"entry level {label} role"),
        hybrid_help=ui.get("hybrid_help") or "Off = embeddings only. Hybrid also rewards exact tool and product names.",
        question=ui.get("question") or f"Which entry level {label} roles are hiring, and what skills do they ask for?",
        groups=[g for g in ui.get("groups", []) if g in groups] or groups,
        over_time=_first6([s for s in ui.get("over_time", []) if s in skills] + core + list(skills)),
        generic_titles=ui.get("generic_titles", set()))
    version = hashlib.sha1(path.read_bytes()).hexdigest()[:8]
    return Profile(name, str(path), skills, role, entry=entry,
                   title_noise=base.title_noise if base else None, ui=ui, version=version)


@lru_cache(maxsize=16)
def _load(name: str, job_radar: str, data_root: str) -> Profile:
    if name != "software":
        # job-radar's file; on the hosted site, the copy published next to the data
        for path in (Path(job_radar) / "profiles" / name / "profile.json", config.data_dir(name) / "profile.json"):
            if not path.exists():
                continue
            try:
                return _from_json(name, path)
            except (ValueError, OSError) as e:
                print(f"echolocate: can't read {path} ({e}); using the built-in profile", file=sys.stderr)
    p = _builtin(name)
    if p is None:
        print(f"echolocate: no profile named {name!r}; using software", file=sys.stderr)
        p = _builtin("software")
    return p


def get(name: str | None = None) -> Profile:
    """The profile by name, or this install's active one (config.PROFILE)."""
    return _load(name or config.PROFILE, str(config.JOB_RADAR), str(config.DATA_ROOT))


def available() -> list[dict]:
    """Career profiles to offer in the app: built-ins plus every job-radar profiles/<name>/profile.json."""
    out = {n: {"name": n, "label": _builtin(n).ui["label"], "description": "", "draft": False} for n in ("software", "it")}
    root = config.JOB_RADAR / "profiles"
    for f in sorted(root.glob("*/profile.json")) if root.exists() else []:
        try:
            data = json.loads(f.read_text())
        except (ValueError, OSError):
            continue
        name = f.parent.name
        entry = out.get(name, {"name": name, "label": data.get("label") or name.replace("-", " ").title()})
        entry.update(description=data.get("description", ""), draft=bool(data.get("draft")))
        out[name] = entry
    return list(out.values())
