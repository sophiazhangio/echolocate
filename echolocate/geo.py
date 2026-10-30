"""Turn free-text job locations into cities with coordinates, offline.

Job boards write locations every which way ("San Francisco, CA | New York City, NY",
"NY or SF; US", "ES - Barcelona, Spain"). We split the text into pieces, look each piece up in
GeoNames' list of cities over 15k people (bundled by `geonamescache`, no network), and use a
following state or country ("Portland, OR") to pick the right one when a name is shared.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

# Shorthand job boards use that GeoNames doesn't list as a name.
ALIASES = {
    "nyc": "new york", "ny": "new york", "new york city": "new york", "nyc metro": "new york", "manhattan": "new york",
    "brooklyn": "new york", "sf": "san francisco", "sfo": "san francisco", "bay area": "san francisco",
    "sf bay area": "san francisco", "san francisco bay area": "san francisco", "dc": "washington",
    "washington dc": "washington", "washington d.c.": "washington", "d.c.": "washington",
    "bangalore": "bengaluru", "gurgaon": "gurugram", "la": "los angeles", "tel-aviv": "tel aviv",
    "tel aviv-yafo": "tel aviv", "montreal": "montréal", "zurich": "zürich", "munich": "munich",
    "sao paulo": "são paulo", "bogota": "bogotá", "mexico city": "mexico city", "cdmx": "mexico city",
    "seoul": "seoul", "hong kong": "hong kong", "silicon valley": "san jose",
}
# Words around a city that aren't part of its name.
_NOISE = re.compile(r"\b(remote|hybrid|on-?site|in[- ]office|office|offices|hq|headquarters|locations?|"
                    r"greater|metro(politan)?|area|based|only|preferred|any|downtown|campus|city of)\b", re.I)
_ZIP = re.compile(r"\b\d{5}(?:-\d{4})?\b")
# Countries we weight up when a bare name is shared (most postings here are US tech companies).
_HOME = {"US": 2.5}
_COUNTRY_ALIASES = {"usa": "US", "us": "US", "u.s.": "US", "u.s.a.": "US", "united states of america": "US",
                    "uk": "GB", "england": "GB", "scotland": "GB", "great britain": "GB", "korea": "KR",
                    "south korea": "KR", "uae": "AE", "the netherlands": "NL", "holland": "NL"}
# Canadian provinces, which GeoNames stores as numeric codes.
_CA_PROVINCES = {"bc": "02", "british columbia": "02", "on": "08", "ontario": "08", "qc": "10",
                 "quebec": "10", "québec": "10", "ab": "01", "alberta": "01", "mb": "03", "ns": "07"}
# Names that are a city and also a place we'd otherwise skip as a state or country.
_CITY_STATES = {"new york", "singapore", "washington", "hong kong", "luxembourg", "monaco"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return re.sub(r"\s+", " ", "".join(c for c in s if not unicodedata.combining(c))).strip(" .-")


@lru_cache(maxsize=1)
def _tables():
    import geonamescache
    gc = geonamescache.GeonamesCache(min_city_population=15000)
    by_name: dict[str, list[dict]] = {}
    keep = ("geonameid", "name", "countrycode", "admin1code", "population", "latitude", "longitude")
    for full in gc.get_cities().values():
        c = {k: full[k] for k in keep}  # the full records (alternate names etc.) take ~300 MB
        keys = {_fold(c["name"])}
        if c["population"] >= 1_000_000:  # well-known alternate spellings of big cities only
            keys |= {_fold(a) for a in full["alternatenames"] if a.isascii() and len(a) > 3}
        for k in keys:
            by_name.setdefault(k, []).append(c)
    for alias, target in ALIASES.items():
        if _fold(target) in by_name:
            by_name.setdefault(_fold(alias), by_name[_fold(target)])
    countries = {}
    for iso, c in gc.get_countries().items():
        countries[_fold(c["name"])] = iso
        countries[iso.lower()] = iso
        countries[c["iso3"].lower()] = iso
    countries.update(_COUNTRY_ALIASES)
    states = {}
    for code, s in gc.get_us_states().items():
        states[code.lower()] = code
        states[_fold(s["name"])] = code
    names = {iso: c["name"] for iso, c in gc.get_countries().items()}
    return by_name, countries, states, names


def _qualifies(city: dict, token: str) -> bool:
    _, countries, states, _ = _tables()
    t = _fold(token)
    if t in states and city["countrycode"] == "US":
        return city["admin1code"] == states[t]
    if t in _CA_PROVINCES and city["countrycode"] == "CA":
        return city["admin1code"] == _CA_PROVINCES[t]
    if t in countries:
        return city["countrycode"] == countries[t]
    return False


def _pick(cands: list[dict], qualifier: str | None, token: str = "") -> tuple[dict, bool]:
    # a city actually named this beats a big city that merely lists it as an alternate name
    exact = [] if token in ALIASES else [c for c in cands if _fold(c["name"]) == token]
    cands = exact or cands
    if qualifier:
        q = [c for c in cands if _qualifies(c, qualifier)]
        if q:
            return max(q, key=lambda c: c["population"]), True
    return max(cands, key=lambda c: c["population"] * _HOME.get(c["countrycode"], 1)), False


def parse(location: str | None) -> list[dict]:
    """Cities mentioned in a location string, each as {city, country, lat, lon}, without repeats."""
    if not location:
        return []
    by_name, countries, states, names = _tables()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", location)
    out, seen = [], set()
    for part in re.split(r"\s*(?:;|\||•|/|\n|\s-\s|–|—|\bor\b|\band\b|&)\s*", text, flags=re.I):
        tokens = [_ZIP.sub(" ", t).strip() for t in part.split(",")]
        # Workday style "US-CA-Irvine" / "United States-California-Torrance": read it as "Irvine, CA, US"
        dashed = part.strip().split("-")
        strict = 2 <= len(dashed) <= 4 and "," not in part and _fold(dashed[0]) in countries | states.keys()
        if strict:  # the city must sit in the country / state it is written with ("US-PERM" is not Perm)
            tokens = [d.strip() for d in reversed(dashed)]
        i = 0
        while i < len(tokens):
            t = _fold(_NOISE.sub(" ", tokens[i]))
            skip = (t in countries or t in states) and t not in _CITY_STATES and t not in ALIASES
            cands = None if skip or len(t) < 2 else by_name.get(t)
            if not cands and not skip:
                # "Irvine Spectrum Office", "Seattle HQ": try the leading words
                words = t.split()
                for n in range(len(words) - 1, 0, -1):
                    head = " ".join(words[:n])
                    # official names only: "Sophia Antipolis" must not become Sofia
                    if (cands := [c for c in by_name.get(head, []) if _fold(c["name"]) == head]):
                        t = head
                        break
            if not cands:
                i += 1
                continue
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            city, used = _pick(cands, nxt, t)
            i += 2 if used else 1
            # an unqualified small town is more likely a false hit than a real office
            if not used and (strict or (city["population"] < 50_000 and t not in ALIASES)):
                continue
            if city["geonameid"] not in seen:
                seen.add(city["geonameid"])
                out.append({"city": city["name"], "country": names.get(city["countrycode"], city["countrycode"]),
                            "lat": city["latitude"], "lon": city["longitude"]})
    return out


def city_table(postings: pd.DataFrame) -> pd.DataFrame:
    """One row per (posting, city) for the postings given; postings with no city are left out."""
    cache = {loc: parse(loc) for loc in postings["location"].dropna().unique()}
    rows = []
    for idx, loc in postings["location"].items():
        for c in cache.get(loc, []):
            rows.append({"row": idx, **c})
    return pd.DataFrame(rows, columns=["row", "city", "country", "lat", "lon"])
