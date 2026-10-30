"""Text helpers: chunking, seniority level, role family, resume reading."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

# MiniLM was trained on inputs up to 128 word pieces, so chunks stay around 100 words.
CHUNK_WORDS = 100
OVERLAP_WORDS = 20


def text_hash(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()[:16]


def boilerplate(rows, min_words: int = 20, min_repeats: int = 5) -> set[tuple[str, str]]:
    """(company, line) pairs that look like company boilerplate.

    A long line (20+ words) that appears in 5+ postings of the same company is almost
    always "About us", benefits, EEO or interview-process text. Short repeated lines are
    kept because they are often real requirements ("3+ years of Python").
    """
    from collections import Counter
    counts: Counter = Counter()
    for company, desc in rows:
        for line in {x.strip() for x in (desc or "").split("\n")}:
            if len(line.split()) >= min_words:
                counts[(company, line)] += 1
    return {k for k, n in counts.items() if n >= min_repeats}


def strip_boilerplate(company: str, desc: str, common: set[tuple[str, str]]) -> str:
    return "\n".join(x for x in (desc or "").split("\n") if (company, x.strip()) not in common)


def chunk(title: str, company: str, description: str,
          size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    """Split a posting into overlapping word windows.

    Every chunk starts with "<title> at <company>:" so a chunk about, say, benefits still
    knows which role it belongs to. Windows break at sentence ends when one is close.
    """
    head = f"{title} at {company}:"
    words = description.split()
    if not words:
        return [head]
    chunks, start = [], 0
    while start < len(words):
        end = min(start + size, len(words))
        if end < len(words):
            # Prefer to stop right after a sentence end in the last quarter of the window.
            for j in range(end, start + size * 3 // 4, -1):
                if re.search(r"[.!?:;]$", words[j - 1]):
                    end = j
                    break
        chunks.append(f"{head} {' '.join(words[start:end])}")
        if end >= len(words):
            break
        start = max(end - overlap, start + 1)
    return chunks


_LEVELS = [
    ("intern", r"\bintern(ship)?\b|\bco-?op\b"),
    ("senior+", r"\b(senior|sr\.?|staff|principal|lead|manager|director|head of|vp|vice president|distinguished|architect)\b"),
    ("entry", r"\b(new grad(uate)?|graduate|entry[- ]level|junior|jr\.?|associate|early career|university|apprentice)\b"
              r"|\b(engineer|developer)\s+(i|1)\b|\b(i|1)\s*$|\blevel 1\b"),
]
LEVELS = ["entry", "unspecified", "intern", "senior+"]


# How IT postings mark entry level ("Help Desk Technician - Tier 1", "Specialist I"); only used
# when the field is "it", so software results stay exactly as they were.
_IT_ENTRY = re.compile(r"\b(technician|specialist|analyst|administrator|support)\s+(i|1)\b"
                       r"|\blevel[- ]?(1|i)\b|\btier[- ]?(1|i|one)\b")


def level(title: str) -> str:
    from . import config
    t = title.lower()
    if config.FIELD == "it" and _IT_ENTRY.search(t) and not re.search(_LEVELS[1][1], t):
        return "entry"
    for name, pat in _LEVELS:
        if re.search(pat, t):
            return name
    if re.search(r"\b(ii|iii|iv|2|3)\s*$|\bmid[- ]level\b", t):
        return "unspecified"
    return "unspecified"


_SOFTWARE = re.compile(r"\b(software|swe|sde|developer|programmer|full[- ]?stack|front[- ]?end|back[- ]?end|"
                       r"web|platform|infrastructure|devops|site reliability|sre|data engineer|ml engineer|"
                       r"machine learning engineer|ai engineer|mobile|ios|android|product engineer|forward deployed)\b")
_ENGINEER = re.compile(r"\bengineer(ing)?\b")
_NOT_SOFTWARE = re.compile(r"\b(sales|solutions?|support|customer|account|recruit|mechanical|electrical|"
                           r"hardware|manufacturing|civil|process|quality assurance technician|field)\b")


def is_software(title: str) -> bool:
    """Rough "is this a software engineering role" check, used by the default filter."""
    t = title.lower()
    if _NOT_SOFTWARE.search(t):
        return bool(re.search(r"\bsoftware\b", t))
    return bool(_SOFTWARE.search(t) or _ENGINEER.search(t))


_IT = re.compile(r"\b(it|i\.t\.|help ?desk|service ?desk|desktop|deskside|end[- ]?user|endpoint|"
                 r"system(s)? (administrator|admin|engineer|analyst|specialist)|sys ?admin|"
                 r"network (administrator|admin|engineer|technician|analyst|specialist)|"
                 r"technical support|tech support|support (technician|specialist|analyst|engineer)|"
                 r"infrastructure (administrator|engineer|analyst|specialist)|"
                 r"microsoft 365|m365|o365|active directory|identity (administrator|engineer)|"
                 r"cloud (administrator|admin)|windows (administrator|engineer)|"
                 r"noc|msp|computer (technician|support)|field (technician|service technician))\b")
_NOT_IT = re.compile(r"\b(sales|account (executive|manager)|recruit\w*|marketing|mechanical|electrical|civil|"
                     r"nurse|medical|clinical|warehouse|driver|attorney|accountant)\b")


def is_it(title: str) -> bool:
    """Rough "is this an IT support / systems / network administration role" check."""
    t = title.lower()
    return bool(_IT.search(t)) and not _NOT_IT.search(t)


def in_field(title: str, field: str = "software") -> bool:
    """The role filter behind "Software roles only" / "IT roles only"."""
    return is_it(title) if field == "it" else is_software(title)


class ResumeError(ValueError):
    """A resume file that can't be read; the message is meant for the person who uploaded it."""


def read_resume(path: Path) -> str:
    """Plain text of a PDF or text resume. Raises ResumeError with a readable message."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(str(path))
            if reader.is_encrypted and not reader.decrypt(""):
                raise ResumeError("This PDF is password-protected. Export a copy without a password and try again.")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ResumeError:
            raise
        except Exception as e:
            raise ResumeError("This file couldn't be opened as a PDF. It may be damaged or not really a PDF; "
                              "try exporting it again from Word or Google Docs.") from e
    else:
        data = path.read_bytes()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1")
    if len(text.split()) < 20:
        raise ResumeError("No readable text was found in this file. If it is a scanned image or a photo, "
                          "export the resume as a text PDF (File > Save as PDF in Word or Google Docs).")
    return text
