"""Skill vocabulary used for demand counts and trends.

Each skill is a regex run on lowercase "title + description". Patterns are deliberately
strict (word boundaries, no bare "go" or "c") so counts stay honest.
"""

from __future__ import annotations

import re

SKILLS: dict[str, tuple[str, str]] = {
    # name: (group, pattern)
    "Python": ("Languages", r"\bpython\b"),
    "TypeScript": ("Languages", r"\btypescript\b"),
    "JavaScript": ("Languages", r"\bjavascript\b"),
    "Java": ("Languages", r"\bjava\b(?!script)"),
    "Go": ("Languages", r"\bgolang\b|\bgo\b(?=\s*(,|/|\)| and | or |programming|developer|services))"),
    "Rust": ("Languages", r"\brust\b"),
    "C++": ("Languages", r"\bc\+\+"),
    "C#/.NET": ("Languages", r"\bc#|\.net\b"),
    "Kotlin": ("Languages", r"\bkotlin\b"),
    "Swift": ("Languages", r"\bswift\b"),
    "Ruby": ("Languages", r"\bruby\b|\brails\b"),
    "Scala": ("Languages", r"\bscala\b"),
    "SQL": ("Data", r"\bsql\b"),
    "PostgreSQL": ("Data", r"\bpostgres(ql)?\b"),
    "MySQL": ("Data", r"\bmysql\b"),
    "MongoDB": ("Data", r"\bmongo(db)?\b"),
    "Redis": ("Data", r"\bredis\b"),
    "Kafka": ("Data", r"\bkafka\b"),
    "Spark": ("Data", r"\b(apache )?spark\b"),
    "Snowflake": ("Data", r"\bsnowflake\b"),
    "dbt": ("Data", r"\bdbt\b"),
    "React": ("Web", r"\breact(\.js|js)?\b(?! native)"),
    "Next.js": ("Web", r"\bnext\.?js\b"),
    "Node.js": ("Web", r"\bnode(\.js|js)\b|\bnode\b(?= ?(and|,|/))"),
    "Vue": ("Web", r"\bvue(\.js|js)?\b"),
    "Angular": ("Web", r"\bangular\b"),
    "GraphQL": ("Web", r"\bgraphql\b"),
    "REST APIs": ("Web", r"\brest(ful)?\b( ?apis?| services)?|\brest\b"),
    "Tailwind": ("Web", r"\btailwind"),
    "Django": ("Web", r"\bdjango\b"),
    "FastAPI/Flask": ("Web", r"\bfastapi\b|\bflask\b"),
    "React Native": ("Mobile", r"\breact native\b"),
    "iOS": ("Mobile", r"\bios\b"),
    "Android": ("Mobile", r"\bandroid\b"),
    "AWS": ("Cloud", r"\baws\b|amazon web services"),
    "GCP": ("Cloud", r"\bgcp\b|google cloud"),
    "Azure": ("Cloud", r"\bazure\b"),
    "Docker": ("Cloud", r"\bdocker\b|\bcontaineri[sz]"),
    "Kubernetes": ("Cloud", r"\bkubernetes\b|\bk8s\b"),
    "Terraform": ("Cloud", r"\bterraform\b"),
    "CI/CD": ("Cloud", r"\bci/cd\b|continuous integration|github actions"),
    "Linux": ("Cloud", r"\blinux\b"),
    "LLMs": ("AI", r"\bllms?\b|large language models?|\bgenerative ai\b|\bgenai\b"),
    "AI agents": ("AI", r"\bai agents?\b|\bagentic\b|\bagents? (framework|workflows?|systems?)\b"),
    "RAG / embeddings": ("AI", r"\brag\b|retrieval[- ]augmented|\bembeddings?\b|vector (db|database|search|store)"),
    "Prompt engineering": ("AI", r"prompt engineering|\bprompting\b"),
    "LLM evals": ("AI", r"\bevals?\b|llm evaluation|model evaluation"),
    "AI coding tools": ("AI", r"claude code|\bcursor\b|copilot|\bcodex\b|ai[- ]assisted (coding|development)"),
    "PyTorch": ("AI", r"\bpytorch\b"),
    "TensorFlow": ("AI", r"\btensorflow\b"),
    "Machine learning": ("AI", r"machine learning|\bml\b"),
    "Testing": ("Practice", r"\bunit tests?\b|\btest(-| )driven\b|\bjest\b|\bpytest\b|\bplaywright\b|\bcypress\b"),
    "Distributed systems": ("Practice", r"distributed systems?"),
    "Microservices": ("Practice", r"micro-?services"),
}

_COMPILED = {name: re.compile(pat) for name, (_, pat) in SKILLS.items()}


def find(text: str) -> list[str]:
    t = text.lower()
    return [name for name, rx in _COMPILED.items() if rx.search(t)]


def group(name: str) -> str:
    return SKILLS[name][0]
