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

# IT support / systems / network administration (used when the field setting is "it").
IT_SKILLS: dict[str, tuple[str, str]] = {
    "Windows Server": ("Microsoft", r"windows server"),
    "Windows 10/11": ("Microsoft", r"windows (10|11)\b|windows desktop"),
    "Active Directory": ("Microsoft", r"active directory|\bad ds\b|\badds\b|group policy|\bgpo\b"),
    "Entra ID / Azure AD": ("Microsoft", r"\bentra\b|azure ad\b|azure active directory"),
    "Microsoft 365": ("Microsoft", r"microsoft 365|\bm365\b|office 365|\bo365\b"),
    "Exchange": ("Microsoft", r"\bexchange( online| server)?\b(?! rate)"),
    "Intune": ("Microsoft", r"\bintune\b|endpoint manager|\bmem\b"),
    "SCCM / MECM": ("Microsoft", r"\bsccm\b|\bmecm\b|configuration manager"),
    "Teams / SharePoint": ("Microsoft", r"\bsharepoint\b|microsoft teams|\bonedrive\b"),
    "macOS / Jamf": ("Endpoints", r"\bjamf\b|\bmacos\b|mac os|apple devices"),
    "Linux": ("Endpoints", r"\blinux\b|\bubuntu\b|\brhel\b|red hat"),
    "Printers / peripherals": ("Endpoints", r"\bprinters?\b|peripherals?"),
    "Mobile devices / MDM": ("Endpoints", r"\bmdm\b|mobile device management|iphones?|ipads?"),
    "Imaging / deployment": ("Endpoints", r"\bimaging\b|\bautopilot\b|\bpxe\b|os deployment"),
    "VMware": ("Infrastructure", r"\bvmware\b|\bvsphere\b|\besxi\b"),
    "Hyper-V": ("Infrastructure", r"hyper-?v\b"),
    "Backup / DR": ("Infrastructure", r"\bbackups?\b|disaster recovery|\bveeam\b|\bdatto\b"),
    "Storage / SAN": ("Infrastructure", r"\bsan\b|\bnas\b|storage arrays?"),
    "DNS / DHCP": ("Networking", r"\bdns\b|\bdhcp\b"),
    "TCP/IP": ("Networking", r"tcp/ip|\bsubnet"),
    "Switches / routers": ("Networking", r"\bswitch(es|ing)\b|\brouters?\b|\bvlans?\b"),
    "Firewalls": ("Networking", r"firewalls?|\bfortinet\b|fortigate|palo alto networks|\bsonicwall\b"),
    "VPN": ("Networking", r"\bvpns?\b"),
    "Wi-Fi": ("Networking", r"wi-?fi|wireless (network|access)|\bmeraki\b"),
    "Cisco": ("Networking", r"\bcisco\b|\bccna\b"),
    "Azure": ("Cloud", r"\bazure\b"),
    "AWS": ("Cloud", r"\baws\b|amazon web services"),
    "Google Workspace": ("Cloud", r"google workspace|g suite|gsuite"),
    "Okta / SSO": ("Cloud", r"\bokta\b|single sign-on|\bsso\b"),
    "PowerShell": ("Scripting", r"powershell"),
    "Bash / shell": ("Scripting", r"\bbash\b|shell script"),
    "Python": ("Scripting", r"\bpython\b"),
    "Help desk / ticketing": ("Service desk", r"help ?desk|service ?desk|ticketing|\btickets?\b"),
    "ServiceNow": ("Service desk", r"servicenow"),
    "Jira Service Management": ("Service desk", r"jira service|\bjsm\b"),
    "ITIL": ("Service desk", r"\bitil\b"),
    "RMM / PSA": ("Service desk", r"\brmm\b|\bpsa\b|connectwise|\bkaseya\b|\bninjaone\b|\bn-able\b"),
    "Remote support": ("Service desk", r"remote (support|desktop|assistance)|\bteamviewer\b|\brdp\b"),
    "Asset management": ("Service desk", r"asset management|inventory of (hardware|devices)"),
    "Endpoint security / EDR": ("Security", r"\bedr\b|crowdstrike|sentinelone|defender for endpoint|antivirus"),
    "MFA": ("Security", r"\bmfa\b|multi-?factor|\b2fa\b"),
    "Patching": ("Security", r"\bpatch(ing|es)?\b|vulnerability management"),
    "CompTIA A+": ("Certifications", r"comptia a\+|\ba\+ cert"),
    "CompTIA Network+": ("Certifications", r"network\+"),
    "CompTIA Security+": ("Certifications", r"security\+"),
    "Microsoft certifications": ("Certifications", r"\bmd-?102\b|\bms-?102\b|\baz-?104\b|\baz-?900\b|\bmcsa\b|microsoft certified"),
}

def for_field(name: str | None = None) -> dict[str, tuple[str, str]]:
    """The skill table of a profile (default: this install's profile)."""
    from . import profiles
    return profiles.get(name).skills


def find(text: str, name: str | None = None) -> list[str]:
    """Skills of the profile mentioned in `text`."""
    from . import profiles
    t = text.lower()
    return [n for n, rx in profiles.get(name).compiled.items() if rx.search(t)]


def group(name: str, profile: str | None = None) -> str:
    from . import profiles
    table = profiles.get(profile).skills
    return (table.get(name) or SKILLS.get(name) or IT_SKILLS.get(name) or ("Other", ""))[0]
