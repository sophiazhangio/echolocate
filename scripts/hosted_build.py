"""Daily build for the hosted site (run by .github/workflows/daily.yml).

For each career profile: start from yesterday's published index, add today's job-radar dump
(only new or changed postings get embedded), precompute the default role maps, and publish
everything to this repo's `data` GitHub Release with a manifest.json the site reads.

    JOB_RADAR_DIR=../job-radar ECHO_DATA=work python scripts/hosted_build.py software it [--upload]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echolocate import config, profiles  # noqa: E402
from echolocate.clusters import MIN_POSTINGS, cluster_map  # noqa: E402
from echolocate.search import Filters  # noqa: E402
from echolocate.store import Index, SentenceEmbedder, build  # noqa: E402

REPO = "sophiazhangio/echolocate"
TAG = "data"
RELEASE = f"https://github.com/{REPO}/releases/download/{TAG}"
LEGACY = f"https://github.com/{REPO}/releases/download/demo-2026-10-29/index.sqlite"  # first seed for software
DEFAULT_LEVELS = ["entry", "unspecified"]  # the app's default level filter


def asset(profile: str, name: str) -> str:
    """Release assets are flat, so files are named <profile>__<file>."""
    return f"{profile}__{name}"


def download(url: str, dest: Path) -> bool:
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for block in r.iter_content(1 << 20):
            fh.write(block)
    return True


def seed(profile: str) -> None:
    """Yesterday's index, so today's run only embeds what is new."""
    if config.DB.exists():
        return
    if download(f"{RELEASE}/{asset(profile, 'index.sqlite')}", config.DB):
        print(f"[{profile}] continuing from the published index")
    elif profile == "software" and download(LEGACY, config.DB):
        print(f"[{profile}] starting from the 2026-10-29 demo index")
    else:
        print(f"[{profile}] starting a new index")


def scrub(db: Path) -> None:
    """Drop runner paths from the dumps table and compact the file."""
    con = sqlite3.connect(db)
    for date, f in con.execute("SELECT date, file FROM dumps").fetchall():
        con.execute("UPDATE dumps SET file = ? WHERE date = ?", (Path(f).name, date))
    con.commit()
    con.execute("VACUUM")
    con.close()


def build_profile(profile: str, embedder: SentenceEmbedder) -> dict | None:
    config.use_profile(profile)
    p = profiles.get(profile)
    seed(profile)
    try:
        stats = build(embedder=embedder)
    except SystemExit as e:  # no dump today (job-radar failed for this profile)
        print(f"[{profile}] skipped: {e}")
        return None
    print(f"[{profile}] {stats}")
    scrub(config.DB)

    ix = Index.load(config.DB)
    files = ["index.sqlite"]
    mask = Filters(software_only=True, levels=DEFAULT_LEVELS).mask(ix.postings)
    if mask.sum() >= MIN_POSTINGS:
        for method in ("hdbscan", "kmeans"):
            before = set(config.DATA.glob("map-*.pkl"))
            cluster_map(ix, mask, method=method)
            files += [f.name for f in set(config.DATA.glob("map-*.pkl")) - before]
    # the profile file travels with the data, so the site tags and groups skills the same way
    if p.source != "built-in":
        shutil.copy(p.source, config.DATA / "profile.json")
        files.append("profile.json")
    return {"label": p.ui["label"], "description": p.ui.get("description", ""), "date": ix.latest_date,
            "postings": int(len(ix.postings)), "companies": int(ix.postings["company"].nunique()),
            "files": files}


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def upload(manifest: dict, built: set[str]) -> None:
    try:
        gh("release", "view", TAG, "--repo", REPO)
    except subprocess.CalledProcessError:
        gh("release", "create", TAG, "--repo", REPO, "--title", "Daily data for the hosted site",
           "--notes", "Updated every day by .github/workflows/daily.yml. Not a software release.", "--latest=false")
    keep = {"manifest.json"}
    for profile, info in manifest["profiles"].items():
        if profile not in built:  # yesterday's files stay as they are
            keep |= {asset(profile, n) for n in info["files"]}
            continue
        data = config.data_dir(profile)
        paths = []
        for name in info["files"]:
            staged = Path("upload") / asset(profile, name)
            staged.parent.mkdir(exist_ok=True)
            shutil.copy(data / name, staged)
            paths.append(str(staged))
            keep.add(staged.name)
        gh("release", "upload", TAG, *paths, "--repo", REPO, "--clobber")
    Path("upload/manifest.json").write_text(json.dumps(manifest, indent=1))
    gh("release", "upload", TAG, "upload/manifest.json", "--repo", REPO, "--clobber")
    # remove yesterday's map caches and profiles that are gone
    for name in json.loads(gh("release", "view", TAG, "--repo", REPO, "--json", "assets"))["assets"]:
        if name["name"] not in keep:
            gh("release", "delete-asset", TAG, name["name"], "--repo", REPO, "--yes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    embedder = SentenceEmbedder(config.MODEL)
    old = {}
    r = requests.get(f"{RELEASE}/manifest.json", timeout=30)
    if r.status_code == 200:
        old = r.json().get("profiles", {})
    manifest = {"updated": datetime.now(timezone.utc).isoformat(timespec="minutes"), "profiles": {}}
    built = set()
    for profile in a.profiles:
        info = build_profile(profile, embedder)
        if info:
            manifest["profiles"][profile] = info
            built.add(profile)
        elif profile in old:  # keep serving yesterday's data rather than dropping the profile
            manifest["profiles"][profile] = old[profile]
            print(f"[{profile}] keeping the published data from {old[profile]['date']}")
    print(json.dumps(manifest, indent=1))
    if a.upload and manifest["profiles"]:
        upload(manifest, built)


if __name__ == "__main__":
    main()
