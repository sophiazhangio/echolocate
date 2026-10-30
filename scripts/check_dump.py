"""Check job-radar's dump result (its last JSON line on stdin) in the daily workflow.

ok:false or an almost empty dump counts as a failure: the file is removed so the build skips
this profile and the site keeps yesterday's data. A few failed boards are only logged.

    npm run --silent dump -- --profile it | python scripts/check_dump.py it
"""
import json
import os
import sys

MIN_POSTINGS = 100

profile = sys.argv[1]
lines = [ln for ln in sys.stdin.read().splitlines() if ln.strip().startswith("{")]
try:
    r = json.loads(lines[-1]) if lines else {}
except ValueError:
    r = {}
if r.get("failed"):
    print(f"[{profile}] failed boards: " + ", ".join(map(str, r["failed"])))
if not r.get("ok") or r.get("postings", 0) < MIN_POSTINGS:
    print(f"::warning::job-radar failed for {profile}: {r.get('error') or r.get('postings', 'no result')}")
    if r.get("file") and os.path.exists(r["file"]):
        os.remove(r["file"])
else:
    print(f"[{profile}] {r['postings']:,} postings, {r.get('failedBoards', 0)} boards failed")
