"""Regenerate docs/*.png from a running app (streamlit run app.py). Needs `pip install playwright`
and Google Chrome; run: python scripts/screenshots.py"""

import os
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get("ECHO_URL", "http://localhost:8501")
OUT = Path(__file__).resolve().parent.parent / "docs"


def wait_idle(page, extra_ms=1500):
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=300_000)
    page.wait_for_timeout(extra_ms)


def tab(page, name):
    page.get_by_role("tab", name=name).click()
    wait_idle(page)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=2, color_scheme="light")
    page.goto(URL)
    page.get_by_text("Understood:").wait_for(timeout=300_000)
    wait_idle(page, 3000)
    page.screenshot(path=OUT / "search.png")

    tab(page, "Match my resume")
    page.get_by_text("Closest postings").wait_for()
    page.screenshot(path=OUT / "resume.png")

    tab(page, "Where the jobs are")
    page.locator(".js-plotly-plot:visible").first.wait_for()
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT / "where.png")

    tab(page, "Skill trends")
    page.locator(".js-plotly-plot:visible").first.wait_for()
    page.screenshot(path=OUT / "trends.png")

    tab(page, "Role map")
    page.locator(".js-plotly-plot:visible").first.wait_for()
    page.wait_for_timeout(3000)
    page.screenshot(path=OUT / "map.png")

    tab(page, "Ask (RAG)")
    page.get_by_role("button", name="Ask").click()
    page.get_by_text("Sources", exact=True).wait_for(timeout=300_000)
    wait_idle(page)
    page.screenshot(path=OUT / "rag.png")
    browser.close()
print("wrote", *sorted(x.name for x in OUT.glob("*.png")))
