"""OSINT scraper: fetch a list of URLs with stealth Playwright and hand
the cleaned text to ChatGPT for cross-source profile correlation.

Sessions are kept in a persistent Chromium profile at the project's
`context/` directory so that previously-logged-in social platforms stay
authenticated across runs. Use `iris.src.scripts.setup_scraper_context`
to do the initial manual logins.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from .chatgpt import ft_call_chatgpt
from .parser import ft_get_and_parse_html

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_DIR = PROJECT_ROOT / "context"
# Chromium leaves multi-GB caches in these subdirs on every run. We nuke
# them after each scrape because they don't carry useful session state.
CACHE_DIRS_TO_PURGE = (
    CONTEXT_DIR / "Default" / "Cache",
    CONTEXT_DIR / "Default" / "Code Cache",
    CONTEXT_DIR / "Default" / "Service Worker",
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


async def ft_scraper(urls: list[str]) -> dict | str:
    """Scrape every URL in parallel, then ask ChatGPT to merge into one profile.

    Returns:
        Either the parsed OSINT profile (dict) or an error string.
        Returns an early error if OPENAI_API_KEY is missing or `urls` is empty.
    """
    # Load .env from the project root, regardless of cwd.
    load_dotenv(PROJECT_ROOT / ".env")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return "Please set OPENAI_API_KEY in .env"
    if not urls:
        return "No URLs to scrape."

    if not CONTEXT_DIR.exists():
        print(
            "[scraper] No persistent context yet — social platforms will "
            "block scraping. Run `python -m iris.src.scripts.setup_scraper_context` once."
        )

    async with Stealth().use_async(async_playwright()) as p:
        browser = await p.chromium.launch_persistent_context(
            headless=True,
            user_agent=USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            user_data_dir=str(CONTEXT_DIR),
            args=["--disable-blink-features=AutomationControlled"],
        )

        tasks = [
            asyncio.create_task(ft_get_and_parse_html(browser, url))
            for url in urls
        ]

        results: list[str] = []
        for finished in asyncio.as_completed(tasks):
            try:
                url, content = await finished
            except Exception as e:
                # A single parse task can crash if the page tears the
                # context down; skip it instead of nuking the whole batch.
                print(f"[scraper] task crashed: {e}")
                continue
            if content is None:
                continue
            results.append(
                f"|START              URL:{url}\nCONTENT:{content}              END|"
            )

        await browser.close()

    # Best-effort cache cleanup; the dirs may not exist on the first run.
    for cache_dir in CACHE_DIRS_TO_PURGE:
        shutil.rmtree(cache_dir, ignore_errors=True)

    # All URLs blocked / failed — short-circuit before burning OpenAI tokens
    # on an empty corpus, which makes GPT fabricate a profile from nothing.
    if not results:
        return "All URLs failed to scrape (anti-bot / login wall / timeout)."

    scraped_data = "\n".join(results)
    return ft_call_chatgpt(scraped_data, openai_api_key)
