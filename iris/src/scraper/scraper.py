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
from instagram import ft_instaloader

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
        
        '''
        INSTRUCTIONS
            if it is your first time running this script you need to uncomment these lines below which will provides you
            10min to connect manually on linkedin insta twitter facebook tiktok or any other platform you need or want
            and ensure that you allow all cookies or close any pop up and then you can close all the tabs
            to do this do not forget to pass the headless argument to False in the lines above
            please do not use your personal accounts use fake ones that you have already created before
            note: you can run it multiple times or increase the sleep time if you want to connect on many other platforms
        '''
        ###await asyncio.sleep(600)
        ###return "Context successfully set! You can now run the script again for better results"

        # launch all scraping tasks for each url
        tasks = []
        for url in urls:
            if "instagram.com/" in url:
                task = asyncio.create_task(asyncio.to_thread(ft_instaloader, url))
                tasks.append(task)
            else:
                task = asyncio.create_task(ft_get_and_parse_html(browser, url))
                tasks.append(task)
        
        # collect tasks results as soon as each one is complete and then format them all
        results = []
        for tsk in asyncio.as_completed(tasks):
            url, content = await tsk

            if "instagram.com/" in url:
                if content != None:
                    results.append(f"|START              URL:{url}\nCONTENT:{content}              END|")
                # if insta function fails then fallback on default function
                else:
                    fallback_url, fallback_content = await ft_get_and_parse_html(browser, url)
                    if fallback_content is None:
                        continue
                    results.append(f"|START              URL:{fallback_url}\nCONTENT:{fallback_content}              END|")
            else:
                if content is None:
                    continue
                results.append(f"|START              URL:{url}\nCONTENT:{content}              END|")
        
        await browser.close()

    # delete useless and heavy cache directories
    for folder in ['Cache', 'Code Cache', 'Service Worker']:
        path = f'./context/Default/{folder}'
        if os.path.exists(path):
            shutil.rmtree(path)

    scraped_data = "\n".join(results)
    return ft_call_chatgpt(scraped_data, openai_api_key)
