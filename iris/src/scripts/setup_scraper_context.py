"""One-time setup for the scraper's persistent browser context.

Run this once to log into the social platforms the scraper visits.
Chromium opens in visible mode; the cookies/sessions you create are
saved in `context/` and reused by the headless scraper afterwards.

Usage:
    python -m iris.src.scripts.setup_scraper_context

Use fake/throwaway accounts only. Never your real ones.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from playwright.async_api import async_playwright
from playwright_stealth import Stealth

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONTEXT_DIR = PROJECT_ROOT / "context"
LANDING_PAGES = (
    "https://www.linkedin.com/login",
    "https://www.instagram.com/accounts/login/",
    "https://x.com/i/flow/login",
    "https://www.facebook.com/login",
    "https://www.tiktok.com/login",
)


async def main() -> None:
    CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[setup] persistent context: {CONTEXT_DIR}")
    print("[setup] log into each tab, accept cookies, then close the browser.")

    async with Stealth().use_async(async_playwright()) as p:
        browser = await p.chromium.launch_persistent_context(
            headless=False,
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            user_data_dir=str(CONTEXT_DIR),
            args=["--disable-blink-features=AutomationControlled"],
        )

        for url in LANDING_PAGES:
            page = await browser.new_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
            except Exception as e:
                print(f"[setup] couldn't preload {url}: {e}")

        closed = asyncio.Event()
        browser.on("close", lambda _: closed.set())
        await closed.wait()
        print("[setup] context saved.")


if __name__ == "__main__":
    asyncio.run(main())
