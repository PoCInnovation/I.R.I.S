"""Single-URL scraper: open a page in a stealth context, extract its
visible text and clean it of structural noise. Always returns a tuple
so the parent `as_completed` loop never has to special-case exceptions.
"""

from __future__ import annotations

import asyncio
import random

from bs4 import BeautifulSoup


async def ft_get_and_parse_html(browser, url: str) -> tuple[str, str | None]:
    """Open `url` in a new page, return (url, cleaned_text) or (url, None).

    Failure modes silenced and surfaced as `None`:
      - browser.new_page() raises (context torn down by anti-bot)
      - page.goto times out
      - target page throws while we evaluate the DOM
    """
    page = None
    try:
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(5000)
        await asyncio.sleep(random.uniform(2, 4))
        await page.mouse.wheel(0, 600)
        await asyncio.sleep(random.uniform(1, 2))

        # Prefer <main> over <body> — most editorial sites place the
        # interesting content inside <main>, which strips global chrome.
        html = await page.evaluate(
            """() => {
                const main = document.querySelector('main');
                if (main) return main.innerHTML;
                return document.body.innerHTML;
            }"""
        )
        soup = BeautifulSoup(html, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "noscript", "svg"]):
            element.decompose()

        text = " || ".join(soup.stripped_strings)
        return url, text
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")
        return url, None
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
