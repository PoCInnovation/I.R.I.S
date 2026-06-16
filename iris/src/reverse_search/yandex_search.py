"""Reverse-image search for face crops against Yandex Images.

Yandex has no public image-search API, so we drive the web UI with
Playwright (stealth mode) and scrape the "sites containing this image"
list. The browser is locked to en-US so the "Select file" button label
the upload flow depends on is stable. Every failure path is caught and
turned into an empty result — the pipeline tolerates that, but a crash
mid-search would tear down the whole run.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

YANDEX_IMAGES_URL = "https://yandex.com/images/"
# The "Search by image" camera button. Yandex serves a few variants
# depending on layout bucket; we try them in order.
CBIR_BUTTON_SELECTORS = (
    "button[aria-label*='image' i]",
    "button.input__button",
    "button.input__cbir-button",
    "div.input__cbir",
)
UPLOAD_INPUT_SELECTOR = "input[type='file']"
SITES_LINK_SELECTORS = (
    "a.CbirSites-ItemTitle",
    "a.cbir-sites__site-link",
    "li.CbirSites-Item a[href^='http']",
)
RESULTS_URL_GLOB = "**/images/search**"
FAILURE_SCREENSHOT = Path("/tmp/iris_yandex_failure.png")


async def yandex_reverse_search(
    face_crop: np.ndarray,
    *,
    max_results: int = 10,
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> list[str]:
    """Upload a face crop to Yandex Images and return matching source URLs.

    Args:
        face_crop: BGR image array, typically a `Face.crop` from `FaceDetector`.
        max_results: cap on URLs returned (downstream pays per URL via ChatGPT).
        headless: set False to watch the browser drive itself.
        timeout_ms: per-step Playwright timeout.

    Returns:
        Distinct source URLs (Yandex's relevance ranking). Empty list on
        any failure — selectors that drift, anti-bot pages, timeouts.
        The pipeline treats an empty list as "no match for this face".
    """
    # Yandex's upload widget requires a real file on disk; we can't stream
    # the numpy array. Write a temp JPEG and clean it up unconditionally.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, face_crop)
        tmp_path = Path(tmp.name)

    browser = None
    try:
        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(headless=headless)
            # Force en-US so the "Select file" button label stays stable.
            context = await browser.new_context(
                locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
                viewport={"width": 1366, "height": 800},
            )
            page = await context.new_page()
            page.set_default_timeout(timeout_ms)

            try:
                await page.goto(YANDEX_IMAGES_URL)
                await page.wait_for_load_state("domcontentloaded")

                # Click the camera icon to surface the upload modal. We try
                # each selector variant; sentinel tracks whether any worked.
                clicked = False
                for sel in CBIR_BUTTON_SELECTORS:
                    btn = page.locator(sel).first
                    if await btn.count() == 0:
                        continue
                    try:
                        await btn.click(timeout=3_000)
                        clicked = True
                        break
                    except Exception:
                        continue
                if not clicked:
                    await _dump_failure(page, "no CBIR button matched")
                    return []

                # The modal exposes a "Select file" button that opens the
                # native file chooser; Playwright intercepts via expect_file_chooser.
                try:
                    select_button = page.get_by_role("button", name="Select file")
                    async with page.expect_file_chooser() as fc_info:
                        await select_button.click(timeout=5_000)
                    chooser = await fc_info.value
                    await chooser.set_files(str(tmp_path))
                except Exception as e:
                    await _dump_failure(page, f"upload click/chooser failed: {e}")
                    return []

                # Upload triggers a nav to /images/search?rpt=imageview&...
                # Anti-bot pages count as "nav happened, no results".
                try:
                    await page.wait_for_url(RESULTS_URL_GLOB, timeout=timeout_ms)
                    await page.wait_for_load_state("networkidle", timeout=10_000)
                except Exception:
                    await _dump_failure(page, "no results page reached")
                    return []

                urls: list[str] = []
                seen: set[str] = set()
                for selector in SITES_LINK_SELECTORS:
                    for anchor in await page.locator(selector).all():
                        href = await anchor.get_attribute("href")
                        if not href or not href.startswith("http") or href in seen:
                            continue
                        seen.add(href)
                        urls.append(href)
                        if len(urls) >= max_results:
                            break
                    if len(urls) >= max_results:
                        break
                return urls
            finally:
                if browser is not None:
                    try:
                        await browser.close()
                    except Exception:
                        pass
    finally:
        tmp_path.unlink(missing_ok=True)


async def _dump_failure(page, reason: str) -> None:
    """Best-effort diagnostic: save a screenshot + log to stderr."""
    try:
        await page.screenshot(path=str(FAILURE_SCREENSHOT), full_page=True)
        print(f"  [yandex] {reason} — saved screenshot to {FAILURE_SCREENSHOT}")
    except Exception as e:
        print(f"  [yandex] {reason} — additionally failed to screenshot: {e}")
