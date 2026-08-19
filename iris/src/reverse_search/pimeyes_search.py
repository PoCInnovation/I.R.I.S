"""Reverse-image face search against PimEyes.

Drives the PimEyes web UI like a human to run a face search: uploads the crop,
accepts the consent boxes, clears the Prosopo "I am human" CAPTCHA, and starts
the search. Runs headless-by-default on an off-screen virtual display (Xvfb) and
tries to solve the CAPTCHA automatically; if a human is required, it reveals the
same live session over VNC (x11vnc + a viewer) so it can be solved in place —
no relaunch, no re-upload. Every failure path returns an empty list so the
pipeline never crashes mid-run.

On the results page it harvests, per match, the thumbnail of the photo and the
domain PimEyes prints on the tile — the pair `iris/src/scripts/crawler.py` needs
to walk that site and recover the page URL the paywall withholds. Records come
back as `PimEyesMatch` (see pimeyes_results.py); the legacy `list[str]` entry
point still exists and returns just the sites.

See docs/pimeyes_search.md for a full walkthrough of the tools and flow.
"""

from __future__ import annotations

import asyncio
import base64
import enum
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from camoufox.async_api import AsyncCamoufox
from camoufox.exceptions import VirtualDisplayError
from camoufox.virtdisplay import VirtualDisplay

from iris.src.reverse_search.pimeyes_results import (
    DOMAIN_RE,
    PimEyesMatch,
    normalize_site,
    save_matches,
    with_local_path,
)

PIMEYES_URL = "https://pimeyes.com/en"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROFILE_DIR = PROJECT_ROOT / "pimeyes_profile"

UPLOAD_INPUT = 'input[type="file"][multiple]'
# The search modal is a full-screen overlay. Its utility classes churn with every
# PimEyes redesign (it used to carry `z-100`, which is now applied by a rule
# instead of a class), so we match only the two stable positioning classes and
# disambiguate by content: the overlay is the one containing Start Search.
SEARCH_MODAL = "div.fixed.inset-0"
PROSOPO_INPUT = "input[type='checkbox'][data-cy='captcha-checkbox']"
PROSOPO_CHECKBOX = "prosopo-procaptcha .prosopo-checkbox"
START_SEARCH_NAME = "Start Search"

HUMANIZE_CAP = 2.5

PROSOPO_CLICK_OFFSETS = ((22, 14), (26, 22), (22, 22), (16, 14), (22, 30), (30, 22))

FAILURE_SCREENSHOT = Path("/tmp/iris_pimeyes_failure.png")

RESULTS_DIR = PROJECT_ROOT / "pimeyes_results"
# A result tile is a square card holding the thumbnail; the same card prints the
# source domain on a button. Utility classes churn, but the grid has been square
# cards since the redesign and `aspect-square` is the one class that says so.
RESULT_TILE_IMG = "div.aspect-square img"
# The grid streams in over a few seconds; treat it as settled once the tile count
# stops moving rather than guessing a fixed sleep.
GRID_SETTLE_POLLS = 3
GRID_SETTLE_INTERVAL_S = 1.5
GRID_MAX_WAIT_S = 25.0

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# Read the tiles in document order (= PimEyes' relevance order), tagging each
# image so a failed byte-fetch can fall back to screenshotting that exact node.
_COLLECT_TILES_JS = """
() => {
    const out = [];
    for (const tile of document.querySelectorAll('div.aspect-square')) {
        const img = tile.querySelector('img');
        if (!img) continue;
        const src = img.currentSrc || img.src || '';
        if (!src) continue;
        // Buttons and links only. The <span> wrapping them has a textContent of
        // "numerama.comMore" — no separator between children — which reads as a
        // perfectly well-formed hostname and would be crawled as one.
        const labels = [...tile.querySelectorAll('button, a')]
            .map(el => (el.textContent || '').trim())
            .filter(Boolean);
        img.setAttribute('data-iris-idx', String(out.length));
        out.push({ src, labels });
    }
    return out;
}
"""

# Thumbnails are blob: URLs minted by the page, so there is nothing to download
# from outside: fetch them in the page's own context and hand back base64. This
# also keeps the original bytes intact — screenshotting the <img> would capture
# the CSS `object-cover` crop instead of the photo PimEyes matched.
_FETCH_BYTES_JS = """
async (src) => {
    const resp = await fetch(src);
    if (!resp.ok) return null;
    const blob = await resp.blob();
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = '';
    const CHUNK = 0x8000;  // chunked: String.fromCharCode blows the stack on big arrays
    for (let i = 0; i < bytes.length; i += CHUNK) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return { type: blob.type || '', data: btoa(binary) };
}
"""


def _is_search_url(url: str) -> bool:
    return "/search/" in url or "/results/" in url


class _Outcome(enum.Enum):
    REACHED = "reached"
    CAPTCHA_UNSOLVED = "captcha_unsolved"
    ERROR = "error"


async def _human_pause(lo: float = 0.3, hi: float = 0.7) -> None:
    await asyncio.sleep(random.uniform(lo, hi))


async def _wander_cursor(page, moves: int = 3) -> None:
    try:
        size = page.viewport_size or {"width": 1280, "height": 720}
        for _ in range(moves):
            x = random.randint(int(size["width"] * 0.2), int(size["width"] * 0.8))
            y = random.randint(int(size["height"] * 0.2), int(size["height"] * 0.8))
            await page.mouse.move(x, y)
            await _human_pause(0.2, 0.6)
    except Exception:
        pass


async def _upload(page, image_path: Path) -> None:
    inp = page.locator(UPLOAD_INPUT).first
    await inp.wait_for(state="attached", timeout=15_000)
    await inp.set_input_files(str(image_path))


async def _accept_consent(modal) -> None:
    boxes = modal.get_by_role("checkbox")
    try:
        await boxes.first.wait_for(state="visible", timeout=15_000)
    except Exception:
        print("  [pimeyes] no consent boxes appeared in the modal")
        return
    for i in range(await boxes.count()):
        box = boxes.nth(i)
        try:
            if await box.get_attribute("aria-checked") == "true":
                continue
            await box.evaluate("el => el.click()")
            await _human_pause(0.2, 0.5)
        except Exception as e:
            print(f"  [pimeyes] consent box {i} not clickable: {str(e)[:60]}")


async def _start_enabled(start) -> bool:
    try:
        return (await start.evaluate("el => el.disabled")) is False
    except Exception:
        return False


async def _click_prosopo_input(page, start) -> bool:
    box = page.locator(PROSOPO_INPUT).first
    try:
        await box.wait_for(state="visible", timeout=8_000)
        bb = await box.bounding_box()
        if not bb:
            return False
        cx, cy = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
        await page.mouse.move(cx, cy)
        await _human_pause(0.1, 0.25)
        await page.mouse.down()
        await _human_pause(0.05, 0.12)
        await page.mouse.up()
    except Exception:
        return False
    await asyncio.sleep(2.0)
    return await _start_enabled(start)


async def _auto_solve_prosopo(page, modal) -> bool:
    start = modal.get_by_role("button", name=START_SEARCH_NAME).first
    if await _start_enabled(start):
        return True

    cb = page.locator(PROSOPO_CHECKBOX).first
    try:
        await cb.wait_for(state="visible", timeout=20_000)
    except Exception:
        print("  [pimeyes] no Prosopo checkbox — assuming no CAPTCHA")
        return await _start_enabled(start)

    await _human_pause(0.6, 1.2)
    await _wander_cursor(page)

    if await _click_prosopo_input(page, start):
        print("  [pimeyes] Prosopo cleared automatically (input click)")
        return True

    for px, py in PROSOPO_CLICK_OFFSETS:
        try:
            await cb.click(position={"x": px, "y": py}, force=True, timeout=6_000)
        except Exception:
            continue
        await asyncio.sleep(2.0)
        if await _start_enabled(start):
            print("  [pimeyes] Prosopo cleared automatically (offset sweep)")
            return True

    return False


async def _wait_for_human(modal, timeout_ms: int) -> bool:
    start = modal.get_by_role("button", name=START_SEARCH_NAME).first
    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
    while asyncio.get_event_loop().time() < deadline:
        if await _start_enabled(start):
            print("  [pimeyes] CAPTCHA cleared")
            return True
        await asyncio.sleep(2.0)
    return False


def _display_port(display: str) -> int:
    return 5900 + int(display.lstrip(":").split(".")[0])


async def _reveal_display(display: str) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    x11vnc = shutil.which("x11vnc")
    if not x11vnc:
        print(
            "  [pimeyes] x11vnc not found (`sudo apt install x11vnc`) — cannot "
            f"reveal the window; the search is stuck on {display}"
        )
        return procs

    port = _display_port(display)
    procs.append(
        subprocess.Popen(
            [x11vnc, "-display", display, "-rfbport", str(port), "-localhost",
             "-nopw", "-forever", "-shared", "-quiet", "-noxdamage"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    )
    await asyncio.sleep(1.0)

    viewer = shutil.which("xtigervncviewer") or shutil.which("vncviewer")
    if viewer:
        print(f"  [pimeyes] opening VNC viewer on localhost::{port} — solve the CAPTCHA")
        procs.append(
            subprocess.Popen(
                [viewer, f"localhost::{port}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    else:
        print(
            f"  [pimeyes] no VNC viewer found — connect one to localhost::{port} "
            "to solve the CAPTCHA"
        )
    return procs


def _stop_reveal(procs: list[subprocess.Popen]) -> None:
    for p in reversed(procs):
        try:
            p.terminate()
        except Exception:
            pass


async def _start_search(page, modal, timeout_ms: int) -> None:
    btn = modal.get_by_role("button", name=START_SEARCH_NAME).first
    await _human_pause(0.3, 0.6)
    await btn.click(position={"x": 20, "y": 20}, force=True)
    print("  [pimeyes] clicked Start Search")
    await page.wait_for_url(_is_search_url, timeout=timeout_ms)


async def _wait_for_grid(page) -> int:
    """Wait until the results grid stops growing; return the tile count."""
    try:
        await page.wait_for_selector(RESULT_TILE_IMG, timeout=int(GRID_MAX_WAIT_S * 1000))
    except Exception:
        return 0

    previous, stable, waited = -1, 0, 0.0
    while waited < GRID_MAX_WAIT_S:
        count = await page.locator(RESULT_TILE_IMG).count()
        stable = stable + 1 if count == previous else 0
        if stable >= GRID_SETTLE_POLLS:
            break
        previous = count
        await asyncio.sleep(GRID_SETTLE_INTERVAL_S)
        waited += GRID_SETTLE_INTERVAL_S
    return await page.locator(RESULT_TILE_IMG).count()


def _pick_domain(labels: list[str]) -> str:
    """First label on the tile that is actually a hostname.

    The button row also carries a "More" action whose text is localized, so we
    match on shape rather than skipping a known word.
    """
    for label in labels:
        if DOMAIN_RE.match(label.strip()):
            return label.strip()
    return ""


async def _save_thumbnail(page, src: str, dom_index: int, dest: Path) -> Path | None:
    """Save one result thumbnail, preferring the original bytes over a repaint."""
    try:
        payload = await page.evaluate(_FETCH_BYTES_JS, src)
    except Exception as e:
        print(f"  [pimeyes] thumbnail {dom_index} fetch failed: {str(e)[:60]}")
        payload = None

    if payload and payload.get("data"):
        path = dest.with_suffix(_MIME_EXT.get(payload.get("type", ""), ".jpg"))
        try:
            path.write_bytes(base64.b64decode(payload["data"]))
            return path
        except Exception as e:
            print(f"  [pimeyes] thumbnail {dom_index} write failed: {str(e)[:60]}")

    # Fallback: repaint the node itself. Loses whatever `object-cover` crops, but
    # a cropped thumbnail still beats no thumbnail.
    path = dest.with_suffix(".png")
    try:
        await page.locator(f'img[data-iris-idx="{dom_index}"]').first.screenshot(
            path=str(path)
        )
        return path
    except Exception as e:
        print(f"  [pimeyes] thumbnail {dom_index} screenshot failed: {str(e)[:60]}")
        return None


async def _collect_matches(page, max_results: int, out_dir: Path) -> list[PimEyesMatch]:
    """Harvest (image, site) pairs from the results grid.

    Never raises: a broken selector or a redesigned tile has to degrade to fewer
    matches, not to a crashed pipeline.
    """
    try:
        count = await _wait_for_grid(page)
        if not count:
            await _dump_failure(page, "results grid never appeared")
            return []

        tiles = await page.evaluate(_COLLECT_TILES_JS)
    except Exception as e:
        await _dump_failure(page, f"results grid unreadable: {e}")
        return []

    matches: list[PimEyesMatch] = []
    skipped = 0
    for dom_index, tile in enumerate(tiles):
        if len(matches) >= max_results:
            break
        domain = _pick_domain(tile.get("labels") or [])
        if not domain:
            # The uploaded face is rendered in a square tile too, but carries no
            # source domain — this is what filters it out.
            skipped += 1
            continue

        index = len(matches)
        out_dir.mkdir(parents=True, exist_ok=True)
        local = await _save_thumbnail(
            page, tile.get("src", ""), dom_index, out_dir / f"{index:03d}"
        )
        matches.append(
            with_local_path(
                PimEyesMatch(
                    index=index,
                    domain=domain,
                    site=normalize_site(domain),
                    thumbnail_url=tile.get("src", ""),
                ),
                local,
            )
        )

    print(f"  [pimeyes] {len(matches)} match(es) harvested ({skipped} tile(s) without a domain)")
    return matches


async def _run_search(
    tmp_path: Path,
    *,
    virtual_display: str | None,
    timeout_ms: int,
    manual_solve_timeout_ms: int,
    max_results: int,
    out_dir: Path,
) -> tuple[_Outcome, list[PimEyesMatch]]:
    async with AsyncCamoufox(
        headless=False,
        virtual_display=virtual_display,
        humanize=HUMANIZE_CAP,
        geoip=True,
        os=("windows",),
        enable_cache=True,
        persistent_context=True,
        user_data_dir=str(PROFILE_DIR),
    ) as browser:
        page = browser.pages[0] if browser.pages else await browser.new_page()
        page.set_default_timeout(timeout_ms)

        try:
            await page.goto(
                PIMEYES_URL, wait_until="domcontentloaded", timeout=timeout_ms
            )
            await _human_pause(0.6, 1.2)

            await _upload(page, tmp_path)

            # Scope by content, not by class: the page also renders a hero
            # "Start Search" button outside any overlay, so filtering on it
            # picks exactly the upload modal.
            modal = (
                page.locator(SEARCH_MODAL)
                .filter(has=page.get_by_role("button", name=START_SEARCH_NAME))
                .first
            )
            await modal.wait_for(state="visible", timeout=20_000)

            await _accept_consent(modal)

            if not await _auto_solve_prosopo(page, modal):
                if not await _human_captcha_handoff(
                    page, modal, virtual_display, manual_solve_timeout_ms
                ):
                    await _dump_failure(page, "Prosopo not cleared")
                    return _Outcome.CAPTCHA_UNSOLVED, []

            await _start_search(page, modal, timeout_ms)
        except Exception as e:
            await _dump_failure(page, f"search failed: {e}")
            return _Outcome.ERROR, []

        await _human_pause(0.8, 1.6)
        print(f"  [pimeyes] search page: {page.url}")
        # Must happen inside the context manager: the thumbnails are blob: URLs
        # that only exist while this page is alive.
        matches = await _collect_matches(page, max_results, out_dir)
        return _Outcome.REACHED, matches


async def _human_captcha_handoff(
    page, modal, virtual_display: str | None, timeout_ms: int
) -> bool:
    reveal: list[subprocess.Popen] = []
    try:
        if virtual_display is not None:
            print(
                "  [pimeyes] CAPTCHA needs a human — revealing the live session "
                "via VNC (solve it in the viewer that opens)..."
            )
            reveal = await _reveal_display(virtual_display)
        else:
            print(
                "  [pimeyes] CAPTCHA needs a human — solve it in the open window; "
                "waiting..."
            )
        return await _wait_for_human(modal, timeout_ms)
    finally:
        _stop_reveal(reveal)


async def pimeyes_reverse_search_matches(
    face_crop: np.ndarray,
    *,
    max_results: int = 10,
    timeout_ms: int = 60_000,
    manual_solve_timeout_ms: int = 180_000,
    headless: bool = True,
    output_dir: Path | None = None,
) -> list[PimEyesMatch]:
    """Run a PimEyes search and return one record per match.

    Each record pairs a matched photo (saved under `output_dir`) with the domain
    PimEyes says it came from — the map the crawler turns into exact page URLs.
    Use `by_image()` / `by_site()` from `pimeyes_results` to view it as a dict.

    Args:
        face_crop: BGR image array, typically a `Face.crop` from `FaceDetector`.
        max_results: cap on matches harvested from the grid.
        timeout_ms: per-step browser timeout.
        manual_solve_timeout_ms: how long to wait for a human on the CAPTCHA.
        headless: True runs on an invisible virtual display with VNC-on-demand.
        output_dir: where thumbnails + matches.json land. Defaults to a fresh
            timestamped directory under `pimeyes_results/`.

    Returns:
        Match records in PimEyes' relevance order — empty on any failure, so a
        broken search degrades to "no match" instead of killing the pipeline.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, face_crop)
        tmp_path = Path(tmp.name)

    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = RESULTS_DIR / stamp

    vdisplay = None
    display: str | None = None
    if headless:
        try:
            vdisplay = VirtualDisplay()
            display = vdisplay.get()
        except VirtualDisplayError as e:
            print(
                f"  [pimeyes] virtual display unavailable ({e}); install Xvfb "
                "(`sudo apt install xvfb`). Running in a visible window instead."
            )
            vdisplay = None
            display = None

    try:
        _, matches = await _run_search(
            tmp_path,
            virtual_display=display,
            timeout_ms=timeout_ms,
            manual_solve_timeout_ms=manual_solve_timeout_ms,
            max_results=max_results,
            out_dir=output_dir,
        )
    except Exception as e:
        print(f"  [pimeyes] search aborted: {e}")
        return []
    finally:
        tmp_path.unlink(missing_ok=True)
        if vdisplay is not None:
            try:
                vdisplay.kill()
            except Exception:
                pass

    if matches:
        manifest = save_matches(matches, output_dir)
        print(f"  [pimeyes] {len(matches)} match(es) -> {manifest}")
    return matches


async def pimeyes_reverse_search(
    face_crop: np.ndarray,
    *,
    max_results: int = 10,
    timeout_ms: int = 60_000,
    manual_solve_timeout_ms: int = 180_000,
    headless: bool = True,
) -> list[str]:
    """Sites a face was found on, deduplicated, in relevance order.

    The flat `list[str]` shape the pipeline's other engine speaks. These are
    origins, not page URLs — PimEyes paywalls the path, so pair them with the
    thumbnails from `pimeyes_reverse_search_matches` and crawl to recover it.
    """
    matches = await pimeyes_reverse_search_matches(
        face_crop,
        max_results=max_results,
        timeout_ms=timeout_ms,
        manual_solve_timeout_ms=manual_solve_timeout_ms,
        headless=headless,
    )
    sites: list[str] = []
    for match in matches:
        if match.site and match.site not in sites:
            sites.append(match.site)
    return sites


async def _dump_failure(page, reason: str) -> None:
    try:
        await page.screenshot(path=str(FAILURE_SCREENSHOT), full_page=True)
        print(f"  [pimeyes] {reason} — saved screenshot to {FAILURE_SCREENSHOT}")
    except Exception as e:
        print(f"  [pimeyes] {reason} — additionally failed to screenshot: {e}")
