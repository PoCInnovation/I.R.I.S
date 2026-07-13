import asyncio
import random
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from camoufox.async_api import AsyncCamoufox

load_dotenv(override=False)
load_dotenv(Path(__file__).parent / ".env.example", override=False)

PIMEYES_URL = "https://pimeyes.com/en"
# Persistent browser profile: cookies/session survive between runs, so after the
# first successful login + CAPTCHA solve PimEyes treats us as a returning user
# and challenges less aggressively. Keep this dir out of git.
PROFILE_DIR = Path(__file__).parent / "pimeyes_profile"
PIMEYES_EMAIL = os.environ["PIMEYES_EMAIL"]
PIMEYES_PASSWORD = os.environ["PIMEYES_PASSWORD"]

# Prosopo's "I am human" widget is rendered INLINE in the page (custom element
# <prosopo-procaptcha>, no iframe). It exposes TWO things matching the data-cy
# attribute: the 302x80 widget wrapper AND the real ~28px <input> checkbox. We
# must target the input — clicking the wrapper's centre lands in dead space
# between the checkbox (left) and the Prosopo logo (right). The input also takes
# a few seconds to render after the upload, so we wait for it to be visible.
# Its label is locale-dependent ("I am human" / "Je suis humain" / ...) so we
# never match on text.
CAPTCHA_CHECKBOX = "input[type='checkbox'][data-cy='captcha-checkbox']"

async def human_pause(lo: float = 0.3, hi: float = 0.7) -> None:
    """Short randomized gap between actions so timing isn't robotic."""
    await asyncio.sleep(random.uniform(lo, hi))


async def warm_up(page) -> None:
    """Settle like a real visitor before acting: scroll down to read the page,
    pause, then drift back up. Per-click cursor motion is left to Camoufox's
    humanize, so there's no scripted cursor wandering here."""
    await human_pause(0.5, 1.0)
    await page.mouse.wheel(0, random.randint(300, 500))
    await human_pause(0.9, 1.6)
    await page.mouse.wheel(0, -random.randint(200, 400))
    await human_pause(0.5, 1.0)


async def login(page) -> None:
    """Click Log In, fill credentials, submit — or skip if already authed.

    Cursor motion is handled by Camoufox's engine-level humanize: every
    `locator.click()` is a fluid arc that lands directly on the target.
    """
    print("[camoufox] checking auth state...")

    # With a persistent profile the saved session may already be authenticated.
    # The nav button gains `.logged-in` when signed in ("My account") and reads
    # "Log In" otherwise. Poll for whichever appears first.
    logged_in_btn = page.locator("button.auth.logged-in").first
    login_btn = page.locator("button.auth:not(.logged-in)").first
    for _ in range(40):  # up to ~4s
        if await logged_in_btn.count() > 0:
            print("[camoufox] already logged in (persistent session), skipping login")
            return
        if await login_btn.count() > 0 and await login_btn.is_visible():
            break
        await asyncio.sleep(0.1)
    else:
        print("[camoufox] no Log In button found, assuming logged in")
        return

    print("[camoufox] logging in...")
    await login_btn.click()

    # Keycloak login form — not the anti-bot gate, so filling instantly is safe.
    await page.wait_for_selector("#username", timeout=30_000)
    await human_pause(0.4, 0.7)
    await page.locator("#username").fill(PIMEYES_EMAIL)
    await human_pause(0.15, 0.3)
    await page.locator("#password").fill(PIMEYES_PASSWORD)
    await human_pause(0.2, 0.4)
    await page.locator("#kc-login").click()

    await page.wait_for_url("**/pimeyes.com/**", timeout=20_000)
    await page.wait_for_load_state("domcontentloaded")
    await human_pause(0.5, 0.9)
    print(f"[camoufox] logged in, current url: {page.url}")


async def upload_image(page, image_path) -> None:
    """Open the upload dialog via the dropzone and feed the file chooser."""
    dropzone = page.locator(".dropzone-blue").first
    await dropzone.wait_for(state="visible", timeout=15_000)
    await human_pause(0.4, 0.8)
    async with page.expect_file_chooser() as fc_info:
        await dropzone.click()
    file_chooser = await fc_info.value
    await file_chooser.set_files(str(image_path))
    print(f"[camoufox] uploaded: {image_path}")


async def click_human_checkbox(page, timeout_s: float = 30.0) -> None:
    """Tick Prosopo's "I am human" box with one real, humanized cursor click.

    Two things matter here, both learned from the live DOM:

      1. Target the actual <input> (a ~28px square), not the 302x80 widget
         wrapper — clicking the wrapper's centre misses the checkbox entirely.
      2. The input renders a few seconds AFTER the upload, so we wait for it to
       become visible (up to timeout_s) instead of giving up immediately.

    The click MUST go through `page.mouse.move()` + down/up, not
    `locator.click()`: with Camoufox's humanize enabled, only an explicit mouse
    move animates the trusted cursor arc Prosopo expects. We read the checkbox's
    box, move the real cursor to its centre, and press once.

    Whether Prosopo accepts the click (stays frictionless) or escalates to an
    image grid is decided by trust/fingerprint, not by us — Start Search
    enabling is the only ground truth, so we don't poll for a "solved" state.
    No checkbox at all (trusted session, no challenge) means nothing to do.
    """
    box = page.locator(CAPTCHA_CHECKBOX).first
    try:
        await box.wait_for(state="visible", timeout=int(timeout_s * 1000))
    except Exception:
        print("[camoufox] no captcha checkbox appeared — nothing to click")
        return

    await box.scroll_into_view_if_needed()
    bb = await box.bounding_box()
    if bb is None:
        print("[camoufox] captcha checkbox has no box — skipping click")
        return
    cx = bb["x"] + bb["width"] / 2
    cy = bb["y"] + bb["height"] / 2

    await human_pause(0.7, 1.4)  # read the challenge like a person before acting
    # Humanized move in (Camoufox draws the arc), brief settle, press, release.
    await page.mouse.move(cx, cy)
    await human_pause(0.1, 0.25)
    await page.mouse.down()
    await human_pause(0.05, 0.12)
    await page.mouse.up()
    print(f"[camoufox] clicked 'I am human' checkbox at ({cx:.0f}, {cy:.0f})")


async def start_search(page, allow_manual: bool = True) -> None:
    """Wait for Start Search to become enabled — the real solved-state signal —
    then click it once and wait for the results page.

    Start Search sheds its disabled attribute/class only once the CAPTCHA is
    cleared, so its enablement is our ground truth. If it doesn't enable quickly,
    Prosopo most likely escalated to an image grid we don't auto-solve: the
    window is visible, so solve it by hand and this picks up automatically.

    If allow_manual is True and the captcha fails, prompts the user to solve it manually.
    """
    start_btn = page.locator(
        "button:has-text('Start Search'):not([disabled]):not(.disabled)"
    ).first
    print("[camoufox] waiting for Start Search to become enabled...")
    
    try:
        await start_btn.wait_for(state="visible", timeout=20_000)
    except Exception:
        print(
            "[camoufox] Start Search still disabled after 20s — captcha likely failed"
        )
        
        if allow_manual:
            print("[camoufox] ================================================")
            print("[camoufox] CAPTCHA FAILED - MANUAL SOLVE REQUIRED")
            print("[camoufox] Please solve the captcha challenge in the browser window.")
            print("[camoufox] Once solved, the script will continue automatically.")
            print("[camoufox] ================================================")
            
            # Wait indefinitely for manual solve, polling every 2 seconds
            while True:
                await asyncio.sleep(2)
                if await start_btn.count() > 0 and await start_btn.is_visible():
                    print("[camoufox] Start Search is now enabled - continuing...")
                    break
        else:
            print(
                "[camoufox] Manual solve disabled. Solve it by hand in the open window; waiting..."
            )
            await start_btn.wait_for(state="visible", timeout=180_000)

    await human_pause(0.3, 0.6)
    print("[camoufox] clicking Start Search")
    await start_btn.click()

    await page.wait_for_url("**/results/**", timeout=60_000)
    await page.wait_for_load_state("networkidle")
    print(f"[camoufox] results page: {page.url}")


async def main() -> None:
    # Anti-bot stack — this is what keeps Prosopo's check frictionless (a single
    # checkbox click) instead of escalating to an image grid:
    #  - humanize: every move/click is a realistic cursor arc, no teleport.
    #  - geoip: derive latitude/longitude/timezone AND locale from the real exit
    #    IP, and spoof WebRTC to it — kills the locale/geo mismatch anti-bot
    #    systems flag, so we deliberately do NOT pin `locale`, geoip sets it.
    #  - persistent_context + user_data_dir: reuse cookies/session so we're a
    #    returning, logged-in user across runs (trusted sessions get challenged
    #    less, and may skip the checkbox entirely).
    #  - enable_cache: real browsers cache assets; a cache-less profile is a tell.
    #  - os=windows: a plausible Windows fingerprint, consistent with the IP.
    # NOTE: headless=False is important — headless is easily detected and makes
    # Prosopo escalate to the image grid (confirmed against the live site).

    # parsing arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode (increases CAPTCHA detection risk)')
    args = parser.parse_args()

    async with AsyncCamoufox(
        headless=args.headless,
        # MAX cursor-travel time in seconds (default True ≈ 1.5s). A slightly
        # higher cap makes moves land at a calm, average-user pace.
        humanize=2.5,
        geoip=True,
        os=("windows",),
        enable_cache=True,
        persistent_context=True,
        user_data_dir=str(PROFILE_DIR),
    ) as browser:
        # A persistent context launches with a page already open; reuse it.
        page = browser.pages[0] if browser.pages else await browser.new_page()

        await page.goto(PIMEYES_URL, wait_until="domcontentloaded", timeout=60_000)
        await human_pause(0.6, 1.2)
        print(f"[camoufox] loaded: {page.url}")

        # Build a natural behavioural trail before doing anything sensitive.
        await warm_up(page)

        #await login(page)
        await upload_image(page, args.image_path)

        # The only gate after upload is the Prosopo captcha — tick it, then let
        # Start Search enabling confirm we're through.
        await click_human_checkbox(page)
        await start_search(page)

        # Keep the window open so the run can be watched. Interactively, Enter
        # closes it; with no TTY (background/piped run), idle instead of crashing.
        if args.headless == False:
            try:
                input("Press Enter to close the browser...")
            except EOFError:
                print("[camoufox] no TTY — keeping browser open for 5 min")
                await asyncio.sleep(300)


if __name__ == "__main__":
    asyncio.run(main())
