"""Per-stage RAM benchmark for sizing the Raspberry Pi 5.

Measures resident memory (RSS) at each pipeline stage and prints a table of
per-stage deltas plus the peak. RSS is summed across the whole *process tree*
(this Python process + every child), because Playwright launches Chromium as
child processes -- on a Pi 5 the browser, not the YOLO model, is usually the
largest single consumer, and a naive `self`-only measurement would miss it.

Stages:

    baseline       -> interpreter + imports, before anything IRIS-specific
    model loaded   -> after the YOLO weights are read into memory
    warm-up        -> after the first inference (graph/allocator init)
    detection      -> after running detection on the input image
    reverse-search -> after one Yandex reverse search (launches Chromium)
    scraper        -> after one scraper run (Chromium + ChatGPT)

The two network stages are best-effort: they need a real crop, network access
and (for the scraper) OPENAI_API_KEY + a logged-in `context/`. Any failure is
reported and the benchmark keeps going, so the detector numbers are always
produced. Use --detect-only to skip the network stages entirely.

We import the pipeline pieces directly rather than iris.src.scripts.run_pipeline
because that module spawns ffmpeg at import time.

No GUI, so it runs over SSH on a headless Pi. Run with:

    uv run python -m iris.src.scripts.ram_benchmark --image test.jpg
    uv run python -m iris.src.scripts.ram_benchmark --image test.jpg --detect-only
"""
from __future__ import annotations

import argparse
import asyncio
import threading
from pathlib import Path

import cv2
import numpy as np
import psutil

from iris.src.detection.face_detector import FaceDetector

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"
DEFAULT_MAX_URLS = 5


def tree_rss(proc: psutil.Process) -> int:
    """Total RSS in bytes for `proc` plus every child, recursively.

    Children (e.g. Chromium spawned by Playwright) are the whole point: on a
    Pi they dominate the budget. We tolerate races -- a child can die between
    listing and reading it -- by skipping any process that vanishes.
    """
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


class PeakSampler(threading.Thread):
    """Background thread recording the peak process-tree RSS.

    Stage snapshots only see memory at stage *boundaries*; a transient spike
    inside a stage (Chromium loading a heavy page) would be invisible to them.
    This sampler polls continuously so the reported peak is real.
    """

    def __init__(self, proc: psutil.Process, interval: float = 0.05) -> None:
        super().__init__(daemon=True)
        self.proc = proc
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, tree_rss(self.proc))
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()


def mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


async def _network_stages(
    crop: np.ndarray,
    snap,
    *,
    max_urls: int,
    show_browser: bool,
    scraper_url: str | None,
) -> None:
    """Reverse-search one crop, then run the scraper. Best-effort.

    Imported lazily so a machine without the reverse-search/scraper deps (or
    with an import-time failure) can still get the detector numbers above.
    """
    try:
        from iris.src.reverse_search import yandex_reverse_search
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, see docstring
        print(f"  [reverse-search skipped] import failed: {exc}")
        return

    urls: list[str] = []
    try:
        urls = await yandex_reverse_search(
            crop, max_results=max_urls, headless=not show_browser
        )
        print(f"  reverse-search returned {len(urls)} URL(s)")
    except Exception as exc:  # noqa: BLE001
        print(f"  [reverse-search errored] {type(exc).__name__}: {exc}")
    snap("reverse-search")

    # Force the scraper stage even with no reverse-search hit, so Chromium's
    # footprint is still measured, by falling back to --scraper-url if given.
    target_urls = urls or ([scraper_url] if scraper_url else [])
    if not target_urls:
        print("  [scraper skipped] no URLs (pass --scraper-url to force this stage)")
        return
    try:
        from iris.src.scraper.scraper import ft_scraper

        await ft_scraper(target_urls)
    except Exception as exc:  # noqa: BLE001
        print(f"  [scraper errored] {type(exc).__name__}: {exc}")
    snap("scraper")


def print_report(stages: list[tuple[str, int]], peak: int) -> None:
    print()
    print("=== RAM by stage (process tree RSS) ===")
    print(f"  {'stage':<16} {'total':>10} {'delta':>11}")
    prev = None
    for label, rss in stages:
        delta = "" if prev is None else f"{mb(rss - prev):+9.1f}M"
        print(f"  {label:<16} {mb(rss):9.1f}M {delta:>11}")
        prev = rss
    print(f"  {'-' * 39}")
    print(f"  peak tree RSS  : {mb(peak):9.1f}M")
    print()
    print("  Note: RSS counts shared pages per process, so the tree total is an")
    print("  upper bound. Compare 'peak' against usable RAM (a 4 GB Pi 5 leaves")
    print("  ~3.7 GB after the OS) to judge headroom.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage RAM benchmark for the Pi 5.")
    parser.add_argument("--image", type=Path, required=True,
                        help="Image to run the pipeline on (a portrait with a clear face).")
    parser.add_argument("--imgsz", type=int, default=320,
                        help="Model input side length (default: 320).")
    parser.add_argument("--detect-only", action="store_true",
                        help="Skip the reverse-search and scraper (Chromium) stages.")
    parser.add_argument("--show-browser", action="store_true",
                        help="Run Chromium visible instead of headless (debug).")
    parser.add_argument("--max-urls", type=int, default=DEFAULT_MAX_URLS,
                        help=f"Max reverse-search URLs to scrape (default: {DEFAULT_MAX_URLS}).")
    parser.add_argument("--scraper-url", default=None,
                        help="Force the scraper stage on this URL if reverse-search finds nothing.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model weights not found at {MODEL_PATH}\n"
            f"Place yolov11n-face.pt in {MODEL_PATH.parent}/ before benchmarking."
        )
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    # Load .env so the scraper stage can see OPENAI_API_KEY.
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:  # noqa: BLE001 - dotenv is optional for detector-only runs
        pass

    proc = psutil.Process()
    sampler = PeakSampler(proc)
    sampler.start()

    stages: list[tuple[str, int]] = []

    def snap(label: str) -> None:
        stages.append((label, tree_rss(proc)))

    try:
        snap("baseline")

        print(f">> loading model from {MODEL_PATH}")
        detector = FaceDetector(model_path=MODEL_PATH, imgsz=args.imgsz)
        snap("model loaded")

        # First inference is the heaviest (graph compile, allocator init);
        # measure it apart from steady-state detection.
        print(">> warm-up inference...")
        detector.detect(image)
        snap("warm-up")

        print(">> detecting...")
        faces = detector.detect(image)
        print(f">> detected {len(faces)} face(s)")
        snap(f"detection ({len(faces)} faces)")

        if not args.detect_only:
            if not faces:
                print("  [network stages skipped] no face detected to search")
            else:
                print(">> running network stages (Chromium)...")
                asyncio.run(
                    _network_stages(
                        faces[0].crop,
                        snap,
                        max_urls=args.max_urls,
                        show_browser=args.show_browser,
                        scraper_url=args.scraper_url,
                    )
                )
    finally:
        sampler.stop()
        sampler.join()

    print_report(stages, sampler.peak)


if __name__ == "__main__":
    main()
