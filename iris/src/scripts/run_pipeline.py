"""End-to-end IRIS pipeline runner.

Input modes:
  * live (default) — read frames from a source, press `s` to run the pipeline
    on every face currently in frame, `q` to quit. Two sources:
      - hololens (default): the HoloLens video stream, decoded by ffmpeg.
        Configured via .env (see IRIS_HOLOLENS_URL below).
      - webcam: a local camera (--source webcam [--camera N]).
  * --image PATH — skip the live source, run the pipeline on a single image
    file. Useful for testing with someone else's portrait.

For every detected face:
    detect -> reverse-search (Yandex, PimEyes, or Social, --backend) -> scraper (Playwright + ChatGPT)

HoloLens is configured through environment variables (loaded from .env):
    IRIS_HOLOLENS_URL      full stream URL, may embed credentials, e.g.
                           https://user:pass@192.168.1.14/api/holographic/stream/live.mp4
    IRIS_FFMPEG_PATH       ffmpeg binary (default: "ffmpeg", i.e. on PATH)
    IRIS_HOLOLENS_WIDTH    stream width in px  (default: 1280)
    IRIS_HOLOLENS_HEIGHT   stream height in px (default: 720)

Usage:
    python -m iris.src.scripts.run_pipeline                      # HoloLens (default)
    python -m iris.src.scripts.run_pipeline --source webcam --camera 0
    python -m iris.src.scripts.run_pipeline --image path/to/portrait.jpg
    python -m iris.src.scripts.run_pipeline --source webcam --backend social
    python -m iris.src.scripts.run_pipeline --source webcam --backend pimeyes --show-browser
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pprint
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from iris.src.detection.face_detector import Face, FaceDetector
from iris.src.reverse_search import pimeyes_reverse_search, social_reverse_search, yandex_reverse_search
from iris.src.scraper.scraper import ft_scraper

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"
DEFAULT_MAX_URLS_PER_FACE = 5
REVERSE_SEARCH_BACKENDS = ("yandex", "pimeyes", "social")


async def run_pipeline_on_crops(
    crops: list[np.ndarray],
    *,
    max_urls: int,
    show_browser: bool,
    backend: str = "yandex",
    auto_only: bool = False,
) -> None:
    """Reverse-search each crop, then feed the URLs to the scraper.

    Sequential by design: both backends tolerate a single stealth/browser
    session better than several in parallel, and the scraper already fans
    out per-URL internally.
    """
    for i, crop in enumerate(crops, start=1):
        print(f"\n--- Face {i}/{len(crops)} ---")
        if crop.size == 0:
            print("  empty crop (bbox at frame edge), skipping.")
            continue

        # Isolate each face: a failure on face #1 must not skip #2 and #3.
        try:
            if backend == "pimeyes":
                urls = await pimeyes_reverse_search(
                    crop, max_results=max_urls, headless=not show_browser, auto_only=auto_only
                )
            elif backend == "social":
                urls = await social_reverse_search(
                    crop, max_results=max_urls, headless=not show_browser
                )
            else:
                urls = await yandex_reverse_search(
                    crop, max_results=max_urls, headless=not show_browser
                )
        except Exception as e:
            print(f"  reverse search crashed: {e}")
            continue
        if not urls:
            print("  no reverse-search match.")
            continue

        print(f"  {len(urls)} candidate URL(s):")
        for u in urls:
            print(f"    - {u}")

        try:
            result = await ft_scraper(urls)
        except Exception as e:
            print(f"  scraper crashed: {e}")
            continue
        print("\n  OSINT profile:")
        if isinstance(result, dict):
            pprint.pprint(result, indent=4, width=100)
            target_path = PROJECT_ROOT / "last_target.json"
            try:
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=4)
                print(f"Profile saved in: {target_path}")
            except Exception as e:
                print(f"Error saving JSON: {e}")
        else:
            print(f"    {result}")


def draw_hud(frame: np.ndarray, faces: list[Face], fps: float) -> None:
    """Corner-bracket face overlay + FPS + key hints."""
    for face in faces:
        L = min(face.width, face.height) // 4
        color = (0, 0, 240)
        cv2.line(frame, (face.x1, face.y1), (face.x1 + L, face.y1), color, 2)
        cv2.line(frame, (face.x1, face.y1), (face.x1, face.y1 + L), color, 2)
        cv2.line(frame, (face.x2, face.y1), (face.x2 - L, face.y1), color, 2)
        cv2.line(frame, (face.x2, face.y1), (face.x2, face.y1 + L), color, 2)
        cv2.line(frame, (face.x1, face.y2), (face.x1 + L, face.y2), color, 2)
        cv2.line(frame, (face.x1, face.y2), (face.x1, face.y2 - L), color, 2)
        cv2.line(frame, (face.x2, face.y2), (face.x2 - L, face.y2), color, 2)
        cv2.line(frame, (face.x2, face.y2), (face.x2, face.y2 - L), color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 0), 2)
    cv2.putText(frame, "s = run pipeline    q = quit", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def image_mode(
    image_path: Path, max_urls: int, show_browser: bool, backend: str, auto_only: bool
) -> None:
    """Run the pipeline on every face found in a single image file."""
    print(f">> loading image {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    print(">> loading model...")
    detector = FaceDetector(model_path=MODEL_PATH, imgsz=320)
    faces = detector.detect(image)
    print(f">> detected {len(faces)} face(s) in {image_path.name}")
    if not faces:
        return

    crops = [face.crop for face in faces]
    asyncio.run(
        run_pipeline_on_crops(
            crops, max_urls=max_urls, show_browser=show_browser, backend=backend, auto_only=auto_only
        )
    )


# --- Frame sources -------------------------------------------------------
# Both are lazy generators: the ffmpeg subprocess / camera is opened only once
# iteration starts, never at import. That keeps --image mode and plain module
# imports side-effect free (the old code spawned ffmpeg at import time). Each
# source releases its resource in a finally block, including on .close().

HOLOLENS_URL_ENV = "IRIS_HOLOLENS_URL"


def _hololens_frames(
    url: str, ffmpeg_path: str, width: int, height: int
) -> Iterator[np.ndarray]:
    """Yield BGR frames from the HoloLens stream, decoded by ffmpeg.

    ffmpeg re-encodes the stream to raw bgr24; we read one full frame's worth
    of bytes per iteration off its stdout. A short read means the stream ended
    or was truncated, so we stop.
    """
    cmd = [ffmpeg_path, "-i", url, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise SystemExit(
            f"ffmpeg not found at {ffmpeg_path!r}. Install ffmpeg or set "
            f"IRIS_FFMPEG_PATH in your .env to its full path."
        )

    frame_size = width * height * 3
    try:
        while True:
            raw = pipe.stdout.read(frame_size)
            if len(raw) != frame_size:
                print("HoloLens stream ended (or frame truncated).")
                break
            yield np.frombuffer(raw, np.uint8).reshape((height, width, 3)).copy()
    finally:
        pipe.terminate()
        try:
            pipe.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pipe.kill()


def _webcam_frames(camera: int) -> Iterator[np.ndarray]:
    """Yield BGR frames from a local webcam via OpenCV."""
    print(f">> opening camera (index={camera})...")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open webcam (index {camera})")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break
            yield frame
    finally:
        cap.release()


def _hololens_frames_from_env() -> Iterator[np.ndarray]:
    """Build the HoloLens frame source from environment (.env) settings."""
    url = os.environ.get(HOLOLENS_URL_ENV)
    if not url:
        raise SystemExit(
            f"{HOLOLENS_URL_ENV} is not set. Add it to your .env, e.g.\n"
            f"  {HOLOLENS_URL_ENV}=https://user:pass@192.168.1.14"
            f"/api/holographic/stream/live.mp4\n"
            f"or run with --source webcam."
        )
    ffmpeg_path = os.environ.get("IRIS_FFMPEG_PATH", "ffmpeg")
    width = int(os.environ.get("IRIS_HOLOLENS_WIDTH", "1280"))
    height = int(os.environ.get("IRIS_HOLOLENS_HEIGHT", "720"))
    print(">> connecting to HoloLens stream via ffmpeg...")
    return _hololens_frames(url, ffmpeg_path, width, height)


def live_loop(
    frames: Iterator[np.ndarray],
    *,
    max_urls: int,
    show_browser: bool,
    backend: str = "yandex",
    auto_only: bool = False,
) -> None:
    """Detect faces on a live frame stream. `s` runs the pipeline, `q` quits."""
    print(">> loading model...")
    detector = FaceDetector(model_path=MODEL_PATH, imgsz=192)

    fps_smoothed = 0.0
    try:
        for frame in frames:
            t0 = time.perf_counter()
            faces = detector.detect(frame)
            # Snapshot crops BEFORE drawing HUD so the pipeline sees clean pixels.
            crops = [face.crop for face in faces]

            draw_hud(frame, faces, fps_smoothed)
            cv2.imshow("I.R.I.S - pipeline", frame)

            elapsed = time.perf_counter() - t0
            inst = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * inst

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not crops:
                    print(">> no face in frame, nothing to search.")
                    continue
                print(f">> running pipeline on {len(crops)} face(s)...")
                # Stop the capture (free camera/ffmpeg) while the slow pipeline runs.
                frames.close()
                cv2.destroyAllWindows()
                asyncio.run(
                    run_pipeline_on_crops(
                        crops,
                        max_urls=max_urls,
                        show_browser=show_browser,
                        backend=backend,
                        auto_only=auto_only,
                    )
                )
                return
    finally:
        frames.close()
        cv2.destroyAllWindows()


def main() -> None:
    # Load .env from project root before any module reads env vars.
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run the IRIS pipeline.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Run the pipeline on this image file instead of the webcam.",
    )
    parser.add_argument(
        "--source",
        choices=["hololens", "webcam"],
        default="hololens",
        help="Live frame source (default: hololens). Ignored when --image is given.",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam index for --source webcam (default: 0)."
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run the reverse-search browser in visible mode (debug).",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=DEFAULT_MAX_URLS_PER_FACE,
        help=f"Max URLs per face to forward to the scraper (default: {DEFAULT_MAX_URLS_PER_FACE}).",
    )
    parser.add_argument(
        "--backend",
        choices=REVERSE_SEARCH_BACKENDS,
        default="yandex",
        help="Reverse-search backend to use (default: yandex). "
        "'social' filters Yandex results to known social-media domains.",
    )
    parser.add_argument(
        "--auto-only",
        action="store_true",
        help="pimeyes backend only: never fall back to the VNC human CAPTCHA "
        "handoff, give up after automated retries instead.",
    )
    args = parser.parse_args()

    if args.image is not None:
        image_mode(args.image, args.max_urls, args.show_browser, args.backend, args.auto_only)
        return

    if args.source == "webcam":
        frames = _webcam_frames(args.camera)
    else:
        frames = _hololens_frames_from_env()
    live_loop(
        frames,
        max_urls=args.max_urls,
        show_browser=args.show_browser,
        backend=args.backend,
        auto_only=args.auto_only,
    )


if __name__ == "__main__":
    main()
