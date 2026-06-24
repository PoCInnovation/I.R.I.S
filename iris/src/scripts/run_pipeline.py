"""End-to-end IRIS pipeline runner.

Two input modes:
  * webcam (default) — open the camera, press `s` to run the pipeline
    on every face currently in frame, `q` to quit.
  * --image PATH — skip the camera, run the pipeline on a single image
    file. Useful for testing with someone else's portrait.

For every detected face:
    detect -> reverse-search (Yandex) -> scraper (Playwright + ChatGPT)

Usage:
    python -m iris.src.scripts.run_pipeline [--camera N] [--show-browser] [--max-urls K]
    python -m iris.src.scripts.run_pipeline --image path/to/portrait.jpg
"""

from __future__ import annotations

import argparse
import asyncio
import pprint
import time
from pathlib import Path

import cv2
import subprocess
import numpy as np
from dotenv import load_dotenv
import json

from iris.src.detection.face_detector import Face, FaceDetector
from iris.src.reverse_search import yandex_reverse_search
from iris.src.scraper.scraper import ft_scraper

## Commande et lien pour récup le flux hololens
ffmpeg_path = r"C:\ffmpeg-2026-06-08-git-6028720d70-full_build\ffmpeg-2026-06-08-git-6028720d70-full_build\bin\ffmpeg.exe"
cmd = [
    ffmpeg_path,
    "-i", "https://Iris:Iris2026*@192.168.1.14/api/holographic/stream/live.mp4",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-"
]
pipe = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL
)
width, height = 1280, 720
frame_size = width * height * 3
## Fin commande et lien hololens

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"
DEFAULT_MAX_URLS_PER_FACE = 5


async def run_pipeline_on_crops(
    crops: list[np.ndarray],
    *,
    max_urls: int,
    show_browser: bool,
) -> None:
    """Reverse-search each crop, then feed the URLs to the scraper.

    Sequential by design: Yandex tolerates a single stealth session
    better than three in parallel, and the scraper already fans out
    per-URL internally.
    """
    for i, crop in enumerate(crops, start=1):
        print(f"\n--- Face {i}/{len(crops)} ---")
        if crop.size == 0:
            print("  empty crop (bbox at frame edge), skipping.")
            continue

        # Isolate each face: a failure on face #1 must not skip #2 and #3.
        try:
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


def image_mode(image_path: Path, max_urls: int, show_browser: bool) -> None:
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
        run_pipeline_on_crops(crops, max_urls=max_urls, show_browser=show_browser)
    )


def webcam_loop(camera: int, max_urls: int, show_browser: bool) -> None:
    print(">> loading model...")
    detector = FaceDetector(model_path=MODEL_PATH, imgsz=192)

    print(f">> opening camera (index={camera})...")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open webcam (index {camera})")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps_smoothed = 0.0
    try:
        while True:
            raw = pipe.stdout.read(frame_size)
            if len(raw) != frame_size:
                print("Failed to read frame from webcam")
                break

            frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3)).copy()
            faces = detector.detect(frame)
            # Snapshot crops BEFORE drawing HUD so the pipeline sees clean pixels.
            crops = [face.crop for face in faces]

            draw_hud(frame, faces, fps_smoothed)
            cv2.imshow("I.R.I.S - pipeline", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not crops:
                    print(">> no face in frame, nothing to search.")
                    continue
                print(f">> running pipeline on {len(crops)} face(s)...")
                # Release the camera while the (slow) pipeline runs.
                cap.release()
                cv2.destroyAllWindows()
                asyncio.run(
                    run_pipeline_on_crops(
                        crops, max_urls=max_urls, show_browser=show_browser
                    )
                )
                return
    finally:
        cap.release()
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
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run Yandex's Chromium in visible mode (debug).",
    )
    parser.add_argument(
        "--max-urls",
        type=int,
        default=DEFAULT_MAX_URLS_PER_FACE,
        help=f"Max URLs per face to forward to the scraper (default: {DEFAULT_MAX_URLS_PER_FACE}).",
    )
    args = parser.parse_args()

    if args.image is not None:
        image_mode(args.image, args.max_urls, args.show_browser)
    else:
        webcam_loop(args.camera, args.max_urls, args.show_browser)


if __name__ == "__main__":
    main()
