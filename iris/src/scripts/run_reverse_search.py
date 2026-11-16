"""Standalone reverse-search runner — no model, no scraper, no HoloLens.

Détecte les visages dans une image (ou utilise l'image entière si aucun visage
n'est détecté), lance la recherche inversée Yandex et/ou PimEyes, et affiche
les URLs trouvées.

Usage:
    uv run python -m iris.src.scripts.run_reverse_search --image portrait.jpg
    uv run python -m iris.src.scripts.run_reverse_search --image portrait.jpg --engine pimeyes
    uv run python -m iris.src.scripts.run_reverse_search --image portrait.jpg --engine both --show-browser
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import cv2
from dotenv import load_dotenv

from iris.src.detection.face_detector import FaceDetector
from iris.src.reverse_search.pimeyes_results import by_site
from iris.src.reverse_search.yandex_search import yandex_reverse_search
from iris.src.reverse_search.pimeyes_search import pimeyes_reverse_search_matches

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "iris" / "src" / "models" / "yolov11n-face.pt"

# .env is loaded for the pipeline's other settings; the PimEyes engine here
# searches anonymously and needs no credentials.
load_dotenv(PROJECT_ROOT / ".env")


async def search(image_path: Path, engine: str, max_urls: int, show_browser: bool) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read image: {image_path}")

    if MODEL_PATH.exists():
        print(">> loading detection model...")
        detector = FaceDetector(model_path=MODEL_PATH, imgsz=320)
        faces = detector.detect(image)
        if faces:
            print(f">> {len(faces)} face(s) detected")
            crops = [face.crop for face in faces]
        else:
            print(">> no face detected — using full image as crop")
            crops = [image]
    else:
        print(f">> model not found at {MODEL_PATH}, using full image as crop")
        crops = [image]

    for i, crop in enumerate(crops, start=1):
        print(f"\n=== Face {i}/{len(crops)} ===")

        if engine in ("yandex", "both"):
            print(">> Yandex reverse search...")
            urls = await yandex_reverse_search(
                crop, max_results=max_urls, headless=not show_browser
            )
            print(f"   {len(urls)} result(s):")
            for u in urls:
                print(f"   - {u}")

        if engine in ("pimeyes", "both"):
            print(">> PimEyes reverse search...")
            # PimEyes runs headless by default (invisible virtual display) with an
            # automatic headed fallback if the CAPTCHA needs a human. --show-browser
            # forces a visible window from the start.
            matches = await pimeyes_reverse_search_matches(
                crop, max_results=max_urls, headless=not show_browser
            )
            print(f"   {len(matches)} result(s):")
            for m in matches:
                target = m.local_path or "<no thumbnail>"
                print(f"   - {target}  ->  {m.site or f'?? ({m.domain})'}")

            unusable = [m for m in matches if not m.usable]
            if unusable:
                print(f"   ({len(unusable)} unusable: no thumbnail or no valid domain)")

            sites = by_site(matches)
            if sites:
                print(f"   {len(sites)} distinct site(s) to crawl:")
                for site, images in sites.items():
                    print(f"   - {site}  ({len(images)} image(s))")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reverse-search on an image.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image file.")
    parser.add_argument(
        "--engine",
        choices=["yandex", "pimeyes", "both"],
        default="both",
        help="Reverse-search engine to use (default: both).",
    )
    parser.add_argument(
        "--max-urls", type=int, default=5, help="Max URLs per face (default: 5)."
    )
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Run browsers in visible mode (debug). Forces PimEyes to a visible window too.",
    )
    args = parser.parse_args()
    asyncio.run(search(args.image, args.engine, args.max_urls, args.show_browser))


if __name__ == "__main__":
    main()
