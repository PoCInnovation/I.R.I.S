"""Targeted site crawler: find pages containing a specific person or image.

Unlike the reverse_search backends (Yandex, PimEyes), which query the whole
web given a face crop, this crawls one known site breadth-first looking for
matches among the <img> tags it finds. Useful when you already suspect a
face (or a specific image file) appears somewhere on a given domain (e.g. a
company or school site) and want to find which page.

Two matching modes:
  * image (default) — perceptual hash (pHash). Fast, but only catches the
    *same image file*, possibly resized/recompressed. Won't match a
    different photo of the same person.
  * face — face_recognition embeddings. Slower, but matches the same
    *person* across different photos (different angle, lighting, crop).

Usage:
    python -m iris.src.scripts.crawler TARGET_IMAGE START_URL [--mode image|face] [--max-pages N] [--tolerance N]
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from collections.abc import Iterator
from io import BytesIO
from urllib.parse import urljoin, urlparse

import imagehash
import requests
from bs4 import BeautifulSoup
from PIL import Image

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
PAGE_TIMEOUT_S = 10
IMAGE_TIMEOUT_S = 5
CRAWL_DELAY_S = 2

# face_recognition's own default tolerance for "same person" (Euclidean
# distance between 128-d embeddings). Lower = stricter.
FACE_MATCH_TOLERANCE = 0.6
# imagehash pHash distance default: 0 = identical, 64 = max possible.
IMAGE_MATCH_TOLERANCE = 8


def _same_site(candidate_netloc: str, site_domain: str) -> bool:
    """True if candidate belongs to site_domain or a subdomain of it.

    A plain substring check (`site_domain in candidate_netloc`) would also
    match unrelated hosts like "site_domain.evil.com" or "evilsite_domain.com",
    letting the crawl escape the intended site.
    """
    candidate = candidate_netloc.lower().removeprefix("www.")
    return candidate == site_domain or candidate.endswith(f".{site_domain}")


def _crawl_images(start_url: str, max_pages: int) -> Iterator[tuple[str, bytes]]:
    """Breadth-first crawl of `start_url`'s site, yielding (image_url, image_bytes).

    Shared by both matching modes so the BFS/scoping logic isn't duplicated.
    """
    site_domain = urlparse(start_url).netloc.lower().removeprefix("www.")
    session = requests.Session()
    session.headers.update(HEADERS)

    visited: set[str] = set()
    queue: deque[str] = deque([start_url])

    while queue and len(visited) < max_pages:
        url = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        print(f"[crawler] scanning {url}")

        try:
            resp = session.get(url, timeout=PAGE_TIMEOUT_S)
        except Exception as e:
            print(f"[crawler] request failed: {e}")
            continue

        if resp.status_code != 200:
            print(f"[crawler] blocked or unavailable ({resp.status_code})")
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        for img in soup.find_all("img"):
            src = img.get("src")
            if not src or src.endswith(".svg") or src.startswith("data:"):
                continue
            full_url = urljoin(url, src)

            try:
                img_resp = session.get(full_url, timeout=IMAGE_TIMEOUT_S)
                content_type = img_resp.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    continue
            except Exception:
                continue

            yield full_url, img_resp.content

        for a in soup.find_all("a", href=True):
            next_url = urljoin(url, a["href"])
            if _same_site(urlparse(next_url).netloc, site_domain) and next_url not in visited:
                queue.append(next_url)

        time.sleep(CRAWL_DELAY_S)


def crawl_for_matching_images(
    target_image_path: str,
    start_url: str,
    max_pages: int = 10,
    tolerance_threshold: int = IMAGE_MATCH_TOLERANCE,
) -> list[str]:
    """Find same-file (possibly resized/recompressed) images via pHash.

    Returns image URLs whose perceptual hash is within `tolerance_threshold`
    of the target image's hash (lower = stricter match). Does NOT match a
    different photo of the same person — use `crawl_for_matching_faces` for
    that.
    """
    try:
        target_hash = imagehash.phash(Image.open(target_image_path))
    except Exception as e:
        print(f"[crawler] could not read target image: {e}")
        return []

    matches: list[str] = []
    for full_url, content in _crawl_images(start_url, max_pages):
        try:
            img_hash = imagehash.phash(Image.open(BytesIO(content)))
            if (target_hash - img_hash) <= tolerance_threshold:
                print(f"[crawler] match: {full_url}")
                matches.append(full_url)
        except Exception:
            pass

    print(f"[crawler] done: {len(matches)} matching image(s)")
    return matches


def crawl_for_matching_faces(
    target_image_path: str,
    start_url: str,
    max_pages: int = 10,
    tolerance: float = FACE_MATCH_TOLERANCE,
) -> list[str]:
    """Find the same PERSON across different photos, via face embeddings.

    Unlike `crawl_for_matching_images`, this recognizes the person even in
    a different photo (different angle, lighting, crop) — it does not
    require the same image file. Requires the `face_recognition` package
    (dlib-backed).

    Returns image URLs containing a face within `tolerance` (Euclidean
    distance between 128-d embeddings; lower = stricter, 0.6 is
    face_recognition's own default for "same person").
    """
    import face_recognition  # heavy/optional import, only needed for this mode

    target_image = face_recognition.load_image_file(target_image_path)
    target_encodings = face_recognition.face_encodings(target_image)
    if not target_encodings:
        print("[crawler] no face detected in target image")
        return []
    target_encoding = target_encodings[0]

    matches: list[str] = []
    for full_url, content in _crawl_images(start_url, max_pages):
        try:
            image = face_recognition.load_image_file(BytesIO(content))
            encodings = face_recognition.face_encodings(image)
        except Exception:
            continue

        for encoding in encodings:
            distance = face_recognition.face_distance([target_encoding], encoding)[0]
            if distance <= tolerance:
                print(f"[crawler] match: {full_url} (distance={distance:.3f})")
                matches.append(full_url)
                break

    print(f"[crawler] done: {len(matches)} matching face(s)")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_image", help="Path to the reference photo to match against")
    parser.add_argument("start_url", help="Site to crawl (breadth-first, same-domain only)")
    parser.add_argument("--mode", choices=["image", "face"], default="image")
    parser.add_argument("--max-pages", type=int, default=50)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Match threshold. pHash distance for --mode image (default 8), "
        "embedding distance for --mode face (default 0.6). Lower = stricter.",
    )
    args = parser.parse_args()

    if args.mode == "face":
        crawl_for_matching_faces(
            args.target_image,
            args.start_url,
            max_pages=args.max_pages,
            tolerance=args.tolerance if args.tolerance is not None else FACE_MATCH_TOLERANCE,
        )
    else:
        crawl_for_matching_images(
            args.target_image,
            args.start_url,
            max_pages=args.max_pages,
            tolerance_threshold=int(args.tolerance) if args.tolerance is not None else IMAGE_MATCH_TOLERANCE,
        )


if __name__ == "__main__":
    main()
