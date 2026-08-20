"""Reverse-image face search targeting social media profiles.

Thin filter on top of `yandex_reverse_search`: runs the same face-crop
search, then keeps only URLs on known social-media / profile domains
(LinkedIn, Instagram, X/Twitter, YouTube, etc). More targeted than the
general Yandex backend when the goal is specifically "which social
profiles does this face appear on", not any matching page.

Deliberately reuses Yandex rather than Google Lens/Images: Google
restricts reverse image search from matching a photo to other photos of
the same person for privacy reasons, so a face-search backend built on it
would systematically underperform on the one thing this module needs to
do. Yandex has no such restriction (see yandex_search.py).
"""

from __future__ import annotations

from urllib.parse import urlparse

import numpy as np

from .yandex_search import yandex_reverse_search

# The filter is intentionally broad — we'd rather over-include and let the
# scraper + ChatGPT sort out relevance than miss a profile because the
# domain wasn't on the list.
SOCIAL_DOMAINS = frozenset({
    # Big social
    "linkedin.com",
    "instagram.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "tiktok.com",
    "pinterest.com",
    "tumblr.com",
    "reddit.com",
    "medium.com",
    "snapchat.com",
    "threads.net",
    # Video / streaming
    "youtube.com",
    "vimeo.com",
    "twitch.tv",
    # Photo / creative
    "flickr.com",
    "deviantart.com",
    "500px.com",
    "unsplash.com",
    # Professional / portfolio
    "github.com",
    "behance.net",
    "dribbble.com",
    "canva.com",
    # Blog / personal site (often host profile pics)
    "blogspot.com",
    "wordpress.com",
    "substack.com",
})


def _is_social_url(url: str) -> bool:
    """True if the URL's domain is (or is a subdomain of) a known social domain."""
    netloc = urlparse(url).netloc.lower().removeprefix("www.")
    if netloc in SOCIAL_DOMAINS:
        return True
    return any(netloc.endswith(f".{domain}") for domain in SOCIAL_DOMAINS)


async def social_reverse_search(
    face_crop: np.ndarray,
    *,
    max_results: int = 15,
    headless: bool = True,
    timeout_ms: int = 30_000,
) -> list[str]:
    """Find social-media profile URLs where a face crop appears.

    Runs a Yandex reverse-image search and keeps only results on known
    social-media domains. Empty list on any failure or simply no social
    match — same contract as the other reverse-search backends.
    """
    # Pull more raw results than we need since most will be filtered out.
    urls = await yandex_reverse_search(
        face_crop,
        max_results=max(max_results * 4, 40),
        headless=headless,
        timeout_ms=timeout_ms,
    )

    social_urls = [u for u in urls if _is_social_url(u)][:max_results]

    if social_urls:
        print(f"  [social] found {len(social_urls)} social media match(es)")
    else:
        print("  [social] no social media matches found")

    return social_urls
