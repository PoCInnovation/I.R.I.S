"""Result records for a PimEyes search — the image <-> site correlation.

PimEyes' paywall is narrower than it looks. "Unlock results" buys the *page*
URL; the free results grid already renders, for every match, the photo itself
and the **domain** it was found on (`numerama.com`, `en.charenteperigord.fr`, …)
as plain text on the tile. Photo + domain is exactly the input
`iris/src/scripts/crawler.py` wants: it walks one site and pHash-matches its way
to the page the photo actually lives on, recovering the URL the paywall hides.

This module holds the record type and the pure functions around it — no browser,
no I/O beyond writing the manifest — so it can be exercised without launching
Camoufox. The scraping half lives in `pimeyes_search.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

# What a tile's domain label looks like. Deliberately strict: the same button row
# also holds a localized "More" label, and matching a bare word would turn that
# into a crawl target.
DOMAIN_RE = re.compile(r"^(?=.{4,253}$)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
                       r"(?:\.[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)+$", re.IGNORECASE)

MANIFEST_NAME = "matches.json"


@dataclass(frozen=True)
class PimEyesMatch:
    """One match from a PimEyes search: an image and the site it was found on."""

    index: int = 0                  # position in the grid, i.e. PimEyes' relevance order
    domain: str = ""                # exactly as shown on the tile, e.g. "numerama.com"
    site: str | None = None         # crawlable origin, e.g. "https://numerama.com"
    thumbnail_url: str = ""         # usually a blob: URL — only valid in the live page
    local_path: Path | None = None  # thumbnail on disk: the crawler's target image

    @property
    def usable(self) -> bool:
        """True when this record can actually be handed to the crawler."""
        return self.site is not None and self.local_path is not None


def normalize_site(domain: str) -> str | None:
    """Turn a tile's domain label into an origin the crawler can start from.

    Accepts a bare host ("numerama.com") or a full URL, and returns None for
    anything that isn't a plausible hostname — button labels like "More", or a
    host with no TLD — so a junk crawl target is never emitted.
    """
    if not domain:
        return None
    text = domain.strip()
    parsed = urlparse(text if "//" in text else f"//{text}", scheme="https")
    host = (parsed.netloc or "").strip().rstrip(".").lower()
    if not host or not DOMAIN_RE.match(host):
        return None
    tld = host.rsplit(".", 1)[-1]
    if not tld.isalpha() or len(tld) < 2:
        return None
    scheme = parsed.scheme if parsed.scheme in ("http", "https") else "https"
    return f"{scheme}://{host}"


def with_local_path(match: PimEyesMatch, path: Path | None) -> PimEyesMatch:
    """Frozen-dataclass-friendly way to attach a saved thumbnail."""
    return replace(match, local_path=path)


def by_image(matches: list[PimEyesMatch]) -> dict[str, str]:
    """{thumbnail path on disk: site} — the dictionary the crawler consumes."""
    return {str(m.local_path): m.site for m in matches if m.usable}


def by_site(matches: list[PimEyesMatch], base: Path | None = None) -> dict[str, list[str]]:
    """{site: [thumbnail paths]} — crawling is per-domain, so group before walking.

    Results cluster hard on one domain (a single search returned nine hits on
    numerama.com), and a crawl is minutes of paced requests. Grouping turns nine
    walks of the same site into one pass that checks all nine hashes.

    `base` writes the paths relative to that directory — how the manifest stores
    them, so a run directory can be moved or archived without breaking. Callers
    that want to open the files leave it None and get absolute paths.
    """
    grouped: dict[str, list[str]] = {}
    for m in matches:
        if not m.usable:
            continue
        path = m.local_path
        if base is not None:
            try:
                path = path.relative_to(base)
            except ValueError:
                pass  # thumbnail outside the run dir: keep it absolute, still usable
        grouped.setdefault(m.site, []).append(str(path))
    return grouped


def save_matches(matches: list[PimEyesMatch], out_dir: Path) -> Path:
    """Write `matches.json` beside the thumbnails so a crawl can run later.

    Keeping the map on disk means the crawl step can be re-run offline instead of
    burning another PimEyes search (and another CAPTCHA) to rebuild the same map.
    So the manifest holds exactly what that crawl needs and nothing else: each
    site to walk, and the thumbnails to hash against it, in relevance order.

    Everything else a match carries is noise here — the blob: `thumbnail_url` is
    dead once the browser closes, and `domain`/`index` are the site and the
    position they're already stored at.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / MANIFEST_NAME
    manifest.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "sites": by_site(matches, base=out_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return manifest


def load_matches(manifest: Path) -> list[PimEyesMatch]:
    """Read back a manifest written by `save_matches`.

    Only the crawlable half survives a round-trip: unusable matches were never
    written, and `thumbnail_url` comes back empty because the blob: URL it held
    stopped resolving when the browser closed.
    """
    data = json.loads(manifest.read_text(encoding="utf-8"))
    out: list[PimEyesMatch] = []
    for site, files in data.get("sites", {}).items():
        for name in files:
            out.append(
                PimEyesMatch(
                    index=len(out),
                    domain=urlparse(site).netloc,
                    site=site,
                    local_path=manifest.parent / name,
                )
            )
    return out
