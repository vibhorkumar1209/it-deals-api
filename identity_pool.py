"""User-agent rotation, delays, domain rate limiting — API-optimised (fast mode)."""

import asyncio
import functools
import logging
import random
import time
from collections import defaultdict
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
]

REFERER_POOL = [
    "https://www.google.com/",
    "https://www.bing.com/",
    None,
]

# API mode: minimal delays — we rely on the per-URL timeout to bound total time
DELAY_PROFILE = {
    "static":   (0.1, 0.3),
    "js":       (0.2, 0.5),
    "linkedin": (1.0, 2.0),
}

# Generous rate limit — 20 req per 10s per domain
MAX_REQUESTS_PER_DOMAIN = 20
DOMAIN_WINDOW_SECONDS = 10


class IdentityPool:
    def __init__(self):
        self._ua_index = 0
        self._domain_hits: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def next_headers(self) -> dict[str, str]:
        ua = USER_AGENT_POOL[self._ua_index % len(USER_AGENT_POOL)]
        self._ua_index += 1
        referer = random.choice(REFERER_POOL)
        headers = {
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    async def wait_for_domain(self, url: str, url_type: str = "static"):
        domain = urlparse(url).netloc
        async with self._lock:
            now = time.monotonic()
            hits = self._domain_hits[domain]
            hits[:] = [t for t in hits if now - t < DOMAIN_WINDOW_SECONDS]
            if len(hits) >= MAX_REQUESTS_PER_DOMAIN:
                # Cap wait at 3s max — never block a slot for longer
                wait_time = min(DOMAIN_WINDOW_SECONDS - (now - hits[0]) + 0.1, 3.0)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                hits[:] = [t for t in hits if time.monotonic() - t < DOMAIN_WINDOW_SECONDS]
            hits.append(time.monotonic())

        lo, hi = DELAY_PROFILE.get(url_type, (0.1, 0.3))
        await asyncio.sleep(random.uniform(lo, hi))


def can_fetch(domain: str, path: str, user_agent: str = "*") -> bool:
    # Skip robots.txt checks in API mode — too slow (network call per domain)
    return True


identity_pool = IdentityPool()
