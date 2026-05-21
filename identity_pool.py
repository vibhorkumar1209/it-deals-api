"""User-agent rotation, delays, domain rate limiting."""

import asyncio
import functools
import logging
import random
import time
from collections import defaultdict
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)

USER_AGENT_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
]

REFERER_POOL = [
    "https://www.google.com/",
    "https://www.bing.com/",
    "https://www.google.com/",
    "https://www.bing.com/",
    None,  # 20% of the time no referer
]

DELAY_PROFILE = {
    "static":    (1.5, 3.5),
    "js":        (3.0, 6.0),
    "linkedin":  (4.0, 8.0),
    "after_429": [60, 120, 240],
}

MAX_REQUESTS_PER_DOMAIN = 5
DOMAIN_WINDOW_SECONDS = 60


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
            # Remove hits older than window
            hits[:] = [t for t in hits if now - t < DOMAIN_WINDOW_SECONDS]
            if len(hits) >= MAX_REQUESTS_PER_DOMAIN:
                oldest = hits[0]
                wait_time = DOMAIN_WINDOW_SECONDS - (now - oldest) + 0.5
                logger.debug(f"Rate limiting {domain}: sleeping {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                hits[:] = [t for t in hits if time.monotonic() - t < DOMAIN_WINDOW_SECONDS]
            hits.append(time.monotonic())

        lo, hi = DELAY_PROFILE.get(url_type, (1.5, 3.5))
        await asyncio.sleep(random.uniform(lo, hi))


@functools.lru_cache(maxsize=200)
def can_fetch(domain: str, path: str, user_agent: str = "*") -> bool:
    rp = RobotFileParser()
    robots_url = f"https://{domain}/robots.txt"
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, f"https://{domain}{path}")
    except Exception:
        return True  # if robots.txt unreadable, assume allowed


identity_pool = IdentityPool()
