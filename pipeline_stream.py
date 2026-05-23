"""Streaming pipeline — yields deal batches of N as they are found."""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

from config_loader import ScraperConfig
from search_engine import discover_all_urls
from website_router import fetch_url, classify_url
from nlp_extractor import build_deal_record

logger = logging.getLogger(__name__)

SEM = asyncio.Semaphore(5)
MAX_URLS = 30
URL_TIMEOUT = 20          # seconds hard kill per URL
HEARTBEAT_EVERY = 3       # emit SSE heartbeat every N URLs (keeps Render connection alive)

URL_CACHE_DIR = "/tmp/url_cache"
os.makedirs(URL_CACHE_DIR, exist_ok=True)


def _cache_key(company_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]", "_", company_name.lower())
    return os.path.join(URL_CACHE_DIR, f"{slug}_urls.json")


def save_url_cache(company_name: str, urls: list[str]) -> None:
    try:
        with open(_cache_key(company_name), "w") as f:
            json.dump({"company": company_name, "urls": urls, "ts": time.time()}, f)
        logger.info(f"Cached {len(urls)} URLs for '{company_name}'")
    except Exception as e:
        logger.warning(f"Failed to save URL cache: {e}")


def load_url_cache(company_name: str) -> list[str] | None:
    path = _cache_key(company_name)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        age_h = (time.time() - data.get("ts", 0)) / 3600
        if age_h > 48:
            logger.info(f"URL cache for '{company_name}' is {age_h:.1f}h old — ignoring")
            return None
        return data.get("urls", [])
    except Exception as e:
        logger.warning(f"Failed to load URL cache: {e}")
        return None

SOURCE_TYPE_MAP = {
    "TYPE_1_STATIC_HTML": "news_article",
    "TYPE_2_JS_RENDERED": "news_article",
    "TYPE_3_SOFT_PAYWALL": "news_article",
    "TYPE_4_HARD_PAYWALL": "hard_paywall_partial",
    "TYPE_6_BOT_PROTECTED": "news_article",
    "TYPE_7_LINKEDIN": "linkedin_post",
    "TYPE_8_PDF": "sec_filing",
    "TYPE_10_ARCHIVE": "news_article",
}


def _infer_source_type(url: str, url_type: str) -> str:
    u = url.lower()
    if "sec.gov" in u:
        return "sec_filing"
    if any(x in u for x in ["investor", "ir.", "/press-release", "newsroom",
                              "businesswire", "prnewswire", "globenewswire"]):
        return "press_release"
    return SOURCE_TYPE_MAP.get(url_type, "news_article")


SKIP_DOMAINS = {
    "linkedin.com", "facebook.com", "instagram.com", "twitter.com", "x.com",
    "youtube.com", "tiktok.com", "reddit.com", "quora.com",
    "danelfin.com", "macrotrends.net", "stockanalysis.com", "wisesheets.io",
    "ambitionbox.com", "glassdoor.com", "indeed.com", "naukri.com",
    "scribd.com", "slideshare.net", "academia.edu",
    "amazon.com", "flipkart.com", "ebay.com",
}

def _should_skip(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lstrip("www.")
        return any(domain == d or domain.endswith("." + d) for d in SKIP_DOMAINS)
    except Exception:
        return False


async def _process_url(url: str, config: ScraperConfig, failures: list) -> list[dict]:
    if _should_skip(url):
        return []  # Silently skip — not a failure, expected behaviour
    async with SEM:
        try:
            text, html, final_type = await asyncio.wait_for(
                fetch_url(url, None, config),
                timeout=URL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            failures.append({"url": url, "failure_type": "timeout"})
            return []
        except Exception as e:
            failures.append({"url": url, "failure_type": "exception", "error": str(e)})
            return []

        if not text:
            failures.append({"url": url, "failure_type": "no_content"})
            return []

        source_type = _infer_source_type(url, final_type)
        soup = None
        if html:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                pass

        deal = build_deal_record(
            text=text, url=url, source_type=source_type,
            company_name=config.company_name,
            company_names=config.all_company_names,
            soup=soup,
        )
        if deal is None:
            return []

        if config.min_deal_value_usd_million is not None:
            dv = deal.get("deal_value_usd")
            if dv is not None and dv < config.min_deal_value_usd_million:
                return []

        return [deal]


async def stream_pipeline(
    config: ScraperConfig,
    batch_size: int = 5,
) -> AsyncGenerator[dict[str, Any], None]:

    # ── Phase 1: URL discovery ────────────────────────────────────────────────
    yield {"type": "progress", "message": "🔍 Searching across press releases, filings, and news sources..."}

    # Send periodic heartbeats so the SSE connection stays alive during discovery
    async def _discover_with_heartbeat():
        task = asyncio.create_task(discover_all_urls(config))
        while not task.done():
            try:
                return await asyncio.wait_for(asyncio.shield(task), timeout=15)
            except asyncio.TimeoutError:
                pass  # heartbeat tick — task still running
        return task.result()

    try:
        # Each strategy already has its own timeout (60/45/30s) — outer is a safety net
        discover_task = asyncio.create_task(discover_all_urls(config))
        heartbeat_interval = 10  # seconds between heartbeat yields
        elapsed = 0
        max_wait = 60   # abort discovery after 60s — proceed with whatever we have
        while not discover_task.done() and elapsed < max_wait:
            try:
                all_urls = await asyncio.wait_for(asyncio.shield(discover_task), timeout=heartbeat_interval)
                break
            except asyncio.TimeoutError:
                elapsed += heartbeat_interval
                yield {"type": "heartbeat", "message": f"⏳ Still searching… ({elapsed}s)"}
        else:
            if not discover_task.done():
                discover_task.cancel()
                all_urls = []
                yield {"type": "progress", "message": "⚠️ Discovery timed out — using known sources only."}
            else:
                all_urls = discover_task.result()
    except Exception:
        all_urls = []
        yield {"type": "progress", "message": "⚠️ Discovery error — using known sources only."}

    all_urls = all_urls[:MAX_URLS]

    # Cache for reuse
    if all_urls:
        save_url_cache(config.company_name, all_urls)

    total_urls = len(all_urls)

    if total_urls == 0:
        yield {
            "type": "complete", "total": 0, "failures": 0,
            "urls_attempted": 0,
            "summary": {"total_deals": 0, "failures": 0, "sources_attempted": 0},
        }
        return

    yield {"type": "progress", "message": f"📋 Found {total_urls} sources — scanning for IT deals..."}

    # ── Phase 2: Fetch + extract ──────────────────────────────────────────────
    failures: list[dict] = []
    seen_hashes: set[str] = set()
    buffer: list[dict] = []
    total_emitted = 0
    done_count = 0

    # Wrap with timeout so as_completed never hangs
    async def _safe(url: str) -> list[dict]:
        try:
            return await _process_url(url, config, failures)
        except Exception:
            return []

    # Wrap every task with a hard outer deadline so one stuck URL
    # can never block the entire batch beyond URL_TIMEOUT + 2s
    async def _guarded(url: str) -> list[dict]:
        try:
            return await asyncio.wait_for(_safe(url), timeout=URL_TIMEOUT + 5)
        except asyncio.TimeoutError:
            failures.append({"url": url, "failure_type": "hard_timeout"})
            return []

    tasks = [_guarded(url) for url in all_urls]

    for coro in asyncio.as_completed(tasks):
        deals = await coro
        done_count += 1

        for deal in deals:
            scope = (deal.get("scope_of_service") or "")[:50]
            vendor_raw = (deal.get("vendor") or "").lower()
            vendor_norm = re.sub(r'\b(ltd|limited|inc|corp|pvt|llc|plc|gmbh|ag|sa)\.?\b', '', vendor_raw).strip()
            key = "|".join([
                deal.get("company_name", "").lower(),
                vendor_norm,
                deal.get("record_type", "").lower(),
                (deal.get("announcement_date") or "")[:7],
                scope.lower(),
            ])
            h = hashlib.sha256(key.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            buffer.append(deal)

        # Emit full batches immediately
        while len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            buffer = buffer[batch_size:]
            total_emitted += len(batch)
            yield {"type": "batch", "deals": batch, "total_so_far": total_emitted}

        # Progress heartbeat every HEARTBEAT_EVERY URLs
        if done_count % HEARTBEAT_EVERY == 0:
            yield {
                "type": "heartbeat",
                "done": done_count,
                "total": total_urls,
                "message": (
                    f"⚡ Scanned {done_count}/{total_urls} sources"
                    + (f" — {total_emitted} deals found so far" if total_emitted else " — scanning...")
                ),
            }

    # ── Phase 3: Flush remainder + complete ───────────────────────────────────
    if buffer:
        total_emitted += len(buffer)
        yield {"type": "batch", "deals": buffer, "total_so_far": total_emitted}

    yield {
        "type": "complete",
        "total": total_emitted,
        "failures": len(failures),
        "urls_attempted": total_urls,
        "summary": {
            "total_deals": total_emitted,
            "failures": len(failures),
            "sources_attempted": total_urls,
        },
    }


async def extract_from_cached_urls(
    config: ScraperConfig,
    urls: list[str],
    batch_size: int = 5,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run extraction only on a provided list of URLs — no search phase."""
    total_urls = len(urls)

    if total_urls == 0:
        yield {"type": "error", "message": "No URLs provided."}
        return

    yield {"type": "progress", "message": f"📋 Re-extracting from {total_urls} cached URLs — scanning for IT deals..."}

    failures: list[dict] = []
    seen_hashes: set[str] = set()
    buffer: list[dict] = []
    total_emitted = 0
    done_count = 0

    async def _safe(url: str) -> list[dict]:
        try:
            return await _process_url(url, config, failures)
        except Exception:
            return []

    tasks = [_safe(url) for url in urls]

    for coro in asyncio.as_completed(tasks):
        deals = await coro
        done_count += 1

        for deal in deals:
            scope = (deal.get("scope_of_service") or "")[:50]
            key = "|".join([
                deal.get("company_name", "").lower(),
                deal.get("vendor", "").lower(),
                deal.get("record_type", "").lower(),
                deal.get("announcement_date", "")[:7],
                scope.lower(),
            ])
            h = hashlib.sha256(key.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            buffer.append(deal)

        while len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            buffer = buffer[batch_size:]
            total_emitted += len(batch)
            yield {"type": "batch", "deals": batch, "total_so_far": total_emitted}

        if done_count % HEARTBEAT_EVERY == 0:
            yield {
                "type": "heartbeat",
                "done": done_count,
                "total": total_urls,
                "message": (
                    f"⚡ Scanned {done_count}/{total_urls} sources"
                    + (f" — {total_emitted} deals found so far" if total_emitted else " — scanning...")
                ),
            }

    if buffer:
        total_emitted += len(buffer)
        yield {"type": "batch", "deals": buffer, "total_so_far": total_emitted}

    yield {
        "type": "complete",
        "total": total_emitted,
        "failures": len(failures),
        "urls_attempted": total_urls,
        "summary": {
            "total_deals": total_emitted,
            "failures": len(failures),
            "sources_attempted": total_urls,
        },
    }
