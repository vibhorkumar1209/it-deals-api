"""Streaming pipeline — yields deal batches of N as they are found."""

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

from config_loader import ScraperConfig
from search_engine import discover_all_urls
from website_router import fetch_url, classify_url
from nlp_extractor import build_deal_record

logger = logging.getLogger(__name__)

SEM = asyncio.Semaphore(8)
MAX_URLS = 100
URL_TIMEOUT = 12          # seconds hard kill per URL
HEARTBEAT_EVERY = 10      # emit progress every N URLs processed

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


async def _process_url(url: str, config: ScraperConfig, failures: list) -> list[dict]:
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

    try:
        all_urls = await asyncio.wait_for(discover_all_urls(config), timeout=90)
    except asyncio.TimeoutError:
        all_urls = []
        yield {"type": "progress", "message": "⚠️ Discovery timed out — using known sources only."}

    all_urls = all_urls[:MAX_URLS]
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

    tasks = [_safe(url) for url in all_urls]

    for coro in asyncio.as_completed(tasks):
        deals = await coro
        done_count += 1

        for deal in deals:
            scope = (deal.get("scope_of_service") or "")[:50]
            key = "|".join([
                deal.get("company_name", "").lower(),
                deal.get("vendor", "").lower(),
                deal.get("announcement_date", "")[:7],
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
