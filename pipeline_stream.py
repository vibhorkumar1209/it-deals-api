"""Streaming pipeline — yields deal batches of N as they are found."""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

from config_loader import ScraperConfig
from search_engine import discover_all_urls
from website_router import fetch_url, classify_url
from nlp_extractor import build_deal_record

logger = logging.getLogger(__name__)

SEM = asyncio.Semaphore(8)       # more concurrency on Render
MAX_URLS = 100                   # hard cap — keeps runs under 5 min
URL_TIMEOUT = 12                 # seconds per URL before giving up
HEARTBEAT_INTERVAL = 15          # send a ping every N seconds so UI stays alive

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


async def _process_url(
    url: str,
    config: ScraperConfig,
    failures: list[dict],
) -> list[dict]:
    async with SEM:
        url_type = classify_url(url)
        try:
            # Hard per-URL timeout — never blocks a slot for more than URL_TIMEOUT seconds
            text, html, final_type = await asyncio.wait_for(
                fetch_url(url, config.proxy_pool or None, config),
                timeout=URL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"TIMEOUT ({URL_TIMEOUT}s): {url[:70]}")
            failures.append({
                "url": url,
                "failure_type": "timeout",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return []
        except Exception as e:
            failures.append({
                "url": url,
                "failure_type": "exception",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            })
            return []

        if not text:
            failures.append({
                "url": url,
                "failure_type": "no_content",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
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
    """
    Async generator. Yields:
      { "type": "progress",   "message": str }
      { "type": "heartbeat",  "done": int, "total": int }
      { "type": "batch",      "deals": [...], "total_so_far": int }
      { "type": "complete",   "total": int, "failures": int, "summary": {...} }
    """
    yield {"type": "progress", "message": "Discovering URLs across search engines and sources..."}

    try:
        all_urls = await asyncio.wait_for(discover_all_urls(config), timeout=90)
    except asyncio.TimeoutError:
        all_urls = []
        yield {"type": "progress", "message": "URL discovery timed out — fetching known sources only."}
    all_urls = all_urls[:MAX_URLS]

    yield {"type": "progress", "message": f"Found {len(all_urls)} URLs — extracting deals (batch of {batch_size})..."}

    failures: list[dict] = []
    seen_hashes: set[str] = set()
    buffer: list[dict] = []
    total_emitted = 0
    done_count = 0
    last_heartbeat = asyncio.get_event_loop().time()

    tasks = {asyncio.ensure_future(_process_url(url, config, failures)): url
             for url in all_urls}

    pending = set(tasks.keys())

    while pending:
        # Wait for whichever finishes first, but re-check every HEARTBEAT_INTERVAL
        done, pending = await asyncio.wait(
            pending,
            timeout=HEARTBEAT_INTERVAL,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Process completed tasks
        for fut in done:
            done_count += 1
            try:
                deals = fut.result()
            except Exception as e:
                logger.error(f"Task error: {e}")
                deals = []

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

            while len(buffer) >= batch_size:
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]
                total_emitted += len(batch)
                yield {
                    "type": "batch",
                    "deals": batch,
                    "total_so_far": total_emitted,
                }

        # Heartbeat — fires every HEARTBEAT_INTERVAL even if no tasks finished
        now = asyncio.get_event_loop().time()
        if now - last_heartbeat >= HEARTBEAT_INTERVAL:
            last_heartbeat = now
            yield {
                "type": "heartbeat",
                "done": done_count,
                "total": len(all_urls),
                "message": f"Analysed {done_count}/{len(all_urls)} URLs — {total_emitted} deals found so far...",
            }

    # Flush remainder
    if buffer:
        total_emitted += len(buffer)
        yield {"type": "batch", "deals": buffer, "total_so_far": total_emitted}

    yield {
        "type": "complete",
        "total": total_emitted,
        "failures": len(failures),
        "urls_attempted": len(all_urls),
        "summary": {
            "total_deals": total_emitted,
            "failures": len(failures),
            "sources_attempted": len(all_urls),
        },
    }
