"""Streaming pipeline — yields deal batches of N as they are found."""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Any

from config_loader import ScraperConfig
from search_engine import discover_all_urls
from website_router import fetch_url, classify_url
from nlp_extractor import build_deal_record
from deduplicator import deduplicate, sort_by_recency_confidence

logger = logging.getLogger(__name__)

SEM = asyncio.Semaphore(5)
MAX_URLS = 300

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
            text, html, final_type = await fetch_url(url, config.proxy_pool or None, config)
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
    Async generator. Yields dicts:
      { "type": "progress", "message": str }
      { "type": "batch",    "deals": [...], "total_so_far": int }
      { "type": "complete", "total": int, "failures": int, "summary": {...} }
    """
    yield {"type": "progress", "message": "Discovering URLs across search engines and sources..."}

    all_urls = await discover_all_urls(config)
    all_urls = all_urls[:MAX_URLS]

    yield {"type": "progress", "message": f"Found {len(all_urls)} URLs to analyse. Extracting deals..."}

    failures: list[dict] = []
    seen_hashes: set[str] = set()
    buffer: list[dict] = []
    total_emitted = 0

    tasks = [_process_url(url, config, failures) for url in all_urls]

    for coro in asyncio.as_completed(tasks):
        deals = await coro
        for deal in deals:
            # Light dedup on the fly
            import hashlib
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

    # Flush remainder
    if buffer:
        total_emitted += len(buffer)
        yield {
            "type": "batch",
            "deals": buffer,
            "total_so_far": total_emitted,
        }

    # Build summary
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
