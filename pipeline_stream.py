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
from search_engine import discover_all_urls, strategy_rss_deals
from website_router import fetch_url, classify_url
from nlp_extractor import build_deal_record

logger = logging.getLogger(__name__)

SEM = asyncio.Semaphore(3)   # Render free: 512MB — keep concurrency low
MAX_URLS = 20                # Only fetchable domains now, so 20 is plenty
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
        logger.info(f"Fetching: {url[:100]}")
        try:
            text, html, final_type = await asyncio.wait_for(
                fetch_url(url, None, config),
                timeout=URL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"TIMEOUT: {url[:100]}")
            failures.append({"url": url, "failure_type": "timeout"})
            return []
        except Exception as e:
            logger.warning(f"EXCEPTION: {url[:100]} — {e}")
            failures.append({"url": url, "failure_type": "exception", "error": str(e)})
            return []

        if not text:
            logger.warning(f"NO_CONTENT: {url[:100]}")
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


async def _run_parallel_research(config: ScraperConfig) -> str | None:
    """Run Parallel.ai research query for the company's IT deals."""
    from parallel_search import parallel_research
    year = config.search_year_range.get("end", 2025) if isinstance(config.search_year_range, dict) else 2025
    query = (
        f"Research and list all IT technology deals, contracts, outsourcing agreements, "
        f"vendor selections, and technology partnerships involving {config.company_name} "
        f"from {year - 2} to {year}. "
        f"For each deal include: vendor/technology company name, deal type, "
        f"technology area (ERP/CRM/cloud/cybersecurity/ATM/managed services), "
        f"deal value if known, announcement date, and source URL. "
        f"Focus on signed contracts and announcements, not analyst reports or stock news."
    )
    return await parallel_research(query)


async def stream_pipeline(
    config: ScraperConfig,
    batch_size: int = 5,
) -> AsyncGenerator[dict[str, Any], None]:

    from parallel_search import (
        parallel_research, parallel_text_to_deals,
        extract_urls_from_text, is_fetchable,
    )

    # ── Phase 1A: Parallel.ai deep research (primary) ────────────────────────
    yield {"type": "progress", "message": "🔍 Researching IT deals via Parallel.ai..."}

    parallel_deals: list[dict] = []
    parallel_urls: list[str] = []

    parallel_task = asyncio.create_task(_run_parallel_research(config))

    # Heartbeat while Parallel.ai runs (up to 90s)
    elapsed = 0
    while not parallel_task.done() and elapsed < 90:
        try:
            parallel_result = await asyncio.wait_for(
                asyncio.shield(parallel_task), timeout=10
            )
            break
        except asyncio.TimeoutError:
            elapsed += 10
            yield {"type": "heartbeat", "message": f"⏳ Researching deals… ({elapsed}s)"}
    else:
        if not parallel_task.done():
            parallel_task.cancel()
            parallel_result = None
        else:
            parallel_result = parallel_task.result()

    if parallel_result:
        parallel_deals = parallel_text_to_deals(
            parallel_result, config.company_name, config.all_company_names
        )
        parallel_urls = extract_urls_from_text(parallel_result)
        yield {"type": "progress", "message": f"✅ Parallel.ai found {len(parallel_deals)} deals — now scanning news sources..."}
    else:
        yield {"type": "progress", "message": "⚠️ Parallel.ai unavailable — scanning news sources directly..."}

    # ── Phase 1B: RSS direct extraction + URL discovery (parallel) ──────────
    yield {"type": "heartbeat", "message": "⏳ Scanning RSS feeds and news sources…"}

    all_urls: list[str] = []
    rss_items: list[dict] = []

    try:
        # Tag each future so we know which result belongs to which task
        discover_fut = asyncio.ensure_future(discover_all_urls(config))
        rss_fut      = asyncio.ensure_future(strategy_rss_deals(config))
        fut_map = {discover_fut: "discover", rss_fut: "rss"}
        pending_1b = set(fut_map.keys())

        elapsed = 0
        while pending_1b and elapsed < 55:
            done_1b, pending_1b = await asyncio.wait(pending_1b, timeout=10)
            elapsed += 10
            for t in done_1b:
                if t.exception():
                    logger.warning(f"Phase 1B {fut_map[t]} error: {t.exception()}")
                    continue
                if fut_map[t] == "discover":
                    all_urls = t.result() or []
                else:
                    rss_items = t.result() or []
            if pending_1b:
                yield {"type": "heartbeat", "message": f"⏳ Scanning feeds… ({elapsed}s)"}

        for t in pending_1b:
            t.cancel()
    except Exception as e:
        logger.warning(f"Phase 1B error: {e}")

    # Merge Parallel.ai URLs + search URLs, filter to fetchable domains only
    all_candidate_urls = list(dict.fromkeys(parallel_urls + all_urls))
    fetchable_urls = [u for u in all_candidate_urls if is_fetchable(u)]

    logger.info(f"URL funnel: {len(all_candidate_urls)} candidates → {len(fetchable_urls)} fetchable | RSS items: {len(rss_items)}")

    # Only attempt URLs from domains where Jina Reader reliably works
    all_urls = fetchable_urls[:MAX_URLS]
    for u in all_urls:
        logger.info(f"  will fetch: {u[:100]}")

    if all_urls:
        save_url_cache(config.company_name, all_urls)

    total_urls = len(all_urls)
    yield {"type": "progress", "message": f"📋 Found {len(rss_items)} RSS deal items + {total_urls} articles to scan..."}

    # ── Emit Parallel.ai deals immediately ───────────────────────────────────
    PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
    seen_hashes: set[str] = set()
    buffer: list[dict] = []
    all_deals_tracked: list[dict] = []   # master list for enrichment phase
    total_emitted = 0
    failures: list[dict] = []

    def _dedup_and_buffer(deal: dict) -> bool:
        vendor_raw = (deal.get("vendor") or "").lower()
        vendor_norm = re.sub(r'\b(ltd|limited|inc|corp|pvt|llc|plc|gmbh|ag|sa)\.?\b', '', vendor_raw).strip()
        company = deal.get("company_name", "").lower()
        date_ym = (deal.get("announcement_date") or "")[:7]  # YYYY-MM
        scope = (deal.get("scope_of_service") or "")[:50].lower()

        # Primary: full key
        key = "|".join([company, vendor_norm, date_ym, scope])
        h = hashlib.sha256(key.encode()).hexdigest()
        if h in seen_hashes:
            return False

        # Secondary: same deal reported by two sources — company + vendor + month
        # (scope wording differs between ET and bfsi.eletsonline for same deal)
        if vendor_norm:
            loose_key = "|".join([company, vendor_norm, date_ym])
            lh = hashlib.sha256(loose_key.encode()).hexdigest()
            if lh in seen_hashes:
                return False
            seen_hashes.add(lh)

        seen_hashes.add(h)
        buffer.append(deal)
        all_deals_tracked.append(deal)   # track for enrichment phase
        return True

    # Buffer Parallel.ai deals
    for deal in parallel_deals:
        _dedup_and_buffer(deal)

    # ── Process RSS deal items directly (no fetch needed) ────────────────────
    rss_deal_count = 0
    for item in rss_items:
        deal = build_deal_record(
            text=item["text"],
            url=item["url"],
            source_type="news_article",
            company_name=config.company_name,
            company_names=config.all_company_names,
            soup=None,
        )
        if deal:
            if _dedup_and_buffer(deal):
                rss_deal_count += 1
    if rss_deal_count:
        logger.info(f"RSS direct extraction: {rss_deal_count} deals")

    while len(buffer) >= batch_size:
        batch = buffer[:batch_size]
        buffer = buffer[batch_size:]
        total_emitted += len(batch)
        yield {"type": "batch", "deals": batch, "total_so_far": total_emitted}

    if total_urls == 0:
        if buffer:
            total_emitted += len(buffer)
            yield {"type": "batch", "deals": buffer, "total_so_far": total_emitted}
        yield {
            "type": "complete", "total": total_emitted, "failures": 0,
            "urls_attempted": 0,
            "summary": {"total_deals": total_emitted, "failures": 0, "sources_attempted": 0},
        }
        return

    # ── Phase 2: Fetch + extract ──────────────────────────────────────────────
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

    # Convert coroutines to Tasks so asyncio.wait can track them
    pending = {asyncio.ensure_future(_guarded(url)) for url in all_urls}
    last_heartbeat = asyncio.get_event_loop().time()

    while pending:
        # Wait up to 8s — then yield a keep-alive heartbeat regardless
        done, pending = await asyncio.wait(pending, timeout=8)

        for task in done:
            deals = task.result() if not task.exception() else []
            done_count += 1

            for deal in deals:
                _dedup_and_buffer(deal)

            # Emit full batches immediately
            while len(buffer) >= batch_size:
                batch = buffer[:batch_size]
                buffer = buffer[batch_size:]
                total_emitted += len(batch)
                yield {"type": "batch", "deals": batch, "total_so_far": total_emitted}

        # Heartbeat at least every 8s to keep Render SSE connection alive
        now = asyncio.get_event_loop().time()
        if now - last_heartbeat >= 8:
            last_heartbeat = now
            yield {
                "type": "heartbeat",
                "done": done_count,
                "total": total_urls,
                "message": (
                    f"⚡ Scanned {done_count}/{total_urls} sources"
                    + (f" — {total_emitted} deals found so far" if total_emitted else " — scanning...")
                ),
            }

    # ── Phase 3: Flush remainder ──────────────────────────────────────────────
    if buffer:
        total_emitted += len(buffer)
        yield {"type": "batch", "deals": buffer, "total_so_far": total_emitted}

    # ── Phase 4: Parallel.ai enrichment ──────────────────────────────────────
    # Collect all emitted deals and enrich missing fields via Parallel.ai
    from parallel_search import parallel_enrich_deals
    all_emitted: list[dict] = []
    # Re-collect from the stream events already yielded — track via a local list
    # Note: we track all buffered deals in all_deals_tracked (set up before phase 2)

    if all_deals_tracked and PARALLEL_API_KEY:
        yield {"type": "heartbeat", "message": "🔬 Enriching deal details via Parallel.ai…"}
        try:
            enriched = await asyncio.wait_for(
                parallel_enrich_deals(
                    all_deals_tracked,
                    config.company_name,
                    config.domain,
                ),
                timeout=90,
            )
            # Emit enriched deals as a patch event
            patched = [d for d in enriched if d.get("announcement_date") or d.get("deal_value_usd")]
            if patched:
                yield {"type": "enriched", "deals": enriched}
        except asyncio.TimeoutError:
            logger.warning("Parallel enrichment timed out")
        except Exception as e:
            logger.warning(f"Parallel enrichment error: {e}")

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
