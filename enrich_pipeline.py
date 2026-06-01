"""
Enrichment pipeline: search → Apify scrape → identify → Claude classify.

Flow for each company input:
  1. Generate search queries: company × top vendors × deal keywords
  2. Run Jina Search to collect URLs
  3. Deduplicate + filter URLs
  4. Scrape each URL via Apify Website Content Crawler
  5. Filter pages for deal relevance (NLP)
  6. Extract schema fields from relevant pages using Claude
  7. Merge results across pages → final row
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator

import httpx

logger = logging.getLogger(__name__)

JINA_KEY      = os.getenv("JINA_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
APIFY_KEY     = os.getenv("APIFY_API_KEY", "")

# ── Load pre-built lists.json (vendors + keywords + sources) ─────────────────
_LISTS: dict = {}

def _load_lists() -> dict:
    global _LISTS
    if _LISTS:
        return _LISTS
    candidates = [
        os.path.join(os.path.dirname(__file__), "lists.json"),
        "/app/lists.json",           # Render
        "lists.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    _LISTS = json.load(f)
                logger.info(f"Loaded lists.json: {len(_LISTS.get('vendors', []))} vendors, "
                            f"{len(_LISTS.get('keyword_frequency', {}))} keywords, "
                            f"{len(_LISTS.get('all_sources', []))} sources")
                return _LISTS
            except Exception as e:
                logger.warning(f"Failed to load lists.json from {path}: {e}")
    logger.warning("lists.json not found — using built-in fallback vendors")
    return {}


# ── Fallback vendor list (used when lists.json not present) ──────────────────
_FALLBACK_VENDORS = [
    "TCS", "Infosys", "Wipro", "HCLTech", "Accenture", "IBM", "Cognizant",
    "Capgemini", "DXC Technology", "Tech Mahindra", "SAP", "Oracle",
    "Microsoft", "AWS", "Google Cloud", "Salesforce", "ServiceNow",
    "Workday", "Palo Alto Networks", "Snowflake", "Temenos", "Finacle",
    "FIS", "Fiserv", "Finastra", "Atos", "NTT", "Unisys",
]

# ── Domains to skip ───────────────────────────────────────────────────────────
SKIP_DOMAINS = {
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "reddit.com", "quora.com", "wikipedia.org",
    "glassdoor.com", "indeed.com", "crunchbase.com",
}

# ── Tiered query budget ───────────────────────────────────────────────────────
# Tier 1 (always): sources + top-N keywords → ~30 queries, fast + cheap
# Tier 2 (if < MIN_DEALS): vendor pairs + more keywords → +25 queries
# Tier 3 (if still thin): broader catch-alls → +15 queries
TIER1_KEYWORDS   = 8    # top keywords by vendor-coverage frequency
TIER1_VENDORS    = 10   # top vendors from lists (or fallback)
TIER2_KEYWORDS   = 15
TIER2_VENDORS    = 20
SOURCE_GROUP_SZ  = 3    # domains per site: query


# ── Step 1: Generate search queries ──────────────────────────────────────────

def build_search_queries(
    company_name: str,
    goal: str,
    year_range: tuple[int, int] = (2022, 2025),
    tier: int = 1,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> list[str]:
    """
    Tiered query builder powered by lists.json.

    Tier 1 (~30 queries): sources + top-8 keywords + top-10 vendors
    Tier 2 (~25 more):    more keywords + more vendors
    Tier 3 (~15 more):    broad catch-alls

    extra_* from the request take priority over lists.json defaults.
    """
    lists   = _load_lists()
    queries: list[str] = []
    years   = list(range(year_range[0], year_range[1] + 1))
    yr_str  = " OR ".join(str(y) for y in years)

    # ── Resolve vendors ───────────────────────────────────────────────────────
    list_vendors = lists.get("vendors", _FALLBACK_VENDORS)
    n_vendors = TIER2_VENDORS if tier >= 2 else TIER1_VENDORS
    # Caller-supplied vendors go first (most specific), then lists.json
    vendors = list(dict.fromkeys((extra_vendors or []) + list_vendors))[:n_vendors]

    # ── Resolve keywords ──────────────────────────────────────────────────────
    ranked_kw = lists.get("top_keywords_ranked", [])
    n_kw = TIER2_KEYWORDS if tier >= 2 else TIER1_KEYWORDS
    keywords = list(dict.fromkeys((extra_keywords or []) + ranked_kw))[:n_kw]

    # ── Resolve sources ───────────────────────────────────────────────────────
    list_sources = lists.get("all_sources", [])
    sources = list(dict.fromkeys((extra_sources or []) + list_sources))

    # ── Build queries ─────────────────────────────────────────────────────────

    # 1. Always: broad deal anchors per year (small, high-recall)
    for year in years:
        queries.append(f'"{company_name}" IT deal contract signed {year}')
        queries.append(f'"{company_name}" technology outsourcing agreement {year}')

    # 2. Always: site: queries against known sources (highest precision)
    #    Group sources 3/query to keep query string short
    for i in range(0, min(len(sources), 36), SOURCE_GROUP_SZ):
        grp = sources[i: i + SOURCE_GROUP_SZ]
        site_expr = " OR ".join(f"site:{s}" for s in grp)
        queries.append(f'({site_expr}) "{company_name}" deal OR contract OR agreement ({yr_str})')

    # 3. Company × top keywords (keyword = deal category signal)
    for kw in keywords:
        queries.append(f'"{company_name}" "{kw}" deal OR contract OR signed ({yr_str})')

    # 4. Company × vendor pairs (vendor name is most precise signal)
    for vendor in vendors:
        queries.append(f'"{company_name}" "{vendor}" deal OR contract OR agreement')

    if tier >= 3:
        # 5. Tier 3: catch-alls with OR-chained keywords
        kw_or = " OR ".join(f'"{k}"' for k in keywords[:6])
        queries.append(f'"{company_name}" ({kw_or}) signed awarded ({yr_str})')
        queries.append(f'"{company_name}" vendor selected partnership announcement ({yr_str})')
        queries.append(f'"{company_name}" outsourcing managed services digital transformation ({yr_str})')

    logger.info(f"Built {len(queries)} tier-{tier} queries for {company_name} "
                f"({len(vendors)} vendors, {len(keywords)} keywords, {len(sources)} sources)")
    return queries


# ── Step 2: Search via Jina → collect URLs ────────────────────────────────────

async def _apify_google_search(queries: list[str], results_per_query: int = 10) -> list[str]:
    """
    Search Google via Apify Google Search Scraper actor.
    Sends all queries in one actor run — much faster than serial searches.
    Returns flat list of unique URLs from organic results.
    """
    if not APIFY_KEY or not queries:
        return []
    try:
        actor_url = (
            "https://api.apify.com/v2/acts/apify~google-search-scraper"
            f"/run-sync-get-dataset-items?token={APIFY_KEY}&timeout=60&memory=256"
        )
        payload = {
            "queries": "\n".join(queries),   # one query per line
            "maxPagesPerQuery": 1,
            "resultsPerPage": results_per_query,
            "countryCode": "us",
            "languageCode": "en",
        }
        async with httpx.AsyncClient(timeout=70) as client:
            r = await client.post(actor_url, json=payload)
            if not r.is_success:
                logger.warning(f"Apify Google Search {r.status_code}: {r.text[:200]}")
                return []
            items = r.json()
            urls: list[str] = []
            seen: set[str] = set()
            for item in items:
                for result in item.get("organicResults", []):
                    url = result.get("url", "")
                    if url and url not in seen:
                        seen.add(url)
                        urls.append(url)
            logger.info(f"Apify Google Search: {len(urls)} URLs from {len(queries)} queries")
            return urls
    except Exception as e:
        logger.warning(f"Apify Google Search error: {e}")
        return []


async def _jina_search_fallback(query: str) -> list[str]:
    """Fallback: Jina semantic search when Apify is unavailable."""
    if not JINA_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                "https://s.jina.ai/",
                params={"q": query},
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {JINA_KEY}",
                    "X-Respond-With": "no-content",
                },
            )
            if not r.is_success:
                return []
            urls = [item.get("url", "") for item in r.json().get("data", []) if item.get("url")]
            return urls
    except Exception as e:
        logger.debug(f"Jina search error: {e}")
        return []


async def collect_urls(queries: list[str], max_urls: int = 30) -> list[str]:
    """
    Collect URLs via Apify Google Search (primary) or Jina (fallback).
    Runs all queries in one Apify call for speed.
    """
    from urllib.parse import urlparse

    def _filter(raw_urls: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for url in raw_urls:
            if not url:
                continue
            domain = urlparse(url).netloc.lstrip("www.")
            if any(domain == s or domain.endswith("." + s) for s in SKIP_DOMAINS):
                continue
            if url not in seen:
                seen.add(url)
                out.append(url)
                if len(out) >= max_urls:
                    break
        return out

    # Primary: Apify Google Search (all queries in one call)
    if APIFY_KEY:
        raw = await _apify_google_search(queries, results_per_query=10)
        urls = _filter(raw)
        if urls:
            return urls
        logger.warning("Apify Google Search returned no URLs — falling back to Jina")

    # Fallback: Jina (batched)
    seen: set[str] = set()
    urls: list[str] = []
    for i in range(0, len(queries), 5):
        batch = queries[i: i + 5]
        results = await asyncio.gather(*[_jina_search_fallback(q) for q in batch], return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            for url in result:
                if not url:
                    continue
                domain = urlparse(url).netloc.lstrip("www.")
                if any(domain == s or domain.endswith("." + s) for s in SKIP_DOMAINS):
                    continue
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
        if len(urls) >= max_urls:
            break

    return urls[:max_urls]


# ── Step 3: Scrape via Apify ──────────────────────────────────────────────────

async def scrape_urls_apify(urls: list[str]) -> list[dict]:
    """
    Scrape a batch of URLs via Apify Website Content Crawler (sync API).
    Returns list of {url, text} dicts.
    """
    if not APIFY_KEY or not urls:
        return []

    actor_url = (
        "https://api.apify.com/v2/acts/apify~website-content-crawler/run-sync-get-dataset-items"
        f"?token={APIFY_KEY}&timeout=120&memory=512"
    )
    payload = {
        "startUrls": [{"url": u} for u in urls],
        "maxCrawlPages": len(urls),
        "crawlerType": "cheerio",
        "removeElementsCssSelector": "nav,footer,header,script,style,.cookie-banner,.ad",
        "htmlTransformer": "readableText",
    }
    try:
        async with httpx.AsyncClient(timeout=130) as client:
            r = await client.post(actor_url, json=payload)
            if not r.is_success:
                logger.warning(f"Apify batch scrape {r.status_code}: {r.text[:200]}")
                return []
            items = r.json()
            results = []
            for item in items:
                text = item.get("text") or item.get("markdown") or ""
                url = item.get("url") or item.get("loadedUrl") or ""
                if text and len(text.split()) >= 80:
                    results.append({"url": url, "text": text})
            logger.info(f"Apify scraped {len(results)}/{len(urls)} URLs with content")
            return results
    except Exception as e:
        logger.warning(f"Apify batch scrape error: {e}")
        return []


async def scrape_urls_jina_fallback(urls: list[str]) -> list[dict]:
    """Fallback scraper using Jina Reader when Apify is not configured."""
    from website_router import fetch_via_jina_reader

    results = []
    tasks = [fetch_via_jina_reader(url) for url in urls]
    fetched = await asyncio.gather(*tasks, return_exceptions=True)
    for url, result in zip(urls, fetched):
        if isinstance(result, Exception) or result is None:
            continue
        text, _ = result
        if text and len(text.split()) >= 80:
            results.append({"url": url, "text": text})
    return results


# ── Step 4: Identify deal-relevant pages ─────────────────────────────────────

def is_deal_page(text: str, company_name: str) -> bool:
    """Quick relevance filter — must mention company + at least one deal signal."""
    tl = text.lower()
    cn = company_name.lower()
    if cn not in tl:
        return False
    deal_signals = [
        "deal", "contract", "signed", "awarded", "selected", "partnership",
        "agreement", "outsourcing", "implementation", "go-live",
        "digital transformation", "managed services", "vendor",
    ]
    return any(sig in tl for sig in deal_signals)


# ── Step 5: Extract schema fields via Claude ──────────────────────────────────

def _claude_extract_deals(pages: list[dict], company_name: str, goal: str, schema_fields: list[dict]) -> list[dict]:
    """
    Call Claude to extract MULTIPLE deal rows from scraped pages.
    Injects the known vendor list so Claude can match vendor names precisely.
    Returns a list of dicts, one per distinct deal found.
    Runs in a thread (blocking Anthropic SDK call).
    """
    import anthropic

    if not ANTHROPIC_KEY:
        return []

    fields_desc = "\n".join(
        f'- {f["key"]}: {f.get("description") or f.get("label", "")} ({f.get("type","string")})'
        for f in schema_fields
    )
    field_keys = [f["key"] for f in schema_fields]

    # Inject vendor list as a recognition aid (first 200 vendors to stay within token budget)
    lists = _load_lists()
    vendor_list = lists.get("vendors", _FALLBACK_VENDORS)
    vendor_hint = ", ".join(vendor_list[:200])

    # Combine page texts, truncate to fit context
    combined = ""
    for p in pages[:12]:
        snippet = p["text"][:2500]
        combined += f"\n\n[Source: {p['url']}]\n{snippet}"
    combined = combined[:24000]  # hard cap

    if combined:
        prompt = (
            f"You are extracting IT deal records for {company_name} from scraped web content.\n\n"
            f"GOAL: {goal}\n\n"
            f"KNOWN VENDOR REFERENCE LIST (use for matching vendor names exactly):\n{vendor_hint}\n\n"
            f"Extract EVERY distinct deal or contract mentioned. Each deal = one JSON object.\n"
            f"Return a JSON ARRAY where each element has these exact keys:\n{fields_desc}\n\n"
            f"Rules:\n"
            f"- One object per deal/contract — do NOT merge multiple deals into one\n"
            f"- Include deals from all years found in the content (2020–2025)\n"
            f"- Match vendor names exactly to the reference list where possible\n"
            f"- Use null for fields not mentioned for that specific deal\n"
            f"- Be specific: exact vendor name, date, value, contract duration where stated\n"
            f"- If no deals found, return an empty array []\n"
            f"- Return ONLY the JSON array, no explanation\n\n"
            f"SCRAPED CONTENT:\n{combined}"
        )
    else:
        # No scraped pages — ask Claude from knowledge
        prompt = (
            f"List all known IT deals and technology contracts for {company_name} from 2020–2025.\n\n"
            f"GOAL: {goal}\n\n"
            f"KNOWN VENDOR REFERENCE LIST (use for matching vendor names):\n{vendor_hint}\n\n"
            f"Return a JSON ARRAY where each element is one deal with these exact keys:\n{fields_desc}\n\n"
            f"Rules:\n"
            f"- One object per deal — do NOT merge multiple deals\n"
            f"- Include as many distinct deals as you know (target 5–15 deals)\n"
            f"- Match vendor names to the reference list where possible\n"
            f"- Use null for fields you are not confident about\n"
            f"- Return ONLY the JSON array, no explanation"
        )

    try:
        ac = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = ac.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw.strip())
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            # Claude returned a single object — wrap it
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        # Normalise: ensure all keys present, convert nulls
        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row = {}
            for key in field_keys:
                val = item.get(key)
                row[key] = str(val) if val not in (None, "null", "") else ""
            out.append(row)
        return out
    except Exception as e:
        logger.warning(f"Claude extraction error: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

MIN_DEALS_THRESHOLD = 10   # escalate to next tier if below this

async def _run_tier(
    company_name: str, goal: str, schema_fields: list[dict],
    year_range: tuple[int, int], tier: int, max_urls: int,
    seen_urls: set,
    extra_vendors: list[str] | None,
    extra_sources: list[str] | None,
    extra_keywords: list[str] | None,
) -> tuple[list[dict], int]:
    """Run one search tier. Returns (deals, relevant_page_count)."""
    queries = build_search_queries(
        company_name, goal, year_range, tier=tier,
        extra_vendors=extra_vendors,
        extra_sources=extra_sources,
        extra_keywords=extra_keywords,
    )
    raw_urls = await collect_urls(queries, max_urls=max_urls)
    urls = [u for u in raw_urls if u not in seen_urls]
    seen_urls.update(urls)
    if not urls:
        return [], 0

    # Scrape
    if APIFY_KEY:
        scraped: list[dict] = []
        for i in range(0, len(urls), 10):
            scraped.extend(await scrape_urls_apify(urls[i: i + 10]))
    else:
        scraped = await scrape_urls_jina_fallback(urls[:15])

    relevant = [p for p in scraped if is_deal_page(p["text"], company_name)]
    pages_to_use = relevant if relevant else scraped[:5]
    deals = await asyncio.to_thread(_claude_extract_deals, pages_to_use, company_name, goal, schema_fields)
    return deals, len(relevant)


async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2022, 2025),
    max_urls: int = 30,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Tiered enrichment pipeline powered by lists.json.

    Tier 1 (~30 queries): all 32 sources + top-8 keywords + top-10 vendors
    Tier 2 (~25 more):    top-15 keywords + top-20 vendors  (if <10 deals found)
    Tier 3 (~15 more):    broad catch-alls                  (if still <10 deals)
    Fallback:             Claude knowledge                  (if no URLs found at all)
    """
    lists = _load_lists()
    n_vendors  = len(lists.get("vendors", _FALLBACK_VENDORS))
    n_keywords = len(lists.get("keyword_frequency", {}))
    n_sources  = len(lists.get("all_sources", []))
    yield {"type": "heartbeat", "message":
           f"📚 Lists loaded: {n_vendors:,} vendors · {n_keywords} keywords · {n_sources} sources"}

    seen_urls: set = set()
    all_deals: list[dict] = []
    total_relevant = 0

    for tier in [1, 2, 3]:
        if tier == 1:
            yield {"type": "heartbeat", "message": f"🔍 Tier 1 search for {company_name} (sources + top keywords + vendors)…"}
        elif tier == 2:
            yield {"type": "heartbeat", "message": f"📈 Only {len(all_deals)} deals — escalating to Tier 2 (broader keywords + more vendors)…"}
        else:
            yield {"type": "heartbeat", "message": f"📈 Still {len(all_deals)} deals — Tier 3 catch-all search…"}

        task = asyncio.ensure_future(_run_tier(
            company_name, goal, schema_fields, year_range, tier, max_urls,
            seen_urls, extra_vendors, extra_sources, extra_keywords,
        ))
        elapsed = 0
        while not task.done() and elapsed < 120:
            done, _ = await asyncio.wait({task}, timeout=8)
            elapsed += 8
            if done:
                break
            yield {"type": "heartbeat", "message": f"🔍 Tier {tier} searching… ({elapsed}s)"}

        if not task.done():
            task.cancel()
            tier_deals, tier_relevant = [], 0
        else:
            try:
                tier_deals, tier_relevant = task.result()
            except Exception as e:
                logger.warning(f"Tier {tier} error: {e}")
                tier_deals, tier_relevant = [], 0

        total_relevant += tier_relevant
        yield {"type": "heartbeat", "message": f"📋 Tier {tier}: {len(tier_deals)} deals from {tier_relevant} relevant pages"}

        for deal in tier_deals:
            row = {"company_name": company_name, "domain": domain,
                   "_status": "ok", "_sources": total_relevant}
            row.update(deal)
            all_deals.append(deal)
            yield {"type": "row_done", "row": row}

        if len(all_deals) >= MIN_DEALS_THRESHOLD:
            break   # enough deals — stop escalating

    # Fallback: Claude knowledge if nothing scraped across all tiers
    if not all_deals:
        yield {"type": "heartbeat", "message": f"⚠️ No scraped deals — using Claude knowledge for {company_name}…"}
        kb_deals = await asyncio.to_thread(_claude_extract_deals, [], company_name, goal, schema_fields)
        if not kb_deals:
            row = {"company_name": company_name, "domain": domain, "_status": "no_result", "_sources": 0}
            for f in schema_fields:
                row[f["key"]] = ""
            yield {"type": "row_done", "row": row}
            return
        for deal in kb_deals:
            row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": 0}
            row.update(deal)
            yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat", "message": f"✅ {company_name}: {len(all_deals)} total deals"}
