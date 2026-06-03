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

JINA_KEY          = os.getenv("JINA_KEY", "")
ANTHROPIC_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
APIFY_KEY         = os.getenv("APIFY_API_KEY", "")
SCRAPER_API_KEY   = os.getenv("SCRAPER_API_KEY", "")
SCRAPER_BASE_URL  = os.getenv("SCRAPER_BASE_URL", "")   # e.g. https://scraper-api-xxxx.onrender.com

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
# Tier 1: all 32 sources + top-30 vendors + top product/process keywords
# Tier 2: top-100 vendors + broader keyword combos     (if < 25 deals)
# Tier 3: top-300 vendors + tech keyword combos        (if still < 25 deals)
TIER1_VENDORS    = 15   # kept small so T1 fits in 1 Apify batch (~40 queries total)
TIER2_VENDORS    = 100
TIER3_VENDORS    = 300
SOURCE_GROUP_SZ  = 3    # domains per site: query
TIER1_MAX_SOURCES = 8   # source groups in T1 (8×3=24 domains, 8 site: queries)

# Suffixes that are geographic or legal entity markers — stripped to produce a
# shorter canonical name used as an OR alias in every search query.
# e.g. "Kubota USA" → "Kubota", "Apple Inc." → "Apple"
_STRIP_SUFFIXES = re.compile(
    r'\s*[,]?\s*\b('
    r'USA|U\.S\.A\.?|US|U\.S\.|UK|U\.K\.|'
    r'Inc\.?|Incorporated|Corp\.?|Corporation|'
    r'Ltd\.?|Limited|LLC|LLP|L\.L\.C\.|'
    r'GmbH|AG|S\.A\.|S\.A|PLC|Plc|NV|BV|'
    r'Co\.?|Company|Group|Holdings|Holding|'
    r'North\s+America|South\s+America|Latin\s+America|'
    r'Asia\s+Pacific|APAC|Europe|Middle\s+East|Africa|Australia|'
    r'Japan|China|India|Germany|France|Canada'
    r')\b\.?\s*$',
    re.IGNORECASE,
)

def _short_name(name: str) -> str:
    """Strip trailing geographic/legal suffixes. Returns '' if same as input."""
    stripped = _STRIP_SUFFIXES.sub("", name).strip(" ,.-")
    # Only return if meaningfully shorter (at least 2 chars removed)
    if stripped and stripped.lower() != name.lower() and len(name) - len(stripped) >= 2:
        return stripped
    return ""

def _co_expr(company_name: str) -> str:
    """Build the company search expression, adding short-name OR alias if applicable."""
    short = _short_name(company_name)
    if short:
        return f'("{company_name}" OR "{short}")'
    return f'"{company_name}"'


# ── Step 1: Generate search queries ──────────────────────────────────────────

def build_search_queries(
    company_name: str,
    goal: str,
    year_range: tuple[int, int] = (2016, 2025),
    tier: int = 1,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
    industry: str = "",
    t2_vendors: list[str] | None = None,  # competitor vendors — injected at T2+ only
) -> list[str]:
    """
    Tiered query builder using three independent lists from lists.json:
      - vendors    (18,906): searched as named parties in deals
      - kw_product (33):     product/platform names — searched as "what was bought"
      - kw_process (120):    business processes    — searched as deal context
      - kw_technology (150): technology terms      — searched as deal type

    Tier 1 (~35q): all sources + product kw + process kw sample + top vendors
    Tier 2 (~30q): technology kw + more vendors
    Tier 3 (~15q): broad catch-alls
    """
    lists  = _load_lists()
    years  = list(range(year_range[0], year_range[1] + 1))
    yr_str = " OR ".join(str(y) for y in years)

    # ── Independent lists ─────────────────────────────────────────────────────
    _full_vendors      = lists.get("vendors", _FALLBACK_VENDORS)
    vendor_meta        = lists.get("vendor_meta", {})
    vendor_cat_map     = lists.get("vendor_cat_map", {})
    industry_vendor_map = lists.get("industry_vendor_map", {})  # industry → [vendors] from Customer Industry file

    # Industry filtering: use industry_vendor_map (customer-industry→vendor mapping)
    # Match user's industry string against the 18 known industry keys (case-insensitive, partial)
    # For T2+: prepend competitor vendors (injected by user's vendor lookup)
    effective_extra = list(dict.fromkeys(
        (extra_vendors or []) + (t2_vendors if tier >= 2 and t2_vendors else [])
    ))

    if industry:
        ind_lower = industry.lower()
        industry_vendors: list[str] = []
        for ind_key, ind_vendors in industry_vendor_map.items():
            # Match if user string contains the industry key or vice versa
            if ind_lower in ind_key.lower() or ind_key.lower() in ind_lower:
                industry_vendors.extend(ind_vendors)
        # Deduplicate while preserving order
        seen_iv: set = set()
        industry_vendors = [v for v in industry_vendors if not (v in seen_iv or seen_iv.add(v))]
        # Put fallback (high-signal) vendors that also serve this industry first,
        # then remaining industry vendors, then fallback vendors not in industry, then rest
        industry_set = set(industry_vendors)
        fallback_in_industry = [v for v in _FALLBACK_VENDORS if v in industry_set]
        fallback_not_in_industry = [v for v in _FALLBACK_VENDORS if v not in industry_set]
        remaining_industry = [v for v in industry_vendors if v not in set(_FALLBACK_VENDORS)]
        all_vendors = list(dict.fromkeys(
            effective_extra + fallback_in_industry + remaining_industry
            + fallback_not_in_industry + _full_vendors
        ))
        logger.info(f"Industry filter '{industry}': {len(industry_vendors)} vendors ({len(fallback_in_industry)} high-signal)")
    else:
        all_vendors = list(dict.fromkeys(effective_extra + _FALLBACK_VENDORS + _full_vendors))
    if tier >= 2 and t2_vendors:
        logger.info(f"T2 competitor vendors injected: {t2_vendors[:5]}")

    kw_product   = list(dict.fromkeys((extra_keywords or []) + lists.get("kw_product",    [])))
    kw_process   = list(dict.fromkeys(                         lists.get("kw_process",   [])))
    kw_technology= list(dict.fromkeys(                         lists.get("kw_technology", [])))
    sources      = list(dict.fromkeys((extra_sources  or []) + lists.get("all_sources",  [])))

    # Tier-based slice sizes
    n_vendors  = {1: TIER1_VENDORS,  2: TIER2_VENDORS,  3: TIER3_VENDORS}.get(tier, TIER1_VENDORS)
    vendors    = all_vendors[:n_vendors]

    # Company search expression — adds short-name OR alias for subsidiary names
    # e.g. "Kubota USA" → ("Kubota USA" OR "Kubota")
    co = _co_expr(company_name)
    short = _short_name(company_name)
    if short:
        logger.info(f"Short-name alias: '{company_name}' → '{short}' (queries use OR alias)")

    # Build separate buckets — most targeted first so Apify cap hits best queries
    queries_vendor: list[str] = []
    queries_kw: list[str] = []
    queries_site: list[str] = []
    queries_broad: list[str] = []

    if tier == 1:
        # T1 target: ≤40 queries (1 Apify batch = 1 parallel run = ~90s max)
        # 15 vendors + 8 product kw + 5 process kw + 8 site groups + 2 broad = 38
        for vendor in vendors:
            meta   = vendor_meta.get(vendor, {})
            market = meta.get("primary_market", "")
            cats   = vendor_cat_map.get(vendor, [])
            if market:
                queries_vendor.append(f'{co} "{vendor}" "{market}" deal OR contract ({yr_str})')
            elif cats:
                queries_vendor.append(f'{co} "{vendor}" "{cats[0]}" deal OR contract')
            else:
                queries_vendor.append(f'{co} "{vendor}" deal OR contract OR agreement')

        # Top product keywords (first 8 — high signal, low noise)
        for kw in kw_product[:8]:
            queries_kw.append(f'{co} "{kw}" deal OR contract OR implementation ({yr_str})')

        # Top process keywords (first 5)
        for kw in kw_process[:5]:
            queries_kw.append(f'{co} "{kw}" vendor OR outsourcing OR contract ({yr_str})')

        # Site: queries — top 8 source groups (24 domains) in T1
        for i in range(0, min(len(sources), TIER1_MAX_SOURCES * SOURCE_GROUP_SZ), SOURCE_GROUP_SZ):
            grp = sources[i: i + SOURCE_GROUP_SZ]
            site_expr = " OR ".join(f"site:{s}" for s in grp)
            queries_site.append(f'({site_expr}) {co} deal OR contract OR agreement ({yr_str})')

        # Broad anchors (2 per year range)
        queries_broad.append(f'{co} IT deal contract signed ({yr_str})')
        queries_broad.append(f'{co} technology outsourcing agreement ({yr_str})')

    elif tier == 2:
        # Vendor + sub-industry + process keyword combos
        for vendor in vendors:
            meta    = vendor_meta.get(vendor, {})
            sub_ind = meta.get("sub_industry", "")
            vkw_pr  = meta.get("kw_process", [])
            if vkw_pr:
                for kw in vkw_pr[:2]:
                    queries_vendor.append(f'{co} "{vendor}" "{kw}" ({yr_str})')
            elif sub_ind:
                queries_vendor.append(f'{co} "{vendor}" "{sub_ind}" contract OR deal')
            else:
                queries_vendor.append(f'{co} "{vendor}" deal OR contract OR agreement')

        # Remaining product keywords + first 50 process keywords
        for kw in kw_product[15:]:
            queries_kw.append(f'{co} "{kw}" deal OR contract OR implementation ({yr_str})')
        for kw in kw_process[10:60]:
            queries_kw.append(f'{co} "{kw}" vendor OR outsourcing OR contract ({yr_str})')

        # Technology keywords (first 40)
        for kw in kw_technology[:40]:
            queries_kw.append(f'{co} "{kw}" deal OR contract OR selected ({yr_str})')

        # No site: queries in T2 — all sources already covered in T1

    elif tier >= 3:
        # T3: keyword catch-alls only — no vendor queries (keeps T3 fast and cheap)
        for kw in kw_process[60:]:
            queries_kw.append(f'{co} "{kw}" vendor OR outsourcing OR contract ({yr_str})')
        for kw in kw_technology[40:]:
            queries_kw.append(f'{co} "{kw}" deal OR contract OR selected ({yr_str})')

        queries_broad.append(f'{co} vendor selected partnership announcement ({yr_str})')
        queries_broad.append(f'{co} outsourcing managed services digital transformation ({yr_str})')

    # Final order: vendor (most targeted) → kw → site → broad
    queries = queries_vendor + queries_kw + queries_site + queries_broad

    logger.info(f"Tier {tier}: {len(queries)} queries for {company_name} "
                f"({len(vendors)} vendors, sources={len(sources)})")
    return queries


# ── Step 2: Search via Bing (custom scraper) / Apify / Jina → collect URLs ────

async def _google_news_rss_search(queries: list[str], results_per_query: int = 10) -> list[str]:
    """
    Search via Google News RSS feed — returns clean XML with real article URLs.
    No API key, no anti-bot issues, no tracking redirects.
    Runs queries in parallel batches of 5.
    """
    from urllib.parse import quote_plus, urlparse

    async def _one(query: str) -> list[str]:
        try:
            rss_url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
            async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
                r = await client.get(rss_url, headers={"User-Agent": "Mozilla/5.0 (compatible; RSS reader)"})
            if not r.is_success:
                return []
            # Parse <link> tags from RSS XML (article URLs)
            # Google News RSS <item><link>URL</link> pattern
            urls = re.findall(r'<link>([^<]+)</link>', r.text)
            # Also catch <guid> which sometimes has the real URL
            guids = re.findall(r'<guid[^>]*>([^<]+)</guid>', r.text)
            all_urls = urls + guids
            out = []
            seen_u: set = set()
            for u in all_urls:
                u = u.strip()
                if not u.startswith("http"):
                    continue
                domain = urlparse(u).netloc.lstrip("www.")
                if any(domain == s or domain.endswith("." + s) for s in SKIP_DOMAINS):
                    continue
                # Skip only the channel-level search page, keep article redirect URLs
                if u == "https://news.google.com/" or "rss/search" in u:
                    continue
                if u not in seen_u:
                    seen_u.add(u)
                    out.append(u)
            return out[:results_per_query]
        except Exception as e:
            logger.debug(f"Google News RSS error: {e}")
            return []

    urls: list[str] = []
    seen: set[str] = set()
    for i in range(0, len(queries), 5):
        batch = queries[i: i + 5]
        results = await asyncio.gather(*[_one(q) for q in batch], return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            for url in (result or []):
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)

    logger.info(f"Google News RSS: {len(urls)} URLs from {len(queries)} queries")
    return urls


APIFY_BATCH_SZ = 40        # max queries per Apify actor run
APIFY_SEARCH_TIMEOUT = 90  # seconds — most batches complete well under this

async def _apify_google_search(queries: list[str], results_per_query: int = 10) -> list[str]:
    """
    Search Google via Apify Google Search Scraper.
    Splits queries into batches of APIFY_BATCH_SZ and fires all batches in PARALLEL.
    All batches complete in ~the same time as one batch (~180s max).
    """
    if not APIFY_KEY or not queries:
        return []

    actor_url = (
        "https://api.apify.com/v2/acts/apify~google-search-scraper"
        f"/run-sync-get-dataset-items?token={APIFY_KEY}&timeout={APIFY_SEARCH_TIMEOUT}&memory=512"
    )
    batches = [queries[i: i + APIFY_BATCH_SZ] for i in range(0, len(queries), APIFY_BATCH_SZ)]

    async def _run_batch(batch: list[str]) -> list[str]:
        try:
            payload = {
                "queries": "\n".join(batch),
                "maxPagesPerQuery": 1,
                "resultsPerPage": results_per_query,
                "countryCode": "us",
                "languageCode": "en",
            }
            async with httpx.AsyncClient(timeout=APIFY_SEARCH_TIMEOUT + 15) as client:
                r = await client.post(actor_url, json=payload)
            if not r.is_success:
                logger.warning(f"Apify batch failed {r.status_code}: {r.text[:200]}")
                return []
            batch_urls = []
            for item in r.json():
                for result in item.get("organicResults", []):
                    url = result.get("url", "")
                    if url:
                        batch_urls.append(url)
            return batch_urls
        except Exception as e:
            logger.warning(f"Apify batch error: {e}")
            return []

    results = await asyncio.gather(*[_run_batch(b) for b in batches])

    seen: set[str] = set()
    urls: list[str] = []
    for batch_urls in results:
        for url in batch_urls:
            if url not in seen:
                seen.add(url)
                urls.append(url)

    logger.info(f"Apify Google Search: {len(urls)} URLs from {len(queries)} queries ({len(batches)} parallel batches)")
    return urls


async def _scraperapi_google_search(queries: list[str], results_per_query: int = 10) -> list[str]:
    """
    Search Google via ScraperAPI by scraping Google search result pages.
    Uses the standard scraping endpoint (works on all plans).
    Parses organic result links from the returned HTML.
    """
    if not SCRAPER_API_KEY or not queries:
        return []

    from urllib.parse import quote_plus
    import re as _re

    async def _one(query: str) -> list[str]:
        try:
            google_url = f"https://www.google.com/search?q={quote_plus(query)}&num={results_per_query}&hl=en"
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    "https://api.scraperapi.com/",
                    params={
                        "api_key": SCRAPER_API_KEY,
                        "url": google_url,
                        "render": "false",
                    },
                )
                if not r.is_success:
                    logger.debug(f"ScraperAPI {r.status_code} for: {query[:60]}")
                    return []
                # Extract URLs from /url?q=... patterns in Google HTML
                raw_links = _re.findall(r'/url\?q=(https?://[^&"]+)', r.text)
                # Decode percent-encoding
                from urllib.parse import unquote
                links = [unquote(l) for l in raw_links]
                # Filter out Google's own domains
                links = [l for l in links if "google.com" not in l and "googleapis.com" not in l]
                return links[:results_per_query]
        except Exception as e:
            logger.debug(f"ScraperAPI error: {e}")
            return []

    urls: list[str] = []
    seen: set[str] = set()
    # Run in batches of 5 parallel requests
    for i in range(0, len(queries), 5):
        batch = capped[i: i + 5]
        results = await asyncio.gather(*[_one(q) for q in batch], return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                continue
            for url in result:
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)

    logger.info(f"ScraperAPI Google Search: {len(urls)} URLs from {len(capped)} queries")
    return urls


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
    Google News RSS is intentionally skipped — its article links are all
    news.google.com redirect URLs which are filtered by SKIP_DOMAINS, making
    it a guaranteed 40s waste before Apify gets called.
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

    # Primary: Apify Google Search Scraper
    if APIFY_KEY:
        raw = await _apify_google_search(queries, results_per_query=5)
        urls = _filter(raw)
        if urls:
            return urls
        logger.warning("Apify returned no URLs — falling back to Jina")

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


async def scrape_urls_custom(urls: list[str]) -> list[dict]:
    """
    Scrape URLs via the custom scraper API (vibhorkumar1209/scraper-api).
    Endpoint: GET {SCRAPER_BASE_URL}/scrape/web?url=...
    Auth:      x-api-key header (SCRAPER_API_KEY)
    Runs in parallel batches of 5.
    """
    if not SCRAPER_BASE_URL or not urls:
        return []

    headers = {}
    if SCRAPER_API_KEY:
        headers["x-api-key"] = SCRAPER_API_KEY

    async def _one(url: str) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=25) as client:
                r = await client.get(
                    f"{SCRAPER_BASE_URL.rstrip('/')}/scrape/web",
                    params={"url": url},
                    headers=headers,
                )
                if not r.is_success:
                    logger.debug(f"Custom scraper {r.status_code} for {url[:60]}")
                    return None
                data = r.json()
                if not data.get("success"):
                    return None
                page = data.get("data", {})
                # Returns: title, bodyText, links, metaTags, statusCode
                text = page.get("bodyText") or page.get("text") or ""
                if text and len(text.split()) >= 80:
                    return {"url": url, "text": text}
        except Exception as e:
            logger.debug(f"Custom scraper error for {url[:60]}: {e}")
        return None

    results = []
    for i in range(0, len(urls), 5):
        batch = urls[i: i + 5]
        items = await asyncio.gather(*[_one(u) for u in batch], return_exceptions=True)
        for item in items:
            if item and not isinstance(item, Exception):
                results.append(item)

    logger.info(f"Custom scraper: {len(results)}/{len(urls)} URLs with content")
    return results


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

    # Inject vendor list as a name-matching reference (first 300 alphabetically, token-budgeted)
    lists = _load_lists()
    vendor_list = lists.get("vendors", _FALLBACK_VENDORS)
    vendor_hint = ", ".join(vendor_list[:300])

    # Combine page texts, truncate to fit context
    combined = ""
    for p in pages[:12]:
        snippet = p["text"][:2500]
        combined += f"\n\n[Source: {p['url']}]\n{snippet}"
    combined = combined[:24000]  # hard cap

    if not combined:
        return []

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
        f"- Only extract deals supported by the scraped content — no guessing\n"
        f"- If no deals found, return an empty array []\n"
        f"- Return ONLY the JSON array, no explanation\n\n"
        f"SCRAPED CONTENT:\n{combined}"
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

MIN_DEALS_THRESHOLD = 25   # escalate to next tier if below this

def _dedup_by_domain(urls: list[str], max_per_domain: int = 2) -> list[str]:
    """Keep at most max_per_domain URLs per root domain to avoid scraping the same site repeatedly."""
    from urllib.parse import urlparse
    domain_count: dict[str, int] = {}
    out: list[str] = []
    for url in urls:
        domain = urlparse(url).netloc.lstrip("www.")
        if domain_count.get(domain, 0) < max_per_domain:
            domain_count[domain] = domain_count.get(domain, 0) + 1
            out.append(url)
    return out


async def _run_tier(
    company_name: str, goal: str, schema_fields: list[dict],
    year_range: tuple[int, int], tier: int, max_urls: int,
    seen_urls: set,
    extra_vendors: list[str] | None,
    extra_sources: list[str] | None,
    extra_keywords: list[str] | None,
    industry: str = "",
    t2_vendors: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """Run one search tier. Yields progress events then a final 'tier_done' event."""
    queries = build_search_queries(
        company_name, goal, year_range, tier=tier,
        extra_vendors=extra_vendors,
        extra_sources=extra_sources,
        extra_keywords=extra_keywords,
        industry=industry,
        t2_vendors=t2_vendors,
    )
    yield {"type": "heartbeat", "message": f"🔍 Tier {tier}: running {len(queries)} search queries…"}

    raw_urls = await collect_urls(queries, max_urls=max_urls)
    urls = _dedup_by_domain([u for u in raw_urls if u not in seen_urls])
    seen_urls.update(urls)
    if not urls:
        yield {"type": "tier_done", "deals": [], "relevant": 0}
        return

    yield {"type": "heartbeat", "message": f"📄 Tier {tier}: scraping {len(urls)} pages…"}

    # Scrape: custom scraper → Apify → Jina (in priority order)
    scraped: list[dict] = []
    if SCRAPER_BASE_URL:
        scraped = await scrape_urls_custom(urls[:20])
    if not scraped and APIFY_KEY:
        for i in range(0, len(urls), 10):
            scraped.extend(await scrape_urls_apify(urls[i: i + 10]))
    if not scraped:
        scraped = await scrape_urls_jina_fallback(urls[:15])

    yield {"type": "heartbeat", "message": f"🧠 Tier {tier}: extracting deals from {len(scraped)} scraped pages…"}
    relevant = [p for p in scraped if is_deal_page(p["text"], company_name)]
    pages_to_use = relevant if relevant else scraped[:5]
    deals = await asyncio.to_thread(_claude_extract_deals, pages_to_use, company_name, goal, schema_fields)
    yield {"type": "tier_done", "deals": deals, "relevant": len(relevant)}


async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2016, 2025),
    max_urls: int = 30,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
    industry: str = "",
    run_t3: bool = False,
    t2_vendors: list[str] | None = None,  # competitor vendors — T2 only
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
    n_keywords = len(lists.get("all_keywords", []))
    n_sources  = len(lists.get("all_sources", []))
    yield {"type": "heartbeat", "message":
           f"📚 Lists loaded: {n_vendors:,} vendors · {n_keywords} keywords · {n_sources} sources"}

    seen_urls: set = set()
    all_deals: list[dict] = []
    total_relevant = 0

    for tier in [1, 2, 3]:
        # T3 is opt-in only — skip unless user explicitly enabled it
        if tier == 3 and not run_t3:
            break

        if tier == 2:
            yield {"type": "heartbeat", "message": f"📈 Only {len(all_deals)} deals — escalating to Tier 2 (broader keywords + more vendors)…"}
        elif tier == 3:
            yield {"type": "heartbeat", "message": f"📈 Tier 3 keyword catch-all enabled — running deep search…"}

        tier_deals: list[dict] = []
        tier_relevant = 0

        # Stream progress events from the tier generator with a watchdog
        tier_gen = _run_tier(
            company_name, goal, schema_fields, year_range, tier, max_urls,
            seen_urls, extra_vendors, extra_sources, extra_keywords, industry,
            t2_vendors=t2_vendors,
        )
        elapsed = 0
        try:
            while elapsed < 1200:
                # Wait up to 8s for next event; emit a heartbeat tick if nothing arrives
                try:
                    event = await asyncio.wait_for(tier_gen.__anext__(), timeout=8)
                except asyncio.TimeoutError:
                    elapsed += 8
                    yield {"type": "heartbeat", "message": f"⏳ Tier {tier} working… ({elapsed}s)"}
                    continue
                except StopAsyncIteration:
                    break

                if event["type"] == "tier_done":
                    tier_deals   = event.get("deals", [])
                    tier_relevant = event.get("relevant", 0)
                    break
                else:
                    elapsed += 0  # progress event resets perceived wait
                    yield event   # forward heartbeat (🔍 Searching / 📄 Scraping / 🧠 Extracting)
        except Exception as e:
            logger.warning(f"Tier {tier} error: {e}")

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

    if not all_deals:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found from scraped content for {company_name} — skipping"}
        row = {"company_name": company_name, "domain": domain, "_status": "no_result", "_sources": total_relevant}
        for f in schema_fields:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat", "message": f"✅ {company_name}: {len(all_deals)} total deals"}
