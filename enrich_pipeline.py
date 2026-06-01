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
TIER1_VENDORS    = 30
TIER2_VENDORS    = 100
TIER3_VENDORS    = 300
SOURCE_GROUP_SZ  = 3    # domains per site: query


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
    # Put fallback (high-signal) vendors first, then full list — prevents obscure vendors in top-N
    _full_vendors = lists.get("vendors", _FALLBACK_VENDORS)
    vendor_meta    = lists.get("vendor_meta", {})
    vendor_cat_map = lists.get("vendor_cat_map", {})

    # Industry filtering: if industry provided, prefer vendors whose sub_industry or
    # primary_market contains a keyword from the industry string (case-insensitive)
    if industry:
        ind_lower = industry.lower()
        ind_tokens = [t.strip() for t in ind_lower.replace(",", " ").split() if len(t.strip()) > 3]
        def _industry_match(v: str) -> bool:
            meta = vendor_meta.get(v, {})
            haystack = (
                meta.get("sub_industry", "") + " " +
                meta.get("primary_market", "") + " " +
                " ".join(vendor_cat_map.get(v, []))
            ).lower()
            return any(tok in haystack for tok in ind_tokens)
        industry_vendors = [v for v in _full_vendors if _industry_match(v)]
        # Industry-matched vendors first, then fallback list, then rest
        all_vendors = list(dict.fromkeys(
            (extra_vendors or []) + industry_vendors + _FALLBACK_VENDORS + _full_vendors
        ))
        logger.info(f"Industry filter '{industry}': {len(industry_vendors)} matched vendors prioritised")
    else:
        all_vendors = list(dict.fromkeys((extra_vendors or []) + _FALLBACK_VENDORS + _full_vendors))

    kw_product   = list(dict.fromkeys((extra_keywords or []) + lists.get("kw_product",    [])))
    kw_process   = list(dict.fromkeys(                         lists.get("kw_process",   [])))
    kw_technology= list(dict.fromkeys(                         lists.get("kw_technology", [])))
    sources      = list(dict.fromkeys((extra_sources  or []) + lists.get("all_sources",  [])))

    # Tier-based slice sizes
    n_vendors  = {1: TIER1_VENDORS,  2: TIER2_VENDORS,  3: TIER3_VENDORS}.get(tier, TIER1_VENDORS)
    vendors    = all_vendors[:n_vendors]

    # Build separate buckets — most targeted first so Apify cap hits best queries
    queries_vendor: list[str] = []
    queries_kw: list[str] = []
    queries_site: list[str] = []
    queries_broad: list[str] = []

    if tier == 1:
        # Vendor + market context (most precise — goes first in final list)
        for vendor in vendors:
            meta   = vendor_meta.get(vendor, {})
            market = meta.get("primary_market", "")
            cats   = vendor_cat_map.get(vendor, [])
            if market:
                queries_vendor.append(f'"{company_name}" "{vendor}" "{market}" deal OR contract ({yr_str})')
            elif cats:
                queries_vendor.append(f'"{company_name}" "{vendor}" "{cats[0]}" deal OR contract')
            else:
                queries_vendor.append(f'"{company_name}" "{vendor}" deal OR contract OR agreement')

        # Top product keywords (first 15)
        for kw in kw_product[:15]:
            queries_kw.append(f'"{company_name}" "{kw}" deal OR contract OR implementation ({yr_str})')

        # Top process keywords (first 10)
        for kw in kw_process[:10]:
            queries_kw.append(f'"{company_name}" "{kw}" vendor OR outsourcing OR contract ({yr_str})')

        # Site: queries — all 32 sources (every group) in T1
        for i in range(0, len(sources), SOURCE_GROUP_SZ):
            grp = sources[i: i + SOURCE_GROUP_SZ]
            site_expr = " OR ".join(f"site:{s}" for s in grp)
            queries_site.append(f'({site_expr}) "{company_name}" deal OR contract OR agreement ({yr_str})')

        # Broad anchors (2 per year range)
        queries_broad.append(f'"{company_name}" IT deal contract signed ({yr_str})')
        queries_broad.append(f'"{company_name}" technology outsourcing agreement ({yr_str})')

    elif tier == 2:
        # Vendor + sub-industry + process keyword combos
        for vendor in vendors:
            meta    = vendor_meta.get(vendor, {})
            sub_ind = meta.get("sub_industry", "")
            vkw_pr  = meta.get("kw_process", [])
            if vkw_pr:
                for kw in vkw_pr[:2]:
                    queries_vendor.append(f'"{company_name}" "{vendor}" "{kw}" ({yr_str})')
            elif sub_ind:
                queries_vendor.append(f'"{company_name}" "{vendor}" "{sub_ind}" contract OR deal')
            else:
                queries_vendor.append(f'"{company_name}" "{vendor}" deal OR contract OR agreement')

        # Remaining product keywords + first 50 process keywords
        for kw in kw_product[15:]:
            queries_kw.append(f'"{company_name}" "{kw}" deal OR contract OR implementation ({yr_str})')
        for kw in kw_process[10:60]:
            queries_kw.append(f'"{company_name}" "{kw}" vendor OR outsourcing OR contract ({yr_str})')

        # Technology keywords (first 40)
        for kw in kw_technology[:40]:
            queries_kw.append(f'"{company_name}" "{kw}" deal OR contract OR selected ({yr_str})')

        # No site: queries in T2 — all sources already covered in T1

    elif tier >= 3:
        # T3: keyword catch-alls only — no vendor queries (keeps T3 fast and cheap)
        for kw in kw_process[60:]:
            queries_kw.append(f'"{company_name}" "{kw}" vendor OR outsourcing OR contract ({yr_str})')
        for kw in kw_technology[40:]:
            queries_kw.append(f'"{company_name}" "{kw}" deal OR contract OR selected ({yr_str})')

        queries_broad.append(f'"{company_name}" vendor selected partnership announcement ({yr_str})')
        queries_broad.append(f'"{company_name}" outsourcing managed services digital transformation ({yr_str})')

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


APIFY_BATCH_SZ = 40   # max queries per Apify actor run (keeps each run under 180s)

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
        f"/run-sync-get-dataset-items?token={APIFY_KEY}&timeout=180&memory=512"
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
            async with httpx.AsyncClient(timeout=200) as client:
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

    # Primary: Google News RSS (free, clean real URLs, no anti-bot)
    raw = await _google_news_rss_search(queries, results_per_query=5)
    urls = _filter(raw)
    if urls:
        logger.info(f"Google News RSS returned {len(urls)} filtered URLs")
        return urls
    logger.warning("Google News RSS returned no URLs — trying Apify")

    # Secondary: Apify Google Search Scraper
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
) -> tuple[list[dict], int]:
    """Run one search tier. Returns (deals, relevant_page_count)."""
    queries = build_search_queries(
        company_name, goal, year_range, tier=tier,
        extra_vendors=extra_vendors,
        extra_sources=extra_sources,
        extra_keywords=extra_keywords,
        industry=industry,
    )
    raw_urls = await collect_urls(queries, max_urls=max_urls)
    urls = _dedup_by_domain([u for u in raw_urls if u not in seen_urls])
    seen_urls.update(urls)
    if not urls:
        return [], 0

    # Scrape: custom scraper → Apify → Jina (in priority order)
    scraped: list[dict] = []
    if SCRAPER_BASE_URL:
        scraped = await scrape_urls_custom(urls[:20])
    if not scraped and APIFY_KEY:
        for i in range(0, len(urls), 10):
            scraped.extend(await scrape_urls_apify(urls[i: i + 10]))
    if not scraped:
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
    year_range: tuple[int, int] = (2016, 2025),
    max_urls: int = 30,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
    industry: str = "",
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
        # T3 is a last-resort catch-all — only run if T1+T2 found nothing at all
        if tier == 3 and len(all_deals) > 0:
            break

        if tier == 1:
            yield {"type": "heartbeat", "message": f"🔍 Tier 1 search for {company_name} (sources + top keywords + vendors)…"}
        elif tier == 2:
            yield {"type": "heartbeat", "message": f"📈 Only {len(all_deals)} deals — escalating to Tier 2 (broader keywords + more vendors)…"}
        else:
            yield {"type": "heartbeat", "message": f"📈 Still 0 deals after T1+T2 — Tier 3 keyword catch-all…"}

        task = asyncio.ensure_future(_run_tier(
            company_name, goal, schema_fields, year_range, tier, max_urls,
            seen_urls, extra_vendors, extra_sources, extra_keywords, industry,
        ))
        elapsed = 0
        while not task.done() and elapsed < 1200:
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

    if not all_deals:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found from scraped content for {company_name} — skipping"}
        row = {"company_name": company_name, "domain": domain, "_status": "no_result", "_sources": total_relevant}
        for f in schema_fields:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat", "message": f"✅ {company_name}: {len(all_deals)} total deals"}
