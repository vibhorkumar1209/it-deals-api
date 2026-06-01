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

# ── Top vendors to combine in searches ───────────────────────────────────────
# A curated shortlist covering the most commonly found IT deal vendors.
# Used to build targeted "company + vendor" search queries.
TOP_VENDORS = [
    # Global SIs
    "TCS", "Infosys", "Wipro", "HCLTech", "Accenture", "IBM", "Cognizant",
    "Capgemini", "DXC Technology", "Tech Mahindra",
    # ERP / Cloud platforms
    "SAP", "Oracle", "Microsoft", "AWS", "Google Cloud", "Salesforce",
    "ServiceNow", "Workday", "SAP S/4HANA",
    # Cybersecurity
    "Palo Alto Networks", "CrowdStrike", "Fortinet", "Zscaler",
    # Analytics / Data
    "Snowflake", "Databricks", "SAS",
    # Banking / Fintech
    "Temenos", "Finacle", "Newgen", "FIS", "Fiserv", "Finastra",
    # Managed Services
    "Atos", "NTT", "Unisys",
]

# ── Deal search keywords ──────────────────────────────────────────────────────
DEAL_KEYWORDS = [
    "deal", "contract", "signed", "awarded", "selected", "partnership",
    "outsourcing", "implementation", "agreement", "go-live", "digital transformation",
]

# ── Domains to skip (social, aggregators with no article text) ────────────────
SKIP_DOMAINS = {
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "youtube.com", "reddit.com", "quora.com", "wikipedia.org",
    "glassdoor.com", "indeed.com", "crunchbase.com",
}

JINA_KEY = os.getenv("JINA_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
APIFY_KEY = os.getenv("APIFY_API_KEY", "")


# ── Step 1: Generate search queries ──────────────────────────────────────────

def build_search_queries(
    company_name: str,
    goal: str,
    year_range: tuple[int, int] = (2022, 2025),
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> list[str]:
    """
    Generate search queries combining:
    - Generic deal templates × years
    - Company × vendors (built-in + caller-supplied)
    - site: queries for caller-supplied source domains
    - Keyword-boosted catch-alls
    """
    queries: list[str] = []
    years = list(range(year_range[0], year_range[1] + 1))
    yr_str = " OR ".join(str(y) for y in years)

    # Merge vendor lists (caller list takes priority / deduped)
    vendors = list(dict.fromkeys((extra_vendors or []) + TOP_VENDORS))

    # Merge keyword signals
    base_kw = ["IT deal", "technology contract", "outsourcing agreement",
                "digital transformation", "managed services", "cloud migration",
                "ERP implementation", "cybersecurity deal", "vendor selected"]
    all_kw = list(dict.fromkeys((extra_keywords or []) + base_kw))

    # 1. Generic deal queries per year
    for year in years:
        queries.append(f'"{company_name}" IT deal contract signed {year}')
        queries.append(f'"{company_name}" technology outsourcing agreement {year}')

    # 2. Keyword × year range (extra keywords get their own targeted queries)
    for kw in (extra_keywords or [])[:10]:
        queries.append(f'"{company_name}" {kw} ({yr_str})')

    # 3. Company × vendor pairs (top vendors + caller vendors)
    for vendor in vendors[:20]:
        queries.append(f'"{company_name}" "{vendor}" deal OR contract OR agreement')

    # 4. site: queries for caller-supplied sources (high-signal)
    if extra_sources:
        # Group up to 3 sources per query to avoid URL length limits
        for i in range(0, min(len(extra_sources), 30), 3):
            group = extra_sources[i: i + 3]
            site_expr = " OR ".join(f'site:{s.strip()}' for s in group)
            queries.append(f'({site_expr}) "{company_name}" deal contract ({yr_str})')

    # 5. Broad catch-all
    kw_sample = " OR ".join(f'"{k}"' for k in all_kw[:5])
    queries.append(f'"{company_name}" ({kw_sample}) ({yr_str})')

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
            f"Extract EVERY distinct deal or contract mentioned. Each deal = one JSON object.\n"
            f"Return a JSON ARRAY where each element has these exact keys:\n{fields_desc}\n\n"
            f"Rules:\n"
            f"- One object per deal/contract — do NOT merge multiple deals into one\n"
            f"- Include deals from all years found in the content (2020–2025)\n"
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
            f"Return a JSON ARRAY where each element is one deal with these exact keys:\n{fields_desc}\n\n"
            f"Rules:\n"
            f"- One object per deal — do NOT merge multiple deals\n"
            f"- Include as many distinct deals as you know (target 5–15 deals)\n"
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

async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2022, 2025),
    max_urls: int = 20,
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Full enrichment pipeline for one company. Yields progress events then final row.
    Accepts optional extra vendors, sources (domains), and keywords to boost coverage.
    """
    yield {"type": "heartbeat", "message": f"🔍 Building search queries for {company_name}…"}

    # Step 1: Generate queries
    queries = build_search_queries(
        company_name, goal, year_range,
        extra_vendors=extra_vendors,
        extra_sources=extra_sources,
        extra_keywords=extra_keywords,
    )
    boost_info = []
    if extra_vendors:  boost_info.append(f"{len(extra_vendors)} vendors")
    if extra_sources:  boost_info.append(f"{len(extra_sources)} sources")
    if extra_keywords: boost_info.append(f"{len(extra_keywords)} keywords")
    boost_str = f" [+{', '.join(boost_info)}]" if boost_info else ""
    yield {"type": "heartbeat", "message": f"🔍 Running {len(queries)} searches for {company_name}{boost_str}…"}

    # Step 2: Collect URLs (run in batches, yield heartbeat between)
    url_collect_task = asyncio.ensure_future(collect_urls(queries, max_urls=max_urls))
    elapsed = 0
    while not url_collect_task.done() and elapsed < 90:
        done, _ = await asyncio.wait({url_collect_task}, timeout=8)
        elapsed += 8
        if done:
            break
        yield {"type": "heartbeat", "message": f"🔍 Searching… ({elapsed}s)"}
    if not url_collect_task.done():
        url_collect_task.cancel()
        urls: list[str] = []
    else:
        try:
            urls = url_collect_task.result()
        except Exception:
            urls = []

    if not urls:
        yield {"type": "heartbeat", "message": f"⚠️ No URLs found — using Claude knowledge for {company_name}…"}
        deals: list[dict] = await asyncio.to_thread(
            _claude_extract_deals, [], company_name, goal, schema_fields
        )
        if not deals:
            row = {"company_name": company_name, "domain": domain, "_status": "no_result", "_sources": 0}
            for f in schema_fields:
                row[f["key"]] = ""
            yield {"type": "row_done", "row": row}
            return
        yield {"type": "heartbeat", "message": f"✅ Found {len(deals)} deals for {company_name} (from knowledge)"}
        for deal in deals:
            row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": 0}
            row.update(deal)
            yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat", "message": f"🕸️ Scraping {len(urls)} URLs for {company_name}…"}

    # Step 3: Scrape via Apify (or Jina fallback)
    if APIFY_KEY:
        # Scrape in batches of 10 to stay within Apify timeout
        scraped: list[dict] = []
        for i in range(0, len(urls), 10):
            batch = urls[i: i + 10]
            batch_result = await scrape_urls_apify(batch)
            scraped.extend(batch_result)
            if i + 10 < len(urls):
                yield {"type": "heartbeat", "message": f"🕸️ Scraped {len(scraped)} pages… ({i+10}/{len(urls)})"}
    else:
        yield {"type": "heartbeat", "message": f"🕸️ Scraping via Jina (no Apify key)…"}
        scraped = await scrape_urls_jina_fallback(urls[:10])

    yield {"type": "heartbeat", "message": f"✅ Scraped {len(scraped)} pages — identifying deals…"}

    # Step 4: Filter deal-relevant pages
    relevant = [p for p in scraped if is_deal_page(p["text"], company_name)]
    yield {"type": "heartbeat", "message": f"📋 {len(relevant)}/{len(scraped)} pages are deal-relevant — extracting…"}

    # Step 5: Extract multiple deals with Claude
    pages_to_use = relevant if relevant else scraped[:5]
    deals: list[dict] = await asyncio.to_thread(
        _claude_extract_deals, pages_to_use, company_name, goal, schema_fields
    )

    if not deals:
        # No deals extracted — yield a no_result row
        row: dict = {
            "company_name": company_name,
            "domain": domain,
            "_status": "no_result",
            "_sources": len(relevant),
        }
        for f in schema_fields:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat", "message": f"✅ Found {len(deals)} deals for {company_name}"}

    # Yield one row per deal
    for deal in deals:
        row = {
            "company_name": company_name,
            "domain": domain,
            "_status": "ok",
            "_sources": len(relevant),
        }
        row.update(deal)
        yield {"type": "row_done", "row": row}
