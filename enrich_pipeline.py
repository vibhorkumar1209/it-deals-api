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

def build_search_queries(company_name: str, goal: str, year_range: tuple[int, int] = (2022, 2025)) -> list[str]:
    """Generate search queries: company-only + company×vendor combinations."""
    queries: list[str] = []
    years = list(range(year_range[0], year_range[1] + 1))[-2:]  # last 2 years

    for year in years:
        # Company + generic deal keywords
        queries += [
            f'"{company_name}" IT deal contract signed {year}',
            f'"{company_name}" technology outsourcing agreement {year}',
            f'"{company_name}" ERP CRM cloud vendor selected {year}',
            f'"{company_name}" digital transformation partnership {year}',
        ]
        # Company + top vendor pairs (high-signal searches)
        for vendor in TOP_VENDORS[:15]:  # top 15 to keep query count manageable
            queries.append(f'"{company_name}" "{vendor}" deal contract {year}')

    return queries


# ── Step 2: Search via Jina → collect URLs ────────────────────────────────────

async def _jina_search(query: str) -> list[str]:
    """Run one Jina search query, return list of URLs."""
    headers = {"Accept": "application/json"}
    if JINA_KEY:
        headers["Authorization"] = f"Bearer {JINA_KEY}"
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            r = await client.get(
                f"https://s.jina.ai/{httpx.URL(query)}",
                headers=headers,
            )
            if not r.is_success:
                return []
            data = r.json()
            results = data if isinstance(data, list) else data.get("data", [])
            return [item.get("url", "") for item in results if item.get("url")]
    except Exception as e:
        logger.debug(f"Jina search error: {e}")
        return []


async def collect_urls(queries: list[str], max_urls: int = 30) -> list[str]:
    """Run queries in batches of 5, collect unique URLs."""
    from urllib.parse import urlparse

    seen: set[str] = set()
    urls: list[str] = []

    batch_size = 5
    for i in range(0, len(queries), batch_size):
        batch = queries[i: i + batch_size]
        results = await asyncio.gather(*[_jina_search(q) for q in batch], return_exceptions=True)
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
                        return urls
        if len(urls) >= max_urls:
            break

    return urls


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

def _claude_extract(pages: list[dict], company_name: str, goal: str, schema_fields: list[dict]) -> dict:
    """
    Call Claude to extract schema fields from all scraped pages combined.
    Runs in a thread (blocking Anthropic SDK call).
    """
    import anthropic

    if not ANTHROPIC_KEY:
        return {}

    fields_desc = "\n".join(
        f'- {f["key"]}: {f.get("description") or f.get("label", "")} ({f.get("type","string")})'
        for f in schema_fields
    )

    # Combine page texts, truncate to fit context
    combined = ""
    for p in pages[:10]:
        snippet = p["text"][:3000]
        combined += f"\n\n[Source: {p['url']}]\n{snippet}"
    combined = combined[:20000]  # hard cap

    prompt = (
        f"You are extracting structured data about {company_name} from web research.\n\n"
        f"GOAL: {goal}\n\n"
        f"Extract the following fields from the research content below.\n"
        f"Return a JSON object with these exact keys:\n{fields_desc}\n\n"
        f"Rules:\n"
        f"- Use null for any field you cannot confirm from the content\n"
        f"- For list fields (e.g. multiple deals), return a semicolon-separated string\n"
        f"- Be specific — include vendor names, dates, values where found\n"
        f"- Return ONLY the JSON object, no explanation\n\n"
        f"RESEARCH CONTENT:\n{combined}"
    )

    try:
        ac = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        msg = ac.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw.strip())
        return json.loads(clean)
    except Exception as e:
        logger.warning(f"Claude extraction error: {e}")
        return {}


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2022, 2025),
    max_urls: int = 20,
) -> AsyncGenerator[dict, None]:
    """
    Full enrichment pipeline for one company. Yields progress events then final row.
    """

    yield {"type": "heartbeat", "message": f"🔍 Building search queries for {company_name}…"}

    # Step 1: Generate queries
    queries = build_search_queries(company_name, goal, year_range)
    yield {"type": "heartbeat", "message": f"🔍 Running {len(queries)} searches for {company_name}…"}

    # Step 2: Collect URLs (run in batches, yield heartbeat between)
    url_collect_task = asyncio.ensure_future(collect_urls(queries, max_urls=max_urls))
    elapsed = 0
    while not url_collect_task.done() and elapsed < 60:
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
        yield {"type": "heartbeat", "message": f"⚠️ No URLs found for {company_name} — trying direct Claude research…"}
        # Fall back to Claude-only with no scraped content
        parsed = await asyncio.to_thread(
            _claude_extract, [], company_name, goal, schema_fields
        )
        row = {"company_name": company_name, "domain": domain, "_status": "ok" if parsed else "no_result"}
        row.update({f["key"]: str(parsed.get(f["key"], "") or "") for f in schema_fields})
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

    # Step 5: Extract with Claude
    pages_to_use = relevant if relevant else scraped[:5]  # fallback to top pages if none flagged relevant
    parsed = await asyncio.to_thread(
        _claude_extract, pages_to_use, company_name, goal, schema_fields
    )

    row: dict = {
        "company_name": company_name,
        "domain": domain,
        "_status": "ok" if any(v for v in parsed.values() if v not in (None, "", "null")) else "no_result",
        "_sources": len(relevant),
    }
    for f in schema_fields:
        val = parsed.get(f["key"], "")
        row[f["key"]] = str(val) if val not in (None, "null") else ""

    yield {"type": "row_done", "row": row}
