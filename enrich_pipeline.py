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
TOP_VENDORS = [
    # Global SIs
    "TCS", "Infosys", "Wipro", "HCLTech", "Accenture", "IBM", "Cognizant",
    "Capgemini", "DXC Technology", "Tech Mahindra", "Hexaware", "Mphasis",
    "L&T Technology", "NIIT Technologies", "Birlasoft",
    # ERP / Cloud platforms
    "SAP", "Oracle", "Microsoft", "AWS", "Google Cloud", "Salesforce",
    "ServiceNow", "Workday", "SAP S/4HANA", "Microsoft Azure", "NetSuite",
    "Infor", "IFS", "Epicor",
    # Cybersecurity
    "Palo Alto Networks", "CrowdStrike", "Fortinet", "Zscaler", "Check Point",
    "Sophos", "Darktrace", "SentinelOne",
    # Analytics / Data / AI
    "Snowflake", "Databricks", "SAS", "Tableau", "Qlik", "MicroStrategy",
    "IBM Watson", "OpenAI", "DataRobot",
    # Banking / Fintech
    "Temenos", "Finacle", "Newgen", "FIS", "Fiserv", "Finastra",
    "Mambu", "Thought Machine", "Intellect Design",
    # Managed / Infra Services
    "Atos", "NTT", "Unisys", "CGI", "Fujitsu", "HPE", "Dell Technologies",
    "Cisco", "VMware",
]

# ── Broad query templates to maximise coverage ────────────────────────────────
BROAD_QUERY_TEMPLATES = [
    '"{company}" IT outsourcing deal {year}',
    '"{company}" technology contract signed {year}',
    '"{company}" digital transformation vendor {year}',
    '"{company}" cloud migration deal {year}',
    '"{company}" ERP implementation {year}',
    '"{company}" managed services contract {year}',
    '"{company}" cybersecurity deal {year}',
    '"{company}" software agreement {year}',
    '"{company}" IT partnership announcement {year}',
    '"{company}" systems integrator selected {year}',
]

# ── Domains to skip ───────────────────────────────────────────────────────────
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

def build_search_queries(company_name: str, goal: str, year_range: tuple[int, int] = (2021, 2025)) -> list[str]:
    """
    Generate a comprehensive set of search queries covering:
    - Broad deal templates × every year in range
    - Company × every top vendor (year-agnostic)
    - Press-release / announcement site-specific searches
    """
    queries: list[str] = []
    years = list(range(year_range[0], year_range[1] + 1))

    yr_str = " OR ".join(str(y) for y in years)

    # 1. High-value catch-alls (run first — fastest signal)
    queries.append(f'"{company_name}" IT deal contract signed awarded ({yr_str})')
    queries.append(f'"{company_name}" vendor selected outsourcing managed services ({yr_str})')
    queries.append(f'"{company_name}" digital transformation technology partnership ({yr_str})')
    queries.append(f'"{company_name}" ERP CRM cloud implementation agreement ({yr_str})')

    # 2. Press-release wires — very high signal, one per year
    for year in years:
        queries.append(f'site:businesswire.com OR site:prnewswire.com "{company_name}" IT deal {year}')
        queries.append(f'site:economictimes.indiatimes.com OR site:financialexpress.com "{company_name}" technology contract {year}')

    # 3. Broad template × year (capped at 6 templates to keep total manageable)
    for year in years:
        for tmpl in BROAD_QUERY_TEMPLATES[:6]:
            queries.append(tmpl.format(company=company_name, year=year))

    # 4. Company × top vendors (year-agnostic — catches undated / evergreen articles)
    for vendor in TOP_VENDORS[:20]:
        queries.append(f'"{company_name}" "{vendor}" deal OR contract OR agreement OR partnership')

    return queries


# ── Step 2: Search via Jina → collect URLs ────────────────────────────────────

async def _apify_google_search(queries: list[str], results_per_query: int = 10) -> list[str]:
    """
    Search Google via Apify Google Search Scraper actor.
    Batches queries in chunks of APIFY_BATCH_SIZE to avoid timeouts.
    Returns flat list of unique URLs from organic results.
    """
    APIFY_BATCH_SIZE = 20
    if not APIFY_KEY or not queries:
        return []

    seen: set[str] = set()
    urls: list[str] = []

    for batch_start in range(0, len(queries), APIFY_BATCH_SIZE):
        batch = queries[batch_start: batch_start + APIFY_BATCH_SIZE]
        try:
            actor_url = (
                "https://api.apify.com/v2/acts/apify~google-search-scraper"
                f"/run-sync-get-dataset-items?token={APIFY_KEY}&timeout=90&memory=512"
            )
            payload = {
                "queries": "\n".join(batch),
                "maxPagesPerQuery": 1,
                "resultsPerPage": results_per_query,
                "countryCode": "us",
                "languageCode": "en",
            }
            async with httpx.AsyncClient(timeout=100) as client:
                r = await client.post(actor_url, json=payload)
            if not r.is_success:
                logger.warning(f"Apify Google Search batch {batch_start}: {r.status_code} {r.text[:150]}")
                continue
            for item in r.json():
                for result in item.get("organicResults", []):
                    url = result.get("url", "")
                    if url and url not in seen:
                        seen.add(url)
                        urls.append(url)
            logger.info(f"Apify batch {batch_start//APIFY_BATCH_SIZE+1}: +{len(urls)} URLs so far")
        except Exception as e:
            logger.warning(f"Apify Google Search batch {batch_start} error: {e}")
            continue

    logger.info(f"Apify Google Search total: {len(urls)} URLs from {len(queries)} queries")
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


async def collect_urls(queries: list[str], max_urls: int = 60) -> list[str]:
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
        raw = await _apify_google_search(queries, results_per_query=15)
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

MIN_DEALS_BEFORE_EXTEND = 10   # if fewer deals found, search previous 5-year window too


async def _search_window(
    company_name: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int],
    max_urls: int,
    seen_urls: set,
) -> tuple[list[dict], list[dict], list[str]]:
    """
    Run one search window: build queries → collect URLs → scrape → filter → return.
    Returns (deals, relevant_pages, urls_found).
    seen_urls is mutated to track already-scraped URLs across windows.
    """
    queries = build_search_queries(company_name, goal, year_range)
    raw_urls = await collect_urls(queries, max_urls=max_urls)
    # Deduplicate against already-seen URLs
    urls = [u for u in raw_urls if u not in seen_urls]
    seen_urls.update(urls)

    if not urls:
        return [], [], []

    # Scrape
    if APIFY_KEY:
        scraped: list[dict] = []
        for i in range(0, len(urls), 10):
            batch_result = await scrape_urls_apify(urls[i: i + 10])
            scraped.extend(batch_result)
    else:
        scraped = await scrape_urls_jina_fallback(urls[:15])

    # Filter
    relevant = [p for p in scraped if is_deal_page(p["text"], company_name)]
    pages_to_use = relevant if relevant else scraped[:5]
    deals = await asyncio.to_thread(_claude_extract_deals, pages_to_use, company_name, goal, schema_fields)
    return deals, relevant, urls


async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2021, 2025),
    max_urls: int = 40,
) -> AsyncGenerator[dict, None]:
    """
    Full enrichment pipeline. Automatically extends to the previous 5-year
    window if fewer than MIN_DEALS_BEFORE_EXTEND deals are found in the first pass.
    """
    seen_urls: set = set()
    all_deals: list[dict] = []

    # ── Window 1: primary year range ─────────────────────────────────────────
    w1_label = f"{year_range[0]}–{year_range[1]}"
    yield {"type": "heartbeat", "message": f"🔍 Searching {company_name} deals ({w1_label})…"}

    w1_task = asyncio.ensure_future(
        _search_window(company_name, goal, schema_fields, year_range, max_urls, seen_urls)
    )
    elapsed = 0
    while not w1_task.done() and elapsed < 180:
        done, _ = await asyncio.wait({w1_task}, timeout=8)
        elapsed += 8
        if done:
            break
        batch_num = (elapsed // 100) + 1
        yield {"type": "heartbeat", "message": f"🔍 Searching {w1_label}… batch {batch_num} ({elapsed}s)"}

    if not w1_task.done():
        w1_task.cancel()
        logger.warning("Window 1 search timed out after 180s")
        w1_deals, w1_relevant, w1_urls = [], [], []
    else:
        try:
            w1_deals, w1_relevant, w1_urls = w1_task.result()
        except Exception as e:
            logger.warning(f"Window 1 error: {e}")
            yield {"type": "heartbeat", "message": f"⚠️ Search error: {e}"}
            w1_deals, w1_relevant, w1_urls = [], [], []

    yield {"type": "heartbeat", "message": f"📋 Window {w1_label}: {len(w1_deals)} deals from {len(w1_relevant)} relevant pages"}

    # Stream window-1 deals immediately
    for deal in w1_deals:
        row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": len(w1_relevant)}
        row.update(deal)
        all_deals.append(deal)
        yield {"type": "row_done", "row": row}

    # ── Window 2: extend back 5 years if not enough deals ────────────────────
    if len(all_deals) < MIN_DEALS_BEFORE_EXTEND:
        w2_range = (year_range[0] - 5, year_range[0] - 1)
        w2_label = f"{w2_range[0]}–{w2_range[1]}"
        yield {"type": "heartbeat", "message": f"📅 Only {len(all_deals)} deals found — extending search to {w2_label}…"}

        w2_task = asyncio.ensure_future(
            _search_window(company_name, goal, schema_fields, w2_range, max_urls, seen_urls)
        )
        elapsed = 0
        while not w2_task.done() and elapsed < 180:
            done, _ = await asyncio.wait({w2_task}, timeout=8)
            elapsed += 8
            if done:
                break
            batch_num = (elapsed // 100) + 1
            yield {"type": "heartbeat", "message": f"🔍 Searching {w2_label}… batch {batch_num} ({elapsed}s)"}

        if not w2_task.done():
            w2_task.cancel()
            logger.warning("Window 2 search timed out after 180s")
            w2_deals, w2_relevant = [], []
        else:
            try:
                w2_deals, w2_relevant, _ = w2_task.result()
            except Exception as e:
                logger.warning(f"Window 2 error: {e}")
                yield {"type": "heartbeat", "message": f"⚠️ Window 2 error: {e}"}
                w2_deals, w2_relevant = [], []

        yield {"type": "heartbeat", "message": f"📋 Window {w2_label}: {len(w2_deals)} deals from {len(w2_relevant)} relevant pages"}

        for deal in w2_deals:
            row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": len(w2_relevant)}
            row.update(deal)
            all_deals.append(deal)
            yield {"type": "row_done", "row": row}

    # ── Fallback: Claude knowledge if nothing scraped ─────────────────────────
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

    yield {"type": "heartbeat", "message": f"✅ {company_name}: {len(all_deals)} total deals found"}
