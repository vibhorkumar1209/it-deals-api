"""Search & discovery engine — Strategies A, B, C, D."""

import asyncio
import logging
import os
import random
from urllib.parse import urlparse

import httpx

from config_loader import ScraperConfig
from identity_pool import identity_pool

logger = logging.getLogger(__name__)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPER_KEY = os.getenv("SERPER_KEY", "")
SCRAPEDO_KEY = os.getenv("SCRAPEDO_KEY", "")

QUERY_TEMPLATES = [
    '"{company}" IT deal signed {year}',
    '"{company}" selects ERP OR CRM OR SAP OR Oracle OR Salesforce OR ServiceNow {year}',
    '"{company}" digital transformation contract awarded {year}',
    '"{company}" outsourcing agreement signed {year}',
    '"{company}" technology partnership announcement {year}',
    '"{company}" {year} annual report technology investment',
    'site:{domain} press release technology {year}',
]

NEWS_AGGREGATOR_DOMAINS = [
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "reuters.com", "ft.com", "techcrunch.com", "zdnet.com", "ciodive.com",
    "computerweekly.com", "theregister.com", "channelweb.co.uk",
]

SEM = asyncio.Semaphore(5)


async def _serpapi_search(query: str) -> list[str]:
    # Try Serper.dev first (serper.dev API)
    if SERPER_KEY:
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                r = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                    json={"q": query, "num": 20, "gl": "us", "hl": "en"},
                )
                if r.status_code == 200:
                    data = r.json()
                    urls = [item.get("link", "") for item in data.get("organic", []) if item.get("link")]
                    urls += [item.get("link", "") for item in data.get("news", []) if item.get("link")]
                    if urls:
                        return [u for u in urls if u]
                else:
                    logger.warning(f"Serper {r.status_code} for '{query[:50]}': {r.text[:100]}")
        except Exception as e:
            logger.warning(f"Serper failed for '{query}': {e}")

    # Fallback: SerpAPI (serpapi.com)
    if not SERPAPI_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://serpapi.com/search",
                params={"q": query, "num": 20, "hl": "en", "gl": "us", "api_key": SERPAPI_KEY},
            )
            data = r.json()
            urls = [item.get("link", "") for item in data.get("organic_results", []) if item.get("link")]
            urls += [item.get("link", "") for item in data.get("news_results", []) if item.get("link")]
            return [u for u in urls if u]
    except Exception as e:
        logger.warning(f"SerpAPI failed for '{query}': {e}")
        return []


async def _ddg_search(query: str) -> list[str]:
    try:
        from duckduckgo_search import DDGS
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: list(DDGS().news(query, max_results=20))
        )
        return [r.get("url", "") for r in results if r.get("url")]
    except Exception as e:
        logger.warning(f"DDG search failed for '{query}': {e}")
        return []


async def _google_search_fallback(query: str) -> list[str]:
    try:
        from googlesearch import search as gsearch
        loop = asyncio.get_event_loop()
        await asyncio.sleep(random.uniform(5, 10))
        results = await loop.run_in_executor(None, lambda: list(gsearch(query, num_results=15, lang="en")))
        return results
    except Exception as e:
        logger.warning(f"googlesearch fallback failed: {e}")
        return []


async def _scrapedo_search(query: str) -> list[str]:
    """Fetch Google SERP via scrape.do — fires when SerpAPI + DDG both fail."""
    if not SCRAPEDO_KEY:
        return []
    from urllib.parse import quote_plus, urlencode, quote
    from bs4 import BeautifulSoup
    google_url = f"https://www.google.com/search?q={quote_plus(query)}&num=20&hl=en&gl=us"
    scrape_url = f"https://api.scrape.do?token={SCRAPEDO_KEY}&url={quote(google_url, safe='')}&render=true"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(scrape_url)
            if r.status_code != 200:
                logger.warning(f"scrape.do {r.status_code} for query '{query[:50]}'")
                return []
            soup = BeautifulSoup(r.text, "lxml")
            urls = []
            # Google organic results are in <a> tags with href=/url?q=...
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/url?q="):
                    actual = href[7:].split("&")[0]
                    if actual.startswith("http") and "google.com" not in actual:
                        urls.append(actual)
            logger.info(f"scrape.do returned {len(urls)} URLs for '{query[:50]}'")
            return urls
    except Exception as e:
        logger.warning(f"scrape.do search failed: {e}")
        return []


async def _run_query(query: str) -> list[str]:
    async with SEM:
        urls = await _serpapi_search(query)
        if not urls:
            urls = await _ddg_search(query)
        if not urls:
            urls = await _google_search_fallback(query)
        if not urls:
            urls = await _scrapedo_search(query)
        return urls


async def strategy_a_search(config: ScraperConfig) -> list[str]:
    """Generate all search queries and collect URLs."""
    all_urls: list[str] = []
    tasks = []

    companies = config.all_company_names
    years = range(config.search_year_range["start"], config.search_year_range["end"] + 1)

    for company in companies:
        for year in years:
            for tmpl in QUERY_TEMPLATES:
                query = tmpl.format(company=company, year=year, domain=config.domain)
                tasks.append(_run_query(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    for r in results:
        if isinstance(r, list):
            all_urls.extend(r)

    # Deduplicate
    return list(dict.fromkeys(u for u in all_urls if u))


async def strategy_b_known_sources(config: ScraperConfig) -> list[str]:
    """Scrape user-supplied known source URLs for article links."""
    from bs4 import BeautifulSoup
    urls: list[str] = []

    async def _scrape_source(source_url: str):
        try:
            async with SEM:
                await identity_pool.wait_for_domain(source_url, "static")
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    r = await client.get(source_url, headers=identity_pool.next_headers())
                    soup = BeautifulSoup(r.text, "lxml")

                    # RSS/feed
                    if "rss" in source_url.lower() or "feed" in source_url.lower():
                        for item in soup.find_all("item"):
                            link = item.find("link")
                            if link:
                                urls.append(link.get_text(strip=True))
                        return

                    # Scan for matching links
                    company_names_lower = [n.lower() for n in config.all_company_names]
                    for a in soup.find_all("a", href=True):
                        href = a["href"]
                        if href.startswith("http") and any(n in a.get_text().lower() for n in company_names_lower):
                            urls.append(href)

                    # Also return the source URL itself for direct parse
                    urls.append(source_url)
        except Exception as e:
            logger.warning(f"Strategy B failed for {source_url}: {e}")

    await asyncio.gather(*[_scrape_source(u) for u in config.known_sources])
    return list(dict.fromkeys(u for u in urls if u))


async def strategy_c_linkedin(config: ScraperConfig) -> list[str]:
    """Return LinkedIn URL for later processing by website_router."""
    if not config.run_linkedin:
        return []
    return [config.linkedin_url]


async def strategy_d_news_aggregators(config: ScraperConfig) -> list[str]:
    """Search news aggregators for company mentions."""
    tasks = []
    company = config.company_name

    for domain in NEWS_AGGREGATOR_DOMAINS:
        query = f'site:{domain} "{company}" deal OR contract OR ERP OR CRM OR SAP OR Oracle'
        tasks.append(_run_query(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    urls = []
    for r in results:
        if isinstance(r, list):
            urls.extend(r)
    return list(dict.fromkeys(u for u in urls if u))


async def discover_all_urls(config: ScraperConfig) -> list[str]:
    """Run all four strategies in parallel, deduplicate, filter skip list."""
    results = await asyncio.gather(
        strategy_a_search(config),
        strategy_b_known_sources(config),
        strategy_c_linkedin(config),
        strategy_d_news_aggregators(config),
        return_exceptions=True,
    )

    all_urls = []
    for r in results:
        if isinstance(r, list):
            all_urls.extend(r)

    # Remove already-processed deals
    skip = set(config.known_deals_to_skip)
    unique = list(dict.fromkeys(u for u in all_urls if u and u not in skip))

    logger.info(f"Discovered {len(unique)} unique URLs across all strategies")
    return unique
