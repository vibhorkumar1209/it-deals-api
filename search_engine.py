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

SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "")
SERPER_KEY     = os.getenv("SERPER_KEY", "")
SCRAPEDO_KEY   = os.getenv("SCRAPEDO_KEY", "")
GOOGLE_PSE_KEY = os.getenv("GOOGLE_PSE_KEY", "")   # Google Custom Search API key
GOOGLE_PSE_CX  = os.getenv("GOOGLE_PSE_CX", "")    # Programmable Search Engine ID
BRAVE_KEY      = os.getenv("BRAVE_KEY", "")         # Brave Search API — 2000 free/month

QUERY_TEMPLATES = [
    # Formal deal / contract — target news wires
    '"{company}" IT contract awarded signed {year}',
    '"{company}" outsourcing agreement managed services {year}',
    '"{company}" technology deal announcement {year}',

    # Vendor selection — split across vendor groups to stay under query length limits
    '"{company}" selects SAP OR Oracle OR Salesforce OR ServiceNow OR Workday {year}',
    '"{company}" selects Microsoft OR AWS OR "Google Cloud" OR IBM OR Accenture {year}',
    '"{company}" chooses OR adopts ERP OR CRM OR HCM {year}',

    # Implementation / go-live
    '"{company}" SAP implementation go-live {year}',
    '"{company}" Oracle OR Workday OR Salesforce deployment rollout {year}',
    '"{company}" "Microsoft Azure" OR AWS migration deployment {year}',

    # Partnership / alliance
    '"{company}" technology partnership "strategic alliance" {year}',
    '"{company}" partners Accenture OR Infosys OR TCS OR Wipro OR Capgemini {year}',

    # Outsourcing / managed services with named SIs
    '"{company}" outsourcing IBM OR DXC OR Infosys OR HCLTech OR Unisys {year}',
    '"{company}" "systems integrator" OR "SI partner" contract {year}',

    # Cybersecurity deals
    '"{company}" cybersecurity contract CrowdStrike OR "Palo Alto" OR Fortinet OR Zscaler {year}',

    # Explicitly target press release newswires
    '"{company}" "signed" OR "selected" site:businesswire.com {year}',
    '"{company}" "signed" OR "selected" site:prnewswire.com {year}',
    '"{company}" technology site:reuters.com OR site:zdnet.com OR site:ciodive.com {year}',
]

NEWS_AGGREGATOR_DOMAINS = [
    # Core newswires
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "reuters.com", "ft.com", "techcrunch.com", "zdnet.com", "ciodive.com",
    "computerweekly.com", "theregister.com", "channelweb.co.uk",
    # Extended source list
    "analyticsindiamag.com",
    "appsruntheworld.com",
    "tenders.gov.au",
    "biovoicenews.com",
    "bloombergquint.com",
    "business-standard.com",
    "businesstoday.in",
    "businesswireindia.com",
    "cioinsight.com",
    "cio.economictimes.indiatimes.com",
    "ciol.com",
    "computer.financialexpress.com",
    "crn.in",
    "cxotoday.com",
    "cyantechnology.com",
    "datacentres.com",
    "data4experts.com",
    "dqindia.com",
    "deccanherald.com",
    "articles.economictimes.indiatimes.com",
    "efytimes.com",
    "enterprisetimes.co.uk",
    "equitybulls.com",
    "etnews.com",
    "ted.europa.eu",
    "exchange4media.com",
    "expresscomputer.in",
    "finalaya.com",
    "finextra.com",
    "fintechfutures.com",
    "globaltelecomsbusiness.com",
    "indiainfoline.com",
    "informationweek.in",
    "mmb.moneycontrol.com",
    "outsourcingdigest.com",
    "psuconnect.in",
    "in.reuters.com",
    "articles.timesofindia.indiatimes.com",
    "contractsfinder.service.gov.uk",
    "automotivelogistics.media",
    "biztech2.in.com",
    "wsj.com",
    "manufacturing.economictimes.indiatimes.com",
]

# Full base URLs for direct scraping (Strategy B style for these sources)
EXTENDED_SOURCE_BASE_URLS = [
    "https://analyticsindiamag.com/",
    "https://www.appsruntheworld.com/",
    "https://www.tenders.gov.au/",
    "https://www.biovoicenews.com/",
    "https://www.bloombergquint.com/business/",
    "https://www.businesswire.com/news",
    "https://www.business-standard.com/article/news-cm/",
    "https://www.businesstoday.in/latest/corporate/story",
    "http://businesswireindia.com/news/",
    "http://www.cioinsight.com/",
    "http://cio.economictimes.indiatimes.com/news/",
    "http://www.ciol.com/ciol/news/",
    "http://computer.financialexpress.com/news/",
    "http://www.crn.in/",
    "http://www.cxotoday.com/story",
    "https://www.ciol.com/",
    "http://www.datacentres.com/news/",
    "http://www.dqindia.com/",
    "https://www.deccanherald.com/business/business-news/",
    "http://articles.economictimes.indiatimes.com/",
    "http://www.efytimes.com/",
    "https://www.enterprisetimes.co.uk/",
    "https://www.equitybulls.com/",
    "https://www.exchange4media.com/marketing-news/",
    "https://www.expresscomputer.in/news/",
    "http://www.finextra.com/News/",
    "https://www.fintechfutures.com/",
    "http://www.outsourcingdigest.com/",
    "https://www.psuconnect.in/news/",
    "https://www.contractsfinder.service.gov.uk/",
    "https://ted.europa.eu/",
    "http://automotivelogistics.media/news/",
    "https://manufacturing.economictimes.indiatimes.com/news/",
]

SEM = asyncio.Semaphore(5)


async def _brave_search(query: str) -> list[str]:
    """Brave Search API — 2,000 free queries/month, no key restrictions.
    Docs: https://api.search.brave.com/app/documentation/web-search/get-started
    """
    if not BRAVE_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": BRAVE_KEY,
                },
                params={"q": query, "count": 20, "country": "us", "search_lang": "en"},
            )
            if r.status_code == 429:
                logger.warning("Brave Search monthly quota reached")
                return []
            if r.status_code != 200:
                logger.warning(f"Brave {r.status_code} for '{query[:50]}': {r.text[:120]}")
                return []
            data = r.json()
            urls = [item.get("url", "") for item in data.get("web", {}).get("results", []) if item.get("url")]
            if urls:
                logger.info(f"Brave returned {len(urls)} URLs for '{query[:50]}'")
            return urls
    except Exception as e:
        logger.warning(f"Brave search failed for '{query[:50]}': {e}")
        return []


async def _google_pse_search(query: str) -> list[str]:
    """Google Programmable Search Engine (site-restricted mode).
    Uses the /siterestrict endpoint which works for engines configured
    with specific domains. Returns up to 20 results per query (2 pages × 10).
    Free tier: 100 queries/day.
    Docs: https://developers.google.com/custom-search/v1/reference/rest/v1/cse.siterestrict/list
    """
    if not GOOGLE_PSE_KEY or not GOOGLE_PSE_CX:
        return []
    urls: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            for start in (1, 11):
                r = await client.get(
                    "https://www.googleapis.com/customsearch/v1/siterestrict",
                    params={
                        "key":   GOOGLE_PSE_KEY,
                        "cx":    GOOGLE_PSE_CX,
                        "q":     query,
                        "num":   10,
                        "start": start,
                        "gl":    "us",
                        "hl":    "en",
                    },
                )
                if r.status_code == 429:
                    logger.warning("Google PSE daily quota reached")
                    break
                if r.status_code != 200:
                    logger.warning(f"Google PSE {r.status_code} for '{query[:50]}': {r.text[:120]}")
                    break
                data = r.json()
                for item in data.get("items", []):
                    link = item.get("link", "")
                    if link:
                        urls.append(link)
        if urls:
            logger.info(f"Google PSE returned {len(urls)} URLs for '{query[:50]}'")
    except Exception as e:
        logger.warning(f"Google PSE failed for '{query[:50]}': {e}")
    return urls


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
        # 1. Brave Search (2000 free/month, no restrictions)
        urls = await _brave_search(query)
        if not urls:
            # 2. Google PSE (100 free/day, site-restricted engine)
            urls = await _google_pse_search(query)
        if not urls:
            # 3. Serper.dev / SerpAPI
            urls = await _serpapi_search(query)
        if not urls:
            # 4. DuckDuckGo (no key needed, rate-limited)
            urls = await _ddg_search(query)
        if not urls:
            # 5. googlesearch-python (slow fallback)
            urls = await _google_search_fallback(query)
        if not urls:
            # 6. scrape.do Google SERP parse
            urls = await _scrapedo_search(query)
        return urls


async def strategy_a_search(config: ScraperConfig) -> list[str]:
    """Generate all search queries and collect URLs.

    PSE quota awareness: Google PSE allows 100 queries/day free.
    With N templates × M years × K company names that can exceed the limit fast.
    When PSE is the only active backend we cap to the 10 highest-value templates
    and the most recent 3 years to stay within ~90 queries for a single company.
    """
    pse_only = (bool(GOOGLE_PSE_KEY) or bool(BRAVE_KEY)) and not SERPAPI_KEY and not SERPER_KEY

    companies = config.all_company_names
    years_full = list(range(config.search_year_range["start"], config.search_year_range["end"] + 1))

    if pse_only:
        # Use primary company name only + last 3 years + first 10 templates
        companies = [config.company_name]
        years_full = sorted(years_full)[-3:]
        templates = QUERY_TEMPLATES[:10]
        logger.info(f"PSE-only mode: {len(templates)} templates × {len(years_full)} years = "
                    f"{len(templates)*len(years_full)} queries")
    else:
        templates = QUERY_TEMPLATES

    tasks = []
    for company in companies:
        for year in years_full:
            for tmpl in templates:
                query = tmpl.format(company=company, year=year, domain=config.domain)
                tasks.append(_run_query(query))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_urls: list[str] = []
    for r in results:
        if isinstance(r, list):
            all_urls.extend(r)

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

    # Combine user-supplied sources with the built-in extended source list
    all_sources = list(dict.fromkeys(config.known_sources + EXTENDED_SOURCE_BASE_URLS))
    await asyncio.gather(*[_scrape_source(u) for u in all_sources])
    return list(dict.fromkeys(u for u in urls if u))


async def strategy_c_linkedin(config: ScraperConfig) -> list[str]:
    """Return LinkedIn URL for later processing by website_router."""
    if not config.run_linkedin:
        return []
    return [config.linkedin_url]


async def strategy_d_news_aggregators(config: ScraperConfig) -> list[str]:
    """Search top news aggregator domains — capped to 15 to limit wall time."""
    TOP_DOMAINS = NEWS_AGGREGATOR_DOMAINS[:15]
    tasks = []
    company = config.company_name  # primary name only — aliases handled in Strategy A
    for domain in TOP_DOMAINS:
        query = (
            f'site:{domain} "{company}" '
            f'deal OR contract OR ERP OR CRM OR SAP OR Oracle OR '
            f'outsourc OR cloud OR cybersecurity OR transformation'
        )
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
