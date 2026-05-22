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
JINA_KEY       = os.getenv("JINA_KEY", "")          # Jina AI Search — s.jina.ai

# Templates for customer-centric search (company is the buyer / enterprise)
QUERY_TEMPLATES_CUSTOMER = [
    # Formal deal / contract
    '"{company}" IT contract awarded signed {year}',
    '"{company}" outsourcing agreement managed services {year}',
    '"{company}" technology deal announcement {year}',

    # Vendor selection
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

    # Outsourcing / managed services
    '"{company}" outsourcing IBM OR DXC OR Infosys OR HCLTech OR Unisys {year}',
    '"{company}" "systems integrator" OR "SI partner" contract {year}',

    # Cybersecurity deals
    '"{company}" cybersecurity contract CrowdStrike OR "Palo Alto" OR Fortinet OR Zscaler {year}',

    # News wire targeting
    '"{company}" "signed" OR "selected" site:businesswire.com {year}',
    '"{company}" "signed" OR "selected" site:prnewswire.com {year}',
    '"{company}" technology site:reuters.com OR site:zdnet.com OR site:ciodive.com {year}',
]

# Templates for vendor-centric search (company is the technology provider / vendor)
QUERY_TEMPLATES_VENDOR = [
    # Who is selecting / adopting this vendor's product
    'selects "{company}" contract deal {year}',
    'adopts "{company}" implementation {year}',
    'chooses "{company}" deployment {year}',
    'signs "{company}" agreement {year}',

    # Named customer segments
    'bank OR insurance OR retail selects "{company}" {year}',
    'government OR healthcare selects "{company}" {year}',
    'enterprise selects OR implements "{company}" {year}',

    # Go-live / deployment
    '"{company}" go-live OR deployment OR rollout customer {year}',
    '"{company}" customer win deal signed {year}',
    '"{company}" contract awarded customer {year}',

    # Partnership with SIs
    '"{company}" Accenture OR Infosys OR Deloitte OR IBM implementation {year}',
    '"{company}" partner ecosystem deal {year}',

    # News wire
    '"{company}" selected site:businesswire.com {year}',
    '"{company}" customer deal site:prnewswire.com OR site:globenewswire.com {year}',
    '"{company}" wins contract site:reuters.com OR site:zdnet.com {year}',
]

# Unified default (backward-compatible)
QUERY_TEMPLATES = QUERY_TEMPLATES_CUSTOMER

# Known vendor names (lowercased) — triggers vendor-centric mode
KNOWN_VENDOR_NAMES = {
    "aws", "amazon web services", "microsoft azure", "azure", "google cloud",
    "sap", "oracle", "salesforce", "servicenow", "workday", "ibm",
    "accenture", "infosys", "tcs", "wipro", "capgemini", "cognizant",
    "deloitte", "pwc", "kpmg", "ey", "hcltech", "dxc", "atos",
    "palo alto networks", "crowdstrike", "fortinet", "zscaler", "splunk",
    "snowflake", "databricks", "tableau", "power bi", "successfactors",
    "ariba", "dynamics 365", "netsuite", "infor", "epicor",
}

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


async def _jina_search(query: str) -> list[str]:
    """Jina AI Search (s.jina.ai) — returns top 10 URLs per query.
    X-Respond-With: no-content means only metadata returned (fast, no scraping).
    """
    if not JINA_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://s.jina.ai/",
                params={"q": query},
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {JINA_KEY}",
                    "X-Respond-With": "no-content",
                },
            )
            if r.status_code == 429:
                logger.warning("Jina AI search rate limited")
                return []
            if r.status_code != 200:
                logger.warning(f"Jina {r.status_code} for '{query[:50]}': {r.text[:120]}")
                return []
            urls = [item.get("url", "") for item in r.json().get("data", []) if item.get("url")]
            if urls:
                logger.info(f"Jina returned {len(urls)} URLs for '{query[:50]}'")
            return urls
    except Exception as e:
        logger.warning(f"Jina search failed for '{query[:50]}': {e}")
        return []


async def _resolve_gnews_urls(gnews_urls: list[str]) -> list[str]:
    """No-op: gnews URLs are already extracted from description <a href> in _google_news_rss."""
    return [u for u in gnews_urls if u]


async def _google_news_rss(query: str) -> list[str]:
    """Search Google News RSS — completely free, no API key, no rate limit for
    reasonable use. Returns news articles from Google's full index.
    URL format: https://news.google.com/rss/search?q=<query>&hl=en-US&gl=US
    """
    from urllib.parse import quote_plus
    from bs4 import BeautifulSoup
    try:
        url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
            })
            if r.status_code != 200:
                logger.warning(f"Google News RSS {r.status_code} for '{query[:50]}'")
                return []
            soup = BeautifulSoup(r.text, "lxml-xml")
            real_urls = []
            for item in soup.find_all("item"):
                # Description contains HTML: <a href="REAL_URL">title</a>&nbsp;&nbsp;<font>source</font>
                desc = item.find("description")
                if desc:
                    desc_text = desc.get_text()
                    # Parse embedded HTML in description
                    desc_soup = BeautifulSoup(desc_text, "html.parser")
                    a = desc_soup.find("a", href=True)
                    if a and a["href"].startswith("http") and "news.google.com" not in a["href"]:
                        real_urls.append(a["href"])
                        continue
                # Fallback: try <link> sibling text (some feeds put URL as text node after <link>)
                link = item.find("link")
                if link and link.next_sibling:
                    href = str(link.next_sibling).strip()
                    if href.startswith("http") and "news.google.com" not in href:
                        real_urls.append(href)

            if real_urls:
                logger.info(f"Google News RSS: {len(real_urls)} real URLs for '{query[:50]}'")
            return real_urls
    except Exception as e:
        logger.warning(f"Google News RSS failed for '{query[:50]}': {e}")
        return []


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


def _simplify_query(query: str) -> str:
    """Strip site:, OR, quotes-within-quotes — keeps Google News RSS happy."""
    import re as _re
    q = _re.sub(r'site:\S+', '', query)       # remove site: operators
    q = _re.sub(r'\bOR\b', '', q)              # remove OR
    q = _re.sub(r'["\']', '', q)               # remove quotes
    return ' '.join(q.split())


async def _run_query(query: str) -> list[str]:
    async with SEM:
        # 1. Jina AI Search — reliable, returns real URLs
        urls = await _jina_search(query)
        if not urls:
            # 2. Google News RSS with simplified query (RSS chokes on OR / site: operators,
            #    and returns gnews redirect URLs that can't be resolved without JS —
            #    so we skip it for query-based search and rely on strategy_rss() feeds)
            pass
        if not urls:
            # 3. Brave Search
            urls = await _brave_search(query)
        if not urls:
            # 4. Google PSE (site-restricted engine)
            urls = await _google_pse_search(query)
        if not urls:
            # 5. Serper.dev / SerpAPI
            urls = await _serpapi_search(query)
        if not urls:
            # 6. DuckDuckGo
            urls = await _ddg_search(query)
        if not urls:
            # 7. googlesearch-python
            urls = await _google_search_fallback(query)
        if not urls:
            # 8. scrape.do Google SERP parse
            urls = await _scrapedo_search(query)
        return urls


async def strategy_a_search(config: ScraperConfig) -> list[str]:
    """Generate all search queries and collect URLs.

    Auto-detects vendor-centric vs customer-centric mode:
    - If the searched company is a known vendor (AWS, SAP, Salesforce…) it uses
      QUERY_TEMPLATES_VENDOR which find "Company X selects AWS" style articles.
    - Otherwise uses QUERY_TEMPLATES_CUSTOMER (the company is the buyer).

    Jina/PSE quota cap: when only free backends are active, caps to 10 templates
    × last 3 years to stay within daily limits.
    """
    pse_only = (bool(JINA_KEY) or bool(GOOGLE_PSE_KEY) or bool(BRAVE_KEY)) and not SERPAPI_KEY and not SERPER_KEY

    # Detect if the primary company name is itself a known vendor
    is_vendor_search = config.company_name.lower() in KNOWN_VENDOR_NAMES
    if is_vendor_search:
        logger.info(f"Vendor-centric mode for '{config.company_name}' — using vendor templates")
        base_templates = QUERY_TEMPLATES_VENDOR
    else:
        base_templates = QUERY_TEMPLATES_CUSTOMER

    companies = config.all_company_names
    years_full = list(range(config.search_year_range["start"], config.search_year_range["end"] + 1))

    if pse_only:
        companies = [config.company_name]
        years_full = sorted(years_full)[-3:]
        templates = base_templates[:10]
        logger.info(f"Quota-cap mode: {len(templates)} templates × {len(years_full)} years = "
                    f"{len(templates)*len(years_full)} queries")
    else:
        templates = base_templates

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


# ── RSS feeds from trusted news sources ──────────────────────────────────────
RSS_FEEDS = [
    "https://feed.businesswire.com/rss/home/?rss=G22",          # BW technology
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://www.globenewswire.com/RssFeed/subjectcode/IT",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.zdnet.com/topic/enterprise-software/rss.xml",
    "https://www.computerweekly.com/rss/IT-industry-news.xml",
    "https://www.ciodive.com/feeds/news/",
    "https://www.finextra.com/rss/headlines.aspx",
    "https://www.theregister.com/enterprise/applications/headlines.atom",
    "https://techcrunch.com/category/enterprise/feed/",
    "https://enterprisetimes.co.uk/feed/",
    "https://www.expresscomputer.in/feed/",
    "https://www.cxotoday.com/feed/",
    "https://www.dqindia.com/feed/",
    "https://www.business-standard.com/rss/technology-10.rss",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
]


async def strategy_rss(config: ScraperConfig) -> list[str]:
    """Fetch RSS feeds from trusted news sources and filter by company mentions.
    Completely free — no API key required. Covers last ~30 days of news.
    """
    from bs4 import BeautifulSoup
    company_names_lower = [cn.lower() for cn in config.all_company_names]
    found_urls: list[str] = []

    async def _fetch_feed(feed_url: str):
        try:
            async with SEM:
                async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                    r = await client.get(feed_url, headers=identity_pool.next_headers())
                    if r.status_code != 200:
                        return
                    soup = BeautifulSoup(r.text, "lxml-xml")

                    # RSS <item> or Atom <entry>
                    items = soup.find_all("item") or soup.find_all("entry")
                    for item in items:
                        # Get article URL
                        link_tag = item.find("link")
                        if link_tag:
                            url = link_tag.get("href") or link_tag.get_text(strip=True)
                        else:
                            url = ""

                        # Check if company mentioned in title or description
                        title = (item.find("title") or item.find("summary") or "")
                        desc  = (item.find("description") or item.find("content") or "")
                        text  = (getattr(title, "get_text", lambda: str(title))() + " " +
                                 getattr(desc,  "get_text", lambda: str(desc))()).lower()

                        if url and any(cn in text for cn in company_names_lower):
                            found_urls.append(url)
                            logger.info(f"RSS hit: {url[:80]}")
        except Exception as e:
            logger.debug(f"RSS feed failed {feed_url}: {e}")

    await asyncio.gather(*[_fetch_feed(f) for f in RSS_FEEDS])
    return list(dict.fromkeys(u for u in found_urls if u))


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
    """Run all strategies in parallel, deduplicate, filter skip list.
    Strategy E (RSS) is free and always runs — gives immediate results even
    when all search API quotas are exhausted.
    """
    results = await asyncio.gather(
        strategy_a_search(config),
        strategy_b_known_sources(config),
        strategy_c_linkedin(config),
        strategy_d_news_aggregators(config),
        strategy_rss(config),             # free, no API key required
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
