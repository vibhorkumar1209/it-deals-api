"""Parallel.ai research API — async task/poll search strategy.

Used as primary research engine; Jina search is fallback when Parallel is unavailable.
"""

import asyncio
import logging
import os
import httpx

logger = logging.getLogger(__name__)

PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
BASE_URL = "https://api.parallel.ai"
TASK_TIMEOUT_S = 55      # max seconds to wait for Parallel task
POLL_INTERVAL_S = 4      # how often to poll for completion


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "x-api-key": PARALLEL_API_KEY,
    }


async def _create_task(query: str) -> str | None:
    """Submit a research task and return run_id."""
    if not PARALLEL_API_KEY:
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(
                f"{BASE_URL}/v1/tasks/runs",
                headers=_headers(),
                json={"input": query, "processor": "base"},
            )
            if not r.is_success:
                logger.warning(f"Parallel task creation failed {r.status_code}: {r.text[:200]}")
                return None
            return r.json().get("run_id")
    except Exception as e:
        logger.warning(f"Parallel task creation exception: {e}")
        return None


async def _poll_task(run_id: str) -> str | None:
    """Poll until task complete; return result text."""
    deadline = asyncio.get_event_loop().time() + TASK_TIMEOUT_S
    async with httpx.AsyncClient(timeout=20) as c:
        while asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)
            try:
                r = await c.get(f"{BASE_URL}/v1/tasks/runs/{run_id}", headers=_headers())
                if not r.is_success:
                    continue
                data = r.json()
                status = data.get("status", "")
                if status == "failed":
                    return None
                if status == "completed" or not data.get("is_active", True):
                    # Fetch result
                    rr = await c.get(f"{BASE_URL}/v1/tasks/runs/{run_id}/result", headers=_headers())
                    if rr.is_success:
                        rd = rr.json()
                        return (
                            rd.get("output", {}).get("content", {}).get("output")
                            or rd.get("result")
                            or str(rd)
                        )
                    return None
            except Exception as e:
                logger.debug(f"Parallel poll error: {e}")
    return None


async def parallel_research(query: str) -> str | None:
    """Run a Parallel.ai research query and return the result text.
    Returns None if Parallel is unavailable or times out.
    """
    if not PARALLEL_API_KEY:
        return None
    run_id = await _create_task(query)
    if not run_id:
        return None
    logger.info(f"Parallel task created: {run_id} for query: {query[:80]}")
    result = await _poll_task(run_id)
    if result:
        logger.info(f"Parallel task {run_id} completed: {len(result)} chars")
    return result


def extract_urls_from_parallel_result(text: str) -> list[str]:
    """Extract URLs from Parallel.ai result text."""
    import re
    return list(dict.fromkeys(re.findall(r'https?://[^\s\)\]\>"\']+', text)))


# ── 47+ hardcoded high-quality IT deal news sources ───────────────────────────
# These are scanned on every run regardless of search results.
KNOWN_IT_DEAL_SOURCES = [
    # Indian banking / fintech news
    "https://bfsi.eletsonline.com/feed/",
    "https://www.expresscomputer.in/feed/",
    "https://cxotoday.com/feed/",
    "https://www.cio.inc/feed/",
    "https://www.bankingfrontiers.com/feed/",
    "https://inc42.com/feed/",
    "https://entrackr.com/feed/",
    "https://www.financialexpress.com/industry/banking-finance/feed/",
    "https://economictimes.indiatimes.com/industry/banking/finance/rssfeeds/13358259.cms",
    "https://www.moneycontrol.com/rss/technology.xml",
    "https://www.livemint.com/rss/technology",

    # Global press release wires
    "https://feed.businesswire.com/rss/home/?rss=G22",
    "https://www.prnewswire.com/rss/news-releases-list.rss",
    "https://www.globenewswire.com/RssFeed/subjectcode/IT",
    "https://www.businesswire.com/rss/home/?rss=G22&rss=G36",

    # Global IT / fintech news
    "https://www.finextra.com/rss/headlines.aspx",
    "https://www.bankingtech.com/feed/",
    "https://www.ciodive.com/feeds/news/",
    "https://www.theregister.com/headlines.atom",
    "https://www.computerweekly.com/rss/IT-industry.xml",
    "https://zdnet.com/news/rss.xml",
    "https://techrepublic.com/rssfeeds/articles/",
    "https://www.cio.com/feed/",
    "https://www.itnews.com.au/feed/",
    "https://www.itpro.com/feed",
    "https://www.paymentssource.com/feed",

    # Vendor IR / press rooms (major IT vendors active in India)
    "https://newsroom.accenture.com/news/rss.xml",
    "https://www.tcs.com/content/tcs/en/newsroom/news/rss.xml",
    "https://www.infosys.com/newsroom/press-releases/rss.xml",
    "https://www.wipro.com/newsroom/press-releases/rss.xml",
    "https://www.hcltech.com/news/press-releases/rss",
    "https://www.cognizant.com/en_us/news/rss.xml",
    "https://www.capgemini.com/news/rss.xml",
    "https://news.sap.com/feed/",
    "https://www.oracle.com/news/rss.xml",
    "https://www.ibm.com/blogs/business-analytics/feed/",
    "https://news.microsoft.com/feed/",
    "https://aws.amazon.com/blogs/aws/feed/",
    "https://cloud.google.com/blog/rss.xml",
    "https://www.servicenow.com/company/media/press-room/rss.xml",
    "https://www.salesforce.com/news/press-releases/rss/",

    # Indian banking IT / RBI / BSE
    "https://www.rbi.org.in/scripts/rss.aspx",
    "https://bseindia.com/xml-data/corpfiling/AttachLive/AllAnnouncement.xml",
    "https://www.nseindia.com/api/corporate-announcements?index=equities",
]
