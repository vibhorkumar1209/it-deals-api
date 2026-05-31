"""Parallel.ai research API.

Uses the official `parallel` SDK when installed; falls back to raw HTTP.
SDK docs: https://docs.parallel.ai

    from parallel import Parallel
    from parallel.types import TaskSpecParam
    client = Parallel(api_key=...)
    run = client.task_run.create(input=..., task_spec=TaskSpecParam(output_schema=...), processor="base")
    result = client.task_run.result(run.run_id, api_timeout=3600)
"""

import asyncio
import logging
import os
import re
import httpx

logger = logging.getLogger(__name__)

PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY", "")
BASE_URL = "https://api.parallel.ai"
TASK_TIMEOUT_S = 200
POLL_INTERVAL_S = 5

# ── SDK client (lazy init) ────────────────────────────────────────────────────
_sdk_client = None

def _get_sdk_client():
    """Return a cached Parallel SDK client, or None if SDK not installed."""
    global _sdk_client
    if _sdk_client is not None:
        return _sdk_client
    if not PARALLEL_API_KEY:
        return None
    try:
        from parallel import Parallel  # type: ignore
        _sdk_client = Parallel(api_key=PARALLEL_API_KEY)
        logger.info("Parallel.ai SDK client initialised")
        return _sdk_client
    except ImportError:
        logger.debug("parallel SDK not installed — using HTTP fallback")
        return None


# ── SDK path (blocking → run in thread) ──────────────────────────────────────

def _sdk_run_sync(query: str, output_schema: str | None = None) -> str | None:
    """Blocking SDK call — run via asyncio.to_thread in async context."""
    client = _get_sdk_client()
    if client is None:
        return None
    try:
        from parallel.types import TaskSpecParam  # type: ignore
        task_spec = TaskSpecParam(output_schema=output_schema) if output_schema else None
        kwargs = dict(input=query, processor="base")
        if task_spec:
            kwargs["task_spec"] = task_spec
        run = client.task_run.create(**kwargs)
        logger.info(f"Parallel SDK run created: {run.run_id}")
        result = client.task_run.result(run.run_id, api_timeout=TASK_TIMEOUT_S)
        output = getattr(result, "output", None) or str(result)
        logger.info(f"Parallel SDK run {run.run_id} done: {len(str(output))} chars")
        return str(output) if output else None
    except Exception as e:
        logger.warning(f"Parallel SDK error: {e}")
        return None


async def _sdk_research(query: str, output_schema: str | None = None) -> str | None:
    """Async wrapper around the blocking SDK call."""
    return await asyncio.to_thread(_sdk_run_sync, query, output_schema)


# ── HTTP fallback path ────────────────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "x-api-key": PARALLEL_API_KEY,
    }


async def _create_task(query: str) -> str | None:
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
    deadline = asyncio.get_running_loop().time() + TASK_TIMEOUT_S
    async with httpx.AsyncClient(timeout=20) as c:
        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(POLL_INTERVAL_S)
            try:
                r = await c.get(f"{BASE_URL}/v1/tasks/runs/{run_id}", headers=_headers())
                if not r.is_success:
                    continue
                data = r.json()
                status = data.get("status", "")
                if status == "failed":
                    logger.warning(f"Parallel task {run_id} failed")
                    return None
                if status == "completed" or not data.get("is_active", True):
                    rr = await c.get(
                        f"{BASE_URL}/v1/tasks/runs/{run_id}/result",
                        headers=_headers()
                    )
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
    logger.warning(f"Parallel task {run_id} timed out after {TASK_TIMEOUT_S}s")
    return None


async def parallel_research(query: str, output_schema: str | None = None) -> str | None:
    """Run a Parallel.ai research query and return result text.

    Uses the official SDK (with output_schema support) when installed,
    falls back to raw HTTP polling otherwise.
    """
    if not PARALLEL_API_KEY:
        return None

    # Try SDK first (supports output_schema for structured output)
    if _get_sdk_client() is not None:
        return await _sdk_research(query, output_schema)

    # HTTP fallback (no output_schema support)
    run_id = await _create_task(query)
    if not run_id:
        return None
    logger.info(f"Parallel HTTP task created: {run_id}")
    result = await _poll_task(run_id)
    if result:
        logger.info(f"Parallel HTTP task {run_id} done: {len(result)} chars")
    return result


async def parallel_enrich_deals(
    deals: list[dict],
    company_name: str,
    domain: str,
) -> list[dict]:
    """Use Parallel.ai to enrich extracted deals with missing fields.

    For each deal missing date / value / description, sends a targeted
    Parallel.ai research query and merges the response back into the record.
    Returns the (possibly enriched) deal list.
    """
    if not PARALLEL_API_KEY or not deals:
        return deals

    # Only enrich deals that are missing at least one key field
    to_enrich = [
        d for d in deals
        if not d.get("announcement_date") or not d.get("deal_description")
           or not d.get("deal_value_usd")
    ]
    if not to_enrich:
        return deals

    # Build a compact summary of each deal to give Parallel context
    deal_lines = []
    for i, d in enumerate(to_enrich, 1):
        vendor = d.get("vendor") or d.get("si_partner") or "unknown vendor"
        scope  = d.get("scope_of_service") or d.get("record_type") or "IT deal"
        deal_lines.append(f"{i}. {company_name} — {vendor} — {scope}")

    query = (
        f"For {company_name} (website: {domain}), please research and provide details "
        f"for each of the following IT deals:\n\n"
        + "\n".join(deal_lines)
        + "\n\nFor each deal provide:\n"
        "- Exact announcement or signing date (YYYY-MM-DD if possible)\n"
        "- Deal value in USD millions (if publicly known)\n"
        "- Contract duration (e.g. '5 years')\n"
        "- One-sentence description of what was agreed\n"
        "- Source URL\n\n"
        "Only return factual information from public sources."
    )

    result = await parallel_research(query)
    if not result:
        return deals

    # Parse enrichment: match numbered items back to deals
    enriched_map: dict[int, dict] = {}
    lines = result.split("\n")
    current_idx: int | None = None
    current_block: list[str] = []

    def _flush(idx, block):
        text = " ".join(block)
        rec: dict = {}
        # Date
        dm = re.search(r'\b(20\d{2}-\d{2}-\d{2})\b', text)
        if not dm:
            dm = re.search(
                r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
                r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
                r'\s+\d{1,2},?\s+20\d{2}\b', text, re.IGNORECASE,
            )
        if dm:
            rec["_date"] = dm.group(0)
        # Value
        vm = re.search(r'\$\s*([\d,]+(?:\.\d+)?)\s*(?:million|mn|M\b)', text, re.IGNORECASE)
        if vm:
            try:
                rec["_value"] = float(vm.group(1).replace(",", ""))
            except ValueError:
                pass
        # Duration
        dur = re.search(r'(\d+[-\s]year|\d+[-\s]month)', text, re.IGNORECASE)
        if dur:
            rec["_duration"] = dur.group(0)
        # Description: first sentence ≥ 20 words
        for sent in re.split(r'(?<=[.!?])\s+', text):
            if len(sent.split()) >= 20:
                rec["_description"] = sent.strip()
                break
        # URL
        urls = re.findall(r'https?://[^\s\)\]\>"\'<,]+', text)
        if urls:
            rec["_url"] = re.sub(r'[.,;:]+$', '', urls[0])
        if rec:
            enriched_map[idx] = rec

    for line in lines:
        m = re.match(r'^(\d+)[.\)]\s+', line.strip())
        if m:
            if current_idx is not None:
                _flush(current_idx, current_block)
            current_idx = int(m.group(1))
            current_block = [line[m.end():].strip()]
        elif current_idx is not None:
            current_block.append(line.strip())

    if current_idx is not None:
        _flush(current_idx, current_block)

    # Merge enrichment back into deal records
    import dateparser as _dp
    from datetime import datetime, timezone as _tz
    today = datetime.now(_tz.utc)

    for i, deal in enumerate(to_enrich, 1):
        patch = enriched_map.get(i, {})
        if not patch:
            continue
        if patch.get("_date") and not deal.get("announcement_date"):
            try:
                dt = _dp.parse(patch["_date"], settings={"RETURN_AS_TIMEZONE_AWARE": True})
                if dt and datetime(2015, 1, 1, tzinfo=_tz.utc) <= dt <= today:
                    deal["announcement_date"] = dt.strftime("%Y-%m-%d")
            except Exception:
                pass
        if patch.get("_value") and not deal.get("deal_value_usd"):
            deal["deal_value_usd"] = patch["_value"]
        if patch.get("_duration") and not deal.get("deal_duration"):
            deal["deal_duration"] = patch["_duration"]
        if patch.get("_description") and not deal.get("deal_description"):
            deal["deal_description"] = patch["_description"]
        if patch.get("_url") and deal.get("source_url") in ("", None, "https://parallel.ai/research"):
            deal["source_url"] = patch["_url"]

    logger.info(f"Parallel enrichment: {len(enriched_map)}/{len(to_enrich)} deals enriched")
    return deals


def extract_urls_from_text(text: str) -> list[str]:
    """Extract all http(s) URLs from Parallel.ai result text."""
    raw = re.findall(r'https?://[^\s\)\]\>"\'<,]+', text)
    # Clean trailing punctuation
    cleaned = [re.sub(r'[.,;:]+$', '', u) for u in raw]
    return list(dict.fromkeys(cleaned))


def parallel_text_to_deals(
    text: str,
    company_name: str,
    company_names: list[str],
) -> list[dict]:
    """
    Parse Parallel.ai research output directly into deal records.
    Parallel already read the source pages — we just extract structure from its summary.
    """
    from nlp_extractor import build_deal_record, is_deal_relevant

    deals = []
    urls = extract_urls_from_text(text)

    # Split on paragraph/section breaks to process each deal block
    # Parallel output is typically structured with one deal per paragraph
    blocks = re.split(r'\n{2,}|\n(?=\d+[\.\)])', text.strip())

    for block in blocks:
        if len(block.split()) < 15:
            continue
        # Must mention the company
        if not any(cn.lower() in block.lower() for cn in company_names):
            continue
        if not is_deal_relevant(block, company_names, []):
            continue

        # Use the first URL in this block as the source, or any extracted URL
        block_urls = extract_urls_from_text(block)
        source_url = block_urls[0] if block_urls else (urls[0] if urls else "https://parallel.ai/research")

        deal = build_deal_record(
            text=block,
            url=source_url,
            source_type="news_article",
            company_name=company_name,
            company_names=company_names,
            soup=None,
        )
        if deal:
            deals.append(deal)

    logger.info(f"Parallel text → {len(deals)} deals from {len(blocks)} blocks")
    return deals


# ── 47+ high-quality IT deal news sources (RSS feeds) ─────────────────────────
KNOWN_IT_DEAL_SOURCES = [
    # Indian banking / fintech news
    "https://bfsi.eletsonline.com/feed/",
    "https://www.expresscomputer.in/feed/",
    "https://cxotoday.com/feed/",
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

    # Global IT / fintech news
    "https://www.finextra.com/rss/headlines.aspx",
    "https://www.bankingtech.com/feed/",
    "https://www.ciodive.com/feeds/news/",
    "https://www.theregister.com/headlines.atom",
    "https://www.computerweekly.com/rss/IT-industry.xml",
    "https://zdnet.com/news/rss.xml",
    "https://www.cio.com/feed/",
    "https://www.paymentssource.com/feed",

    # Vendor IR / press rooms
    "https://newsroom.accenture.com/news/rss.xml",
    "https://www.infosys.com/newsroom/press-releases/rss.xml",
    "https://www.wipro.com/newsroom/press-releases/rss.xml",
    "https://news.sap.com/feed/",
    "https://www.oracle.com/news/rss.xml",
    "https://news.microsoft.com/feed/",
    "https://aws.amazon.com/blogs/aws/feed/",
    "https://www.servicenow.com/company/media/press-room/rss.xml",
]

# Domains where Jina Reader reliably returns article text
FETCHABLE_DOMAINS = {
    # Press wires (very reliable)
    "businesswire.com", "prnewswire.com", "globenewswire.com",
    "businesswireindia.com",
    # Indian financial / IT news
    "economictimes.indiatimes.com", "economictimes.com",
    "financialexpress.com", "livemint.com", "moneycontrol.com",
    "business-standard.com", "thehindu.com", "hindustantimes.com",
    "ndtv.com", "deccanherald.com",
    "bfsi.eletsonline.com", "expresscomputer.in", "cxotoday.com",
    "dqindia.com", "ciol.com", "crn.in", "etcio.com",
    "inc42.com", "entrackr.com", "yourstory.com",
    "bankingfrontiers.com",
    # Global IT / fintech news
    "finextra.com", "bankingtech.com", "ciodive.com", "paymentssource.com",
    "theregister.com", "computerweekly.com", "zdnet.com",
    "cio.com", "techrepublic.com", "channelweb.co.uk",
    # Vendor newsrooms / IR
    "newsroom.accenture.com", "infosys.com", "wipro.com",
    "news.sap.com", "oracle.com", "news.microsoft.com",
    # Regulatory
    "sec.gov", "bseindia.com", "nseindia.com",
}


def is_fetchable(url: str) -> bool:
    """Return True only for domains where Jina Reader reliably works."""
    try:
        from urllib.parse import urlparse
        d = urlparse(url).netloc.lstrip("www.")
        return any(d == fd or d.endswith("." + fd) for fd in FETCHABLE_DOMAINS)
    except Exception:
        return False
