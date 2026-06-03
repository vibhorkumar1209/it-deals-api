"""IT Deals Intelligence API — FastAPI with SSE streaming."""

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config_loader import ScraperConfig
from pipeline_stream import stream_pipeline, extract_from_cached_urls, load_url_cache, save_url_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="IT Deals Intelligence API", version="1.0.0")

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class ScrapeRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    company_aliases: list[str] = Field(default_factory=list)
    domain: str = Field(..., min_length=3)
    secondary_domains: list[str] = Field(default_factory=list)
    linkedin_url: str = Field(default="")
    stock_ticker: str | None = None
    exchange: str | None = None
    industry_sector: str = ""
    hq_country: str = ""
    hq_city: str = ""
    search_year_range: dict[str, int] = Field(default={"start": 2020, "end": 2025})
    known_sources: list[str] = Field(default_factory=list)
    known_deals_to_skip: list[str] = Field(default_factory=list)
    focus_deal_types: list[str] = Field(default_factory=list)
    min_deal_value_usd_million: float | None = None
    run_linkedin: bool = False
    batch_size: int = Field(default=5, ge=1, le=20)


def _request_to_config(req: ScrapeRequest) -> ScraperConfig:
    return ScraperConfig(
        company_name=req.company_name,
        company_aliases=req.company_aliases,
        domain=req.domain,
        secondary_domains=req.secondary_domains,
        linkedin_url=req.linkedin_url,
        stock_ticker=req.stock_ticker,
        exchange=req.exchange,
        industry_sector=req.industry_sector,
        hq_country=req.hq_country,
        hq_city=req.hq_city,
        search_year_range=req.search_year_range,
        known_sources=req.known_sources,
        known_deals_to_skip=req.known_deals_to_skip,
        focus_deal_types=req.focus_deal_types or [
            "ERP","CRM","HCM","SCM","cloud_migration","managed_services",
            "cybersecurity","digital_transformation","infrastructure",
            "analytics","AI_ML","outsourcing","SaaS","SI_contract"
        ],
        min_deal_value_usd_million=req.min_deal_value_usd_million,
        output_dir="/tmp/deals/",
        run_linkedin=req.run_linkedin,
        run_pdf_extraction=True,
        use_proxies=False,
        proxy_pool=[],
        notify_email=None,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "it-deals-api"}


@app.post("/api/scrape")
async def scrape(req: ScrapeRequest):
    """Stream deal results via SSE. Each event is a JSON object."""
    config = _request_to_config(req)

    async def event_stream():
        try:
            async for event in stream_pipeline(config, batch_size=req.batch_size):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


class ReExtractRequest(BaseModel):
    company_name: str
    urls: list[str] = Field(default_factory=list)
    batch_size: int = Field(default=5, ge=1, le=20)
    # Full company config needed for extraction context
    company_aliases: list[str] = Field(default_factory=list)
    domain: str = ""
    focus_deal_types: list[str] = Field(default_factory=list)
    min_deal_value_usd_million: float | None = None


@app.get("/api/cached-urls/{company_name}")
async def get_cached_urls(company_name: str):
    """Return cached URLs for a company (discovered in last 48h)."""
    urls = load_url_cache(company_name)
    if urls is None:
        raise HTTPException(status_code=404, detail=f"No cached URLs for '{company_name}' (or cache expired)")
    return {"company_name": company_name, "url_count": len(urls), "urls": urls}


@app.post("/api/extract")
async def extract(req: ReExtractRequest):
    """Re-run extraction on a provided list of URLs — skips search phase."""
    # Build minimal config for extraction context
    config = ScraperConfig(
        company_name=req.company_name,
        company_aliases=req.company_aliases,
        domain=req.domain or f"{req.company_name.lower().replace(' ', '')}.com",
        secondary_domains=[],
        linkedin_url="",
        stock_ticker=None,
        exchange=None,
        industry_sector="",
        hq_country="",
        hq_city="",
        search_year_range={"start": 2020, "end": 2025},
        known_sources=[],
        known_deals_to_skip=[],
        focus_deal_types=req.focus_deal_types or [
            "ERP","CRM","HCM","SCM","cloud_migration","managed_services",
            "cybersecurity","digital_transformation","infrastructure",
            "analytics","AI_ML","outsourcing","SaaS","SI_contract"
        ],
        min_deal_value_usd_million=req.min_deal_value_usd_million,
        output_dir="/tmp/deals/",
        run_linkedin=False,
        run_pdf_extraction=True,
        use_proxies=False,
        proxy_pool=[],
        notify_email=None,
    )

    # Use provided URLs or fall back to cache
    urls = req.urls
    if not urls:
        cached = load_url_cache(req.company_name)
        if not cached:
            raise HTTPException(status_code=404, detail="No URLs provided and no cache found for this company.")
        urls = cached

    async def event_stream():
        try:
            async for event in extract_from_cached_urls(config, urls, batch_size=req.batch_size):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error(f"Extract error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )



# ── Enrichment Task ───────────────────────────────────────────────────────────

class EnrichInput(BaseModel):
    company_name: str
    domain: str
    industry: str = ""   # e.g. "banking", "retail", "manufacturing" — filters vendor list

class SchemaField(BaseModel):
    key: str
    label: str
    type: str = "string"          # string | number | date | boolean
    description: str = ""

class EnrichTaskRequest(BaseModel):
    goal: str = Field(..., min_length=10)
    schema_fields: list[SchemaField] = Field(..., min_length=1)
    inputs: list[EnrichInput] = Field(..., min_length=1, max_length=50)
    # Optional enrichment boosters — merged with pipeline defaults when provided
    vendors: list[str] = Field(default_factory=list)              # extra vendor names to search
    sources: list[str] = Field(default_factory=list)              # domains for site: queries
    keywords: list[str] = Field(default_factory=list)             # extra deal signal keywords
    run_t3: bool = Field(default=False)                           # opt-in: run Tier 3 keyword catch-all
    competitor_vendors: list[str] = Field(default_factory=list)   # competitor vendors — injected at T2 only


@app.post("/api/enrich-task")
async def enrich_task(req: EnrichTaskRequest):
    """SSE stream: search → Apify scrape → identify → Claude classify."""
    from enrich_pipeline import enrich_company

    async def _generate():
        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            yield _sse({"type": "error", "message": "ANTHROPIC_API_KEY not set on server."})
            return

        schema_fields = [f.model_dump() for f in req.schema_fields]

        yield _sse({"type": "progress", "message": f"🚀 Starting enrichment for {len(req.inputs)} companies…"})

        results: list[dict] = []

        deal_index = 0  # global row counter across all companies

        for i, inp in enumerate(req.inputs):
            yield _sse({"type": "heartbeat", "message": f"🔍 Company {i+1}/{len(req.inputs)}: {inp.company_name}"})

            company_deals: list[dict] = []

            try:
                async for event in enrich_company(
                    company_name=inp.company_name,
                    domain=inp.domain,
                    goal=req.goal,
                    schema_fields=schema_fields,
                    year_range=(2016, 2025),
                    extra_vendors=req.vendors,
                    extra_sources=req.sources,
                    extra_keywords=req.keywords,
                    industry=inp.industry,
                    run_t3=req.run_t3,
                    t2_vendors=req.competitor_vendors or None,
                ):
                    if event["type"] == "row_done":
                        deal_row = event["row"]
                        company_deals.append(deal_row)
                        results.append(deal_row)
                        yield _sse({"type": "row", "row": deal_row, "index": deal_index, "total": None})
                        deal_index += 1
                    else:
                        yield _sse(event)
            except Exception as e:
                logger.error(f"Enrich pipeline error for {inp.company_name}: {e}", exc_info=True)
                # emit a no_result row so frontend doesn't hang
                fallback = {"company_name": inp.company_name, "domain": inp.domain, "_status": "error"}
                for f in req.schema_fields:
                    fallback[f.key] = ""
                results.append(fallback)
                yield _sse({"type": "row", "row": fallback, "index": deal_index, "total": None})
                deal_index += 1

            yield _sse({"type": "heartbeat", "message": f"✅ {inp.company_name}: {len(company_deals)} deals found"})

        yield _sse({
            "type": "complete",
            "results": results,
            "total": len(results),
            "succeeded": sum(1 for r in results if r.get("_status") == "ok"),
        })

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/fetch-test")
async def fetch_test(url: str = "https://m.economictimes.com/industry/banking/finance/banking/hdfc-bank-signs-multi-year-data-and-technology-deal-with-refinitiv/articleshow/94349454.cms"):
    """Quick test: fetch one URL via Jina Reader and show result."""
    from website_router import fetch_via_jina_reader
    from nlp_extractor import is_deal_relevant
    import asyncio
    try:
        result = await asyncio.wait_for(fetch_via_jina_reader(url), timeout=25)
        if result is None:
            return {"status": "failed", "url": url, "reason": "Jina returned None"}
        text, _ = result
        return {
            "status": "ok",
            "url": url,
            "chars": len(text),
            "words": len(text.split()),
            "preview": text[:500],
            "is_deal_relevant": is_deal_relevant(text, ["HDFC Bank"], []),
        }
    except asyncio.TimeoutError:
        return {"status": "timeout", "url": url}
    except Exception as e:
        return {"status": "error", "url": url, "error": str(e)}


@app.get("/api/debug")
async def debug(company: str = "HDFC Bank"):
    """
    Full pipeline diagnostic.
    Returns: env vars, discovered URLs, per-URL fetch result, per-URL extraction result.
    Hit: /api/debug?company=HDFC+Bank
    """
    import asyncio
    from search_engine import _jina_search, JINA_KEY, QUERY_TEMPLATES_CUSTOMER
    from website_router import fetch_via_jina_reader, classify_url
    from nlp_extractor import is_deal_relevant, build_deal_record, NON_DEAL_DISQUALIFIERS, DEAL_ACTION_PHRASES

    year = 2025
    out: dict = {
        "company": company,
        "env": {},
        "queries": [],
        "urls_discovered": [],
        "url_results": [],
    }

    # ── Env check ─────────────────────────────────────────────────────────────
    out["env"] = {
        "JINA_KEY_set": bool(JINA_KEY),
        "JINA_KEY_prefix": JINA_KEY[:12] + "..." if JINA_KEY else None,
    }

    # ── Run 1 query only (keep under 30s Render timeout) ─────────────────────
    q = QUERY_TEMPLATES_CUSTOMER[0].format(company=company, year=year)
    out["queries"].append(q)
    try:
        urls = await asyncio.wait_for(_jina_search(q), timeout=10)
        out["urls_discovered"] = [{"url": u} for u in urls[:8]]
    except Exception as e:
        out["urls_discovered"] = [{"error": str(e)}]
        urls = []

    # Skip social, take first 3 real URLs
    _SKIP = {"linkedin.com","facebook.com","instagram.com","twitter.com","x.com","youtube.com","danelfin.com"}
    from urllib.parse import urlparse as _up
    unique_urls = [u for u in urls if not any(_up(u).netloc.lstrip("www.").endswith(s) for s in _SKIP)][:3]

    # ── Fetch + extract each URL (parallel) ───────────────────────────────────
    async def _probe(url: str) -> dict:
        entry: dict = {"url": url}
        try:
            result = await asyncio.wait_for(fetch_via_jina_reader(url), timeout=10)
            if result is None:
                entry["fetch"] = "empty"
            else:
                text, _ = result
                entry["fetch"] = "ok"
                entry["chars"] = len(text)
                entry["preview"] = text[:200]
                entry["relevant"] = is_deal_relevant(text, [company], [])
                if entry["relevant"]:
                    deal = build_deal_record(text=text, url=url, source_type="news_article",
                                            company_name=company, company_names=[company], soup=None)
                    entry["deal"] = deal
                else:
                    tl = text.lower()
                    entry["disq"] = [d for d in NON_DEAL_DISQUALIFIERS if d in tl]
                    entry["phrases"] = [p for p in DEAL_ACTION_PHRASES if p in tl][:5]
                    entry["company_found"] = company.lower() in tl
        except asyncio.TimeoutError:
            entry["fetch"] = "timeout"
        except Exception as e:
            entry["fetch"] = f"error: {e}"
        return entry

    out["url_results"] = await asyncio.gather(*[_probe(u) for u in unique_urls])
    return out


@app.get("/api/debug-enrich")
async def debug_enrich():
    """Diagnose enrichment: check env keys and test Claude via thread."""
    import anthropic as _anthropic

    anthropic_key   = os.getenv("ANTHROPIC_API_KEY", "")
    apify_key       = os.getenv("APIFY_API_KEY", "")
    parallel_key    = os.getenv("PARALLEL_API_KEY", "")
    jina_key        = os.getenv("JINA_KEY", "")
    scraper_api_key  = os.getenv("SCRAPER_API_KEY", "")
    scraper_base_url = os.getenv("SCRAPER_BASE_URL", "")

    out: dict = {
        "env": {
            "ANTHROPIC_API_KEY": "set" if anthropic_key    else "MISSING",
            "APIFY_API_KEY":     "set" if apify_key        else "MISSING",
            "PARALLEL_API_KEY":  "set" if parallel_key     else "MISSING",
            "JINA_KEY":          "set" if jina_key         else "MISSING",
            "SCRAPER_API_KEY":   "set" if scraper_api_key  else "MISSING",
            "SCRAPER_BASE_URL":  scraper_base_url          if scraper_base_url else "MISSING",
        },
        "claude_test": "skipped — key missing",
        "anthropic_version": _anthropic.__version__,
    }

    if anthropic_key:
        def _test_claude():
            ac = _anthropic.Anthropic(api_key=anthropic_key)
            msg = ac.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=20,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return msg.content[0].text
        try:
            result = await asyncio.to_thread(_test_claude)
            out["claude_test"] = result
        except Exception as e:
            out["claude_test"] = f"ERROR: {e}"

    # Test Google News RSS search
    import httpx as _httpx, re as _re2
    from urllib.parse import quote_plus as _qp
    try:
        rss_url = f"https://news.google.com/rss/search?q={_qp('HDFC Bank IBM deal 2023')}&hl=en-US&gl=US&ceid=US:en"
        async with _httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            r = await client.get(rss_url, headers={"User-Agent": "Mozilla/5.0 (compatible; RSS reader)"})
        urls = [u for u in _re2.findall(r'<link>([^<]+)</link>', r.text) if u.startswith("http") and "rss/search" not in u and u != "https://news.google.com/"][:4]
        out["gnews_rss_test"] = {"status_code": r.status_code, "ok": r.is_success,
                                  "urls_found": len(urls), "sample": urls}
    except Exception as e:
        out["gnews_rss_test"] = {"error": str(e)}

    # Test custom scraper API (vibhorkumar1209/scraper-api)
    if scraper_base_url:
        import httpx as _httpx
        try:
            headers = {"x-api-key": scraper_api_key} if scraper_api_key else {}
            async with _httpx.AsyncClient(timeout=25) as client:
                r = await client.get(
                    f"{scraper_base_url.rstrip('/')}/scrape/web",
                    params={"url": "https://economictimes.indiatimes.com/industry/banking/finance/banking/hdfc-bank-signs-multi-year-data-deal/articleshow/94349454.cms"},
                    headers=headers,
                )
            body = r.json() if r.is_success else {}
            text = (body.get("data") or {}).get("bodyText", "")
            out["custom_scraper_test"] = {
                "status_code": r.status_code,
                "ok": r.is_success,
                "words_scraped": len(text.split()) if text else 0,
                "preview": text[:200] if text else None,
                "error": body.get("error") if not r.is_success else None,
            }
        except Exception as e:
            out["custom_scraper_test"] = {"error": str(e)}

    # Test Apify Google Search with 1 query
    if apify_key:
        import httpx as _httpx
        try:
            actor_url = (
                f"https://api.apify.com/v2/acts/apify~google-search-scraper"
                f"/run-sync-get-dataset-items?token={apify_key}&timeout=30&memory=256"
            )
            async with _httpx.AsyncClient(timeout=40) as client:
                r = await client.post(actor_url, json={
                    "queries": '"HDFC Bank" IBM deal 2023',
                    "maxPagesPerQuery": 1,
                    "resultsPerPage": 3,
                    "countryCode": "us",
                })
            out["apify_test"] = {
                "status_code": r.status_code,
                "ok": r.is_success,
                "response_preview": r.text[:300] if not r.is_success else f"{len(r.json())} result sets, first set has {len(r.json()[0].get('organicResults', [])) if r.json() else 0} URLs",
            }
        except Exception as e:
            out["apify_test"] = {"error": str(e)}

    # Test Jina search
    if jina_key:
        import httpx as _httpx
        try:
            async with _httpx.AsyncClient(timeout=15) as client:
                r = await client.get("https://s.jina.ai/", params={"q": "HDFC Bank IBM deal 2023"},
                    headers={"Accept": "application/json", "Authorization": f"Bearer {jina_key}", "X-Respond-With": "no-content"})
            body = r.json()
            out["jina_test"] = {
                "status_code": r.status_code,
                "ok": r.is_success,
                "urls_found": len(body.get("data") or []),
                "error": body.get("message") if not r.is_success else None,
            }
        except Exception as e:
            out["jina_test"] = {"error": str(e)}

    return out


@app.get("/api/vendor-competitors")
async def vendor_competitors(vendor: str = ""):
    """
    Given a vendor name, return its top 8-10 direct competitors.
    Primary: Claude (targeted, knows IT vendor landscape).
    Fallback: kw_process/sub_industry overlap scoring from lists.json.
    """
    if not vendor.strip():
        return {"vendor": vendor, "competitors": [], "industries": []}

    from enrich_pipeline import _load_lists, _FALLBACK_VENDORS
    import anthropic as _anthropic

    vendor_clean = vendor.strip()
    vendor_lower = vendor_clean.lower()
    lists = _load_lists()
    vendor_meta: dict = lists.get("vendor_meta", {})
    vendor_industry_map: dict = lists.get("vendor_industry_map", {})

    # ── 1. Find vendor record (exact → partial match) ──────────────────────────
    matched_name = ""
    for v in vendor_meta:
        if v.lower() == vendor_lower:
            matched_name = v
            break
    if not matched_name:
        for v in vendor_meta:
            if vendor_lower in v.lower() or v.lower() in vendor_lower:
                matched_name = v
                break

    meta = vendor_meta.get(matched_name, {})
    industries = vendor_industry_map.get(matched_name, [])
    sub_industry = meta.get("sub_industry", "")
    kw_process = meta.get("kw_process", [])

    # ── 2. Ask Claude for targeted competitors ─────────────────────────────────
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    claude_competitors: list[str] = []
    if anthropic_key:
        # Build context from known metadata to help Claude be precise
        context_parts = []
        if sub_industry:
            context_parts.append(f"sub-industry: {sub_industry}")
        if kw_process:
            context_parts.append(f"specializes in: {', '.join(kw_process[:6])}")
        if industries:
            context_parts.append(f"serves: {', '.join(industries[:3])}")
        context = f" ({'; '.join(context_parts)})" if context_parts else ""

        prompt = (
            f"List the top 8 direct competitors of {vendor_clean}{context} in the IT services / enterprise software space. "
            f"Return ONLY a JSON array of vendor/company names, no explanation. "
            f"Focus on vendors competing for the same customer deals, not adjacent markets. "
            f"Example format: [\"Vendor A\", \"Vendor B\", ...]"
        )
        try:
            def _ask_claude():
                ac = _anthropic.Anthropic(api_key=anthropic_key)
                msg = ac.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                return msg.content[0].text.strip()

            raw = await asyncio.to_thread(_ask_claude)
            # Parse JSON array from response
            import re as _re
            m = _re.search(r'\[.*?\]', raw, _re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                claude_competitors = [str(c).strip() for c in parsed if c][:10]
        except Exception as e:
            logger.warning(f"Claude competitor lookup failed: {e}")

    if claude_competitors:
        return {
            "vendor": vendor_clean,
            "matched_name": matched_name or vendor_clean,
            "industries": industries[:5],
            "competitors": claude_competitors,
            "source": "claude",
        }

    # ── 3. Fallback: kw_process specificity scoring ───────────────────────────
    target_proc = set(kw_process)
    target_tech = set(meta.get("kw_technology", []))
    scores: dict[str, float] = {}
    for vname, vmeta in vendor_meta.items():
        if vendor_lower in vname.lower():
            continue
        vproc = vmeta.get("kw_process", [])
        vtech = vmeta.get("kw_technology", [])
        proc_overlap = len(target_proc & set(vproc))
        tech_overlap = len(target_tech & set(vtech))
        sub_match = 1 if vmeta.get("sub_industry") == sub_industry and sub_industry else 0
        # Specificity weight: penalise vendors with huge process lists (generic SI)
        proc_spec = (proc_overlap / max(len(vproc), 1)) * proc_overlap * 3
        tech_spec = (tech_overlap / max(len(vtech), 1)) * tech_overlap
        score = proc_spec + tech_spec + sub_match * 2
        if score > 0:
            scores[vname] = score

    sorted_fallback = sorted(scores, key=lambda v: -scores[v])[:10]
    return {
        "vendor": vendor_clean,
        "matched_name": matched_name or vendor_clean,
        "industries": industries[:5],
        "competitors": sorted_fallback,
        "source": "dataset",
        "message": "Claude unavailable — using dataset scoring" if not anthropic_key else "Claude failed — using dataset scoring",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4001, reload=True)
