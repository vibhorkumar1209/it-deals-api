"""IT Deals Intelligence API — FastAPI with SSE streaming."""

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

class SchemaField(BaseModel):
    key: str
    label: str
    type: str = "string"          # string | number | date | boolean
    description: str = ""

class EnrichTaskRequest(BaseModel):
    goal: str = Field(..., min_length=10)
    schema_fields: list[SchemaField] = Field(..., min_length=1)
    inputs: list[EnrichInput] = Field(..., min_length=1, max_length=50)


@app.post("/api/enrich-task")
async def enrich_task(req: EnrichTaskRequest):
    """SSE stream: run Parallel.ai enrichment for each input against a user-defined schema."""
    import asyncio
    from parallel_search import parallel_research

    async def _generate():
        import re as _re
        import json as _json

        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        # Early check — fail fast with a clear message
        parallel_key = os.getenv("PARALLEL_API_KEY", "")
        if not parallel_key:
            yield _sse({"type": "error", "message": "PARALLEL_API_KEY not set on server. Add it in Render → Environment."})
            return

        yield _sse({"type": "progress", "message": f"🚀 Starting enrichment for {len(req.inputs)} companies via Parallel.ai…"})

        schema_lines = "\n".join(
            f'- {f.key}: {f.description or f.label} ({f.type})'
            for f in req.schema_fields
        )
        output_schema_tmpl = (
            "A JSON object with these exact fields based on the research:\n"
            f"{schema_lines}\n"
            "Use null for any field that cannot be confirmed from public sources. "
            "Return ONLY the JSON object, no markdown fences."
        )
        results: list[dict] = []

        for i, inp in enumerate(req.inputs):
            yield _sse({
                "type": "heartbeat",
                "message": f"⏳ Researching {inp.company_name} ({i+1}/{len(req.inputs)})…"
            })

            query = (
                f"{inp.company_name} (website: {inp.domain})\n\n"
                f"{req.goal}"
            )
            company_schema = output_schema_tmpl

            # Run Parallel.ai as a background task, yield heartbeats every 8s while waiting
            research_task = asyncio.ensure_future(
                parallel_research(query, output_schema=company_schema)
            )
            elapsed = 0
            raw = None
            while not research_task.done() and elapsed < 220:
                try:
                    done, _ = await asyncio.wait({research_task}, timeout=8)
                    if done:
                        raw = research_task.result()
                        break
                except Exception:
                    break
                elapsed += 8
                yield _sse({
                    "type": "heartbeat",
                    "message": f"⏳ Researching {inp.company_name}… ({elapsed}s)"
                })
            if not research_task.done():
                research_task.cancel()

            # Fallback to Claude when Parallel.ai times out or returns nothing
            if not raw:
                yield _sse({"type": "heartbeat", "message": f"⚡ Switching to Claude for {inp.company_name}…"})
                try:
                    import anthropic as _anthropic
                    _ac = _anthropic.Anthropic()
                    _fields_desc = "\n".join(
                        f'- {f.key}: {f.description or f.label} ({f.type})'
                        for f in req.schema_fields
                    )
                    _claude_resp = await asyncio.to_thread(
                        lambda: _ac.messages.create(
                            model="claude-haiku-20240307",
                            max_tokens=1024,
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Research the following for {inp.company_name} (website: {inp.domain}):\n\n"
                                    f"{req.goal}\n\n"
                                    f"Return a JSON object with these exact fields:\n{_fields_desc}\n\n"
                                    f"Use null for any field you cannot confirm. Return ONLY the JSON object."
                                ),
                            }],
                        )
                    )
                    raw = _claude_resp.content[0].text
                    logger.info(f"Claude fallback for {inp.company_name}: {len(raw)} chars")
                except Exception as _ce:
                    logger.warning(f"Claude fallback failed for {inp.company_name}: {_ce}")

            row: dict = {
                "company_name": inp.company_name,
                "domain": inp.domain,
                "_status": "ok" if raw else "no_result",
            }

            if raw:
                # Try JSON parse first
                parsed: dict = {}
                try:
                    clean = _re.sub(r'```(?:json)?\s*|\s*```', '', raw.strip())
                    parsed = _json.loads(clean)
                except Exception:
                    pass

                # If JSON parse failed or missing keys, use Claude Haiku to extract fields
                fields_missing = [
                    f for f in req.schema_fields
                    if f.key not in parsed or parsed.get(f.key) in (None, "null", "", "N/A", "Unknown")
                ]
                if fields_missing:
                    try:
                        import anthropic as _anthropic
                        _ac = _anthropic.Anthropic()
                        _fields_desc = "\n".join(
                            f'- {f.key}: {f.description or f.label}'
                            for f in fields_missing
                        )
                        _msg = _ac.messages.create(
                            model="claude-haiku-20240307",
                            max_tokens=512,
                            messages=[{
                                "role": "user",
                                "content": (
                                    f"Extract these fields from the research text. "
                                    f"Return ONLY a JSON object, no explanation.\n\n"
                                    f"Fields:\n{_fields_desc}\n\n"
                                    f"Research text:\n{raw[:3000]}\n\n"
                                    f"Use null for any field not found. Return JSON only."
                                ),
                            }],
                        )
                        _claude_text = _msg.content[0].text
                        _clean2 = _re.sub(r'```(?:json)?\s*|\s*```', '', _claude_text.strip())
                        _claude_parsed = _json.loads(_clean2)
                        parsed.update({k: v for k, v in _claude_parsed.items() if v not in (None, "null", "")})
                    except Exception as _ce:
                        logger.debug(f"Claude parse fallback failed: {_ce}")

                for field in req.schema_fields:
                    val = parsed.get(field.key, "")
                    if val in (None, "null", "N/A", "Unknown", "n/a"):
                        val = ""
                    row[field.key] = str(val) if val else ""

            results.append(row)
            yield _sse({"type": "row", "row": row, "index": i, "total": len(req.inputs)})

        yield _sse({
            "type": "complete",
            "results": results,
            "total": len(results),
            "succeeded": sum(1 for r in results if r["_status"] == "ok"),
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4001, reload=True)
