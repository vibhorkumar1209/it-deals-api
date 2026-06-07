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



# ── Tech Stack Finder ────────────────────────────────────────────────────────

class TechStackInput(BaseModel):
    company_name: str
    domain: str
    linkedin_url: str = ""
    focus_categories: list[str] = Field(default_factory=list)
    focus_vendors: list[str] = Field(default_factory=list)

class TechStackRequest(BaseModel):
    inputs: list[TechStackInput] = Field(..., min_length=1, max_length=20)


@app.post("/api/tech-stack")
async def tech_stack(req: TechStackRequest):
    """SSE stream: technographic profile per company."""
    from tech_stack_pipeline import find_tech_stack, TECH_STACK_FIELDS

    async def _generate():
        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        if not os.getenv("GOOGLE_AI_API_KEY"):
            yield _sse({"type": "error", "message": "GOOGLE_AI_API_KEY not set on server."})
            return

        yield _sse({"type": "progress", "message": f"🚀 Scanning tech stack for {len(req.inputs)} companies…"})

        results: list[dict] = []
        row_index = 0

        for i, inp in enumerate(req.inputs):
            yield _sse({"type": "heartbeat", "message": f"🔍 Company {i+1}/{len(req.inputs)}: {inp.company_name}"})
            try:
                async for event in find_tech_stack(
                    company_name=inp.company_name,
                    domain=inp.domain,
                    linkedin_url=inp.linkedin_url,
                    focus_categories=inp.focus_categories,
                    focus_vendors=inp.focus_vendors,
                ):
                    if event["type"] == "row_done":
                        results.append(event["row"])
                        yield _sse({"type": "row", "row": event["row"], "index": row_index})
                        row_index += 1
                    else:
                        yield _sse(event)
            except Exception as e:
                logger.error(f"Tech stack error for {inp.company_name}: {e}", exc_info=True)
                yield _sse({"type": "heartbeat", "message": f"⚠️ Error for {inp.company_name}: {e}"})

        yield _sse({"type": "complete", "results": results, "total": len(results)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── Enrichment Task ───────────────────────────────────────────────────────────

class EnrichInput(BaseModel):
    company_name: str
    domain: str
    linkedin_url: str = ""
    focus_tech: list[str] = Field(default_factory=list)
    focus_vendor: list[str] = Field(default_factory=list)

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
                    linkedin_url=inp.linkedin_url,
                    focus_tech=inp.focus_tech or [],
                    focus_vendor=inp.focus_vendor or [],
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
    """Diagnose enrichment: check env keys and test Gemini."""
    google_key = os.getenv("GOOGLE_AI_API_KEY", "")

    out: dict = {
        "env": {
            "GOOGLE_AI_API_KEY": "set" if google_key else "MISSING",
        },
        "gemini_test": "skipped — key missing",
    }

    if google_key:
        def _test_gemini():
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=google_key)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents='Return ONLY this JSON array: [{"ok": true}]',
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=50),
            )
            return resp.text or ""
        try:
            result = await asyncio.wait_for(asyncio.to_thread(_test_gemini), timeout=30)
            out["gemini_test"] = result
        except Exception as e:
            out["gemini_test"] = f"ERROR: {e}"

    return out


@app.get("/api/debug-gemini-search")
async def debug_gemini_search(company: str = "Daimler Truck North America", full_prompt: bool = False):
    """Test Gemini with Google Search Grounding. full_prompt=true uses the real pipeline prompt."""
    google_key = os.getenv("GOOGLE_AI_API_KEY", "")
    if not google_key:
        return {"error": "GOOGLE_AI_API_KEY not set"}

    def _run():
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=google_key)
        if full_prompt:
            from enrich_pipeline import _build_prompts
            prompts = _build_prompts(company, "", "", [], [])
            prompt = prompts[0]   # use first call prompt for quick test
            max_tok = 16384
        else:
            prompt = (
                f'Find 3 recent IT technology deals for {company}. '
                f'Return ONLY a JSON array: [{{"vendor":"..","deal_type":"..","date_signed":"..","description":".."}}]'
            )
            max_tok = 2048
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=max_tok,
            ),
        )
        # Try .text, fall back to parts
        try:
            text = resp.text or ""
        except Exception:
            text = ""
            try:
                for part in resp.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
            except Exception:
                pass
        finish = str(resp.candidates[0].finish_reason) if resp.candidates else "unknown"
        return text, finish

    try:
        raw, finish = await asyncio.wait_for(asyncio.to_thread(_run), timeout=180)
        import re as _re
        has_array = bool(_re.search(r"\[.*?\]", raw, _re.DOTALL))
        return {
            "company": company,
            "full_prompt": full_prompt,
            "finish_reason": finish,
            "chars": len(raw),
            "has_json_array": has_array,
            "preview": raw[:1200],
            "tail": raw[-600:] if len(raw) > 1200 else "",
        }
    except asyncio.TimeoutError:
        return {"error": "timeout after 60s"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/debug-tech-stack")
async def debug_tech_stack(company: str = "Kubota North America", call_num: int = 1):
    """Test one tech stack Gemini call. call_num=1,2,3."""
    google_key = os.getenv("GOOGLE_AI_API_KEY", "")
    if not google_key:
        return {"error": "GOOGLE_AI_API_KEY not set"}

    def _run():
        from google import genai
        from google.genai import types
        from tech_stack_pipeline import _build_tech_stack_prompt, _gemini_tech_stack_sync
        prompt = _build_tech_stack_prompt(company, "", "", [], [], call_num)
        result = _gemini_tech_stack_sync(prompt, company)
        return result

    def _run_raw():
        from google import genai
        from google.genai import types
        from tech_stack_pipeline import _build_tech_stack_prompt
        prompt = _build_tech_stack_prompt(company, "", "", [], [], call_num)
        client = genai.Client(api_key=google_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1, max_output_tokens=65536,
            ),
        )
        raw = ""
        try:
            for cand in (resp.candidates or []):
                for part in (cand.content.parts or []):
                    t = getattr(part, "text", None)
                    if t: raw += t
        except Exception:
            pass
        return raw

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_run_raw), timeout=180)
        tools = await asyncio.wait_for(asyncio.to_thread(_run), timeout=5)
        return {
            "company": company,
            "call_num": call_num,
            "raw_chars": len(raw),
            "raw_preview": raw[:2000],
            "tools_found": len(tools),
            "first_tool_keys": list(tools[0].keys()) if tools else [],
            "preview": tools[:3],
        }
    except asyncio.TimeoutError:
        return {"error": f"timeout after 180s on call {call_num}"}
    except Exception as e:
        return {"error": str(e)}


# ── GCC Intelligence Hub ──────────────────────────────────────────────────────

class GCCIntelRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    domain: str = Field(default="")
    location: str = Field(default="")
    target_vendor: str = Field(default="")
    focus_domains: list[str] = Field(default_factory=list)


@app.post("/api/gcc-intel")
async def gcc_intel(req: GCCIntelRequest):
    """SSE stream: two-phase GCC Intelligence Hub."""
    from gcc_pipeline import run_gcc_intelligence

    async def _generate():
        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        if not os.getenv("GOOGLE_AI_API_KEY"):
            yield _sse({"type": "error", "message": "GOOGLE_AI_API_KEY not set."})
            return

        try:
            async for event in run_gcc_intelligence(
                company_name=req.company_name,
                domain=req.domain,
                location=req.location,
                target_vendor=req.target_vendor,
                focus_domains=req.focus_domains or None,
            ):
                yield _sse(event)
        except Exception as e:
            logger.error(f"GCC Intel error: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ── Aftermarket Deep Dive ─────────────────────────────────────────────────────

class AftermarketRequest(BaseModel):
    company_name: str = Field(..., min_length=1)
    domain: str = Field(default="")
    industry: str = Field(default="")
    competitors: str = Field(default="")
    target_vendor: str = Field(default="")
    sections_to_run: list[str] = Field(default_factory=list)  # empty = all sections


@app.post("/api/aftermarket-dive")
async def aftermarket_dive(req: AftermarketRequest):
    """SSE stream: Aftermarket Deep Dive analysis."""
    from aftermarket_pipeline import run_aftermarket_deep_dive

    async def _generate():
        def _sse(obj: dict) -> str:
            return f"data: {json.dumps(obj)}\n\n"

        if not os.getenv("GOOGLE_AI_API_KEY"):
            yield _sse({"type": "error", "message": "GOOGLE_AI_API_KEY not set."})
            return

        try:
            from aftermarket_pipeline import ALL_SECTIONS
            sections = set(req.sections_to_run) & ALL_SECTIONS if req.sections_to_run else None
            async for event in run_aftermarket_deep_dive(
                company_name=req.company_name,
                domain=req.domain,
                industry=req.industry,
                competitors=req.competitors,
                target_vendor=req.target_vendor,
                sections_to_run=sections,
            ):
                yield _sse(event)
        except Exception as e:
            logger.error(f"Aftermarket dive error: {e}", exc_info=True)
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/api/debug-aftermarket-section")
async def debug_aftermarket_section(company: str = "Daimler Truck North America", section: str = "spend_module"):
    """Debug a specific aftermarket section — returns rows + raw Gemini response preview."""
    import re as _re, json as _json
    from aftermarket_pipeline import _spend_module_prompt, _readiness_tam_prompt
    from google import genai
    from google.genai import types
    import os

    GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
    prompts = {
        "spend_module": _spend_module_prompt(company, ""),
        "readiness": _readiness_tam_prompt(company, "", ""),
    }
    prompt = prompts.get(section)
    if not prompt:
        return {"error": f"Unknown section: {section}"}

    def _run():
        client = genai.Client(api_key=GOOGLE_AI_KEY)
        max_tok = 32768 if section == "readiness" else 16384
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.15,
                max_output_tokens=max_tok,
                tools=[types.Tool(google_search=types.GoogleSearch())],
            ),
        )
        # Collect all parts with metadata
        parts_info = []
        raw_no_thoughts = ""
        raw_all = ""
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                thought = getattr(part, "thought", None)
                txt = getattr(part, "text", "") or ""
                parts_info.append({"thought": thought, "text_len": len(txt), "text_preview": txt[:100]})
                raw_all += txt
                if not thought:
                    raw_no_thoughts += txt
        finish = getattr(response.candidates[0], "finish_reason", None) if response.candidates else None
        return {
            "parts": parts_info,
            "raw_all_preview": raw_all[:500],
            "raw_no_thoughts_preview": raw_no_thoughts[:500],
            "finish_reason": str(finish),
        }

    try:
        result = await asyncio.wait_for(asyncio.to_thread(_run), timeout=120)
        return {"company": company, "section": section, **result}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4001, reload=True)
