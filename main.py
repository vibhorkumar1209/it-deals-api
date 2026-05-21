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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4001, reload=True)
