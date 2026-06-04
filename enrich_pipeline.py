"""
Enrichment pipeline: Google Search Grounding → Gemini extraction.

Single-call architecture:
  1. Build a focused research prompt with company × vendor × keyword queries
  2. Call gemini-2.5-flash with googleSearch tool enabled
  3. Gemini searches live web, reads pages, and returns structured JSON deals
  4. Parse + normalise → yield one row_done per deal
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

# ── Top vendors for query construction ───────────────────────────────────────
TOP_VENDORS = [
    # Global SIs
    "TCS", "Infosys", "Wipro", "HCLTech", "Accenture", "IBM", "Cognizant",
    "Capgemini", "DXC Technology", "Tech Mahindra",
    # ERP / Cloud platforms
    "SAP", "Oracle", "Microsoft", "AWS", "Google Cloud", "Salesforce",
    "ServiceNow", "Workday", "SAP S/4HANA",
    # Cybersecurity
    "Palo Alto Networks", "CrowdStrike", "Fortinet", "Zscaler",
    # Analytics / Data
    "Snowflake", "Databricks", "SAS",
    # Banking / Fintech
    "Temenos", "Finacle", "Newgen", "FIS", "Fiserv", "Finastra",
    # Managed Services
    "Atos", "NTT", "Unisys",
]


def _build_research_prompt(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int],
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> str:
    """
    Construct a rich research prompt that Gemini + Google Search Grounding will act on.
    The prompt tells Gemini exactly what to search for and what JSON to return.
    """
    start_yr, end_yr = year_range
    vendors = list(dict.fromkeys((extra_vendors or []) + TOP_VENDORS))[:20]
    keywords = extra_keywords or [
        "IT deal", "technology contract", "outsourcing agreement",
        "digital transformation", "managed services", "cloud migration",
        "ERP implementation", "cybersecurity contract", "vendor selected",
    ]
    sources = extra_sources or []

    fields_desc = "\n".join(
        f'  "{f["key"]}": "{f.get("description") or f.get("label", "")}"'
        for f in schema_fields
    )
    field_keys_example = {
        f["key"]: f"<{f.get('type', 'string')}>" for f in schema_fields
    }

    vendor_list = ", ".join(vendors[:15])
    keyword_list = ", ".join(keywords[:10])
    source_hint = (
        f"\nPrioritise results from these sources: {', '.join(sources[:10])}"
        if sources else ""
    )

    return f"""You are an enterprise IT deal researcher. Your task is to find every IT deal,
technology contract, outsourcing agreement, and digital transformation initiative
involving {company_name} ({domain}) announced between {start_yr} and {end_yr}.

RESEARCH INSTRUCTIONS:
- Search Google for deals involving {company_name} and vendors such as: {vendor_list}
- Look for keywords such as: {keyword_list}
- Cover press releases, business news, vendor announcements, and IR filings{source_hint}
- Search multiple queries — by year, by vendor, by deal type — to maximise coverage
- Read the actual article pages, not just headlines

GOAL: {goal}

OUTPUT FORMAT:
Return ONLY a valid JSON array. Each element represents one distinct deal:
[
  {{
{fields_desc}
  }},
  ...
]

RULES:
- One JSON object per deal — never merge two deals into one
- Use null for any field not found in the sources
- Be specific: exact vendor names, exact dates (YYYY-MM-DD), exact values in USD millions
- Include source URL where available
- If no deals are found, return an empty array []
- Do NOT include any text outside the JSON array

Example element shape:
{json.dumps(field_keys_example, indent=2)}
"""


def _gemini_extract_deals_sync(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int],
    extra_vendors: list[str] | None,
    extra_sources: list[str] | None,
    extra_keywords: list[str] | None,
) -> list[dict]:
    """
    Blocking call to Gemini 2.5 Flash with Google Search grounding.
    Runs inside asyncio.to_thread so it doesn't block the event loop.
    Returns list of normalised deal dicts.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed. Run: pip install google-genai")
        return []

    if not GOOGLE_AI_KEY:
        logger.error("GOOGLE_AI_API_KEY env var not set")
        return []

    prompt = _build_research_prompt(
        company_name, domain, goal, schema_fields,
        year_range, extra_vendors, extra_sources, extra_keywords,
    )
    field_keys = [f["key"] for f in schema_fields]

    try:
        client = genai.Client(api_key=GOOGLE_AI_KEY)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,          # low temp for factual extraction
                max_output_tokens=8192,
            ),
        )

        raw_text = response.text or ""
        logger.info(f"Gemini response length: {len(raw_text)} chars for {company_name}")

    except Exception as e:
        logger.error(f"Gemini API error for {company_name}: {e}")
        return []

    # ── Parse JSON from response ──────────────────────────────────────────────
    try:
        # Strip markdown code fences if present
        clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text.strip())

        # Extract the JSON array (handles prose before/after the array)
        array_match = re.search(r"\[.*\]", clean, re.DOTALL)
        if not array_match:
            logger.warning(f"No JSON array found in Gemini response for {company_name}")
            return []

        parsed = json.loads(array_match.group(0))

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []

        # Normalise: ensure all schema keys present, coerce nulls to ""
        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row: dict = {}
            for key in field_keys:
                val = item.get(key)
                row[key] = str(val) if val not in (None, "null", "None", "") else ""
            out.append(row)

        logger.info(f"Gemini extracted {len(out)} deals for {company_name}")
        return out

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {company_name}: {e}\nRaw: {raw_text[:500]}")
        return []
    except Exception as e:
        logger.error(f"Unexpected parse error for {company_name}: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def enrich_company(
    company_name: str,
    domain: str,
    goal: str,
    schema_fields: list[dict],
    year_range: tuple[int, int] = (2021, 2025),
    max_urls: int = 20,                 # kept for API compatibility, unused
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Single-call enrichment pipeline using Gemini 2.5 Flash + Google Search Grounding.

    Replaces the previous multi-step: Apify search → Apify scrape → Claude extract.
    Gemini handles searching, reading, and structured extraction in one API call.
    """
    boost_parts = []
    if extra_vendors:  boost_parts.append(f"{len(extra_vendors)} vendors")
    if extra_sources:  boost_parts.append(f"{len(extra_sources)} sources")
    if extra_keywords: boost_parts.append(f"{len(extra_keywords)} keywords")
    boost_str = f" [+{', '.join(boost_parts)}]" if boost_parts else ""

    yield {
        "type": "heartbeat",
        "message": f"🔍 Searching {company_name} deals ({year_range[0]}–{year_range[1]}){boost_str}…",
    }

    # Run blocking Gemini call in a thread — yield heartbeats while waiting
    gemini_task = asyncio.ensure_future(
        asyncio.to_thread(
            _gemini_extract_deals_sync,
            company_name, domain, goal, schema_fields,
            year_range, extra_vendors, extra_sources, extra_keywords,
        )
    )

    elapsed = 0
    while not gemini_task.done() and elapsed < 120:
        done, _ = await asyncio.wait({gemini_task}, timeout=8)
        elapsed += 8
        if done:
            break
        yield {"type": "heartbeat", "message": f"🌐 Gemini searching & reading sources… ({elapsed}s)"}

    if not gemini_task.done():
        gemini_task.cancel()
        logger.error(f"Gemini task timed out for {company_name}")
        row = {
            "company_name": company_name,
            "domain": domain,
            "_status": "timeout",
            "_sources": 0,
        }
        for f in schema_fields:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    try:
        deals: list[dict] = gemini_task.result()
    except Exception as e:
        logger.error(f"Gemini task raised: {e}")
        deals = []

    if not deals:
        yield {
            "type": "heartbeat",
            "message": f"⚠️ No deals found for {company_name}",
        }
        row = {
            "company_name": company_name,
            "domain": domain,
            "_status": "no_result",
            "_sources": 0,
        }
        for f in schema_fields:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {
        "type": "heartbeat",
        "message": f"✅ Found {len(deals)} deals for {company_name}",
    }

    for deal in deals:
        row = {
            "company_name": company_name,
            "domain": domain,
            "_status": "ok",
            "_sources": 1,   # Gemini used live grounding (sources embedded in response)
        }
        row.update(deal)
        yield {"type": "row_done", "row": row}
