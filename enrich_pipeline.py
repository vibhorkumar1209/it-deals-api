"""
IT Deal enrichment pipeline — Gemini 2.5 Flash + Google Search Grounding.

Single-call architecture:
  1. Build a research prompt: company × industry × vendors × keywords (2010–2026)
  2. Call gemini-2.5-flash with googleSearch tool so it searches live web
  3. Parse JSON array from response → yield one row_done per deal
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

# Fixed output schema — matches the IT Deal Details preset in the frontend
SCHEMA_FIELDS = [
    {"key": "vendor",      "label": "Vendor Name",  "type": "string",
     "description": "Technology vendor or service provider name"},
    {"key": "deal_type",   "label": "Deal Type",    "type": "string",
     "description": "e.g. ERP, Cloud Migration, Cybersecurity, Outsourcing, Managed Services"},
    {"key": "deal_value",  "label": "Deal Value",   "type": "string",
     "description": "Contract value in USD millions if publicly known"},
    {"key": "date_signed", "label": "Date Signed",  "type": "date",
     "description": "Announcement or contract signing date (YYYY-MM-DD or YYYY-MM or YYYY)"},
    {"key": "description", "label": "Description",  "type": "string",
     "description": "One sentence describing what was agreed"},
    {"key": "source",      "label": "Source URL",   "type": "string",
     "description": "URL of the press release, news article, or filing"},
]

FIXED_GOAL = (
    "Find every IT and technology deal, contract, outsourcing agreement, and digital "
    "transformation initiative involving this company."
)

# ── Industry-aware vendor + tech-area maps ────────────────────────────────────
# Gemini will auto-detect industry; these give it richer context per sector.

INDUSTRY_CONTEXT = {
    "banking_finance": {
        "tech_areas": ["core banking", "digital banking", "payments", "risk & compliance",
                       "anti-money laundering", "trade finance", "wealth management platform",
                       "open banking", "cloud migration", "cybersecurity"],
        "vendors": ["Temenos", "Finacle", "FIS", "Fiserv", "Finastra", "Mambu",
                    "Thought Machine", "Intellect Design", "TCS BaNCS", "Newgen",
                    "Oracle FLEXCUBE", "SAP Banking", "IBM", "Accenture", "Infosys",
                    "Wipro", "TCS", "HCLTech", "Cognizant", "AWS", "Microsoft Azure"],
    },
    "insurance": {
        "tech_areas": ["policy administration", "claims management", "underwriting automation",
                       "insurtech", "telematics", "digital distribution", "cloud migration"],
        "vendors": ["Guidewire", "Duck Creek", "Majesco", "EIS", "SAP Insurance",
                    "Oracle Insurance", "Accenture", "Capgemini", "TCS", "Infosys"],
    },
    "manufacturing": {
        "tech_areas": ["ERP", "MES", "supply chain", "IoT", "industry 4.0",
                       "digital twin", "PLM", "warehouse management", "procurement"],
        "vendors": ["SAP", "Oracle", "Siemens", "PTC", "Dassault Systèmes", "IBM",
                    "Accenture", "Wipro", "TCS", "HCLTech", "Infor", "IFS"],
    },
    "retail_ecommerce": {
        "tech_areas": ["e-commerce platform", "POS", "loyalty programme", "supply chain",
                       "omnichannel", "demand forecasting", "CRM", "personalisation"],
        "vendors": ["Salesforce", "SAP", "Oracle", "Manhattan Associates", "Blue Yonder",
                    "Shopify", "commercetools", "Infosys", "TCS", "Cognizant"],
    },
    "telecom": {
        "tech_areas": ["BSS/OSS", "5G", "network virtualisation", "billing", "CRM",
                       "network managed services", "cloud transformation"],
        "vendors": ["Ericsson", "Nokia", "Amdocs", "Netcracker", "Huawei", "IBM",
                    "Accenture", "TCS", "Infosys", "Wipro"],
    },
    "healthcare": {
        "tech_areas": ["EMR/EHR", "hospital information system", "revenue cycle",
                       "telemedicine", "AI diagnostics", "pharmacy management", "claims"],
        "vendors": ["Epic", "Cerner", "Meditech", "Allscripts", "Philips", "Siemens Healthineers",
                    "Oracle Health", "TCS", "Cognizant", "Accenture"],
    },
    "default": {
        "tech_areas": ["ERP", "CRM", "cloud migration", "managed services", "cybersecurity",
                       "digital transformation", "analytics", "AI/ML", "outsourcing",
                       "infrastructure", "SaaS implementation"],
        "vendors": ["SAP", "Oracle", "Microsoft", "AWS", "Google Cloud", "Salesforce",
                    "ServiceNow", "Workday", "IBM", "Accenture", "TCS", "Infosys",
                    "Wipro", "HCLTech", "Cognizant", "Capgemini", "DXC Technology",
                    "Palo Alto Networks", "CrowdStrike", "Snowflake"],
    },
}


def _build_research_prompt(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_tech: list[str],
    focus_vendor: list[str],
) -> str:
    """
    Build the Gemini research prompt. Broad baseline sweep always runs first;
    focus_tech / focus_vendor are additive extra search passes.
    """
    linkedin_block = f"\n  LinkedIn: {linkedin_url}" if linkedin_url else ""

    fields_json_keys = {f["key"]: f"<{f['type']}>" for f in SCHEMA_FIELDS}
    fields_desc = "\n".join(
        f'  "{f["key"]}": "{f["description"]}"' for f in SCHEMA_FIELDS
    )

    # ── Focus blocks (additive only) ──────────────────────────────────────────
    focus_block = ""
    if focus_tech or focus_vendor:
        lines = ["\nPASS 4 — USER-SPECIFIED FOCUS (run these searches IN ADDITION to passes 1-3):"]
        if focus_tech:
            for t in focus_tech:
                lines.append(f'  - "{company_name}" {t} deal contract agreement')
        if focus_vendor:
            for v in focus_vendor:
                lines.append(f'  - "{company_name}" {v} deal contract partnership')
        focus_block = "\n".join(lines)

    return f"""You are a senior enterprise IT deal research analyst with live Google Search access.
Your goal: find UP TO 50 distinct IT and technology deals for {company_name}. Be exhaustive.

COMPANY:
  Name: {company_name}
  Website: {domain}{linkedin_block}

━━━ MANDATORY SEARCH PASSES (always execute all three) ━━━

PASS 1 — BROAD DEAL SWEEP (run ALL of these searches):
  - "{company_name}" IT outsourcing deal signed
  - "{company_name}" technology contract award
  - "{company_name}" digital transformation program
  - "{company_name}" cloud migration agreement
  - "{company_name}" managed services contract
  - "{company_name}" ERP implementation SAP Oracle
  - "{company_name}" infrastructure deal data center
  - "{company_name}" cybersecurity contract
  - "{company_name}" IT carve-out separation spin-off technology
  - site:businesswire.com "{company_name}" technology deal
  - site:prnewswire.com "{company_name}" IT contract
  - site:globenewswire.com "{company_name}" technology

PASS 2 — MAJOR VENDOR SWEEP (search each vendor paired with company name):
  Accenture, Infosys, TCS, Wipro, HCLTech, Cognizant, Capgemini, DXC Technology,
  IBM, SAP, Oracle, Microsoft, AWS, Google Cloud, ServiceNow, Salesforce,
  Workday, Dell Technologies, HPE, Atos, CGI, NTT DATA, Unisys, Fujitsu,
  T-Systems, Siemens, Bosch, PTC, Dassault Systèmes, Palo Alto Networks

PASS 3 — YEAR-BY-YEAR SWEEP (search by year to surface older deals):
  For each year from 2015 to 2025, search:
  - "{company_name}" IT deal {"{year}"}
  - "{company_name}" technology contract {"{year}"}
{focus_block}

━━━ EXTRACTION RULES ━━━
- Read the FULL article for every promising result — do not stop at headlines
- Include: outsourcing contracts, cloud migrations, ERP/CRM rollouts, managed services,
  infrastructure deals, IT carve-outs, SaaS agreements, vendor selections, SI contracts,
  digital transformation programmes, joint ventures with tech angle
- Each distinct deal = one JSON object. Never merge two deals.
- If a deal spans multiple years, use the announcement/signing date

━━━ OUTPUT ━━━
Return ONLY a valid JSON array — no prose, no markdown, no explanation:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- "vendor": exact vendor or product name (e.g. "Infosys", "SAP S/4HANA", "Microsoft Azure")
- "deal_type": ERP | CRM | Cloud Migration | Managed Services | Cybersecurity | Outsourcing |
  Analytics/AI | Digital Transformation | Infrastructure | SaaS | HCM | SCM | Network |
  IT Carve-out | Other
- "deal_value": e.g. "$3.2 billion" — omit if not publicly disclosed
- "date_signed": YYYY-MM-DD preferred; YYYY-MM or YYYY if exact date unknown
- "description": one concise sentence — what was agreed and why it matters
- "source": direct URL to press release, news article, or filing

HARD RULES:
- TARGET 50 deals — do not stop at 10 or 15, keep searching until exhausted
- Return [] only if genuinely zero deals exist
- Return ONLY the raw JSON array

Example shape: {json.dumps(fields_json_keys)}
"""


def _gemini_extract_deals_sync(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_tech: list[str],
    focus_vendor: list[str],
) -> list[dict]:
    """
    Blocking Gemini 2.5 Flash call with Google Search grounding.
    Runs inside asyncio.to_thread.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        return []

    if not GOOGLE_AI_KEY:
        logger.error("GOOGLE_AI_API_KEY env var not set")
        return []

    prompt = _build_research_prompt(company_name, domain, linkedin_url, focus_tech, focus_vendor)
    field_keys = [f["key"] for f in SCHEMA_FIELDS]

    try:
        client = genai.Client(api_key=GOOGLE_AI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=16384,
            ),
        )
        # response.text can raise ValueError when grounding is active and the
        # response has non-text parts — extract text manually from parts.
        try:
            raw_text = response.text or ""
        except Exception:
            raw_text = ""
            try:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        raw_text += part.text
            except Exception:
                pass

        logger.info(f"Gemini response: {len(raw_text)} chars for {company_name}")
        if not raw_text:
            logger.warning(f"Empty Gemini response for {company_name}. "
                           f"Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}")
            return []
    except Exception as e:
        logger.error(f"Gemini API error for {company_name}: {e}")
        return []

    # ── Parse JSON array from response ───────────────────────────────────────
    try:
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", raw_text.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

        # Try direct parse first
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Find the outermost [...] array
            array_match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not array_match:
                logger.warning(f"No JSON array in Gemini response for {company_name}. Preview: {clean[:400]}")
                return []
            try:
                parsed = json.loads(array_match.group(0))
            except json.JSONDecodeError:
                # Try to find the LAST complete array by scanning from the end
                # This handles trailing commas or truncated JSON
                text = array_match.group(0)
                # Remove trailing commas before ] or }
                text = re.sub(r",\s*([\]}])", r"\1", text)
                parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []

        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row: dict = {}
            for key in field_keys:
                val = item.get(key)
                row[key] = str(val) if val not in (None, "null", "None", "") else ""
            out.append(row)

        logger.info(f"Parsed {len(out)} deals for {company_name}")
        return out

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {company_name}: {e} | raw: {raw_text[:400]}")
        return []
    except Exception as e:
        logger.error(f"Parse error for {company_name}: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def enrich_company(
    company_name: str,
    domain: str,
    goal: str = FIXED_GOAL,           # kept for API compat, not used in prompt
    schema_fields: list[dict] | None = None,   # kept for API compat, not used
    year_range: tuple[int, int] = (2010, 2026),
    max_urls: int = 20,               # kept for API compat, unused
    linkedin_url: str = "",
    focus_tech: list[str] | None = None,
    focus_vendor: list[str] | None = None,
    # legacy aliases accepted but ignored
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Find IT deals for one company (2010–2026) via Gemini + Google Search Grounding.
    Yields heartbeat events then row_done events (one per deal).
    """
    ft = focus_tech or []
    fv = focus_vendor or []

    focus_parts = []
    if ft:  focus_parts.append(f"{len(ft)} tech focus")
    if fv:  focus_parts.append(f"{len(fv)} vendor focus")
    focus_str = f" [{', '.join(focus_parts)}]" if focus_parts else ""

    yield {"type": "heartbeat",
           "message": f"🔍 Researching {company_name} IT deals{focus_str}…"}
    await asyncio.sleep(0)  # flush SSE before blocking call

    # Run Gemini in a thread; send heartbeats every 10s so SSE stays alive
    loop = asyncio.get_event_loop()
    gemini_future = loop.run_in_executor(
        None,
        _gemini_extract_deals_sync,
        company_name, domain, linkedin_url, ft, fv,
    )

    elapsed = 0
    deals: list[dict] = []
    TIMEOUT = 240

    while elapsed < TIMEOUT:
        try:
            deals = await asyncio.wait_for(asyncio.shield(gemini_future), timeout=10)
            break  # finished successfully
        except asyncio.TimeoutError:
            elapsed += 10
            yield {"type": "heartbeat",
                   "message": f"🌐 Gemini searching & reading sources… ({elapsed}s)"}
            await asyncio.sleep(0)  # flush heartbeat to client
        except Exception as e:
            logger.error(f"Gemini task error for {company_name}: {e}", exc_info=True)
            yield {"type": "heartbeat", "message": f"⚠️ Gemini error: {e}"}
            deals = []
            break
    else:
        # True timeout
        gemini_future.cancel()
        logger.error(f"Gemini timed out ({TIMEOUT}s) for {company_name}")
        row = {"company_name": company_name, "domain": domain,
               "_status": "timeout", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    if not deals:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found for {company_name}"}
        row = {"company_name": company_name, "domain": domain,
               "_status": "no_result", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat",
           "message": f"✅ {company_name}: {len(deals)} deals found — populating table…"}

    for i, deal in enumerate(deals):
        row = {"company_name": company_name, "domain": domain,
               "_status": "ok", "_sources": 1}
        row.update(deal)
        yield {"type": "row_done", "row": row}
        await asyncio.sleep(0.05)   # force SSE flush between each row
