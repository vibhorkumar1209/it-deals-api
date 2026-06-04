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
    Build the Gemini research prompt. Industry detection is delegated to Gemini
    (it knows the company). User focus lists get boosted priority.
    """
    # Compose industry hint section
    industry_sections = []
    for sector, ctx in INDUSTRY_CONTEXT.items():
        if sector == "default":
            continue
        industry_sections.append(
            f"  {sector}: tech={', '.join(ctx['tech_areas'][:5])} | "
            f"vendors={', '.join(ctx['vendors'][:6])}"
        )

    focus_tech_block = ""
    if focus_tech:
        focus_tech_block = (
            f"\nUSER-SPECIFIED FOCUS TECHNOLOGIES (search these first, find ALL deals):\n"
            + "\n".join(f"  - {t}" for t in focus_tech)
        )

    focus_vendor_block = ""
    if focus_vendor:
        focus_vendor_block = (
            f"\nUSER-SPECIFIED FOCUS VENDORS (search these first, find ALL deals):\n"
            + "\n".join(f"  - {v}" for v in focus_vendor)
        )

    linkedin_block = f"\nLinkedIn company page: {linkedin_url}" if linkedin_url else ""

    fields_json_keys = {f["key"]: f"<{f['type']}>" for f in SCHEMA_FIELDS}
    fields_desc = "\n".join(
        f'  "{f["key"]}": "{f["description"]}"' for f in SCHEMA_FIELDS
    )

    return f"""You are an enterprise IT deal research analyst with access to live Google Search.

COMPANY TO RESEARCH:
  Name: {company_name}
  Website: {domain}{linkedin_block}

TASK: Find EVERY IT and technology deal, contract, outsourcing agreement, vendor selection,
and digital transformation initiative involving {company_name}.

RESEARCH STRATEGY:
1. First, identify {company_name}'s industry sector.
2. Based on the industry, determine the most relevant technology areas and major vendors.
   Use these industry-to-vendor mappings as guidance:
{chr(10).join(industry_sections)}

3. Run multiple Google searches to maximise coverage:
   - "{company_name}" + major vendor names (by industry)
   - "{company_name}" + key technology areas (by industry)
   - "{company_name}" IT deal contract signed announcement
   - "{company_name}" digital transformation outsourcing agreement
   - site:businesswire.com OR site:prnewswire.com "{company_name}" technology
   - "{company_name}" annual report technology spend
{focus_tech_block}{focus_vendor_block}

4. Read the actual articles — not just headlines. Extract deal specifics.
5. Include deals from press releases, news articles, vendor announcements, IR filings.

OUTPUT FORMAT — return ONLY a valid JSON array, one object per deal:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- "vendor": exact vendor/product name (e.g. "SAP S/4HANA", "TCS", "Microsoft Azure")
- "deal_type": one of: ERP | CRM | Cloud Migration | Managed Services | Cybersecurity |
  Outsourcing | Analytics/AI | Digital Transformation | Infrastructure | SaaS | Core Banking |
  HCM | SCM | Network | Other
- "deal_value": e.g. "$45 million" or "$1.2 billion" — null if not public
- "date_signed": YYYY-MM-DD if exact, YYYY-MM if month known, YYYY if year only
- "description": one clear sentence stating what was agreed
- "source": full URL of press release or news article — null if not found

CRITICAL RULES:
- One JSON object per distinct deal — never merge two deals into one object
- Cover ALL years 2010–2026 — do not limit to recent years
- Return [] if genuinely no deals found
- Return ONLY the JSON array — no prose, no markdown fences, nothing else

Example shape:
{json.dumps(fields_json_keys, indent=2)}
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
                max_output_tokens=8192,
            ),
        )
        raw_text = response.text or ""
        logger.info(f"Gemini response: {len(raw_text)} chars for {company_name}")
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

    gemini_task = asyncio.ensure_future(
        asyncio.to_thread(
            _gemini_extract_deals_sync,
            company_name, domain, linkedin_url, ft, fv,
        )
    )

    elapsed = 0
    while not gemini_task.done() and elapsed < 150:
        done, _ = await asyncio.wait({gemini_task}, timeout=8)
        elapsed += 8
        if done:
            break
        yield {"type": "heartbeat",
               "message": f"🌐 Gemini searching & reading sources… ({elapsed}s)"}

    if not gemini_task.done():
        gemini_task.cancel()
        logger.error(f"Gemini timed out for {company_name}")
        row = {"company_name": company_name, "domain": domain,
               "_status": "timeout", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    try:
        deals: list[dict] = gemini_task.result()
    except Exception as e:
        logger.error(f"Gemini task error for {company_name}: {e}")
        deals = []

    if not deals:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found for {company_name}"}
        row = {"company_name": company_name, "domain": domain,
               "_status": "no_result", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
        return

    yield {"type": "heartbeat",
           "message": f"✅ {company_name}: {len(deals)} deals found"}

    for deal in deals:
        row = {"company_name": company_name, "domain": domain,
               "_status": "ok", "_sources": 1}
        row.update(deal)
        yield {"type": "row_done", "row": row}
