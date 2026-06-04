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


def _make_prompt(company_name: str, domain: str, linkedin_block: str,
                 search_focus: str, known_vendors: str,
                 extra_searches: str, fields_desc: str, fields_json_keys: dict) -> str:
    return f"""You are an enterprise IT deal research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}{linkedin_block}

TASK: Find IT and technology deals for {company_name}. Search these topics:
{search_focus}
Vendor pairs to search (each vendor + company name):
{known_vendors}
{extra_searches}

EXTRACTION RULES:
- Read full articles, not just headlines
- One JSON object per distinct deal
- Include: outsourcing, cloud, ERP/CRM, managed services, infrastructure,
  IT carve-outs, SaaS, vendor selections, digital transformation, acquisitions with tech angle

Return ONLY a valid JSON array:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- vendor: exact name (e.g. "Infosys", "SAP S/4HANA", "Microsoft Azure")
- deal_type: ERP | CRM | Cloud Migration | Managed Services | Cybersecurity | Outsourcing |
  Analytics/AI | Digital Transformation | Infrastructure | SaaS | HCM | SCM | IT Carve-out | Other
- deal_value: e.g. "$3.2 billion" or omit if not public
- date_signed: YYYY-MM-DD or YYYY-MM or YYYY
- description: one clear sentence on what was agreed
- source: direct URL to press release or news article

Return ONLY the raw JSON array. No prose. No markdown fences.
Example shape: {json.dumps(fields_json_keys)}
"""


def _build_prompts(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_tech: list[str],
    focus_vendor: list[str],
) -> list[str]:
    """
    Return 2 focused prompts instead of one huge one.
    Call 1: broad IT deals + major SI/cloud vendors.
    Call 2: year-by-year sweep + focus tech/vendors.
    Each completes in ~45-75s.
    """
    linkedin_block = f" | LinkedIn: {linkedin_url}" if linkedin_url else ""
    fields_json_keys = {f["key"]: f"<{f['type']}>" for f in SCHEMA_FIELDS}
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["description"]}"' for f in SCHEMA_FIELDS)

    # ── Prompt 1: broad sweep + top vendors ───────────────────────────────────
    p1_searches = f"""  - "{company_name}" IT outsourcing deal signed
  - "{company_name}" technology contract award
  - "{company_name}" digital transformation program
  - "{company_name}" cloud migration agreement
  - "{company_name}" managed services contract
  - "{company_name}" ERP SAP Oracle implementation
  - "{company_name}" infrastructure data center deal
  - "{company_name}" cybersecurity contract
  - "{company_name}" IT carve-out spin-off technology
  - site:businesswire.com OR site:prnewswire.com "{company_name}" technology"""

    p1_vendors = (
        "Accenture, Infosys, TCS, Wipro, HCLTech, Cognizant, Capgemini, DXC Technology, "
        "IBM, SAP, Oracle, Microsoft, AWS, Google Cloud, ServiceNow, Salesforce, Workday"
    )

    prompt1 = _make_prompt(company_name, domain, linkedin_block,
                           p1_searches, p1_vendors, "",
                           fields_desc, fields_json_keys)

    # ── Prompt 2: year sweep + more vendors + user focus ─────────────────────
    year_searches = "\n".join(
        f'  - "{company_name}" IT deal technology contract {y}' for y in range(2015, 2026)
    )

    p2_vendors = (
        "Dell Technologies, HPE, Atos, NTT DATA, Unisys, Fujitsu, T-Systems, CGI, "
        "Siemens, Bosch, PTC, Dassault Systèmes, Palo Alto Networks, CrowdStrike, Snowflake, "
        "Trimble, Hexagon, Esri, Bentley Systems"
    )

    extra = ""
    if focus_tech or focus_vendor:
        lines = ["Additional user-specified searches:"]
        for t in focus_tech:
            lines.append(f'  - "{company_name}" {t} deal contract')
        for v in focus_vendor:
            lines.append(f'  - "{company_name}" {v} deal partnership')
        extra = "\n".join(lines)

    prompt2 = _make_prompt(company_name, domain, linkedin_block,
                           year_searches, p2_vendors, extra,
                           fields_desc, fields_json_keys)

    return [prompt1, prompt2]


def _gemini_extract_deals_sync(
    prompt: str,
    company_name: str,
) -> list[dict]:
    """
    Blocking Gemini 2.5 Flash call with Google Search grounding.
    Accepts a pre-built prompt. Runs inside run_in_executor.
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

        # ── Extract raw text — ALWAYS use parts iteration, never rely on .text ──
        raw_text = ""
        try:
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts or []):
                    t = getattr(part, "text", None)
                    if t:
                        raw_text += t
        except Exception as parts_err:
            logger.warning(f"Parts extraction failed for {company_name}: {parts_err}")
            # Last-ditch: try .text property
            try:
                raw_text = response.text or ""
            except Exception:
                pass

        finish_reason = "unknown"
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass

        logger.info(f"Gemini response: {len(raw_text)} chars, finish={finish_reason} for {company_name}")

        if not raw_text:
            logger.warning(f"Empty response for {company_name}. finish_reason={finish_reason}")
            return []

    except Exception as e:
        logger.error(f"Gemini API error for {company_name}: {e}", exc_info=True)
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
            # Find the outermost [ ... ] array
            array_match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not array_match:
                # Response might be truncated — try to recover partial objects
                logger.warning(f"No complete JSON array for {company_name}. Attempting recovery. Preview: {clean[:400]}")
                # Find start of array, close it after last complete object
                start = clean.find("[")
                if start == -1:
                    return []
                fragment = clean[start:]
                # Find last complete '}' and close array there
                last_brace = fragment.rfind("}")
                if last_brace == -1:
                    return []
                fragment = fragment[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError:
                    logger.error(f"Recovery failed for {company_name}")
                    return []
            else:
                candidate_text = array_match.group(0)
                try:
                    parsed = json.loads(candidate_text)
                except json.JSONDecodeError:
                    # Fix trailing commas + try again
                    candidate_text = re.sub(r",\s*([\]}])", r"\1", candidate_text)
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        # Truncated: recover last complete object
                        last_brace = candidate_text.rfind("}")
                        if last_brace == -1:
                            return []
                        fragment = candidate_text[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                        fragment = "[" + fragment.lstrip("[")
                        parsed = json.loads(fragment)

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

    prompts = _build_prompts(company_name, domain, linkedin_url, ft, fv)
    seen_keys: set[str] = set()   # deduplicate across the two calls
    total_deals = 0
    CALL_TIMEOUT = 120            # 2 min per call — well within Gemini's capacity

    for call_idx, prompt in enumerate(prompts, 1):
        label = "broad IT & cloud deals" if call_idx == 1 else "year-by-year + vendor sweep"
        yield {"type": "heartbeat",
               "message": f"🔍 [{call_idx}/{len(prompts)}] Searching {company_name}: {label}…"}
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_extract_deals_sync, prompt, company_name)

        elapsed = 0
        call_deals: list[dict] = []

        while elapsed < CALL_TIMEOUT:
            try:
                call_deals = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat",
                       "message": f"🌐 [{call_idx}/{len(prompts)}] Gemini searching… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Gemini call {call_idx} error for {company_name}: {e}", exc_info=True)
                yield {"type": "heartbeat", "message": f"⚠️ Call {call_idx} error: {e}"}
                call_deals = []
                break
        else:
            future.cancel()
            logger.warning(f"Call {call_idx} timed out for {company_name}")
            yield {"type": "heartbeat", "message": f"⏱ Call {call_idx} timed out — partial results below"}

        # Deduplicate and stream new deals immediately
        new_deals = 0
        for deal in call_deals:
            dedup_key = f"{deal.get('vendor','').lower()}|{deal.get('date_signed','')}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": 1}
            row.update(deal)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.05)
            new_deals += 1
            total_deals += 1

        yield {"type": "heartbeat",
               "message": f"✅ Call {call_idx} done: +{new_deals} deals (total {total_deals})"}
        await asyncio.sleep(0)

    if total_deals == 0:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found for {company_name}"}
        row = {"company_name": company_name, "domain": domain,
               "_status": "no_result", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
