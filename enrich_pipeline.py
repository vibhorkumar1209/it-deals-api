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
                 extra_searches: str, fields_desc: str, fields_json_keys: dict,
                 sector_block: str = "") -> str:
    return f"""You are an enterprise IT deal research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}{linkedin_block}

TASK: Find IT, technology, and strategic tech deals for {company_name}.

DEAL CATEGORIES TO CAPTURE (find ALL of these):
1. IT Outsourcing & Managed Services — SI contracts, BPO, ITO, infrastructure managed services
2. Cloud & Digital Transformation — cloud migration, SaaS rollouts, digital programmes
3. ERP / CRM / HCM / SCM — enterprise application implementations and upgrades
4. IT Acquisitions — tech company acquisitions, acqui-hires, asset purchases with IT angle
5. Strategic Joint Ventures & Corporate Venture — JVs with tech firms, CVC investments in tech cos
6. Enterprise Operations Partnerships — long-term IT ops partnerships, co-innovation agreements
7. Technology Disinvestments — IT asset sales, carve-outs, spin-offs, divestitures of tech units
8. Cybersecurity & Compliance — security platform contracts, SOC outsourcing, compliance tools
9. Analytics, AI & Data — AI/ML platform deals, data lake, BI, advanced analytics contracts
{sector_block}

SEARCHES TO RUN:
{search_focus}
Vendor/partner searches (pair each with company name):
{known_vendors}
{extra_searches}

EXTRACTION RULES:
- Read full articles, not just headlines
- One JSON object per distinct deal — never merge two deals
- Capture deals across ALL years available, not just recent ones
- Include press releases, news, vendor announcements, IR filings, annual reports
- PAY SPECIAL ATTENTION to: post spin-off / demerger IT separation programmes,
  nine-figure or billion-dollar IT outsourcing restructurings, IT carve-out programmes
  that replace systems from a former parent company, large infrastructure consolidation deals

Return ONLY a valid JSON array:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- vendor: exact name (e.g. "Infosys", "SAP S/4HANA", "Microsoft Azure", "Trimble")
- deal_type: IT Outsourcing | Cloud Migration | ERP | CRM | HCM | SCM | Cybersecurity |
  Analytics/AI | Digital Transformation | Infrastructure | SaaS | IT Acquisition |
  Joint Venture | Corporate Venture | Disinvestment | Managed Services | Other
- deal_value: e.g. "$3.2 billion" — omit if not public
- date_signed: YYYY-MM-DD or YYYY-MM or YYYY
- description: one concise sentence — what was agreed and why it matters
- source: direct URL to press release, article, or filing

Return ONLY the raw JSON array. No prose. No markdown fences.
Example shape: {json.dumps(fields_json_keys)}
"""


# ── Sector detection keywords → extra search topics ──────────────────────────
_SECTOR_KEYWORDS = {
    "agriculture_agtech": {
        "triggers": ["agco", "deere", "cnh", "claas", "kubota", "trimble agriculture",
                     "precision ag", "agtech", "farm", "agriculture", "agricultural",
                     "crop", "harvest", "livestock", "grain"],
        "deal_types": [
            "10. AgTech Acquisitions — precision ag, farm management software, autonomous machinery startups",
            "11. Precision Agriculture Platform — FMIS, guidance, telematics, yield monitoring deals",
            "12. Autonomous Farming Tech — autonomous tractor, robotics, AI harvesting JVs",
            "13. Farm Data & Analytics — farm data platforms, satellite/sensor analytics deals",
        ],
        "extra_vendors": (
            "Trimble, Climate Corporation, Monsanto, Bayer Crop Science, Raven Industries, "
            "Precision Planting, 640 Labs, 84.51°, Ag Leader Technology, CNH Industrial, "
            "AGCO Fendt, Hexagon Agriculture, Farmers Edge, Granular, Proagrica"
        ),
        "extra_searches": [
            "precision agriculture technology acquisition",
            "farm management software deal acquisition",
            "autonomous farming startup acquisition",
            "agtech acquisition startup buy",
            "precision ag platform partnership",
            "farm data analytics deal",
            "autonomous tractor robotics technology",
        ],
    },
    "automotive_manufacturing": {
        "triggers": ["truck", "automotive", "vehicle", "motor", "daimler",
                     "caterpillar", "volvo", "ford", "gm", "stellantis",
                     "manufacturer", "manufacturing", "industrial"],
        "deal_types": [
            "10. Telematics & Fleet Tech — OEM telematics, connected vehicle platforms, fleet SaaS",
            "11. Autonomous & ADAS Tech — autonomous driving JVs, ADAS software, sensor partnerships",
            "12. Heavy Hardware & AI SaaS — AI-powered machinery, precision agriculture/construction tech",
            "13. Manufacturing Execution & IoT — MES, IIoT platforms, Industry 4.0, digital twin deals",
        ],
        "extra_vendors": (
            "Trimble, Hexagon, Geotab, Samsara, Mobileye, NVIDIA, Qualcomm, HERE Technologies, "
            "Aptiv, Verizon Connect, PTC, Dassault Systèmes, Siemens Digital Industries, "
            "Bosch Connected Devices, Continental, ZF Friedrichshafen"
        ),
        "extra_searches": [
            "telematics fleet management deal",
            "autonomous vehicle technology partnership",
            "connected truck platform agreement",
            "precision agriculture technology deal",
            "AI SaaS manufacturing contract",
            "digital twin industrial IoT agreement",
            "spin-off IT separation carve-out independent infrastructure",
            "post spin-off IT systems standalone architecture",
            "demerger IT outsourcing restructuring billion",
        ],
    },
    "banking_finance": {
        "triggers": ["bank", "financial", "insurance", "hdfc", "icici", "fintech",
                     "payment", "lending", "asset management"],
        "deal_types": [
            "10. Core Banking & Payments — core banking platform migrations, payment rails, open banking",
            "11. RegTech & Compliance — AML, KYC, risk platform contracts",
            "12. Digital Banking — mobile/internet banking platform deals, neobank JVs",
        ],
        "extra_vendors": (
            "Temenos, Finastra, FIS, Fiserv, Mambu, Thought Machine, Backbase, "
            "TCS BaNCS, Oracle FLEXCUBE, Intellect Design, Newgen, Finacle"
        ),
        "extra_searches": [
            "core banking platform deal",
            "digital banking transformation",
            "payment technology partnership",
            "fintech investment acquisition",
        ],
    },
    "telecom": {
        "triggers": ["telecom", "telco", "wireless", "network", "5g", "broadband",
                     "spectrum", "operator"],
        "deal_types": [
            "10. Network & BSS/OSS — 5G rollout contracts, BSS/OSS transformation, network managed services",
            "11. Digital Services Platform — content, cloud, enterprise ICT deals",
        ],
        "extra_vendors": (
            "Ericsson, Nokia, Amdocs, Netcracker, Huawei, Mavenir, Rakuten Symphony"
        ),
        "extra_searches": [
            "5G network contract deal",
            "BSS OSS transformation",
            "network managed services agreement",
        ],
    },
}


def _detect_sector_block(company_name: str) -> str:
    """Return sector-specific deal categories and searches if company matches a sector."""
    name_lower = company_name.lower()
    for sector, cfg in _SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in cfg["triggers"]):
            lines = cfg["deal_types"]
            return "\n".join(lines)
    return ""


def _detect_sector_extra(company_name: str) -> tuple[str, str]:
    """Return (extra_vendors, extra_searches_block) for detected sector."""
    name_lower = company_name.lower()
    for sector, cfg in _SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in cfg["triggers"]):
            searches = "\n".join(
                f'  - "{company_name}" {s}' for s in cfg["extra_searches"]
            )
            return cfg["extra_vendors"], searches
    return "", ""


def _build_prompts(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_tech: list[str],
    focus_vendor: list[str],
) -> list[str]:
    """
    Return 2 focused prompts.
    Call 1: broad IT + acquisitions/JVs/disinvestments + top SI vendors.
    Call 2: year sweep + sector-specific + user focus.
    Each targets ~60s completion.
    """
    linkedin_block = f" | LinkedIn: {linkedin_url}" if linkedin_url else ""
    fields_json_keys = {f["key"]: f"<{f['type']}>" for f in SCHEMA_FIELDS}
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["description"]}"' for f in SCHEMA_FIELDS)

    sector_block = _detect_sector_block(company_name)
    sector_vendors, sector_searches = _detect_sector_extra(company_name)

    # ── Prompt 1: broad sweep + acquisitions/JVs/disinvestments + top vendors ─
    p1_searches = f"""  - "{company_name}" IT outsourcing contract deal signed
  - "{company_name}" technology acquisition acqui-hire
  - "{company_name}" joint venture technology partner
  - "{company_name}" corporate venture fund investment tech startup
  - "{company_name}" IT disinvestment carve-out spin-off asset sale
  - "{company_name}" digital transformation program cloud migration
  - "{company_name}" managed services ERP SAP Oracle implementation
  - "{company_name}" cybersecurity infrastructure data center contract
  - "{company_name}" IT separation spin-off independent infrastructure
  - "{company_name}" post spin-off IT systems carve-out restructuring
  - "{company_name}" IT outsourcing restructuring nine-figure billion
  - "{company_name}" independent digital architecture enterprise systems
  - "{company_name}" demerger IT separation standalone systems
  - "{company_name}" acquires technology company startup
  - "{company_name}" acquisition technology software hardware
  - "{company_name}" bought acquired tech firm stake joint venture
  - site:businesswire.com OR site:prnewswire.com "{company_name}" acquires
  - site:businesswire.com OR site:prnewswire.com "{company_name}" technology deal"""

    p1_vendors = (
        "Accenture, Infosys, TCS, Wipro, HCLTech, Cognizant, Capgemini, DXC Technology, "
        "IBM, SAP, Oracle, Microsoft, AWS, Google Cloud, ServiceNow, Salesforce, Workday, "
        "Baker McKenzie, EY, Deloitte, KPMG, PwC"   # advisory/legal for large IT restructurings
    )

    prompt1 = _make_prompt(company_name, domain, linkedin_block,
                           p1_searches, p1_vendors, "",
                           fields_desc, fields_json_keys, sector_block)

    # ── Prompt 2: year sweep + sector vendors + user focus ────────────────────
    # Combine IT deal + acquisition into one query per year to halve search count
    year_searches = "\n".join(
        f'  - "{company_name}" IT deal OR technology acquisition OR tech contract {y}' for y in range(2016, 2026)
    )

    p2_vendors = "Dell Technologies, HPE, Atos, NTT DATA, Unisys, Fujitsu, T-Systems, CGI, Snowflake, CrowdStrike, Palo Alto Networks"
    if sector_vendors:
        p2_vendors += f", {sector_vendors}"

    extra_parts = []
    if sector_searches:
        extra_parts.append(f"Sector-specific searches:\n{sector_searches}")
    if focus_tech or focus_vendor:
        lines = ["User-specified focus searches:"]
        for t in focus_tech:
            lines.append(f'  - "{company_name}" {t} deal contract')
        for v in focus_vendor:
            lines.append(f'  - "{company_name}" {v} deal partnership acquisition')
        extra_parts.append("\n".join(lines))
    extra = "\n\n".join(extra_parts)

    prompt2 = _make_prompt(company_name, domain, linkedin_block,
                           year_searches, p2_vendors, extra,
                           fields_desc, fields_json_keys, sector_block)

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

    import time as _time
    MAX_RETRIES = 3

    HTTP_TIMEOUT = 90   # hard HTTP-level timeout — prevents indefinite hangs

    response = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            try:
                http_opts = types.HttpOptions(timeout=HTTP_TIMEOUT)
            except Exception:
                http_opts = {"timeout": HTTP_TIMEOUT}
            client = genai.Client(api_key=GOOGLE_AI_KEY, http_options=http_opts)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=16384,
                ),
            )
            break   # success
        except Exception as api_err:
            err_str = str(api_err)
            is_quota = "RESOURCE_EXHAUSTED" in err_str or "free_tier" in err_str
            is_timeout = any(x in err_str.lower() for x in ("timeout", "timed out", "deadline"))
            is_retryable = not is_quota and (any(x in err_str for x in ("503", "UNAVAILABLE", "overloaded")) or is_timeout)
            logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed for {company_name}: {api_err}")
            if is_quota:
                # Quota exhausted — not retryable, raise so callers can surface to user
                raise RuntimeError(f"Gemini quota exhausted (free tier limit reached). "
                                   f"Please upgrade your Google AI API key to a paid plan.") from api_err
            if is_retryable and attempt < MAX_RETRIES:
                _time.sleep(15 * attempt)
                continue
            logger.error(f"Gemini API error for {company_name}: {api_err}", exc_info=True)
            return []

    if response is None:
        return []

    try:

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

    # Strip common subsidiary suffixes to derive a broader search name for fallback
    _STRIP = re.compile(
        r"\s+(north america|south america|europe|asia|apac|latam|"
        r"inc\.?|llc\.?|ltd\.?|corp\.?|corporation|group|holdings|"
        r"gmbh|ag|plc|sa|bv|nv|pty|co\.?)\s*$",
        re.IGNORECASE,
    )
    brand_name = _STRIP.sub("", company_name).strip()

    prompts = _build_prompts(company_name, domain, linkedin_url, ft, fv)
    seen_keys: set[str] = set()   # deduplicate across the two calls
    total_deals = 0
    CALL_TIMEOUT = 150            # 2.5 min per call

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

    # ── Fallback: retry with brand name if subsidiary name returned nothing ──────
    if total_deals == 0 and brand_name.lower() != company_name.lower():
        yield {"type": "heartbeat",
               "message": f"🔄 No deals for '{company_name}' — retrying as '{brand_name}'…"}
        await asyncio.sleep(0)

        fallback_prompts = _build_prompts(brand_name, domain, linkedin_url, ft, fv)
        for call_idx, prompt in enumerate(fallback_prompts, 1):
            yield {"type": "heartbeat",
                   "message": f"🔍 Fallback [{call_idx}/{len(fallback_prompts)}] Searching '{brand_name}'…"}
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, _gemini_extract_deals_sync, prompt, brand_name)
            elapsed = 0
            call_deals: list[dict] = []

            while elapsed < CALL_TIMEOUT:
                try:
                    call_deals = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                    break
                except asyncio.TimeoutError:
                    elapsed += 10
                    yield {"type": "heartbeat", "message": f"🌐 Fallback searching… ({elapsed}s)"}
                    await asyncio.sleep(0)
                except Exception as e:
                    logger.error(f"Fallback error for {brand_name}: {e}", exc_info=True)
                    call_deals = []
                    break
            else:
                future.cancel()

            for deal in call_deals:
                dedup_key = f"{deal.get('vendor','').lower()}|{deal.get('date_signed','')}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                row = {"company_name": company_name, "domain": domain,
                       "_status": "ok", "_sources": 1}
                row.update(deal)
                yield {"type": "row_done", "row": row}
                await asyncio.sleep(0.05)
                total_deals += 1

        if total_deals > 0:
            yield {"type": "heartbeat",
                   "message": f"✅ Fallback found {total_deals} deals via '{brand_name}'"}

    if total_deals == 0:
        yield {"type": "heartbeat", "message": f"⚠️ No deals found for {company_name}"}
        row = {"company_name": company_name, "domain": domain,
               "_status": "no_result", "_sources": 0}
        for f in SCHEMA_FIELDS:
            row[f["key"]] = ""
        yield {"type": "row_done", "row": row}
