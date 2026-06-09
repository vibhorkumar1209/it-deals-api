"""
GCC Intelligence Hub pipeline — v2

Two independent input modes:
  Mode A (Company Search):  list of known companies → enrich each
  Mode B (Industry Discovery): industry + location → discover companies → enrich

Enrichment produces 6 sections per company:
  profile · capabilities · projects · talent · financials · techstack
"""

import asyncio
import json
import logging
import os
import re
import time as _time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")


# ── Shared Gemini caller ──────────────────────────────────────────────────────

def _gemini_call_sync(prompt: str, label: str, max_output_tokens: int = 8192):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    if not GOOGLE_AI_KEY:
        return None

    TOTAL_BUDGET = 240
    call_start = _time.time()

    for attempt in range(1, 6):
        if _time.time() - call_start > TOTAL_BUDGET:
            logger.warning(f"[{label}] budget {TOTAL_BUDGET}s exceeded")
            return None
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=max_output_tokens,
                ),
            )
            break
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "free_tier" in err:
                raise RuntimeError("Gemini quota exhausted") from e
            if any(x in err for x in ("503", "UNAVAILABLE", "overloaded", "timeout")) and attempt < 5:
                remaining = TOTAL_BUDGET - (_time.time() - call_start)
                wait = min(15 * attempt, max(remaining - 5, 1))
                _time.sleep(wait)
                continue
            logger.error(f"[{label}] error: {e}")
            return None
    else:
        return None

    raw = ""
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                t = getattr(part, "text", None)
                if t and not getattr(part, "thought", False):
                    raw += t
    except Exception:
        try:
            raw = response.text or ""
        except Exception:
            pass

    if not raw:
        return None

    # Try to parse JSON (array or object)
    clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, (list, dict)):
                return parsed
        except Exception:
            pass
        try:
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
                parsed = json.loads(text)
                if isinstance(parsed, (list, dict)):
                    return parsed
        except Exception:
            pass

    logger.warning(f"[{label}] no JSON found")
    return None


async def _collect(loop, fn_args: tuple, label: str, timeout: int = 180):
    """Run a sync Gemini call in executor, wait up to timeout seconds."""
    fut = loop.run_in_executor(None, _gemini_call_sync, *fn_args)
    elapsed = 0
    while elapsed < timeout:
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=15)
        except asyncio.TimeoutError:
            elapsed += 15
        except Exception as e:
            logger.error(f"[{label}] collect error: {e}")
            return None
    fut.cancel()
    logger.warning(f"[{label}] timed out after {timeout}s")
    return None


# ── DISCOVERY PROMPT ─────────────────────────────────────────────────────────

def _discovery_prompt(industry: str, location: str) -> str:
    loc_clause = f" in {location}" if location else (
        " — focus on top GCC destinations: India, Poland, Philippines, Romania, Malaysia, Mexico, Hungary"
    )
    return f"""You are a GCC (Global Capability Center) research analyst with live Google Search.

TASK: Find companies from the {industry} industry that have established GCCs{loc_clause}.

Run ALL of these searches:
- "{industry}" industry companies GCC "global capability center" {location or "India OR Poland OR Philippines"}
- "{industry}" company captive center offshore {location or "India"} 2022 OR 2023 OR 2024 OR 2025
- "{industry}" "global delivery center" OR "center of excellence" {location or "India OR Philippines"}
- site:nasscom.in OR site:globalcapabilitycenters.com "{industry}"
- "{industry}" GCC headcount employees engineers established

Return a JSON array of up to 50 companies. Each object:
[
  {{
    "company_name": "<parent company name>",
    "gcc_location": "<city, country where GCC is located>",
    "gcc_name": "<official GCC entity name or '-'>",
    "industry": "{industry}",
    "estimated_headcount": "<e.g. 500-1000 or Unknown>",
    "established_year": "<year or Unknown>",
    "operating_model": "<Pure Captive | BOT | GCC-as-a-Service | Unknown>",
    "source": "<URL>"
  }}
]

Return ONLY the JSON array. No prose. No markdown."""


# ── ENRICHMENT PROMPTS (6 sections per company) ───────────────────────────────

def _profile_prompt(company_name: str, gcc_location: str, domain: str) -> str:
    loc = f" — GCC location: {gcc_location}" if gcc_location else ""
    dom_hint = f" (website: {domain})" if domain else ""
    return f"""You are a GCC research analyst with live Google Search.

TARGET: {company_name}{dom_hint} GCC{loc}

Run these searches:
- "{company_name}" GCC OR "global capability center" location established year operating model
- "{company_name}" captive center OR offshore center founded history headcount
- "{company_name}" GCC BOT OR "build operate transfer" OR "GCC-as-a-service"
- site:linkedin.com/company "{company_name}" global capability center
- "{company_name}" GCC India OR Poland OR Philippines employees

Return a JSON object:
{{
  "gcc_name": "<official GCC entity name or '-'>",
  "gcc_locations": "<all known GCC city/country locations, comma-separated>",
  "primary_location": "<main GCC city, country>",
  "established_year": "<year or decade or Unknown>",
  "operating_model": "<Pure Captive | BOT (Build-Operate-Transfer) | GCC-as-a-Service | Hybrid | Unknown>",
  "total_headcount": "<number or range or Unknown>",
  "parent_hq": "<parent company HQ city, country>",
  "gcc_overview": "<2-3 sentence summary of the GCC role and strategic purpose>",
  "source": "<URL>"
}}
Return ONLY the JSON object."""


def _capabilities_prompt(company_name: str, gcc_location: str) -> str:
    loc = f" ({gcc_location})" if gcc_location else ""
    return f"""You are a GCC capability analyst with live Google Search.

TARGET: {company_name} GCC{loc}

Searches:
- "{company_name}" GCC capabilities engineering R&D shared services digital AI analytics
- "{company_name}" global capability center functions teams customer experience finance HR
- site:linkedin.com "{company_name}" GCC technology center capabilities
- "{company_name}" GCC centre of excellence 2023 OR 2024 OR 2025

Return a JSON array — one object per capability area:
[
  {{
    "capability_area": "<Engineering & R&D | Shared Services | Digital Transformation | AI/ML & Data Science | Analytics & BI | Customer Experience | IT Infrastructure & Cloud | Cybersecurity | Product Development | Finance & Accounting | HR Services | Other>",
    "description": "<what this team does at the GCC in 1-2 sentences>",
    "team_size_estimate": "<number or range or Unknown>",
    "maturity_level": "<Nascent | Growing | Mature | Centre of Excellence>",
    "key_functions": "<comma-separated list of specific functions/roles>",
    "source": "<URL>"
  }}
]
Return ONLY the JSON array."""


def _projects_prompt(company_name: str, gcc_location: str) -> str:
    loc = f" ({gcc_location})" if gcc_location else ""
    return f"""You are a GCC project intelligence analyst with live Google Search.

TARGET: {company_name} GCC{loc}

Searches:
- "{company_name}" GCC project initiative technology investment 2023 OR 2024 OR 2025
- "{company_name}" GCC partnership expansion hiring surge announcement
- site:businesswire.com OR site:prnewswire.com "{company_name}" GCC OR "capability center"
- "{company_name}" GCC new investment digital transformation program
- "{company_name}" GCC hiring velocity growth plan

Return a JSON array — one object per project/initiative:
[
  {{
    "project_name": "<specific project or initiative name>",
    "category": "<Technology Investment | Partnership | Expansion | Hiring Surge | Digital Initiative | R&D Program | Automation | Other>",
    "description": "<what the project/initiative involves — be specific>",
    "status": "<Active | Announced | Completed | Planning>",
    "investment_value": "<$ amount if known or Unknown>",
    "partner_vendor": "<partner or vendor name or '-'>",
    "timeline": "<year or date range or Unknown>",
    "source": "<URL>"
  }}
]
Return ONLY the JSON array."""


def _talent_prompt(company_name: str, gcc_location: str) -> str:
    loc = f" ({gcc_location})" if gcc_location else ""
    return f"""You are a GCC talent intelligence analyst with live Google Search.

TARGET: {company_name} GCC{loc}

Searches:
- site:linkedin.com "{company_name}" GCC OR "global capability center" Head VP Director Managing
- "{company_name}" GCC leadership team Head of Center Managing Director CTO CIO
- "{company_name}" GCC key executives decision makers 2024 OR 2025
- "{company_name}" GCC talent hiring trends headcount skill distribution
- "{company_name}" GCC reporting structure parent HQ

Return a JSON array — mix of individual leaders and talent insights:
[
  {{
    "type": "<Leader | Decision Maker | Talent Insight>",
    "name": "<person full name — or '-' for talent insights>",
    "title": "<exact job title>",
    "seniority": "<C-Suite | VP | Director | Senior Manager | Manager | N/A>",
    "function": "<Technology | Finance | HR | Operations | AI/Data | Engineering | Legal | Other>",
    "linkedin_url": "<LinkedIn profile URL or '-'>",
    "reporting_to": "<reports to role/name or '-'>",
    "contact_hint": "<email pattern e.g. firstname.lastname@company.com or '-'>",
    "insight": "<key fact: role scope, team size, or talent trend>"
  }}
]
Return ONLY the JSON array."""


def _financials_prompt(company_name: str, gcc_location: str) -> str:
    loc = f" ({gcc_location})" if gcc_location else ""
    return f"""You are a GCC financial intelligence analyst with live Google Search.

TARGET: {company_name} GCC{loc}

Searches:
- "{company_name}" annual report revenue global turnover 2023 OR 2024
- "{company_name}" GCC budget operating cost investment ROI
- "{company_name}" GCC patents IP proprietary platform intellectual property
- "{company_name}" global capability center cost savings financial contribution
- "{company_name}" GCC R&D investment innovation

Return a JSON object:
{{
  "parent_global_revenue": "<annual revenue with year e.g. '$45B FY2024' or Unknown>",
  "gcc_operational_budget": "<estimated annual GCC budget e.g. '$80-120M' or Unknown>",
  "gcc_revenue_generated": "<revenue or cost savings attributed to GCC or Unknown>",
  "cost_arbitrage": "<estimated cost savings vs onshore e.g. '30-40%' or Unknown>",
  "ip_patents_filed": "<number of patents filed or owned at GCC or Unknown>",
  "proprietary_platforms": "<list of proprietary tools/platforms developed at GCC or '-'>",
  "r_and_d_investment": "<R&D spend figure or Unknown>",
  "financial_notes": "<any other notable financial facts about the GCC>",
  "source": "<URL>"
}}
Return ONLY the JSON object."""


def _techstack_prompt(company_name: str, gcc_location: str) -> str:
    loc = f" ({gcc_location})" if gcc_location else ""
    return f"""You are a GCC technology analyst with live Google Search.

TARGET: {company_name} GCC{loc}

Searches:
- "{company_name}" GCC technology stack cloud AWS Azure GCP DevOps
- "{company_name}" GCC AI ML platform automation tools 2024 OR 2025
- "{company_name}" global capability center modernization legacy cloud migration
- "{company_name}" GCC DevSecOps microservices Kubernetes containers
- site:linkedin.com "{company_name}" GCC engineer cloud devops AI

Return a JSON object:
{{
  "cloud_providers": "<AWS | Azure | GCP | Multi-Cloud | On-Premise | Unknown>",
  "cloud_maturity_score": <integer 0-100 or 0 if unknown>,
  "cloud_migration_maturity": "<Cloud-Native | Hybrid | Migrating | Legacy-Heavy | Unknown>",
  "automation_index": "<Low | Medium | High | Hyper-Automated>",
  "automation_notes": "<brief description of automation/hyper-automation posture>",
  "devops_adoption": "<description of DevOps/CI-CD maturity and key tools>",
  "ai_ml_platforms": "<AI/ML platforms in use — comma-separated or '-'>",
  "tech_vendors": "<key technology vendors — comma-separated>",
  "modern_vs_legacy_split": "<e.g. '70% modern / 30% legacy' or Unknown>",
  "digital_maturity_level": "<Foundational | Developing | Advanced | Leading>",
  "tech_highlights": "<2-3 sentence summary of the GCC's technology posture>",
  "source": "<URL>"
}}
Return ONLY the JSON object."""


# ── DISCOVERY ────────────────────────────────────────────────────────────────

async def run_gcc_discovery(
    industry: str,
    location: str = "",
) -> AsyncGenerator[dict, None]:
    """Discover companies with GCCs in a given industry + location."""
    yield {"type": "heartbeat", "message": f"🔍 Discovering GCCs in {industry}{f' — {location}' if location else ''}…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    result = await _collect(
        loop,
        (_discovery_prompt(industry, location), "gcc_discovery", 8192),
        "gcc_discovery",
        timeout=180,
    )

    companies = result if isinstance(result, list) else []
    for company in companies[:50]:
        if isinstance(company, dict):
            yield {"type": "discovery_company", "company": company}
            await asyncio.sleep(0.05)

    yield {"type": "complete", "total_found": len(companies)}


# ── ENRICHMENT ────────────────────────────────────────────────────────────────

async def _enrich_one(company_name: str, gcc_location: str, domain: str, loop) -> dict:
    """Run all 6 enrichment sections in parallel for one company."""
    sections = [
        ("profile",       _profile_prompt(company_name, gcc_location, domain),      "profile",       6144),
        ("capabilities",  _capabilities_prompt(company_name, gcc_location),          "capabilities",  8192),
        ("projects",      _projects_prompt(company_name, gcc_location),              "projects",      8192),
        ("talent",        _talent_prompt(company_name, gcc_location),                "talent",        8192),
        ("financials",    _financials_prompt(company_name, gcc_location),            "financials",    4096),
        ("techstack",     _techstack_prompt(company_name, gcc_location),             "techstack",     4096),
    ]

    tasks = [
        _collect(loop, (prompt, label, tokens), label, timeout=200)
        for _, prompt, label, tokens in sections
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out = {}
    for (key, _, _, _), res in zip(sections, results):
        if isinstance(res, Exception):
            logger.error(f"[{company_name}] {key} error: {res}")
            out[key] = None
        else:
            out[key] = res
    return out


async def run_gcc_enrichment(
    companies: list[dict],
    max_concurrent: int = 2,
) -> AsyncGenerator[dict, None]:
    """
    Enrich each company with 6 GCC intelligence sections.
    companies: list of {company_name, gcc_location?, domain?}
    """
    if not companies:
        yield {"type": "error", "message": "No companies provided"}
        return

    total = min(len(companies), 50)
    yield {"type": "heartbeat", "message": f"🚀 Enriching {total} compan{'y' if total == 1 else 'ies'} — 6 intel sections each…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(max_concurrent)

    async def _enrich_with_sem(company: dict, idx: int):
        async with sem:
            cname = company.get("company_name") or company.get("name") or "Unknown"
            cloc  = company.get("gcc_location") or company.get("location") or ""
            cdom  = company.get("domain") or ""
            return cname, cloc, await _enrich_one(cname, cloc, cdom, loop)

    tasks = [_enrich_with_sem(c, i) for i, c in enumerate(companies[:50])]

    for coro in asyncio.as_completed(tasks):
        try:
            cname, cloc, sections = await coro
            yield {
                "type": "gcc_enriched",
                "company_name": cname,
                "gcc_location": cloc,
                **sections,
            }
            await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Enrichment task error: {e}")

    yield {"type": "complete", "total_enriched": total}


# ── LEGACY: single-company GCC search (kept for backward compat) ───────────

def _gcc_search_prompt_legacy(company_name: str, domain: str, location: str) -> str:
    loc_filter = f"\nLOCATION FILTER: Only return GCCs in or near: {location}" if location else \
                 "\nCOVERAGE: Return ALL GCC locations worldwide."
    return f"""You are a GCC (Global Capability Centre) research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}
{loc_filter}

Search for all GCC/technology centres/offshore development centres of {company_name}:
- "{company_name}" "Global Capability Centre" location
- "{company_name}" GCC site:linkedin.com
- "{company_name}" captive centre engineering hub India Poland Philippines
- "{company_name}" GCC CEO head director executive

Return ONLY a JSON array — one object per distinct GCC location:
[
  {{
    "company_name": "{company_name}",
    "gcc_name": "<centre name>",
    "location": "<City, Country>",
    "size": "<headcount estimate>",
    "established": "<year>",
    "tech_projects": "<specific named projects with tech stack>",
    "languages": "<languages/frameworks>",
    "cloud": "<cloud/containers>",
    "data_mlops": "<data/mlops tools>",
    "executives": "<Name (Title), Name (Title), Name (Title)>",
    "source": "<URL>"
  }}
]
Return ONLY the raw JSON array."""


async def run_gcc_intelligence(
    company_name: str,
    domain: str = "",
    location: str = "",
    target_vendor: str = "",
    focus_domains=None,
) -> AsyncGenerator[dict, None]:
    """Legacy single-company GCC search (backward compat)."""
    yield {"type": "heartbeat", "message": f"🔍 Searching GCCs for {company_name}…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    prompt = _gcc_search_prompt_legacy(company_name, domain, location)
    result = await _collect(loop, (prompt, "gcc_legacy", 8192), "gcc_legacy", timeout=150)

    rows = result if isinstance(result, list) else []
    for row in rows:
        if isinstance(row, dict):
            row.setdefault("company_name", company_name)
            yield {"type": "gcc_row", "row": row}
            await asyncio.sleep(0.05)

    yield {"type": "complete", "total": len(rows)}
