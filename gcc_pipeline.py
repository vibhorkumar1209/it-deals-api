"""
GCC Intelligence Hub pipeline — v3

Flow:
  Enrichment: per company → discover all GCC locations → enrich each location in parallel
              → stream one gcc_location_result per location as it completes

Discovery: industry + location → find companies with GCCs
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


async def _collect(loop, fn_args: tuple, label: str, timeout: int = 200):
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


# ── LOCATION DISCOVERY (per company) ─────────────────────────────────────────

def _location_discovery_prompt(company_name: str, domain: str, location_filter: str) -> str:
    loc_clause = f"\nLOCATION FILTER: Only return GCC locations in or near: {location_filter}" if location_filter else \
                 "\nCOVERAGE: Return ALL worldwide GCC locations."
    return f"""You are a GCC location researcher with live Google Search.

TARGET COMPANY: {company_name}{f" (website: {domain})" if domain else ""}
{loc_clause}

Run these searches to find every distinct GCC/captive center/technology hub location:
- "{company_name}" GCC OR "global capability center" locations city country
- "{company_name}" technology center hub engineering center India China Poland Philippines
- "{company_name}" captive center offshore development center city
- site:linkedin.com/company "{company_name}" offices locations
- "{company_name}" GCC Bengaluru OR Pune OR Chennai OR Hyderabad OR Mumbai
- "{company_name}" GCC Warsaw OR Krakow OR Bucharest OR Budapest OR Prague OR Lisbon
- "{company_name}" GCC Kuala Lumpur OR Manila OR Singapore OR Shanghai OR Beijing
- "{company_name}" GCC Mexico City OR Guadalajara OR Bogota OR Buenos Aires

Return a JSON array — one object per distinct GCC city location:
[
  {{
    "gcc_name": "<official name of this GCC entity e.g. 'Volkswagen India Technology Center'>",
    "city": "<city>",
    "country": "<country>",
    "gcc_location": "<City, Country>",
    "established_year": "<year or Unknown>",
    "headcount": "<number/range or Unknown>",
    "operating_model": "<Pure Captive | BOT | GCC-as-a-Service | Unknown>",
    "primary_focus": "<brief: e.g. Engineering & R&D, Shared Services, Digital>",
    "source": "<URL>"
  }}
]

Return ONLY the raw JSON array. No prose. No markdown."""


# ── PER-LOCATION ENRICHMENT PROMPTS ──────────────────────────────────────────

def _capabilities_prompt(company_name: str, gcc_location: str) -> str:
    city = gcc_location.split(",")[0].strip()
    return f"""You are a GCC capability analyst with live Google Search.

TARGET: {company_name} GCC in {gcc_location}

Run these searches:
- "{company_name}" {city} GCC capabilities functions teams engineering shared services
- "{company_name}" {city} technology center what does it do R&D digital AI analytics
- site:linkedin.com "{company_name}" {city} technology center capabilities
- "{company_name}" {city} centre of excellence services delivered
- "{company_name}" annual report {city} center function

Return a JSON array — one object per capability area this specific {gcc_location} GCC handles:
[
  {{
    "capability_area": "<Engineering & R&D | Product Development | Shared Services | Digital Transformation | AI/ML & Data Science | Analytics & BI | Customer Experience | IT Infrastructure | Cybersecurity | Finance & Accounting | HR Services | Supply Chain | Other>",
    "description": "<specific description of what this team does at {gcc_location}>",
    "team_size_estimate": "<headcount or range or Unknown>",
    "key_functions": "<comma-separated specific roles/functions>"
  }}
]

Return ONLY the JSON array. No prose. No markdown. Return [] if no data found."""


def _projects_prompt(company_name: str, gcc_location: str) -> str:
    city = gcc_location.split(",")[0].strip()
    return f"""You are a GCC projects analyst with live Google Search.

TARGET: {company_name} GCC in {gcc_location}

Search broadly — include press releases, LinkedIn announcements, job postings, news:
- "{company_name}" {city} GCC project initiative announcement 2023 OR 2024 OR 2025
- "{company_name}" {city} technology center expansion investment hiring
- site:linkedin.com "{company_name}" {city} technology project announcement
- site:businesswire.com OR site:prnewswire.com "{company_name}" {city}
- "{company_name}" {city} GCC new hiring 1000 OR 500 OR 2000 employees expansion
- "{company_name}" {city} center partnership technology vendor deal

Return a JSON array — one per project, initiative, expansion, or key announcement:
[
  {{
    "project_name": "<specific project/initiative name or announcement>",
    "category": "<Technology Investment | Expansion | Hiring Surge | Digital Initiative | Partnership | R&D Program | Automation | Infrastructure>",
    "description": "<what this project/initiative involves — be specific>",
    "status": "<Active | Announced | Completed | Planning>",
    "investment_value": "<$ amount if mentioned or Unknown>",
    "partner_vendor": "<partner/vendor name or '-'>",
    "timeline": "<year or date range>",
    "hiring_signal": "<number of jobs being added if mentioned or '-'>",
    "source": "<URL>"
  }}
]

IMPORTANT: If no formal project announcements found, search for:
- Any plans to expand headcount or facilities in {city}
- Any technology vendor partnerships signed for {gcc_location}
- Any digital transformation programs running at this center
Return [] only if truly nothing found after all searches."""


def _talent_prompt(company_name: str, gcc_location: str) -> str:
    city = gcc_location.split(",")[0].strip()
    return f"""You are a GCC talent intelligence analyst with live Google Search.

TARGET: {company_name} GCC in {gcc_location}

Run these searches:
- site:linkedin.com "{company_name}" {city} Head OR VP OR Director OR "Managing Director" GCC OR technology
- "{company_name}" {city} GCC head center director managing technology leader
- "{company_name}" {city} technology center leadership team executives
- site:linkedin.com/in "{company_name}" {city} chief technology officer VP engineering
- "{company_name}" {city} GCC hiring talent engineers headcount skill

Return a JSON array with leaders AND talent insights:
[
  {{
    "type": "<Leader | Talent Insight>",
    "name": "<full name or '-' for talent insights>",
    "title": "<exact job title>",
    "seniority": "<C-Suite | VP | Director | Senior Manager | N/A>",
    "function": "<Technology | Engineering | Finance | HR | Operations | AI/Data | Legal | Other>",
    "linkedin_url": "<LinkedIn profile URL or '-'>",
    "reporting_to": "<reports to role or name or '-'>",
    "contact_hint": "<email pattern e.g. firstname.lastname@company.com or '-'>",
    "insight": "<key fact about this person's scope OR a talent/hiring trend at {gcc_location}>"
  }}
]

Return ONLY the JSON array. No prose. No markdown."""


def _financials_prompt(company_name: str, gcc_location: str) -> str:
    city = gcc_location.split(",")[0].strip()
    return f"""You are a GCC financial analyst with live Google Search.

TARGET: {company_name} GCC in {gcc_location}

Run these searches:
- "{company_name}" annual report 2023 OR 2024 revenue turnover global
- "{company_name}" {city} center budget investment cost 2023 OR 2024
- "{company_name}" {city} GCC IP patents intellectual property innovation
- "{company_name}" {city} cost savings offshore arbitrage technology investment
- "{company_name}" {city} proprietary platform product developed

Return a JSON object with available data (use "Unknown" only if search yields nothing, use benchmarks with a note if actual data unavailable):
{{
  "parent_global_revenue": "<annual revenue with year e.g. '$45B FY2024' or Unknown>",
  "gcc_operational_budget": "<estimated annual budget for {gcc_location} GCC e.g. '$40-80M' or Unknown>",
  "gcc_cost_to_parent": "<cost savings or value generated or Unknown>",
  "cost_arbitrage_estimate": "<% savings vs onshore e.g. '30-40%' — use industry benchmark if not found>",
  "ip_patents_at_location": "<patents filed from {city} or Unknown>",
  "proprietary_platforms": "<tools/platforms built at {gcc_location} or '-'>",
  "r_and_d_investment": "<R&D spend for this location or Unknown>",
  "financial_notes": "<any other financial facts about this GCC>",
  "source": "<URL>"
}}
Return ONLY the JSON object."""


def _techstack_prompt(company_name: str, gcc_location: str) -> str:
    city = gcc_location.split(",")[0].strip()
    return f"""You are a GCC tech stack analyst with live Google Search.

TARGET: {company_name} GCC in {gcc_location}

Search job postings and tech announcements — job postings reveal actual tech stack:
- site:linkedin.com/jobs "{company_name}" {city} engineer developer requirements AWS OR Azure OR GCP
- "{company_name}" {city} technology center cloud DevOps AI automation stack 2024 OR 2025
- site:glassdoor.com "{company_name}" {city} technology interview tech stack
- "{company_name}" {city} GCC technology tools platform modernization
- "{company_name}" {city} job openings skills required python java react kubernetes

Return a JSON object:
{{
  "cloud_providers": "<AWS | Azure | GCP | Multi-Cloud | On-Premise | Hybrid — found in job postings or news>",
  "cloud_maturity_score": <integer 0-100 based on cloud adoption signals, 0 if truly unknown>,
  "cloud_migration_maturity": "<Cloud-Native | Hybrid | Migrating | Legacy-Heavy | Unknown>",
  "automation_index": "<Hyper-Automated | High | Medium | Low — based on DevOps/RPA/AI signals>",
  "automation_notes": "<specific tools: Jenkins, GitHub Actions, Ansible, RPA, etc.>",
  "devops_adoption": "<CI/CD maturity and key tools found in job postings>",
  "programming_languages": "<languages from job postings: e.g. Java, Python, TypeScript, Go>",
  "frameworks_tools": "<frameworks/tools: e.g. Spring Boot, React, Kubernetes, Terraform>",
  "ai_ml_platforms": "<AI/ML tools found: TensorFlow, PyTorch, Azure ML, SageMaker, etc. or '-'>",
  "tech_vendors": "<key ISV partners/vendors: SAP, Salesforce, ServiceNow, etc.>",
  "modern_vs_legacy_split": "<e.g. '70% modern / 30% legacy' — estimate from signals>",
  "digital_maturity_level": "<Foundational | Developing | Advanced | Leading>",
  "tech_highlights": "<2-3 sentences on the GCC's tech posture based on findings>",
  "source": "<URL>"
}}
Return ONLY the JSON object."""


# ── DISCOVERY PROMPT ──────────────────────────────────────────────────────────

def _discovery_prompt(industry: str, location: str) -> str:
    loc_clause = f" in {location}" if location else \
                 " — focus on: India, Poland, Philippines, Romania, Malaysia, Mexico, Hungary"
    return f"""You are a GCC research analyst with live Google Search.

TASK: Find companies from the {industry} industry that have established GCCs{loc_clause}.

Run ALL these searches:
- "{industry}" companies GCC "global capability center" {location or "India OR Poland OR Philippines"}
- "{industry}" company captive center offshore 2022 OR 2023 OR 2024 OR 2025
- "{industry}" "global delivery center" OR "center of excellence" {location or "India"}
- site:nasscom.in OR site:globalcapabilitycenters.com "{industry}"
- "{industry}" GCC headcount employees established

Return a JSON array of up to 50 companies:
[
  {{
    "company_name": "<parent company name>",
    "gcc_location": "<primary GCC city, country>",
    "gcc_name": "<official GCC entity name or '-'>",
    "industry": "{industry}",
    "estimated_headcount": "<e.g. 500-1000 or Unknown>",
    "established_year": "<year or Unknown>",
    "operating_model": "<Pure Captive | BOT | GCC-as-a-Service | Unknown>",
    "source": "<URL>"
  }}
]
Return ONLY the JSON array. No prose. No markdown."""


# ── DISCOVERY ────────────────────────────────────────────────────────────────

async def run_gcc_discovery(
    industry: str,
    location: str = "",
) -> AsyncGenerator[dict, None]:
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


# ── ENRICHMENT (per company → per location) ───────────────────────────────────

async def _enrich_one_location(
    company_name: str,
    location_info: dict,
    domain: str,
    loop,
) -> dict:
    """Run 4 parallel enrichment calls for one GCC location."""
    gcc_location = location_info.get("gcc_location", "")
    label = f"{company_name[:20]}@{gcc_location[:15]}"

    caps_task  = _collect(loop, (_capabilities_prompt(company_name, gcc_location), f"cap_{label}",  6144), f"cap_{label}",  180)
    proj_task  = _collect(loop, (_projects_prompt(company_name, gcc_location),     f"proj_{label}", 6144), f"proj_{label}", 180)
    tal_task   = _collect(loop, (_talent_prompt(company_name, gcc_location),        f"tal_{label}",  6144), f"tal_{label}",  180)
    fin_task   = _collect(loop, (_financials_prompt(company_name, gcc_location),    f"fin_{label}",  4096), f"fin_{label}",  180)
    tech_task  = _collect(loop, (_techstack_prompt(company_name, gcc_location),     f"tech_{label}", 4096), f"tech_{label}", 180)

    caps, projs, talent, fin, tech = await asyncio.gather(
        caps_task, proj_task, tal_task, fin_task, tech_task,
        return_exceptions=True,
    )

    def safe(v):
        return None if isinstance(v, Exception) else v

    return {
        "company_name":  company_name,
        "gcc_location":  gcc_location,
        "gcc_name":      location_info.get("gcc_name", "-"),
        "established_year": location_info.get("established_year", "Unknown"),
        "headcount":     location_info.get("headcount", "Unknown"),
        "operating_model": location_info.get("operating_model", "Unknown"),
        "primary_focus": location_info.get("primary_focus", "-"),
        "capabilities":  safe(caps) if isinstance(safe(caps), list) else [],
        "projects":      safe(projs) if isinstance(safe(projs), list) else [],
        "talent":        safe(talent) if isinstance(safe(talent), list) else [],
        "financials":    safe(fin) if isinstance(safe(fin), dict) else {},
        "techstack":     safe(tech) if isinstance(safe(tech), dict) else {},
    }


async def run_gcc_enrichment(
    companies: list[dict],
    max_concurrent: int = 2,
) -> AsyncGenerator[dict, None]:
    """
    For each company:
      1. Discover all GCC locations
      2. Enrich each location in parallel (4 calls per location)
      3. Stream each gcc_location_result as it completes
    """
    if not companies:
        yield {"type": "error", "message": "No companies provided"}
        return

    total_cos = min(len(companies), 50)
    yield {"type": "heartbeat", "message": f"🚀 Starting GCC enrichment for {total_cos} compan{'y' if total_cos == 1 else 'ies'}…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    company_sem = asyncio.Semaphore(max_concurrent)

    async def process_company(company: dict):
        async with company_sem:
            cname  = company.get("company_name") or company.get("name") or "Unknown"
            domain = company.get("domain", "")
            loc_filter = company.get("gcc_location") or company.get("location") or ""

            yield {"type": "heartbeat", "message": f"🔍 Finding GCC locations for {cname}…"}
            await asyncio.sleep(0)

            # Step 1: discover locations for this company
            loc_result = await _collect(
                loop,
                (_location_discovery_prompt(cname, domain, loc_filter), f"locs_{cname[:20]}", 6144),
                f"locs_{cname[:20]}",
                timeout=150,
            )
            locations = loc_result if isinstance(loc_result, list) else []

            # Fallback: use provided location if discovery returned nothing
            if not locations:
                fallback_loc = loc_filter or f"India"
                locations = [{"gcc_name": f"{cname} GCC", "gcc_location": fallback_loc, "established_year": "Unknown", "headcount": "Unknown", "operating_model": "Unknown", "primary_focus": "-"}]
                yield {"type": "heartbeat", "message": f"⚠️ {cname}: no locations found via search, using {fallback_loc}"}
            else:
                loc_names = ", ".join(l.get("gcc_location", "") for l in locations[:5])
                yield {"type": "heartbeat", "message": f"📍 {cname}: {len(locations)} location{'s' if len(locations) > 1 else ''} found — {loc_names}"}

            await asyncio.sleep(0)

            # Step 2: enrich each location (all in parallel, capped at 4 concurrent per company)
            loc_sem = asyncio.Semaphore(4)

            async def enrich_loc(loc_info):
                async with loc_sem:
                    return await _enrich_one_location(cname, loc_info, domain, loop)

            loc_tasks = [enrich_loc(loc) for loc in locations[:10]]
            for coro in asyncio.as_completed(loc_tasks):
                try:
                    result = await coro
                    yield {"type": "gcc_location_result", "result": result}
                    await asyncio.sleep(0.05)
                except Exception as e:
                    logger.error(f"Location enrichment error: {e}")

    # Process companies sequentially (semaphore inside handles concurrency)
    for company in companies[:50]:
        async for event in process_company(company):
            yield event

    yield {"type": "complete", "total_companies": total_cos}


# ── LEGACY ────────────────────────────────────────────────────────────────────

def _gcc_search_prompt_legacy(company_name: str, domain: str, location: str) -> str:
    loc_filter = f"\nLOCATION FILTER: Only return GCCs in or near: {location}" if location else \
                 "\nCOVERAGE: Return ALL GCC locations worldwide."
    return f"""You are a GCC research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}
{loc_filter}

Search for all GCC/technology centres of {company_name}:
- "{company_name}" "Global Capability Centre" location
- "{company_name}" GCC site:linkedin.com
- "{company_name}" captive centre engineering hub India Poland Philippines

Return ONLY a JSON array — one object per distinct GCC location:
[{{
  "company_name": "{company_name}",
  "gcc_name": "<centre name>",
  "location": "<City, Country>",
  "size": "<headcount estimate>",
  "established": "<year>",
  "tech_projects": "<specific named projects>",
  "languages": "<languages/frameworks>",
  "cloud": "<cloud/containers>",
  "data_mlops": "<data/mlops tools>",
  "executives": "<Name (Title), Name (Title)>",
  "source": "<URL>"
}}]
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
    result = await _collect(
        loop,
        (_gcc_search_prompt_legacy(company_name, domain, location), "gcc_legacy", 8192),
        "gcc_legacy",
        timeout=150,
    )

    rows = result if isinstance(result, list) else []
    for row in rows:
        if isinstance(row, dict):
            row.setdefault("company_name", company_name)
            yield {"type": "gcc_row", "row": row}
            await asyncio.sleep(0.05)

    yield {"type": "complete", "total": len(rows)}
