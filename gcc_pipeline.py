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

    CALL_TIMEOUT = 90    # per-call HTTP timeout — prevents indefinite hangs
    TOTAL_BUDGET = 220
    call_start = _time.time()

    # Only set thinking_config for 2.5 models — 2.0-flash doesn't support it
    # and passing it causes an API error that returns None silently.
    cfg_extra = {}
    if "2.5" in "gemini-2.5-flash":  # gcc always uses 2.5
        try:
            cfg_extra["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass

    for attempt in range(1, 5):
        if _time.time() - call_start > TOTAL_BUDGET:
            logger.warning(f"[{label}] budget {TOTAL_BUDGET}s exceeded")
            return None
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY, http_options={"timeout": CALL_TIMEOUT})
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=max_output_tokens,
                    **cfg_extra,
                ),
            )
            break
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err or "free_tier" in err:
                raise RuntimeError("Gemini quota exhausted") from e
            if any(x in err for x in ("503", "UNAVAILABLE", "overloaded", "timeout", "TimeoutError", "DeadlineExceeded", "429", "500", "502", "504")) and attempt < 4:
                remaining = TOTAL_BUDGET - (_time.time() - call_start)
                wait = min(15 * attempt, max(remaining - 5, 1))
                logger.warning(f"[{label}] attempt {attempt} failed ({err[:60]}), retry in {wait:.0f}s")
                _time.sleep(wait)
                continue
            logger.error(f"[{label}] error: {err[:120]}")
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
    loc_clause = f"\nLOCATION FILTER: Only return GCC/center locations in or near: {location_filter}" if location_filter else \
                 "\nCOVERAGE: Return ALL worldwide GCC/center locations."
    return f"""You are a GCC location researcher with live Google Search.

TARGET COMPANY: {company_name}{f" (website: {domain})" if domain else ""}
{loc_clause}

Run ALL of these searches to find every distinct center/captive/offshore hub location:
- "{company_name}" "global capability center" OR "technology center" OR "captive center" location city
- "{company_name}" engineering center OR development center OR shared services center city country
- "{company_name}" India center Bengaluru OR Pune OR Chennai OR Hyderabad OR Mumbai OR Noida OR Gurugram
- "{company_name}" Poland OR Romania OR Hungary OR Czech Republic OR Portugal technology center
- "{company_name}" Philippines OR Malaysia OR Singapore OR Vietnam technology hub center
- "{company_name}" Mexico OR Brazil OR Colombia OR Argentina OR Costa Rica technology center
- "{company_name}" China OR Shanghai OR Shenzhen OR Beijing technology center
- site:linkedin.com/company "{company_name}" offices engineering center locations headcount
- "{company_name}" GCC established employees headcount site:nasscom.in OR site:globalcapabilitycenters.com

KEY RULE: Return ONE entry per distinct GCC/center ENTITY. If a company has an Engineering Center AND a Shared Services Center in the same city, return BOTH as separate entries. Use the official entity name to distinguish them.

Return a JSON array — one object per distinct GCC entity:
[
  {{
    "gcc_name": "<official entity name e.g. 'Volkswagen India Technology Center, Pune' — be specific>",
    "city": "<city>",
    "country": "<country>",
    "gcc_location": "<City, Country>",
    "established_year": "<year or Unknown>",
    "headcount": "<number/range or Unknown>",
    "operating_model": "<Pure Captive | BOT | GCC-as-a-Service | Unknown>",
    "primary_focus": "<what this specific entity does: e.g. Engineering & R&D, Finance Shared Services, Digital Transformation>",
    "source": "<URL>"
  }}
]

Return ONLY the raw JSON array. No prose. No markdown."""


# ── PER-LOCATION ENRICHMENT PROMPTS ──────────────────────────────────────────

def _capabilities_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    return f"""You are a GCC capability analyst with live Google Search.

TARGET: {company_name} — center in {gcc_location}{f' (entity: {gcc_name})' if gcc_name and gcc_name != '-' else ''}

You MUST run ALL of these searches before responding:
1. site:linkedin.com "{company_name}" "{city}" engineer OR developer OR analyst — what roles exist here?
2. "{company_name}" "{city}" technology center capabilities services functions
3. "{company_name}" {city} operations finance HR shared services teams
4. "{entity}" capabilities OR functions OR services OR teams
5. "{company_name}" {country} center engineering digital AI shared services what does it do
6. site:linkedin.com/jobs "{company_name}" {city} — scan job titles to infer what teams exist

From these searches, identify what functions/teams operate at this specific {gcc_location} location.
Infer from job titles, press releases, LinkedIn posts, and company descriptions.

Return a JSON array — one object per capability area identified at {gcc_location}:
[
  {{
    "capability_area": "<Engineering & R&D | Product Development | Shared Services | Digital Transformation | AI/ML & Data Science | Analytics & BI | Customer Experience | IT Infrastructure & Cloud | Cybersecurity | Finance & Accounting | HR Services | Supply Chain | Legal & Compliance | Other>",
    "description": "<what this specific team does at {gcc_location} — be concrete, cite evidence>",
    "team_size_estimate": "<headcount or role count or Unknown>",
    "key_functions": "<comma-separated specific job titles or functions found in searches>"
  }}
]

Return ONLY the JSON array. No prose. No markdown."""


def _projects_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    return f"""You are a GCC projects and initiatives analyst with live Google Search.

TARGET: {company_name} — center in {gcc_location}{f' (entity: {gcc_name})' if gcc_name and gcc_name != '-' else ''}

Run ALL of these searches — do not skip any:
1. "{company_name}" {city} expansion investment hiring announcement 2023 OR 2024 OR 2025
2. "{company_name}" {country} center project initiative technology investment 2024
3. site:businesswire.com OR site:prnewswire.com "{company_name}" {city} OR {country}
4. "{company_name}" {city} jobs hiring 500 OR 1000 OR 2000 OR 3000 employees
5. "{company_name}" {country} digital transformation AI automation program 2024 OR 2025
6. "{entity}" project initiative expansion partnership announcement
7. "{company_name}" {city} new office facility campus expansion
8. "{company_name}" {country} technology vendor SAP OR Salesforce OR Microsoft OR AWS partnership

IMPORTANT: Any of these count as a "project/initiative":
- Headcount expansion announcements (e.g., "will hire 5000 in India by 2025")
- New facility or campus openings
- Technology transformation programs
- Vendor partnerships or platform deployments
- Digital/AI/cloud initiatives
- Center of Excellence launches

Return a JSON array — one per project, initiative, expansion, or announcement found:
[
  {{
    "project_name": "<specific name or short title of the announcement/initiative>",
    "category": "<Headcount Expansion | New Facility | Technology Investment | Digital Initiative | Partnership | R&D Program | Automation | Centre of Excellence | Other>",
    "description": "<concrete description — include numbers, dates, technologies if found>",
    "status": "<Active | Announced | Completed | Planning>",
    "investment_value": "<$ amount or headcount target or Unknown>",
    "partner_vendor": "<partner/vendor if mentioned or '-'>",
    "timeline": "<year or date range>",
    "hiring_signal": "<jobs being added e.g. '2,000 engineers by 2025' or '-'>",
    "source": "<URL of press release, news article, or LinkedIn post>"
  }}
]

Return ONLY the JSON array. If genuinely nothing found after all searches, return []."""


def _talent_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    # Derive likely email domain from company name
    email_domain = company_name.lower().replace(" ", "") + ".com"
    return f"""You are a GCC talent intelligence analyst with live Google Search.

TARGET: {company_name} — center in {gcc_location}{f' (entity: {gcc_name})' if gcc_name and gcc_name != '-' else ''}

Run ALL of these LinkedIn and web searches:
1. site:linkedin.com/in "{company_name}" "{city}" "Head of" OR "Managing Director" OR "VP" OR "Vice President"
2. site:linkedin.com/in "{company_name}" "{city}" "Country Head" OR "Site Lead" OR "Center Head" OR "CTO" OR "CFO"
3. site:linkedin.com/in "{company_name}" "{country}" Director OR VP engineering OR technology OR digital
4. "{company_name}" {city} managing director OR head of center OR country head name
5. "{entity}" leadership team management {city}
6. "{company_name}" {city} executive appointment announcement 2023 OR 2024 OR 2025
7. "{company_name}" {city} talent hiring engineers data scientists skill shortage trend

For each leader found, search: site:linkedin.com/in "<their name>" "{company_name}" to get their profile URL.

Return a JSON array — mix of named leaders AND talent/hiring insights:
[
  {{
    "type": "<Leader | Talent Insight>",
    "name": "<full name — or '-' for talent insights>",
    "title": "<exact job title from LinkedIn or press release>",
    "seniority": "<C-Suite | VP | Director | Senior Manager | N/A>",
    "function": "<Technology | Engineering | Finance | HR | Operations | AI/Data | Legal | Sales | Other>",
    "linkedin_url": "<full LinkedIn profile URL or '-'>",
    "reporting_to": "<reports to role/name or '-'>",
    "contact_hint": "<likely email e.g. firstname.lastname@{email_domain} or '-'>",
    "insight": "<scope of role, team size managed, or a key talent/hiring trend for {gcc_location}>"
  }}
]

Return ONLY the JSON array. No prose. No markdown."""


def _financials_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    return f"""You are a GCC financial intelligence analyst with live Google Search.

TARGET: {company_name} — center in {gcc_location}

Run ALL of these searches:
1. "{company_name}" annual revenue OR turnover 2023 OR 2024 — find parent company global revenue
2. "{company_name}" {country} investment OR budget OR spend 2023 OR 2024 OR 2025
3. "{company_name}" {city} expansion investment "$" million OR billion announcement
4. "{company_name}" intellectual property OR patent OR trademark {country} OR {city}
5. "{company_name}" {country} R&D investment OR research spend
6. "{company_name}" {city} proprietary platform OR product OR technology developed
7. site:annualreports.com OR site:ir.{company_name.lower().replace(" ","")}.com annual report revenue
8. "{company_name}" cost arbitrage OR offshore savings India OR {country}

CRITICAL: Parent company global revenue is ALWAYS findable for public companies.
Search investor relations pages and financial news. Never leave parent_global_revenue as Unknown
if the company is publicly traded.

For GCC budget: if no specific figure found, estimate using industry benchmarks:
- Small GCC (500-1000 staff): $20-50M/year
- Mid GCC (1000-5000 staff): $50-200M/year  
- Large GCC (5000+): $200M+/year
Flag as "(estimated)" if using benchmarks.

Return a JSON object:
{{
  "parent_global_revenue": "<annual revenue with FY year e.g. '€45.3B FY2024' — REQUIRED for public companies>",
  "gcc_operational_budget": "<annual GCC budget e.g. '$40-80M (estimated)' or actual if found>",
  "gcc_cost_to_parent": "<cost savings or value attributed to this GCC or Unknown>",
  "cost_arbitrage_estimate": "<% labor cost savings vs onshore — use benchmark if not found e.g. '60-70% vs US rates'>",
  "ip_patents_at_location": "<number of patents or IP assets at {city} or Unknown>",
  "proprietary_platforms": "<specific tools/platforms/products built at this center or '-'>",
  "r_and_d_investment": "<R&D spend globally or for this region or Unknown>",
  "financial_notes": "<any other financial insight: contracts won, cost savings reported, investment announcements>",
  "source": "<URL of annual report, press release, or financial news>"
}}
Return ONLY the JSON object."""


def _techstack_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    return f"""You are a GCC technology stack analyst with live Google Search.

TARGET: {company_name} — center in {gcc_location}

Job postings are the BEST source for tech stack. Run ALL of these searches:
1. site:linkedin.com/jobs "{company_name}" {city} software engineer developer — read job requirements
2. site:naukri.com "{company_name}" {city} engineer developer — for India locations
3. site:glassdoor.com "{company_name}" {city} engineer interview — tech stack mentions
4. "{company_name}" {city} AWS OR Azure OR "Google Cloud" cloud infrastructure
5. "{company_name}" {city} kubernetes OR docker OR microservices OR DevOps
6. "{company_name}" {city} python OR java OR javascript OR golang developer jobs
7. "{company_name}" technology blog OR tech stack OR engineering blog — look for official tech blogs
8. github.com "{company_name}" — check public repos for language usage
9. "{company_name}" {country} SAP OR Salesforce OR ServiceNow OR Oracle OR Microsoft deployment
10. "{company_name}" {city} AI machine learning deep learning LLM 2024 OR 2025

From job postings, extract: required skills, preferred tools, frameworks listed in JD requirements.

Return a JSON object:
{{
  "cloud_providers": "<AWS | Azure | GCP | Multi-Cloud | On-Premise | Hybrid — cite evidence>",
  "cloud_maturity_score": <integer 0-100: 80+=Cloud-Native, 60-79=Hybrid, 40-59=Migrating, <40=Legacy-Heavy, 0 only if truly no signal>,
  "cloud_migration_maturity": "<Cloud-Native | Hybrid | Migrating | Legacy-Heavy | Unknown>",
  "automation_index": "<Hyper-Automated | High | Medium | Low>",
  "devops_tools": "<specific CI/CD and DevOps tools found: e.g. Jenkins, GitHub Actions, GitLab CI, Terraform, Ansible>",
  "programming_languages": "<languages from job postings — comma separated: e.g. Java, Python, TypeScript, Go, C++>",
  "frameworks_tools": "<frameworks and tools: e.g. Spring Boot, React, Node.js, Kubernetes, Kafka, Spark>",
  "ai_ml_platforms": "<AI/ML stack: TensorFlow, PyTorch, Azure ML, SageMaker, LangChain, or '-'>",
  "enterprise_vendors": "<SAP, Salesforce, Oracle, ServiceNow, Microsoft, Workday etc. in use>",
  "modern_vs_legacy_split": "<estimate e.g. '70% modern cloud / 30% legacy' — base on job posting signals>",
  "digital_maturity_level": "<Foundational | Developing | Advanced | Leading>",
  "tech_highlights": "<2-3 sentence summary of the tech posture — cite specific tools and evidence found>",
  "source": "<URL of job posting, tech blog, or news article used>"
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
    gcc_name     = location_info.get("gcc_name", "")
    label = f"{company_name[:20]}@{gcc_location[:15]}"

    caps_task  = _collect(loop, (_capabilities_prompt(company_name, gcc_location, gcc_name), f"cap_{label}",  6144), f"cap_{label}",  180)
    proj_task  = _collect(loop, (_projects_prompt(company_name, gcc_location, gcc_name),     f"proj_{label}", 6144), f"proj_{label}", 180)
    tal_task   = _collect(loop, (_talent_prompt(company_name, gcc_location, gcc_name),        f"tal_{label}",  6144), f"tal_{label}",  180)
    fin_task   = _collect(loop, (_financials_prompt(company_name, gcc_location, gcc_name),    f"fin_{label}",  4096), f"fin_{label}",  180)
    tech_task  = _collect(loop, (_techstack_prompt(company_name, gcc_location, gcc_name),     f"tech_{label}", 4096), f"tech_{label}", 180)

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
