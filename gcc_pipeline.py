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
        logger.error(f"[{label}] google-genai not installed")
        return None

    if not GOOGLE_AI_KEY:
        logger.error(f"[{label}] GOOGLE_AI_API_KEY not set")
        return None

    TOTAL_BUDGET = 200
    call_start = _time.time()

    cfg_extra = {}
    try:
        cfg_extra["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception:
        pass

    for attempt in range(1, 5):
        elapsed_so_far = _time.time() - call_start
        if elapsed_so_far > TOTAL_BUDGET:
            logger.warning(f"[{label}] budget {TOTAL_BUDGET}s exceeded after {elapsed_so_far:.0f}s")
            return None
        try:
            # NOTE: Do NOT pass http_options — the SDK interprets timeout as milliseconds,
            # causing calls to fail in <1s. Let the SDK use its default (no timeout).
            client = genai.Client(api_key=GOOGLE_AI_KEY)
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
            retryable = any(x in err for x in (
                "503", "UNAVAILABLE", "overloaded", "timeout", "TimeoutError",
                "DeadlineExceeded", "429", "500", "502", "504", "Read timed out",
            ))
            if retryable and attempt < 4:
                remaining = TOTAL_BUDGET - (_time.time() - call_start)
                wait = min(12 * attempt, max(remaining - 5, 1))
                logger.warning(f"[{label}] attempt {attempt} failed ({err[:80]}), retry in {wait:.0f}s")
                _time.sleep(wait)
                continue
            logger.error(f"[{label}] non-retryable error: {err[:150]}")
            return None
    else:
        logger.warning(f"[{label}] all 4 attempts failed")
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
        logger.warning(f"[{label}] empty response text")
        return None

    clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

    # Try parsing cleaned text directly first
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, (list, dict)):
            return parsed
    except Exception:
        pass

    # Then try extracting JSON array or object via regex
    for pattern in [r"\[[\s\S]*?\]", r"\{[\s\S]*?\}"]:
        try:
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
                parsed = json.loads(text)
                if isinstance(parsed, (list, dict)):
                    return parsed
        except Exception:
            pass

    # Last resort: greedy match for largest JSON block
    for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
        try:
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
                parsed = json.loads(text)
                if isinstance(parsed, (list, dict)):
                    return parsed
        except Exception:
            pass

    logger.warning(f"[{label}] no valid JSON found in response ({len(raw)} chars)")
    return None


async def _collect(loop, fn_args: tuple, label: str, timeout: int = 200):
    """Run _gemini_call_sync in executor with hard timeout. No asyncio.shield."""
    fut = loop.run_in_executor(None, _gemini_call_sync, *fn_args)
    try:
        result = await asyncio.wait_for(fut, timeout=timeout)
        return result
    except asyncio.TimeoutError:
        logger.warning(f"[{label}] timed out after {timeout}s")
        return None
    except Exception as e:
        logger.error(f"[{label}] collect error: {e}")
        return None


# ── LOCATION DISCOVERY (per company) ─────────────────────────────────────────

def _location_discovery_prompt(company_name: str, domain: str, location_filter: str) -> str:
    loc_clause = f"LOCATION FILTER: Only include locations in or near: {location_filter}" if location_filter else \
                 "SCOPE: Return ALL worldwide locations — cover every continent equally. Do NOT bias toward any single region."
    domain_hint = f" (official website: {domain})" if domain else ""
    return f"""Find every Global Capability Center (GCC), technology hub, shared services center, and offshore delivery center that {company_name} operates worldwide.{domain_hint}

{loc_clause}

CRITICAL INSTRUCTION: You MUST run ALL searches below before answering. Many companies have GCCs outside India — in the Americas, Europe, and Southeast Asia. Do not default to India-only results.

Run EVERY one of these 14 searches:
1. "{company_name}" "global capability center" OR "GCC" locations worldwide full list 2024 2025
2. "{company_name}" "technology center" OR "tech hub" OR "engineering center" OR "development center" global locations
3. "{company_name}" "shared services" OR "global business services" OR "GBS" OR "captive center" locations worldwide
4. site:linkedin.com/company "{company_name}" offices locations worldwide — check the Locations tab
5. "{company_name}" global offices employees headcount 2024 site:linkedin.com
6. "{company_name}" annual report 2024 OR 2025 global offices technology hubs locations operations
7. "{company_name}" United States OR Canada OR Mexico OR "Costa Rica" OR Brazil OR Colombia OR Chile technology center hub
8. "{company_name}" United Kingdom OR Ireland OR Poland OR Romania OR Hungary OR Portugal OR Spain OR Germany OR "Czech Republic" technology center
9. "{company_name}" India OR "Bengaluru" OR "Hyderabad" OR "Chennai" OR "Pune" OR "Gurugram" OR "Mumbai" engineering center
10. "{company_name}" Philippines OR Malaysia OR Singapore OR Vietnam OR Indonesia OR Thailand hub operations center
11. "{company_name}" China OR Shanghai OR Beijing OR "Hong Kong" OR Taiwan technology center
12. "{company_name}" Egypt OR Morocco OR "South Africa" OR Kenya OR UAE OR Israel technology center
13. site:nasscom.in OR site:globalcapabilitycenters.com OR site:zinnov.com OR site:everestgrp.com "{company_name}" GCC
14. "{company_name}" "center of excellence" OR "CoE" OR "innovation hub" OR "delivery center" location city country

RULES:
- Include EVERY city where {company_name} has any GCC/tech/ops center regardless of size
- List centers in ALL regions found — Americas, Europe, Asia-Pacific, MEA — not just India
- If a company has multiple centers in the same country, list each city separately
- Headcount: use the most recent figure; if unknown write "Unknown"

Return a JSON array. Each element must have exactly these fields:
[
  {{
    "gcc_name": "<official center name, e.g. 'Walmart Global Tech, Bengaluru'>",
    "gcc_location": "<City, Country>",
    "city": "<city only>",
    "country": "<country only>",
    "headcount": "<number or range e.g. '13,000' or '2,000–3,000' or 'Unknown'>",
    "established_year": "<4-digit year or 'Unknown'>",
    "operating_model": "<Pure Captive | BOT | GCC-as-a-Service | Unknown>",
    "primary_focus": "<Engineering & R&D | Shared Services | Digital Transformation | AI/ML & Data | Customer Experience | Mixed>",
    "source": "<URL of press release, LinkedIn, annual report, or news article>"
  }}
]
Return ONLY the raw JSON array. No explanation. No markdown. No prose before or after."""


def _location_discovery_prompt_simple(company_name: str) -> str:
    """Simpler fallback discovery prompt — used on retry when full prompt fails to parse."""
    return f"""Search for all GCC, technology hub, and shared services center locations of {company_name} worldwide.

Run ALL of these searches — do not skip any:
1. "{company_name}" "global capability center" OR "technology center" OR "tech hub" locations worldwide list
2. "{company_name}" offices global locations employees 2024 site:linkedin.com
3. "{company_name}" United States OR Canada OR Mexico OR "Costa Rica" technology center hub
4. "{company_name}" India OR Poland OR Philippines OR Malaysia technology center 2024
5. "{company_name}" annual report 2024 global offices locations
6. "{company_name}" GCC OR "shared services" OR "engineering center" OR "delivery center" city country

IMPORTANT: Include ALL regions — Americas, Europe, Asia-Pacific, and MEA. Do not focus only on India.

Return a JSON array — one entry per city:
[{{"gcc_name": "<name>", "gcc_location": "<City, Country>", "city": "<city>", "country": "<country>", "headcount": "<number or Unknown>", "established_year": "<year or Unknown>", "operating_model": "Unknown", "primary_focus": "Mixed", "source": "<URL>"}}]
Return ONLY the JSON array. No text outside the array."""


# ── PER-LOCATION ENRICHMENT PROMPTS ──────────────────────────────────────────

def _capabilities_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    return f"""What functions does {company_name}'s GCC in {gcc_location} handle?{f" (center: {gcc_name})" if gcc_name and gcc_name != '-' else ""}

Provide a breakdown across: software engineering and R&D, shared services (finance, HR, legal), digital transformation, AI/ML and data analytics, and customer experience operations. Include the primary mandate of the center.

Run ALL of these searches before responding:
1. "{company_name}" "{city}" GCC capabilities OR functions OR services — what does this center do?
2. site:linkedin.com "{company_name}" "{city}" engineer OR developer OR analyst OR manager — what roles exist?
3. site:linkedin.com/jobs "{company_name}" {city} — scan job titles to infer teams and functions
4. "{company_name}" {city} operations finance HR shared services legal digital teams
5. "{entity}" capabilities mandate primary focus engineering AI data analytics
6. "{company_name}" {country} center engineering digital AI shared services customer experience
7. "{company_name}" {city} center of excellence OR CoE OR R&D OR product development

From searches, identify ALL functions/teams at {gcc_location}. Infer from job titles, org charts, press releases, and company descriptions.

Return a JSON array — one object per capability area at {gcc_location}:
[
  {{
    "capability_area": "<Engineering & R&D | Product Development | Shared Services | Digital Transformation | AI/ML & Data Science | Analytics & BI | Customer Experience | IT Infrastructure & Cloud | Cybersecurity | Finance & Accounting | HR Services | Supply Chain | Legal & Compliance | Other>",
    "description": "<concrete description of what this team does at {gcc_location} — cite evidence>",
    "team_size_estimate": "<headcount/role count or Unknown>",
    "key_functions": "<specific job titles or sub-functions found in searches>"
  }}
]
Return ONLY the JSON array. No prose. No markdown."""


def _projects_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else gcc_location.strip()
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    return f"""What are the active projects, technology investments, strategic partnerships, and expansion plans at {company_name}'s center in {gcc_location} as of 2024–2026?{f" (center: {gcc_name})" if gcc_name and gcc_name != '-' else ""}

Include facility openings, headcount expansions, new technology programs, vendor partnerships, AI/cloud/digital initiatives, CoE launches, and LinkedIn hiring signals.

Run ALL of these searches — do not skip any:
1. "{company_name}" "{city}" expansion OR investment OR initiative OR hiring announcement 2024 OR 2025 OR 2026
2. "{company_name}" "{country}" expansion OR "center of excellence" OR CoE OR facility announcement 2024 OR 2025
3. site:businesswire.com OR site:prnewswire.com OR site:globenewswire.com "{company_name}" "{city}" OR "{country}" 2024 OR 2025
4. "{company_name}" "{city}" OR "{country}" AI OR "artificial intelligence" OR cloud OR "digital transformation" program 2024 OR 2025
5. "{company_name}" "{city}" OR "{country}" new office OR campus OR facility OR "square feet" 2024 OR 2025
6. "{company_name}" "{city}" OR "{country}" hiring engineers OR developers OR architects 2025 jobs
7. site:linkedin.com/jobs "{company_name}" "{city}" 2025 — scan job titles for new initiative signals
8. "{company_name}" "{country}" Microsoft OR AWS OR Azure OR "Google Cloud" OR SAP OR ServiceNow partnership 2024 OR 2025
9. "{entity}" technology investment OR digital program OR innovation lab announcement 2024 OR 2025
10. "{company_name}" global technology OR engineering strategy 2024 OR 2025 — any global programs that this center participates in

IMPORTANT: If {city}-specific results are sparse, broaden to {country}-wide and global company initiatives. Any project where {gcc_location} plays a role counts.

Count as a project/initiative: headcount expansions, facility openings, technology programs, vendor partnerships, digital/AI/cloud initiatives, CoE launches, R&D programs, automation rollouts.

Return a JSON array — one object per distinct project, initiative, or announcement:
[
  {{
    "project_name": "<specific name or short descriptive title>",
    "category": "<Headcount Expansion | New Facility | Technology Investment | Digital Initiative | AI/ML Program | Partnership | R&D Program | Automation | Centre of Excellence | Other>",
    "description": "<concrete description — include numbers, dates, technologies, impact>",
    "status": "<Active | Announced | Completed | Planning>",
    "investment_value": "<$ amount or headcount target or Unknown>",
    "partner_vendor": "<partner/vendor if mentioned or '-'>",
    "timeline": "<year or date range>",
    "hiring_signal": "<new roles being added e.g. '2,000 engineers by 2025' or '-'>",
    "source": "<URL of press release, news article, or LinkedIn post>"
  }}
]
Return ONLY the JSON array. If genuinely nothing found after all searches, return []."""


def _talent_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    entity = gcc_name if gcc_name and gcc_name != "-" else company_name
    email_domain = company_name.lower().replace(" ", "") + ".com"
    return f"""Who are the key leaders at {company_name}'s GCC in {gcc_location}?{f" (center: {gcc_name})" if gcc_name and gcc_name != '-' else ""}

Provide names and titles for: GCC Head / MD, VPs, and functional leaders. Include their LinkedIn profiles if available, reporting lines to parent HQ, current headcount, dominant skill sets, and top open roles being hired for in 2025–2026.

Run ALL of these searches:
1. site:linkedin.com/in "{company_name}" "{city}" "Managing Director" OR "Head of" OR "VP" OR "Vice President" OR "Country Head" OR "Site Lead" OR "Center Head"
2. site:linkedin.com/in "{company_name}" "{city}" "CTO" OR "CFO" OR "COO" OR "Director" engineering OR technology OR digital
3. "{company_name}" {city} managing director OR GCC head OR country head appointed OR named 2023 OR 2024 OR 2025
4. "{entity}" leadership team management {city} executive
5. "{company_name}" {city} executive appointment announcement 2024 OR 2025
6. site:linkedin.com/jobs "{company_name}" {city} 2025 — what are the top roles being hired now?
7. "{company_name}" {city} talent hiring engineers data scientists dominant skills 2025
8. "{company_name}" {country} headcount total employees GCC 2024 OR 2025

Return a JSON array — named leaders first, then talent/hiring insights:
[
  {{
    "type": "<Leader | Talent Insight>",
    "name": "<full name or '-' for talent insights>",
    "title": "<exact job title from LinkedIn or press release>",
    "seniority": "<C-Suite | VP | Director | Senior Manager | N/A>",
    "function": "<Technology | Engineering | Finance | HR | Operations | AI/Data | Legal | Sales | Other>",
    "linkedin_url": "<full LinkedIn profile URL or '-'>",
    "reporting_to": "<reports to HQ role/name or '-'>",
    "contact_hint": "<likely email e.g. firstname.lastname@{email_domain} or '-'>",
    "insight": "<scope of role, team size, or key talent/hiring trend for {gcc_location}>"
  }}
]
Return ONLY the JSON array. No prose. No markdown."""


def _financials_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else ""
    return f"""What is the estimated operational budget and revenue contribution of {company_name}'s GCC in {gcc_location}?

Include: parent company global annual revenue, any disclosed GCC-specific financials, IP or patents developed at the center, and proprietary platforms built there.

Run ALL of these searches:
1. "{company_name}" annual revenue OR turnover 2024 OR 2025 — find verified parent company global revenue
2. site:annualreports.com OR site:ir.*.com "{company_name}" annual report 2024 revenue earnings
3. "{company_name}" {country} investment OR budget OR technology spend 2024 OR 2025
4. "{company_name}" {city} expansion investment "$" million OR billion announcement
5. "{company_name}" intellectual property OR patent OR trademark {country} OR {city}
6. "{company_name}" {country} R&D investment OR research and development spend
7. "{company_name}" {city} proprietary platform OR product OR technology developed built
8. "{company_name}" cost arbitrage OR offshore savings OR labor cost {country}
9. "{company_name}" GCC revenue contribution OR value delivered OR savings reported

CRITICAL: Parent global revenue is ALWAYS findable for public companies — search investor relations and financial news.

For GCC operational budget: if no specific figure, estimate from headcount benchmarks:
- Small GCC (500–1,000 staff): $20–50M/year
- Mid GCC (1,000–5,000 staff): $50–200M/year
- Large GCC (5,000+): $200M+/year
Flag estimated figures as "(estimated)".

Return a JSON object:
{{
  "parent_global_revenue": "<verified annual revenue with FY year e.g. '€45.3B FY2024'>",
  "gcc_operational_budget": "<annual GCC budget e.g. '$40–80M (estimated)' or actual>",
  "gcc_cost_to_parent": "<cost savings or value attributed to this GCC or Unknown>",
  "cost_arbitrage_estimate": "<% labor cost savings vs onshore e.g. '60–70% vs US rates'>",
  "ip_patents_at_location": "<number of patents or IP assets at {city} or Unknown>",
  "proprietary_platforms": "<specific tools/platforms/products built at this center or '-'>",
  "r_and_d_investment": "<R&D spend for this region or globally or Unknown>",
  "financial_notes": "<any other financial insight: contracts won, cost savings reported, investment announcements>",
  "source": "<URL of annual report, press release, or financial news>"
}}
Return ONLY the JSON object."""


def _techstack_prompt(company_name: str, gcc_location: str, gcc_name: str = "") -> str:
    city = gcc_location.split(",")[0].strip()
    country = gcc_location.split(",")[-1].strip() if "," in gcc_location else gcc_location.strip()
    return f"""What is the technology stack and digital maturity of {company_name}'s center in {gcc_location}?

Cover: cloud providers and migration status, automation and RPA/hyper-automation, DevOps/DevSecOps maturity, key technology vendors, programming languages and frameworks in active use, and estimated modern-cloud vs legacy split.

Run ALL of these searches — do not skip any:
1. site:linkedin.com/jobs "{company_name}" "{city}" software engineer OR developer OR architect — extract required tech skills from job descriptions
2. site:linkedin.com/jobs "{company_name}" "{country}" engineer developer data scientist — broaden to country if city sparse
3. site:glassdoor.com "{company_name}" "{city}" OR "{country}" engineer interview tech stack
4. site:indeed.com OR site:monster.com "{company_name}" "{city}" software engineer developer 2025
5. "{company_name}" "{city}" OR "{country}" AWS OR Azure OR "Google Cloud" OR GCP cloud infrastructure migration 2024 OR 2025
6. "{company_name}" "{city}" OR "{country}" Kubernetes OR Docker OR microservices OR DevOps OR DevSecOps OR CI/CD
7. "{company_name}" "{city}" OR "{country}" RPA OR automation OR UiPath OR "Automation Anywhere" OR "Power Automate"
8. "{company_name}" engineering blog OR tech blog OR open source GitHub technology stack 2024 2025
9. github.com "{company_name}" repositories — check public repos for primary languages and frameworks
10. "{company_name}" global SAP OR Salesforce OR ServiceNow OR Oracle OR Microsoft OR Workday — enterprise vendors
11. "{company_name}" "{city}" OR "{country}" AI OR "machine learning" OR LLM OR "generative AI" technology platform 2024 OR 2025
12. "{company_name}" technology strategy 2024 2025 — global tech investments relevant to {gcc_location}

NOTE: Job postings on LinkedIn/Indeed/Glassdoor are the BEST signal for tech stack — search those first. Also use company engineering blogs and GitHub.

Extract from job postings: required programming languages, preferred tools, frameworks, cloud platforms, and certifications mentioned.

Return a JSON object:
{{
  "cloud_providers": "<AWS | Azure | GCP | Multi-Cloud | On-Premise | Hybrid — cite evidence source>",
  "cloud_maturity_score": <integer 0-100: 80+=Cloud-Native, 60-79=Hybrid, 40-59=Migrating, <40=Legacy-Heavy>,
  "cloud_migration_maturity": "<Cloud-Native | Hybrid | Migrating | Legacy-Heavy | Unknown>",
  "automation_index": "<Hyper-Automated | High | Medium | Low>",
  "devops_tools": "<specific CI/CD and DevOps tools: e.g. Jenkins, GitHub Actions, GitLab CI, Terraform, Ansible>",
  "programming_languages": "<from job postings — comma separated: e.g. Java, Python, TypeScript, Go, C++>",
  "frameworks_tools": "<e.g. Spring Boot, React, Node.js, Kubernetes, Kafka, Spark>",
  "ai_ml_platforms": "<AI/ML stack: TensorFlow, PyTorch, Azure ML, SageMaker, LangChain, or '-'>",
  "enterprise_vendors": "<SAP, Salesforce, Oracle, ServiceNow, Microsoft, Workday etc.>",
  "modern_vs_legacy_split": "<estimate e.g. '70% modern cloud / 30% legacy' — based on job posting evidence>",
  "digital_maturity_level": "<Foundational | Developing | Advanced | Leading>",
  "tech_highlights": "<2-3 sentence summary citing specific tools and sources found>",
  "source": "<URL of job posting, tech blog, GitHub, or news article used>"
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

def _fallback_capabilities(company_name: str, gcc_location: str) -> list:
    """Return a minimal placeholder capabilities list so the column is never blank."""
    return [{"capability_area": "Engineering & R&D", "description": f"{company_name} GCC in {gcc_location} — data unavailable, search did not return results.", "team_size_estimate": "Unknown", "key_functions": "Requires manual verification"}]

def _fallback_projects(company_name: str, gcc_location: str) -> list:
    return [{"project_name": "No public projects found", "category": "Other", "description": f"No announced projects found for {company_name} GCC in {gcc_location} at this time.", "status": "Unknown", "investment_value": "Unknown", "partner_vendor": "-", "timeline": "-", "hiring_signal": "-", "source": "-"}]

def _fallback_talent(company_name: str, gcc_location: str) -> list:
    return [{"type": "Talent Insight", "name": "-", "title": "GCC Leadership", "seniority": "N/A", "function": "Technology", "linkedin_url": "-", "reporting_to": "-", "contact_hint": "-", "insight": f"No publicly listed leaders found for {company_name} GCC in {gcc_location}. Check LinkedIn directly."}]

def _fallback_financials(company_name: str, gcc_location: str) -> dict:
    return {"parent_global_revenue": "Not found — check investor relations", "gcc_operational_budget": "Estimated based on headcount benchmarks", "gcc_cost_to_parent": "Unknown", "cost_arbitrage_estimate": "60–70% vs onshore (industry average)", "ip_patents_at_location": "Unknown", "proprietary_platforms": "-", "r_and_d_investment": "Unknown", "financial_notes": f"No disclosed financials found for {company_name} GCC in {gcc_location}.", "source": "-"}

def _fallback_techstack(company_name: str, gcc_location: str) -> dict:
    return {"cloud_providers": "Unknown — check job postings", "cloud_maturity_score": 0, "cloud_migration_maturity": "Unknown", "automation_index": "Unknown", "devops_tools": "-", "programming_languages": "-", "frameworks_tools": "-", "ai_ml_platforms": "-", "enterprise_vendors": "-", "modern_vs_legacy_split": "Unknown", "digital_maturity_level": "Unknown", "tech_highlights": f"No tech stack data found for {company_name} GCC in {gcc_location}. No public job postings or tech blog entries available.", "source": "-"}


async def _enrich_one_location(
    company_name: str,
    location_info: dict,
    domain: str,
    loop,
) -> dict:
    """Run 5 parallel enrichment calls for one GCC location. Retry empty sections once."""
    gcc_location = location_info.get("gcc_location", "")
    gcc_name     = location_info.get("gcc_name", "")
    label = f"{company_name[:20]}@{gcc_location[:15]}"

    caps_task  = _collect(loop, (_capabilities_prompt(company_name, gcc_location, gcc_name), f"cap_{label}",  6144), f"cap_{label}",  160)
    proj_task  = _collect(loop, (_projects_prompt(company_name, gcc_location, gcc_name),     f"proj_{label}", 6144), f"proj_{label}", 160)
    tal_task   = _collect(loop, (_talent_prompt(company_name, gcc_location, gcc_name),        f"tal_{label}",  6144), f"tal_{label}",  160)
    fin_task   = _collect(loop, (_financials_prompt(company_name, gcc_location, gcc_name),    f"fin_{label}",  4096), f"fin_{label}",  160)
    tech_task  = _collect(loop, (_techstack_prompt(company_name, gcc_location, gcc_name),     f"tech_{label}", 4096), f"tech_{label}", 160)

    caps, projs, talent, fin, tech = await asyncio.gather(
        caps_task, proj_task, tal_task, fin_task, tech_task,
        return_exceptions=True,
    )

    def safe(v):
        return None if isinstance(v, Exception) else v

    caps_v  = safe(caps)  if isinstance(safe(caps), list)  and len(safe(caps) or []) > 0  else None
    projs_v = safe(projs) if isinstance(safe(projs), list) and len(safe(projs) or []) > 0 else None
    tal_v   = safe(talent) if isinstance(safe(talent), list) and len(safe(talent) or []) > 0 else None
    fin_v   = safe(fin)   if isinstance(safe(fin), dict)   and len(safe(fin) or {}) > 0   else None
    tech_v  = safe(tech)  if isinstance(safe(tech), dict)  and len(safe(tech) or {}) > 0  else None

    # Retry any sections that returned empty — one additional attempt each
    retry_tasks = {}
    if caps_v  is None: retry_tasks["caps"]  = _collect(loop, (_capabilities_prompt(company_name, gcc_location, gcc_name), f"cap2_{label}",  6144), f"cap2_{label}",  120)
    if projs_v is None: retry_tasks["projs"] = _collect(loop, (_projects_prompt(company_name, gcc_location, gcc_name),     f"proj2_{label}", 6144), f"proj2_{label}", 120)
    if tal_v   is None: retry_tasks["tal"]   = _collect(loop, (_talent_prompt(company_name, gcc_location, gcc_name),        f"tal2_{label}",  6144), f"tal2_{label}",  120)
    if fin_v   is None: retry_tasks["fin"]   = _collect(loop, (_financials_prompt(company_name, gcc_location, gcc_name),    f"fin2_{label}",  4096), f"fin2_{label}",  120)
    if tech_v  is None: retry_tasks["tech"]  = _collect(loop, (_techstack_prompt(company_name, gcc_location, gcc_name),     f"tech2_{label}", 4096), f"tech2_{label}", 120)

    if retry_tasks:
        logger.info(f"[{label}] retrying empty sections: {list(retry_tasks.keys())}")
        retry_results = await asyncio.gather(*retry_tasks.values(), return_exceptions=True)
        retry_map = dict(zip(retry_tasks.keys(), retry_results))

        def from_retry(key, check_type):
            v = safe(retry_map.get(key))
            if check_type is list:
                return v if isinstance(v, list) and len(v) > 0 else None
            return v if isinstance(v, dict) and len(v) > 0 else None

        if caps_v  is None: caps_v  = from_retry("caps",  list)
        if projs_v is None: projs_v = from_retry("projs", list)
        if tal_v   is None: tal_v   = from_retry("tal",   list)
        if fin_v   is None: fin_v   = from_retry("fin",   dict)
        if tech_v  is None: tech_v  = from_retry("tech",  dict)

    # Use fallback stubs for any still-empty sections so no column is blank
    if caps_v  is None: caps_v  = _fallback_capabilities(company_name, gcc_location)
    if projs_v is None: projs_v = _fallback_projects(company_name, gcc_location)
    if tal_v   is None: tal_v   = _fallback_talent(company_name, gcc_location)
    if fin_v   is None: fin_v   = _fallback_financials(company_name, gcc_location)
    if tech_v  is None: tech_v  = _fallback_techstack(company_name, gcc_location)

    return {
        "company_name":     company_name,
        "gcc_location":     gcc_location,
        "gcc_name":         location_info.get("gcc_name", "-"),
        "established_year": location_info.get("established_year", "Unknown"),
        "headcount":        location_info.get("headcount", "Unknown"),
        "operating_model":  location_info.get("operating_model", "Unknown"),
        "primary_focus":    location_info.get("primary_focus", "-"),
        "capabilities":     caps_v,
        "projects":         projs_v,
        "talent":           tal_v,
        "financials":       fin_v,
        "techstack":        tech_v,
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

            # Step 1: discover locations for this company — full prompt first
            loc_result = await _collect(
                loop,
                (_location_discovery_prompt(cname, domain, loc_filter), f"locs_{cname[:20]}", 8192),
                f"locs_{cname[:20]}",
                timeout=160,
            )
            locations = loc_result if isinstance(loc_result, list) and len(loc_result) > 0 else []

            # Retry with simpler prompt if first attempt failed or returned empty
            if not locations:
                yield {"type": "heartbeat", "message": f"🔄 {cname}: retrying location discovery with simplified prompt…"}
                await asyncio.sleep(0)
                loc_result2 = await _collect(
                    loop,
                    (_location_discovery_prompt_simple(cname), f"locs2_{cname[:20]}", 4096),
                    f"locs2_{cname[:20]}",
                    timeout=120,
                )
                locations = loc_result2 if isinstance(loc_result2, list) and len(loc_result2) > 0 else []

            if not locations:
                # Last resort: synthesize entries for the most common worldwide GCC hubs
                if loc_filter:
                    fallback_locs = [loc_filter]
                else:
                    # Balanced global fallback — Americas + Europe + Asia (not India-only)
                    fallback_locs = [
                        "Bengaluru, India", "Hyderabad, India",
                        "Warsaw, Poland", "Mexico City, Mexico",
                        "Kuala Lumpur, Malaysia", "Manila, Philippines",
                    ]
                locations = [
                    {"gcc_name": f"{cname} GCC", "gcc_location": fl, "city": fl.split(",")[0].strip(),
                     "country": fl.split(",")[-1].strip() if "," in fl else fl,
                     "established_year": "Unknown", "headcount": "Unknown",
                     "operating_model": "Unknown", "primary_focus": "-"}
                    for fl in fallback_locs
                ]
                yield {"type": "heartbeat", "message": f"⚠️ {cname}: location search returned no results — enriching {len(locations)} probable hub{'s' if len(locations)>1 else ''}"}
            else:
                loc_names = ", ".join(l.get("gcc_location", "") for l in locations[:6])
                extra = f" (+{len(locations)-6} more)" if len(locations) > 6 else ""
                yield {"type": "heartbeat", "message": f"📍 {cname}: {len(locations)} location{'s' if len(locations) > 1 else ''} found — {loc_names}{extra}"}

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
