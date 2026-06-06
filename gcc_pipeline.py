"""
GCC Intelligence Hub pipeline — Two-phase Gemini orchestration.

Phase 1 (Fact-Finding): Gemini + Google Search → raw facts per domain
Phase 2 (Synthesis):    Gemini grounded on Phase 1 facts → structured JSON
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

# ── Domain taxonomy for manufacturing / automotive companies ──────────────────
AFTERMARKET_DOMAINS = [
    "Warranty Management",
    "Service Operations & Field Service",
    "Quality Management",
    "Knowledge Management & Technical Documentation",
    "Parts & Spare Parts Management",
    "Dealer Management System (DMS)",
    "Supply Chain & Procurement",
    "Manufacturing Execution & IoT",
    "Engineering & PLM",
    "Customer Experience & CRM",
    "Finance & ERP",
    "HR & Workforce Management",
    "Data Foundation & Analytics",
    "AI & Automation Platform",
    "Cybersecurity & Compliance",
]

# ── Output schemas ────────────────────────────────────────────────────────────

TECH_STACK_FIELDS = [
    {"key": "domain",           "label": "Domain"},
    {"key": "layer",            "label": "Layer"},          # e.g. Application | Data | Infrastructure
    {"key": "tool_vendor",      "label": "Tool / Vendor"},
    {"key": "current_status",   "label": "Status"},          # Active | Legacy | Evaluating | Planned
    {"key": "notes",            "label": "Notes"},
    {"key": "source",           "label": "Source"},
]

VENDOR_SIGNAL_FIELDS = [
    {"key": "domain",                "label": "Domain"},
    {"key": "signal_strength",       "label": "Signal"},      # High | Medium | Low | None
    {"key": "opportunity_type",      "label": "Opportunity"},  # Displacement | Expansion | Greenfield | Partnership
    {"key": "existing_competitor",   "label": "Incumbent"},
    {"key": "readiness_score",       "label": "Score"},        # 0-100
    {"key": "rationale",             "label": "Rationale"},
    {"key": "source",                "label": "Source"},
]

IT_BUDGET_FIELDS = [
    {"key": "domain",            "label": "Domain"},
    {"key": "estimated_budget",  "label": "Est. Budget (USD)"},
    {"key": "budget_basis",      "label": "Basis"},
    {"key": "source",            "label": "Source"},
]


# ── Phase 1: Fact-Finding ─────────────────────────────────────────────────────

def _phase1_prompt(company_name: str, domain: str, domain_str: str) -> str:
    return f"""You are a deep enterprise IT research analyst with live Google Search.

TARGET COMPANY: {company_name}
RESEARCH DOMAIN: {domain}

Run targeted searches to extract verified facts about {company_name}'s technology tools,
vendors, and initiatives in the domain of "{domain}".

Search queries to execute:
- "{company_name}" {domain} software system vendor platform
- "{company_name}" {domain} technology solution implementation 2022 OR 2023 OR 2024 OR 2025
- "{company_name}" {domain_str} tool platform deployed partner
- site:businesswire.com OR site:prnewswire.com "{company_name}" {domain}
- "{company_name}" annual report {domain} technology spend

Extract ALL of:
1. Specific software products and vendors used in this domain
2. Whether they are active/legacy/being evaluated
3. Any announced implementations, upgrades, or replacements
4. Any public budget figures or project cost mentions related to this domain
5. Source URLs for every fact

Return a JSON object:
{{
  "domain": "{domain}",
  "facts": [
    {{
      "tool_vendor": "<exact product or vendor name>",
      "layer": "<Application | Data | Infrastructure | AI/Analytics | Integration>",
      "current_status": "<Active | Legacy | Evaluating | Planned | Replaced>",
      "notes": "<brief note on how it is used>",
      "source": "<URL>"
    }}
  ],
  "budget_signals": [
    {{
      "amount": "<e.g. $45 million>",
      "context": "<what this spend covers>",
      "source": "<URL>"
    }}
  ]
}}

Return ONLY the raw JSON object. No prose. No markdown.
"""


def _run_phase1_sync(company_name: str, domain: str) -> dict:
    """Single domain fact-finding call. Runs in thread pool."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"domain": domain, "facts": [], "budget_signals": []}

    if not GOOGLE_AI_KEY:
        return {"domain": domain, "facts": [], "budget_signals": []}

    # Shorten domain for search query
    domain_str = domain.split("(")[0].strip().replace(" & ", " ")

    prompt = _phase1_prompt(company_name, domain, domain_str)

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=4096,
                ),
            )
            break
        except Exception as e:
            err = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in err or "free_tier" in err
            is_retry = not is_quota and any(x in err for x in ("503", "UNAVAILABLE", "overloaded", "timeout"))
            if is_quota:
                raise RuntimeError("Gemini quota exhausted — upgrade to paid API plan.") from e
            if is_retry and attempt < MAX_RETRIES:
                _time.sleep(10 * attempt)
                continue
            logger.error(f"Phase1 error for {company_name}/{domain}: {e}")
            return {"domain": domain, "facts": [], "budget_signals": []}
    else:
        return {"domain": domain, "facts": [], "budget_signals": []}

    # Extract text from parts
    raw = ""
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                t = getattr(part, "text", None)
                if t:
                    raw += t
    except Exception:
        try:
            raw = response.text or ""
        except Exception:
            pass

    if not raw:
        return {"domain": domain, "facts": [], "budget_signals": []}

    # Parse JSON
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try to extract inner JSON object
    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass

    return {"domain": domain, "facts": [], "budget_signals": []}


# ── Phase 2: Synthesis ────────────────────────────────────────────────────────

def _phase2_prompt_tech_stack(company_name: str, phase1_data: list[dict]) -> str:
    facts_json = json.dumps(phase1_data, indent=2)
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in TECH_STACK_FIELDS)
    return f"""You are an enterprise IT analyst synthesising research findings.

COMPANY: {company_name}

PHASE 1 RESEARCH DATA (verified facts only — do not add information not in this data):
{facts_json}

TASK: From the research data above, produce a comprehensive tech stack table.
Extract every tool and vendor mentioned into structured rows.

Return ONLY a JSON array of objects with EXACTLY these keys:
[
  {{
{fields_desc}
  }}
]

RULES:
- domain: use the exact domain name from the research data
- layer: Application | Data | Infrastructure | AI/Analytics | Integration | Security
- tool_vendor: exact product or vendor name
- current_status: Active | Legacy | Evaluating | Planned | Replaced
- notes: brief description of use case
- source: URL from research data — use "-" only if genuinely not found

Return ONLY the raw JSON array. No prose. No markdown.
"""


def _phase2_prompt_vendor_signals(
    company_name: str, target_vendor: str, phase1_data: list[dict]
) -> str:
    facts_json = json.dumps(phase1_data, indent=2)
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in VENDOR_SIGNAL_FIELDS)
    return f"""You are a vendor readiness analyst assessing go-to-market opportunities.

TARGET COMPANY: {company_name}
TARGET VENDOR: {target_vendor}

PHASE 1 RESEARCH DATA:
{facts_json}

TASK: For each domain, assess {target_vendor}'s readiness signal and opportunity
at {company_name} based ONLY on the research data provided.

Readiness signal logic:
- High: {target_vendor} is already present, or a direct competitor is present and
  {target_vendor} has a strong displacement case
- Medium: Adjacent technology is present; {target_vendor} has relevant capability
- Low: Domain uses entrenched incumbents with low churn likelihood
- None: No signal found in research data

Return ONLY a JSON array:
[
  {{
{fields_desc}
  }}
]

RULES:
- domain: exact domain name
- signal_strength: High | Medium | Low | None
- opportunity_type: Displacement | Expansion | Greenfield | Partnership | Upsell
- existing_competitor: current incumbent tool/vendor in this domain, or "-"
- readiness_score: integer 0-100 reflecting signal strength + opportunity size
- rationale: 1-2 sentences grounded only in the research facts above
- source: URL from research data supporting the assessment, or "-"

Return ONLY the raw JSON array. No prose. No markdown.
"""


def _phase2_prompt_budget(company_name: str, phase1_data: list[dict]) -> str:
    facts_json = json.dumps(phase1_data, indent=2)
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in IT_BUDGET_FIELDS)
    return f"""You are an enterprise IT financial analyst.

COMPANY: {company_name}

PHASE 1 RESEARCH DATA:
{facts_json}

TASK: For each domain, extract or estimate IT budget spend based ONLY on
the budget signals in the research data. Use annual reports, press releases,
or contract values mentioned in the data.

Return ONLY a JSON array:
[
  {{
{fields_desc}
  }}
]

RULES:
- domain: exact domain name
- estimated_budget: e.g. "$15–25 million" or "-" if no signal found
- budget_basis: e.g. "Annual report FY2024", "Contract announcement", "Industry benchmark" or "-"
- source: URL from research data or "-"

Return ONLY the raw JSON array. No prose. No markdown.
"""


def _run_phase2_sync(prompt: str, label: str) -> list[dict]:
    """Phase 2 synthesis (no search grounding needed — uses facts from Phase 1)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return []

    if not GOOGLE_AI_KEY:
        return []

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            break
        except Exception as e:
            err = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in err or "free_tier" in err
            is_retry = not is_quota and any(x in err for x in ("503", "UNAVAILABLE", "overloaded"))
            if is_quota:
                raise RuntimeError("Gemini quota exhausted.") from e
            if is_retry and attempt < MAX_RETRIES:
                _time.sleep(10 * attempt)
                continue
            logger.error(f"Phase2 error for {label}: {e}")
            return []
    else:
        return []

    raw = ""
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                t = getattr(part, "text", None)
                if t:
                    raw += t
    except Exception:
        try:
            raw = response.text or ""
        except Exception:
            pass

    if not raw:
        return []

    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
    except Exception:
        pass

    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
            return json.loads(text)
    except Exception:
        pass

    return []


# ── Main async generator ──────────────────────────────────────────────────────

async def run_gcc_intelligence(
    company_name: str,
    domain: str,
    target_vendor: str = "",
    focus_domains: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Two-phase GCC Intelligence pipeline.
    Yields: heartbeat | tech_stack_row | vendor_signal_row | budget_row | complete
    """
    domains = focus_domains or AFTERMARKET_DOMAINS
    yield {"type": "heartbeat", "message": f"🔍 Starting GCC Intel for {company_name} across {len(domains)} domains…"}
    await asyncio.sleep(0)

    # ── Phase 1: parallel fact-finding per domain ─────────────────────────────
    yield {"type": "heartbeat", "message": f"🌐 Phase 1: Live research across {len(domains)} domains…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()
    phase1_results: list[dict] = []
    DOMAIN_TIMEOUT = 90

    for i, dom in enumerate(domains):
        yield {"type": "heartbeat", "message": f"🔎 [{i+1}/{len(domains)}] Researching: {dom}…"}
        await asyncio.sleep(0)

        future = loop.run_in_executor(None, _run_phase1_sync, company_name, dom)
        elapsed = 0
        result = {"domain": dom, "facts": [], "budget_signals": []}

        while elapsed < DOMAIN_TIMEOUT:
            try:
                result = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat", "message": f"🌐 Researching {dom}… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Phase1 domain error {dom}: {e}", exc_info=True)
                yield {"type": "heartbeat", "message": f"⚠️ {dom}: {e}"}
                break
        else:
            future.cancel()
            yield {"type": "heartbeat", "message": f"⏱ {dom} timed out — skipping"}

        facts_count = len(result.get("facts", []))
        yield {"type": "heartbeat", "message": f"✅ {dom}: {facts_count} tools found"}
        phase1_results.append(result)
        await asyncio.sleep(0)

    total_facts = sum(len(r.get("facts", [])) for r in phase1_results)
    yield {"type": "heartbeat",
           "message": f"✅ Phase 1 complete — {total_facts} facts across {len(domains)} domains"}
    await asyncio.sleep(0)

    # ── Phase 2a: Tech Stack synthesis ────────────────────────────────────────
    yield {"type": "heartbeat", "message": "🧠 Phase 2: Synthesising tech stack…"}
    await asyncio.sleep(0)

    p2_tech_prompt = _phase2_prompt_tech_stack(company_name, phase1_results)
    tech_future = loop.run_in_executor(None, _run_phase2_sync, p2_tech_prompt, "tech_stack")
    elapsed = 0
    tech_rows: list[dict] = []
    while elapsed < 120:
        try:
            tech_rows = await asyncio.wait_for(asyncio.shield(tech_future), timeout=10)
            break
        except asyncio.TimeoutError:
            elapsed += 10
            yield {"type": "heartbeat", "message": f"🧠 Synthesising tech stack… ({elapsed}s)"}
            await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Phase2 tech stack error: {e}")
            break
    else:
        tech_future.cancel()

    for row in tech_rows:
        yield {"type": "tech_stack_row", "row": row}
        await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Tech stack: {len(tech_rows)} tools mapped"}
    await asyncio.sleep(0)

    # ── Phase 2b: IT Budget synthesis ─────────────────────────────────────────
    yield {"type": "heartbeat", "message": "💰 Phase 2: Estimating IT budgets…"}
    await asyncio.sleep(0)

    p2_budget_prompt = _phase2_prompt_budget(company_name, phase1_results)
    budget_future = loop.run_in_executor(None, _run_phase2_sync, p2_budget_prompt, "budget")
    elapsed = 0
    budget_rows: list[dict] = []
    while elapsed < 90:
        try:
            budget_rows = await asyncio.wait_for(asyncio.shield(budget_future), timeout=10)
            break
        except asyncio.TimeoutError:
            elapsed += 10
            yield {"type": "heartbeat", "message": f"💰 Estimating budgets… ({elapsed}s)"}
            await asyncio.sleep(0)
        except Exception as e:
            logger.error(f"Phase2 budget error: {e}")
            break
    else:
        budget_future.cancel()

    for row in budget_rows:
        yield {"type": "budget_row", "row": row}
        await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Budget: {len(budget_rows)} domain estimates"}
    await asyncio.sleep(0)

    # ── Phase 2c: Vendor Readiness (optional) ─────────────────────────────────
    vendor_rows: list[dict] = []
    if target_vendor:
        yield {"type": "heartbeat", "message": f"🎯 Phase 2: Scoring {target_vendor} readiness signals…"}
        await asyncio.sleep(0)

        p2_vendor_prompt = _phase2_prompt_vendor_signals(company_name, target_vendor, phase1_results)
        vendor_future = loop.run_in_executor(None, _run_phase2_sync, p2_vendor_prompt, "vendor_signals")
        elapsed = 0
        while elapsed < 90:
            try:
                vendor_rows = await asyncio.wait_for(asyncio.shield(vendor_future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat", "message": f"🎯 Scoring {target_vendor}… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Phase2 vendor error: {e}")
                break
        else:
            vendor_future.cancel()

        for row in vendor_rows:
            yield {"type": "vendor_signal_row", "row": row}
            await asyncio.sleep(0.04)

        yield {"type": "heartbeat", "message": f"✅ Vendor signals: {len(vendor_rows)} domains scored"}
        await asyncio.sleep(0)

    yield {
        "type": "complete",
        "tech_stack": tech_rows,
        "budget": budget_rows,
        "vendor_signals": vendor_rows,
        "total_tools": len(tech_rows),
        "domains_researched": len(domains),
    }
