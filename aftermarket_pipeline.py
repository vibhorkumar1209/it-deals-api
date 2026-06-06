"""
Aftermarket Deep Dive pipeline — Gemini 2.5 Flash + Google Search.

Tables:
  Table 1 — Capabilities: domain × capability × technology (sub-rows), use case, install base
  Table 2 — Tech Gaps: domain, gap, priority, recommended tech, benchmark
  Table 3 — Spend by Module: domain, current spend estimate, rationale + math shown
  Table 4 — Readiness Matrix + TAM: domain, current system, readiness score, TAM
  Bonus  — Competitive positioning (if competitors provided)
  Bonus  — Aggregate spend estimates: IT / AI / Cloud / Aftermarket spend
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

# ── Output schemas ────────────────────────────────────────────────────────────

# Table 1 — Capabilities (one row per capability × technology pair)
CAPABILITY_FIELDS = [
    {"key": "domain",        "label": "Domain"},
    {"key": "capability",    "label": "Capability"},
    {"key": "technology",    "label": "Technology"},
    {"key": "use_case",      "label": "Use Case"},
    {"key": "install_base",  "label": "Install Base"},
    {"key": "source",        "label": "Source"},
]

# Table 2 — Tech Gaps
GAP_FIELDS = [
    {"key": "domain",           "label": "Domain"},
    {"key": "gap_description",  "label": "Gap / Opportunity"},
    {"key": "priority",         "label": "Priority"},
    {"key": "recommended_tech", "label": "Recommended Technology"},
    {"key": "benchmark",        "label": "Industry Benchmark"},
    {"key": "source",           "label": "Source"},
]

# Table 3 — Spend by Module
SPEND_MODULE_FIELDS = [
    {"key": "domain",          "label": "Module / Domain"},
    {"key": "current_spend",   "label": "Current Spend (Est.)"},
    {"key": "spend_math",      "label": "Calculation / Rationale"},
    {"key": "market_benchmark","label": "Market Benchmark"},
    {"key": "source",          "label": "Source"},
]

# Table 4 — Readiness Matrix + TAM
READINESS_FIELDS = [
    {"key": "domain",                "label": "Module / Domain"},
    {"key": "current_system",        "label": "Current System"},
    {"key": "readiness_score",       "label": "Readiness Score"},
    {"key": "displacement_opp",      "label": "Displacement Opportunity"},
    {"key": "addressable_tam",       "label": "Addressable TAM"},
    {"key": "tam_rationale",         "label": "TAM Rationale"},
    {"key": "source",                "label": "Source"},
]

# Aggregate Spend — IT / AI / Cloud / Aftermarket
AGGREGATE_SPEND_FIELDS = [
    {"key": "spend_type",   "label": "Spend Category"},
    {"key": "estimate",     "label": "Estimate (USD)"},
    {"key": "basis",        "label": "Calculation Basis"},
    {"key": "source",       "label": "Source"},
]

# IT Deals & Partnerships (supporting spend rationale)
SPEND_DEAL_FIELDS = [
    {"key": "vendor",       "label": "Vendor / Partner"},
    {"key": "deal_type",    "label": "Deal Type"},
    {"key": "deal_value",   "label": "Deal Value"},
    {"key": "date",         "label": "Date"},
    {"key": "spend_link",   "label": "Linked Spend Category"},
    {"key": "rationale",    "label": "Spend Rationale"},
    {"key": "source",       "label": "Source"},
]

# Competitive (optional)
COMPETITOR_FIELDS = [
    {"key": "competitor",     "label": "Competitor"},
    {"key": "domain",         "label": "Domain"},
    {"key": "their_advantage","label": "Their Advantage"},
    {"key": "technology",     "label": "Technology"},
    {"key": "implication",    "label": "Implication"},
    {"key": "source",         "label": "Source"},
]

AFTERMARKET_DOMAINS = [
    "Warranty Management",
    "Service & Repair Operations",
    "Parts & Inventory Management",
    "Field Service Management",
    "Technical Knowledge & Documentation",
    "Dealer & Distribution Network",
    "Customer Service & Support",
    "Telematics & Connected Products",
    "Predictive Maintenance & IoT",
    "Digital Commerce & Self-Service",
    "Analytics & Business Intelligence",
    "AI & Automation",
]


# ── Shared Gemini call ────────────────────────────────────────────────────────

def _gemini_call_sync(prompt: str, use_search: bool, label: str):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return []

    if not GOOGLE_AI_KEY:
        return []

    config_kwargs = dict(temperature=0.15, max_output_tokens=8192)
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    MAX_RETRIES = 3
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
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
            logger.error(f"Aftermarket Gemini [{label}]: {e}")
            return []
    else:
        return []

    raw = ""
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                t = getattr(part, "text", None)
                if t: raw += t
    except Exception:
        try: raw = response.text or ""
        except Exception: pass

    if not raw:
        return []

    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
        parsed = json.loads(clean)
        return parsed
    except Exception:
        pass

    try:
        m = re.search(r"[\[\{].*[\]\}]", raw, re.DOTALL)
        if m:
            text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
            return json.loads(text)
    except Exception:
        pass
    return []


async def _run_async(prompt: str, use_search: bool, label: str, timeout: int = 110) -> list:
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, _gemini_call_sync, prompt, use_search, label)
    elapsed = 0
    while elapsed < timeout:
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=10)
        except asyncio.TimeoutError:
            elapsed += 10
    future.cancel()
    return []


# ── Table 1: Capability Assessment ───────────────────────────────────────────

def _cap_prompt(company_name: str, domain: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    return f"""You are an aftermarket service technology analyst with live Google Search.

COMPANY: {company_name}{ind}
DOMAIN: {domain}

Search for the specific technologies and platforms used by {company_name} in "{domain}":
- "{company_name}" {domain} software technology platform
- "{company_name}" {domain} system vendor tool 2022 OR 2023 OR 2024 OR 2025
- "{company_name}" {domain} implementation partner case study

For each technology found in this domain, return ONE JSON object per capability × technology pair.
Each row must describe HOW that technology is used (use case) and ROUGHLY how many users/licenses exist.

Return ONLY a JSON array:
[
  {{
    "domain": "{domain}",
    "capability": "<specific business capability e.g. 'Claim Submission', 'Parts Ordering', 'Work Order Management'>",
    "technology": "<exact product/vendor name e.g. 'Tavant Warranty', 'SAP S/4HANA', 'Salesforce Service Cloud'>",
    "use_case": "<one sentence: how this technology is specifically used for this capability>",
    "install_base": "<estimated users/licenses e.g. '~500 dealer users', '2,000-5,000 technicians', 'Enterprise-wide'>",
    "source": "<URL to press release, case study, or job posting — or '-'>"
  }}
]

Rules:
- Multiple rows per domain if multiple technologies found
- install_base: base estimate on dealer count, employee count, or public data
- If no technology found for a capability, still list the capability with technology="Unknown/Not Disclosed"
- Return [] only if domain has zero public evidence

Return ONLY the raw JSON array.
"""


# ── Table 2: Gap Analysis ─────────────────────────────────────────────────────

def _gap_prompt(company_name: str, industry: str, cap_data: list) -> str:
    ind = f" ({industry})" if industry else ""
    cap_summary = json.dumps(cap_data[:25] if cap_data else [], indent=1)
    return f"""You are an aftermarket technology consultant.

COMPANY: {company_name}{ind}

CURRENT CAPABILITIES (from research):
{cap_summary}

Search for aftermarket technology best practices and gaps:
- aftermarket technology trends 2024 2025 manufacturing automotive service
- warranty management AI automation benchmark best practice
- predictive maintenance IoT connected service industry leader

Identify the top 10-15 technology gaps and investment opportunities.

Return ONLY a JSON array:
[
  {{
    "domain": "<aftermarket domain>",
    "gap_description": "<clear description of the gap or opportunity>",
    "priority": "<Critical | High | Medium | Low>",
    "recommended_tech": "<specific vendor or product that addresses this>",
    "benchmark": "<what leading companies do in this area>",
    "source": "<URL or '-'>"
  }}
]

Return ONLY the raw JSON array.
"""


# ── Table 3: Spend by Module ──────────────────────────────────────────────────

def _spend_module_prompt(company_name: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    domains_list = "\n".join(f"  - {d}" for d in AFTERMARKET_DOMAINS)
    return f"""You are an enterprise IT financial analyst specialising in aftermarket operations.

COMPANY: {company_name}{ind}

Search for {company_name} financial and technology spend data:
- "{company_name}" annual report IT spend technology 2023 OR 2024
- "{company_name}" revenue operating expenses technology budget
- "{company_name}" aftermarket service revenue technology spend
- "{company_name}" warranty management system spend cost
- "{company_name}" dealer management field service technology investment

For EACH of the following aftermarket modules, estimate annual technology spend with explicit math:
{domains_list}

Return ONLY a JSON array — one row per module:
[
  {{
    "domain": "<module name from list above>",
    "current_spend": "<estimated annual spend e.g. '$15M-$25M'>",
    "spend_math": "<calculation logic e.g. '500 dealers × $3k/yr + $2M infra = $3.5M' or '0.3% of $16B revenue × 8% aftermarket allocation = $3.8M'>",
    "market_benchmark": "<typical range for comparable companies>",
    "source": "<URL to annual report, filing or '-'>"
  }}
]

Always show the math in spend_math. Return ONLY the raw JSON array.
"""


# ── Aggregate Spend: IT / AI / Cloud / Aftermarket ────────────────────────────

def _spend_deals_prompt(company_name: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    return f"""You are an IT deal research analyst with live Google Search.

COMPANY: {company_name}{ind}

Search for IT deals, contracts, and technology partnerships that reveal or support spend estimates:
- "{company_name}" IT outsourcing contract deal technology spend
- "{company_name}" cloud migration AWS Azure Google Cloud deal
- "{company_name}" AI machine learning platform deal investment
- "{company_name}" aftermarket service technology vendor deal
- "{company_name}" digital transformation program spend billion million
- site:businesswire.com OR site:prnewswire.com "{company_name}" technology deal

For each deal found, identify which spend category (IT / AI / Cloud / Aftermarket) it supports.

Return ONLY a JSON array:
[
  {{
    "vendor": "<vendor or partner name>",
    "deal_type": "<type e.g. Cloud Migration | Managed Services | AI Platform | IT Outsourcing | Digital Transformation>",
    "deal_value": "<value e.g. '$45 million' or 'Undisclosed'>",
    "date": "<year or YYYY-MM>",
    "spend_link": "<IT Spend | AI Spend | Cloud Spend | Aftermarket Tech Spend>",
    "rationale": "<one sentence: how this deal supports the spend estimate for that category>",
    "source": "<URL to press release or announcement>"
  }}
]

Return ONLY the raw JSON array. No prose. No markdown.
"""


def _aggregate_spend_prompt(company_name: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    return f"""You are an enterprise IT financial analyst.

COMPANY: {company_name}{ind}

Search for financial data:
- "{company_name}" annual report technology IT spend 2023 OR 2024
- "{company_name}" revenue total operating expenses
- "{company_name}" cloud spend AWS Azure Google Cloud
- "{company_name}" AI investment artificial intelligence budget
- "{company_name}" aftermarket service revenue

Estimate the following four spend categories with calculation basis:

Return ONLY a JSON array:
[
  {{
    "spend_type": "IT Spend",
    "estimate": "<total annual IT spend e.g. '$450M-$500M'>",
    "basis": "<e.g. '2.8% of $16.2B revenue (manufacturing industry avg per Gartner)'>",
    "source": "<URL to annual report or analyst benchmark>"
  }},
  {{
    "spend_type": "AI Spend",
    "estimate": "<AI/ML specific spend>",
    "basis": "<e.g. '~8% of total IT spend based on 2024 AI adoption benchmarks'>",
    "source": "<URL>"
  }},
  {{
    "spend_type": "Cloud Spend",
    "estimate": "<cloud infrastructure spend>",
    "basis": "<e.g. '~22% of IT budget (industry avg cloud allocation for manufacturing)'>",
    "source": "<URL>"
  }},
  {{
    "spend_type": "Aftermarket IT Spend",
    "estimate": "<IT spend specifically on aftermarket/service operations>",
    "basis": "<e.g. 'Aftermarket = ~18% of revenue; IT for aftermarket ~1.5% of aftermarket revenue'>",
    "source": "<URL>"
  }}
]

Return ONLY the raw JSON array.
"""


# ── Table 4: Readiness Matrix + TAM ──────────────────────────────────────────

def _readiness_tam_prompt(company_name: str, industry: str, target_vendor: str) -> str:
    ind = f" ({industry})" if industry else ""
    vendor_hint = f" Score readiness specifically from {target_vendor}'s perspective." if target_vendor else ""
    domains_list = "\n".join(f"  - {d}" for d in AFTERMARKET_DOMAINS)
    return f"""You are a technology readiness and market sizing analyst.

COMPANY: {company_name}{ind}{vendor_hint}

Search for technology and market size data:
- "{company_name}" aftermarket technology systems legacy modern platform
- "{company_name}" warranty field service parts technology readiness
- aftermarket software market size TAM warranty management field service DMS
- dealer management system market size manufacturing automotive

For EACH of the following modules, assess readiness and estimate addressable TAM:
{domains_list}

For each module, assess technology readiness and estimate addressable TAM.

Return ONLY a JSON array — one row per module:
[
  {{
    "domain": "<aftermarket module name>",
    "current_system": "<current primary technology/vendor in this domain, or 'Legacy/Unknown'>",
    "readiness_score": <integer 0-100 where 100 = fully modern, 0 = completely legacy/absent>,
    "displacement_opp": "<High | Medium | Low | None — opportunity to replace/upgrade current system>",
    "addressable_tam": "<estimated addressable market for this module e.g. '$12M-$20M'>",
    "tam_rationale": "<show sizing math e.g. '~200 dealer locations × $60k/yr license = $12M base; with expansion 20% uplift = $14.4M'>",
    "source": "<URL to market data or '-'>"
  }}
]

Readiness score guide:
  80-100: Modern cloud-native, well-integrated, recent implementation
  60-79: Established system, some modernisation, minor gaps
  40-59: Partially modernised, legacy components, integration challenges
  20-39: Predominantly legacy, known replacement need
  0-19: No clear system or severely outdated

Return ONLY the raw JSON array.
"""


# ── Competitive (optional) ────────────────────────────────────────────────────

def _comp_prompt(company_name: str, industry: str, competitors: str) -> str:
    ind = f" ({industry})" if industry else ""
    comp_list = [c.strip() for c in competitors.split(",") if c.strip()]
    searches = "\n".join(f'- "{c}" aftermarket service technology platform capability 2023 OR 2024' for c in comp_list[:4])
    return f"""You are a competitive intelligence analyst.

TARGET: {company_name}{ind}
COMPETITORS: {competitors}

{searches}

Return ONLY a JSON array:
[
  {{
    "competitor": "<name>",
    "domain": "<aftermarket domain>",
    "their_advantage": "<what they do better>",
    "technology": "<technology they use>",
    "implication": "<what this means for {company_name}>",
    "source": "<URL or '-'>"
  }}
]

Return ONLY the raw JSON array.
"""


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run_aftermarket_deep_dive(
    company_name: str,
    domain: str,
    industry: str = "",
    competitors: str = "",
    target_vendor: str = "",
) -> AsyncGenerator[dict, None]:
    """
    Yields: heartbeat | capability_row | gap_row | spend_module_row |
            aggregate_spend_row | readiness_row | competitor_row | complete
    """
    yield {"type": "heartbeat", "message": f"🔍 Starting Aftermarket Deep Dive for {company_name}…"}
    await asyncio.sleep(0)

    # ── Step 1: Capability Assessment (Table 1) — per domain ─────────────────
    yield {"type": "heartbeat", "message": f"📋 Table 1: Researching capabilities across {len(AFTERMARKET_DOMAINS)} domains…"}
    await asyncio.sleep(0)

    all_cap_rows = []
    for i, dom in enumerate(AFTERMARKET_DOMAINS):
        yield {"type": "heartbeat", "message": f"🔎 [{i+1}/{len(AFTERMARKET_DOMAINS)}] {dom}…"}
        await asyncio.sleep(0)

        rows = await _run_async(_cap_prompt(company_name, dom, industry), True, f"cap_{dom}", timeout=90)
        for row in (rows if isinstance(rows, list) else []):
            if isinstance(row, dict):
                all_cap_rows.append(row)
                yield {"type": "capability_row", "row": row}
                await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 1 complete — {len(all_cap_rows)} capability × technology rows"}
    await asyncio.sleep(0)

    # ── Step 2: Aggregate Spend (IT / AI / Cloud / Aftermarket) ──────────────
    yield {"type": "heartbeat", "message": "💰 Estimating aggregate IT / AI / Cloud / Aftermarket spend…"}
    await asyncio.sleep(0)

    agg_rows = await _run_async(_aggregate_spend_prompt(company_name, industry), True, "agg_spend", timeout=90)
    for row in (agg_rows if isinstance(agg_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "aggregate_spend_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Aggregate spend: {len(agg_rows) if isinstance(agg_rows, list) else 0} categories estimated"}
    await asyncio.sleep(0)

    # ── Step 2b: IT Deals & Partnerships supporting spend rationale ───────────
    yield {"type": "heartbeat", "message": "🤝 Finding IT deals & partnerships to validate spend estimates…"}
    await asyncio.sleep(0)

    spend_deal_rows = await _run_async(_spend_deals_prompt(company_name, industry), True, "spend_deals", timeout=90)
    for row in (spend_deal_rows if isinstance(spend_deal_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_deal_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ IT deals: {len(spend_deal_rows) if isinstance(spend_deal_rows, list) else 0} deals found"}
    await asyncio.sleep(0)

    # ── Step 3: Spend by Module (Table 3) ────────────────────────────────────
    yield {"type": "heartbeat", "message": "📊 Table 3: Estimating spend by module with rationale…"}
    await asyncio.sleep(0)

    spend_rows = await _run_async(_spend_module_prompt(company_name, industry), True, "spend_module", timeout=110)
    for row in (spend_rows if isinstance(spend_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_module_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 3 complete — {len(spend_rows) if isinstance(spend_rows, list) else 0} module spend estimates"}
    await asyncio.sleep(0)

    # ── Step 4: Readiness Matrix + TAM (Table 4) ─────────────────────────────
    yield {"type": "heartbeat", "message": "🎯 Table 4: Building readiness matrix and TAM estimates…"}
    await asyncio.sleep(0)

    readiness_rows = await _run_async(_readiness_tam_prompt(company_name, industry, target_vendor), True, "readiness_tam", timeout=110)
    for row in (readiness_rows if isinstance(readiness_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "readiness_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 4 complete — {len(readiness_rows) if isinstance(readiness_rows, list) else 0} modules assessed"}
    await asyncio.sleep(0)

    # ── Step 5: Tech Gaps (Table 2) ───────────────────────────────────────────
    yield {"type": "heartbeat", "message": "🔎 Table 2: Identifying technology gaps…"}
    await asyncio.sleep(0)

    gap_rows = await _run_async(_gap_prompt(company_name, industry, []), True, "gaps", timeout=100)
    for row in (gap_rows if isinstance(gap_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "gap_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 2 complete — {len(gap_rows) if isinstance(gap_rows, list) else 0} gaps identified"}
    await asyncio.sleep(0)

    # ── Step 6: Competitive (optional) ────────────────────────────────────────
    comp_rows = []
    if competitors:
        yield {"type": "heartbeat", "message": f"🏆 Competitive benchmarking vs {competitors}…"}
        await asyncio.sleep(0)

        comp_rows = await _run_async(_comp_prompt(company_name, industry, competitors), True, "competitive", timeout=100)
        for row in (comp_rows if isinstance(comp_rows, list) else []):
            if isinstance(row, dict):
                yield {"type": "competitor_row", "row": row}
                await asyncio.sleep(0.04)

        yield {"type": "heartbeat", "message": f"✅ Competitive: {len(comp_rows) if isinstance(comp_rows, list) else 0} findings"}
        await asyncio.sleep(0)

    yield {
        "type": "complete",
        "capabilities": all_cap_rows,
        "gaps": gap_rows if isinstance(gap_rows, list) else [],
        "spend_modules": spend_rows if isinstance(spend_rows, list) else [],
        "aggregate_spend": agg_rows if isinstance(agg_rows, list) else [],
        "spend_deals": spend_deal_rows if isinstance(spend_deal_rows, list) else [],
        "readiness": readiness_rows if isinstance(readiness_rows, list) else [],
        "competitors": comp_rows if isinstance(comp_rows, list) else [],
    }
