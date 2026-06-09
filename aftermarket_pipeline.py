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

# Vendor Footprint
VENDOR_FOOTPRINT_FIELDS = [
    {"key": "domain",               "label": "Domain"},
    {"key": "footprint_status",     "label": "Footprint Status"},
    {"key": "evidence",             "label": "Evidence"},
    {"key": "product_deployed",     "label": "Product Deployed"},
    {"key": "opportunity_size",     "label": "Opportunity"},
    {"key": "opportunity_rationale","label": "Rationale"},
    {"key": "source",               "label": "Source"},
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

def _gemini_call_sync(prompt: str, use_search: bool, label: str, max_output_tokens: int = 16384, model: str = "gemini-2.5-flash", return_raw: bool = False):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return []

    if not GOOGLE_AI_KEY:
        return []

    config_kwargs = dict(temperature=0.15, max_output_tokens=max_output_tokens)
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    # Only disable thinking for 2.5 models — 2.0-flash has no thinking capability
    # and passing ThinkingConfig to it causes an API error.
    if "2.5" in model:
        try:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass  # older SDK version — skip

    MAX_RETRIES = 4
    CALL_TIMEOUT = 120   # hard per-call HTTP timeout in seconds
    TOTAL_BUDGET = 300   # all attempts combined must finish within this
    call_start = _time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        if _time.time() - call_start > TOTAL_BUDGET:
            logger.warning(f"Gemini [{label}] total budget {TOTAL_BUDGET}s exceeded — giving up")
            return []
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            logger.info(f"Gemini [{label}] attempt {attempt}/{MAX_RETRIES} starting (model={model})")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            logger.info(f"Gemini [{label}] attempt {attempt} succeeded")
            break
        except Exception as e:
            err = str(e)
            is_quota = "RESOURCE_EXHAUSTED" in err or "free_tier" in err
            is_retry = not is_quota and any(x in err for x in ("503", "UNAVAILABLE", "overloaded", "timeout", "TimeoutError", "DeadlineExceeded", "429", "500", "502", "503", "504", "ConnectionError", "ConnectionReset", "RemoteDisconnected"))
            if is_quota:
                raise RuntimeError("Gemini quota exhausted — upgrade to paid API plan.") from e
            if is_retry and attempt < MAX_RETRIES:
                elapsed = _time.time() - call_start
                remaining = TOTAL_BUDGET - elapsed
                wait = min(15 * attempt, 45, max(0, remaining - 15))
                if wait <= 0:
                    logger.warning(f"Gemini [{label}] no time left for retry — giving up")
                    return []
                logger.warning(f"Gemini [{label}] attempt {attempt}/{MAX_RETRIES} failed ({err[:60]}), retry in {wait:.0f}s")
                _time.sleep(wait)
                continue
            logger.error(f"Aftermarket Gemini [{label}] TERMINAL FAIL (attempt {attempt}, model={model}): {err}")
            return []
    else:
        return []

    # Collect non-thought text parts. Gemini 2.5 Flash may return multiple parts
    # (e.g. two identical-start parts when search grounding is active). Try each
    # part individually so we don't concatenate two separate JSON arrays into one
    # invalid blob. Return on the first part that parses successfully.
    text_parts = []
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    continue  # skip internal reasoning
                t = getattr(part, "text", None)
                if t:
                    text_parts.append(t)
    except Exception:
        pass

    # Fallback: response.text excludes thought parts in Google GenAI SDK
    if not text_parts:
        try:
            t = response.text
            if t:
                text_parts = [t]
        except Exception:
            pass

    if not text_parts:
        logger.warning(f"_gemini_call_sync [{label}]: empty response from Gemini")
        return "" if return_raw else []

    # return_raw: caller wants the raw text, not parsed JSON (e.g. research step)
    if return_raw:
        return "\n\n".join(text_parts)

    def _try_parse(raw: str) -> list | None:
        """Try multiple strategies to extract a JSON array from raw text."""
        try:
            clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
            clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
            parsed = json.loads(clean)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
        try:
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
                result = json.loads(text)
                if isinstance(result, list):
                    return result
        except Exception:
            pass
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
                obj = json.loads(text)
                if isinstance(obj, dict):
                    return [obj]
        except Exception:
            pass
        return None

    # Try each part independently first (avoids concatenation of two full arrays)
    best: list | None = None
    for part_text in text_parts:
        result = _try_parse(part_text)
        if result:
            if best is None or len(result) > len(best):
                best = result  # keep the part with the most rows

    if best is not None:
        return best

    # Last resort: concatenate all parts and try again
    combined = "".join(text_parts)
    result = _try_parse(combined)
    if result:
        return result

    logger.warning(f"_gemini_call_sync [{label}]: no JSON found. Parts={len(text_parts)}, preview: {text_parts[0][:200] if text_parts else ''}")
    return []


async def _collect_future(future, label: str, timeout: int = 90) -> list:
    """Await an asyncio Future from run_in_executor with timeout. Module-level."""
    try:
        return await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"_collect_future: {label} timed out after {timeout}s")
        return []
    except Exception as e:
        logger.error(f"_collect_future: {label} error: {e}")
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

def _cap_prompt(company_name: str, domain: str, industry: str, target_vendor: str = "") -> str:
    ind = f" ({industry})" if industry else ""
    vendor_search = f'\n- "{company_name}" "{target_vendor}" {domain}' if target_vendor else ""
    return f"""You are an aftermarket service technology analyst with live Google Search.

COMPANY: {company_name}{ind}
DOMAIN: {domain}

Search for the specific technologies and platforms used by {company_name} in "{domain}":
- "{company_name}" {domain} software technology platform
- "{company_name}" {domain} system vendor tool 2022 OR 2023 OR 2024 OR 2025
- "{company_name}" {domain} implementation partner case study{vendor_search}

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
- Only include a technology row if there is ACTUAL evidence that {company_name} uses it for this domain
- If the target vendor search returns no evidence of deployment at {company_name}, do NOT add a row for it
- Do NOT add rows saying "vendor offers products generally" or "no specific deployment found"
- Return [] if no confirmed technology deployments found in this domain

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

def _spend_module_prompt(company_name: str, industry: str, cap_data: list[dict] | None = None, agg_data: list[dict] | None = None) -> str:
    ind = f" ({industry})" if industry else ""
    # Phase 2 synthesis — uses capability + aggregate spend data, NO search grounding
    cap_summary = ""
    if cap_data:
        # Group by domain for compact summary
        from collections import defaultdict
        by_domain: dict = defaultdict(list)
        for r in (cap_data or []):
            by_domain[r.get("domain","")].append(f"{r.get('technology','')} ({r.get('install_base','')})")
        cap_summary = "\n".join(f"  {d}: {', '.join(tools[:3])}" for d,tools in list(by_domain.items())[:12])

    agg_summary = ""
    if agg_data:
        agg_summary = "\n".join(f"  {r.get('spend_type')}: {r.get('estimate')} — {r.get('basis','')}" for r in agg_data)

    return f"""You are an IT financial analyst synthesising research findings. No web search needed.

COMPANY: {company_name}{ind}

KNOWN TECHNOLOGY STACK (from research):
{cap_summary or "  (No capability data available — use industry benchmarks)"}

AGGREGATE IT SPEND CONTEXT:
{agg_summary or "  (No aggregate data — estimate from industry benchmarks)"}

TASK: For each aftermarket module below, estimate annual technology spend.
Base calculations on: (a) known technology vendors/products from the tech stack above,
(b) typical SaaS/license pricing for those specific tools, (c) headcount ratios,
(d) the aggregate spend context to validate totals.

Example calculation: If Tavant WarrantyXchange is deployed for 500 dealers at ~$5k/dealer/yr = $2.5M + $500k infra.

Return a JSON array. Each object must have exactly these keys:
[
  {{
    "domain": "Warranty Management",
    "current_spend": "$8M-$15M",
    "spend_math": "Tavant WarrantyXchange: 500 dealers × $5k/yr = $2.5M + SAP integration $1M + infra $500k = ~$4M. Add managed services 2×: $8M total",
    "market_benchmark": "$5M-$20M for Tier-1 OEMs"
  }}
]

Modules: Warranty Management, Service & Repair Operations, Parts & Inventory Management, Field Service Management, Dealer & Distribution Network, Telematics & Connected Products, Predictive Maintenance & IoT, Analytics & Business Intelligence, AI & Automation.

IMPORTANT: Return ONLY the JSON array starting with [. No prose, no source field needed.
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

SIGNAL_WEIGHTS = {
    "existing_relationship": 0.30,
    "it_signals": 0.15,
    "company_signals": 0.20,
    "executive_signals": 0.15,
    "budget_signals": 0.20,
}

READINESS_FIELDS = [
    {"key": "domain",               "label": "Module"},
    {"key": "current_system",       "label": "Current System"},
    # Signal scores (0-100)
    {"key": "existing_rel_score",   "label": "Existing Rel. (30%)"},
    {"key": "it_signals_score",     "label": "IT Signals (15%)"},
    {"key": "company_signals_score","label": "Company Signals (20%)"},
    {"key": "exec_signals_score",   "label": "Exec Signals (15%)"},
    {"key": "budget_signals_score", "label": "Budget Signals (20%)"},
    # Top 3 signals per category (array of {text, source})
    {"key": "existing_rel_signals",     "label": "Existing Rel. Signals"},
    {"key": "it_signals_signals",       "label": "IT Signal Evidence"},
    {"key": "company_signals_signals",  "label": "Company Signal Evidence"},
    {"key": "exec_signals_signals",     "label": "Exec Signal Evidence"},
    {"key": "budget_signals_signals",   "label": "Budget Signal Evidence"},
    # Calculated outputs
    {"key": "weighted_readiness",   "label": "Weighted Readiness Score"},
    {"key": "displacement_opp",     "label": "Displacement Opportunity"},
    {"key": "total_domain_spend",   "label": "Total Domain Spend"},
    {"key": "vendor_adjusted_tam",  "label": "Vendor-Adjusted TAM"},
    {"key": "tam_rationale",        "label": "TAM Rationale"},
]


def _readiness_research_prompt(company_name: str, industry: str) -> str:
    """Step 1: Broad search-grounded signal gathering for IT investment, exec, budget, hiring."""
    return f"""You are a technology intelligence analyst researching {company_name} ({industry}).

Use Google Search to find REAL, RECENT evidence on the following topics. Search broadly — use company name, abbreviations, and parent/subsidiary names.

SEARCH TOPICS:
1. Digital transformation, IT modernisation, ERP/CRM/field service technology investments by {company_name} (2022–2025)
2. Technology announcements, partnerships with software vendors, system implementations at {company_name}
3. CTO, CIO, CDO, VP Technology statements or interviews about {company_name}'s technology strategy
4. Annual reports, earnings calls, investor presentations mentioning IT spend or digital at {company_name}
5. Job postings: {company_name} hiring for IT, software, data, digital roles (2023–2025)
6. News: {company_name} RFP, tender, contract award for technology systems

For EVERY piece of evidence found, record:
[CATEGORY] One-sentence finding with specific detail. Source: <URL>

Use these categories:
- IT_INVESTMENT: technology system purchases, ERP/CRM/platform go-lives, vendor contracts
- EXEC_AGENDA: executive quotes or strategy statements on digital/technology
- BUDGET_SIGNAL: IT budget announcements, RFPs, tenders, capex for technology
- HIRING_SIGNAL: open roles or hiring patterns in technology/digital/software

If genuinely nothing found for a category after searching: [CATEGORY] No public evidence found.

Report every real finding — aim for at least 2–3 findings per category if they exist."""


def _readiness_score_prompt(company_name: str, industry: str, target_vendor: str,
                             research_text: str, cap_data: list,
                             modules: list[str], spend_lines: str, agg_ref: str) -> str:
    """Step 2: No-search scoring. Vendor deployment from cap_data; other signals from research."""
    vendor = target_vendor or "the vendor"
    n = len(modules)
    modules_list = ", ".join(modules)

    # Build vendor deployment context from already-researched capabilities
    from collections import defaultdict
    cap_by_domain: dict = defaultdict(lambda: {"techs": [], "vendor_hit": False, "signals": []})
    for r in (cap_data or []):
        d = r.get("domain", "")
        tech = r.get("technology", "")
        rel = (r.get("vendor_relationship") or r.get("relationship") or "").lower()
        vendor_lower = vendor.lower()
        if tech and tech not in ("Unknown/Not Disclosed", "-"):
            cap_by_domain[d]["techs"].append(tech)
        if vendor_lower and (vendor_lower in tech.lower() or vendor_lower in rel):
            cap_by_domain[d]["vendor_hit"] = True
            cap_by_domain[d]["signals"].append(tech or rel)

    vendor_lines = []
    for d, info in cap_by_domain.items():
        if info["vendor_hit"]:
            vendor_lines.append(f"  {d}: {vendor} DEPLOYED — {', '.join(info['signals'][:2])}")
        elif info["techs"]:
            vendor_lines.append(f"  {d}: {vendor} NOT found — current tech: {', '.join(info['techs'][:2])}")
    vendor_context = "\n".join(vendor_lines) if vendor_lines else f"  No capabilities data available for {vendor}."

    pre_research = research_text.strip() if research_text and research_text.strip() else ""

    return f"""You are a vendor displacement readiness analyst. Use Google Search to find evidence for each module below.

COMPANY: {company_name} ({industry})
VENDOR BEING EVALUATED: {vendor}

VENDOR DEPLOYMENT — ALREADY VERIFIED (use for existing_rel_score only):
{vendor_context}

PRE-FETCHED SIGNALS (may be partial — search for more if needed):
{pre_research or "Not available — use Google Search to find evidence."}

SPEND BY MODULE:
{spend_lines}

INSTRUCTIONS:
For each module in MODULES, search Google for recent (2022–2025) evidence specific to {company_name}:
- IT_INVESTMENT: technology implementations, ERP/platform go-lives, vendor contracts in this domain
- EXEC_AGENDA: executive statements on digital/technology strategy relevant to this domain
- BUDGET_SIGNAL: IT budget, RFP, tender, capex for technology in this domain
- HIRING_SIGNAL: open roles or hiring in technology/digital relevant to this domain

Use PRE-FETCHED SIGNALS above first. If a signal type is missing or thin, search Google now for that specific signal for {company_name}.

SCORING RULES:
- existing_rel_score (0.30 weight): VENDOR DEPLOYMENT only. DEPLOYED→≥75. Partial/mentioned→40-70. Not found→≤20.
- it_signals_score (0.15): IT_INVESTMENT or HIRING_SIGNAL for this specific domain
- company_signals_score (0.20): IT_INVESTMENT or growth signals for this domain
- exec_signals_score (0.15): EXEC_AGENDA for this domain
- budget_signals_score (0.20): BUDGET_SIGNAL for this domain
- weighted_readiness = (existing_rel×0.30)+(it×0.15)+(company×0.20)+(exec×0.15)+(budget×0.20)
- displacement_opp: High if ≥65, Medium if 40–64, Low if <40
- For evidence fields: quote the actual finding with source URL, or "No evidence found after search"
- total_domain_spend: copy from SPEND BY MODULE, else industry benchmark
- vendor_adjusted_tam = total_domain_spend × weighted_readiness/100
- tam_rationale: show midpoint × readiness% calculation

MODULES: {modules_list}

Return ONLY a JSON array — exactly {n} objects:
[
  {{
    "domain": "<module name>",
    "current_system": "<known system or Unknown>",
    "existing_rel_score": <0-100>,
    "existing_rel_evidence": "<quote from VENDOR DEPLOYMENT or Not deployed>",
    "it_signals_score": <0-100>,
    "it_signals_evidence": "<finding with source URL or No evidence found after search>",
    "company_signals_score": <0-100>,
    "company_signals_evidence": "<finding with source URL or No evidence found after search>",
    "exec_signals_score": <0-100>,
    "exec_signals_evidence": "<finding with source URL or No evidence found after search>",
    "budget_signals_score": <0-100>,
    "budget_signals_evidence": "<finding with source URL or No evidence found after search>",
    "weighted_readiness": <0-100>,
    "displacement_opp": "<High|Medium|Low>",
    "total_domain_spend": "<from SPEND BY MODULE or benchmark>",
    "vendor_adjusted_tam": "<range × readiness%>",
    "tam_rationale": "<midpoint × readiness% calculation>"
  }}
]
Return ONLY the JSON array. No prose."""


def _readiness_tam_prompt(company_name: str, industry: str, target_vendor: str,
                          cap_data: list[dict] | None = None,
                          agg_data: list[dict] | None = None,
                          spend_module_rows: list[dict] | None = None,
                          modules: list[str] | None = None) -> str:
    ind = f" ({industry})" if industry else ""
    vendor = target_vendor or "the vendor"
    from collections import defaultdict

    # Build per-domain context from capabilities: tech + existing vendor relationship
    cap_by_domain: dict = defaultdict(lambda: {"techs": [], "vendor_hit": False, "vendor_signals": []})
    for r in (cap_data or []):
        d = r.get("domain", "")
        tech = r.get("technology", "")
        if tech and tech not in ("Unknown/Not Disclosed", "-"):
            cap_by_domain[d]["techs"].append(tech)
        # Flag if target vendor appears in technology or vendor relationship fields
        rel = (r.get("vendor_relationship") or r.get("relationship") or "").lower()
        tech_lower = tech.lower()
        vendor_lower = vendor.lower()
        # Only flag if vendor is non-empty (empty string matches everything via Python's `in`)
        if vendor_lower and (vendor_lower in tech_lower or vendor_lower in rel):
            cap_by_domain[d]["vendor_hit"] = True
            if tech:
                cap_by_domain[d]["vendor_signals"].append(tech)

    cap_lines = []
    for d, info in list(cap_by_domain.items())[:9]:
        techs_str = ", ".join(info["techs"][:3]) or "Unknown"
        vendor_flag = f" ⚠️ {vendor} ALREADY DEPLOYED" if info["vendor_hit"] else ""
        cap_lines.append(f"  {d}: {techs_str}{vendor_flag}")
    cap_intel = "\n".join(cap_lines) or "  limited data"

    # Build per-domain spend reference from Spend by Module (authoritative figures)
    spend_by_domain: dict[str, str] = {}
    for r in (spend_module_rows or []):
        dm = r.get("domain", "")
        cs = r.get("current_spend", "")
        if dm and cs:
            spend_by_domain[dm] = cs

    spend_lines = "\n".join(f"  {d}: {v}" for d, v in spend_by_domain.items()) if spend_by_domain else "  (not available — estimate from market benchmarks)"

    agg_ref = " | ".join(f"{r.get('spend_type')}: {r.get('estimate','?')}" for r in (agg_data or [])[:4])

    modules_to_run = modules or ["Warranty Management", "Service & Repair Operations", "Parts & Inventory Management", "Field Service Management", "Dealer & Distribution Network", "Telematics & Connected Products", "Predictive Maintenance & IoT", "Analytics & Business Intelligence", "AI & Automation"]
    modules_list = ", ".join(modules_to_run)
    n_modules = len(modules_to_run)

    return f"""You are a vendor displacement readiness analyst with deep knowledge of enterprise software markets.

COMPANY: {company_name}{ind}
VENDOR BEING EVALUATED: {vendor}

KNOWN TECH STACK (from prior research — use for existing_rel_score):
{cap_intel}

SPEND BY MODULE (use EXACTLY for total_domain_spend):
{spend_lines}

AGGREGATE SPEND CONTEXT: {agg_ref}

SPEND & TAM RULES:
1. total_domain_spend: copy exactly from SPEND BY MODULE above. If missing, estimate from industry benchmarks.
2. vendor_adjusted_tam = total_domain_spend × (weighted_readiness / 100) applied to both ends of range.
3. tam_rationale: show midpoint = (low+high)/2, then × readiness%.
4. existing_rel_score: if KNOWN TECH STACK shows "{vendor} ALREADY DEPLOYED" → score ≥ 80. Pilot/eval → 40-70. No evidence → ≤ 30.

SCORING RULES (use your knowledge of {company_name} and the {industry} sector):
- existing_rel_score (weight 0.30): Is {vendor} known to be deployed or piloted at {company_name}?
- it_signals_score (weight 0.15): Does {company_name} show technology investment signals in this domain (hiring, RFPs, known projects)?
- company_signals_score (weight 0.20): Is {company_name} growing, undergoing digital transformation, or investing in this domain?
- exec_signals_score (weight 0.15): Are executives at {company_name} known to prioritise this domain?
- budget_signals_score (weight 0.20): Does {company_name} have budget signals or IT spend patterns supporting this domain?
- weighted_readiness = (existing_rel_score×0.30)+(it_signals_score×0.15)+(company_signals_score×0.20)+(exec_signals_score×0.15)+(budget_signals_score×0.20)

Base scores on your training knowledge of {company_name}, its known technology landscape, and typical patterns for {industry} companies of its size. Be concrete and specific in evidence fields.

MODULES: {modules_list}

Return ONLY a valid JSON array — one entry per module listed above. Each element:

[
  {{
    "domain": "Warranty Management",
    "current_system": "<e.g. SAP Warranty or Unknown>",
    "existing_rel_score": 70,
    "existing_rel_evidence": "<one sentence: specific evidence of vendor deployment or absence>",
    "it_signals_score": 55,
    "it_signals_evidence": "<one sentence: job postings, RFPs, or tech evaluations>",
    "company_signals_score": 60,
    "company_signals_evidence": "<one sentence: growth, M&A, or transformation signal>",
    "exec_signals_score": 45,
    "exec_signals_evidence": "<one sentence: exec statement or LinkedIn post>",
    "budget_signals_score": 50,
    "budget_signals_evidence": "<one sentence: budget announcement or contract renewal>",
    "weighted_readiness": 58,
    "displacement_opp": "High",
    "total_domain_spend": "$0.9M-$1.2M",
    "vendor_adjusted_tam": "$0.52M-$0.70M",
    "tam_rationale": "Midpoint = ($0.9M+$1.2M)/2 = $1.05M × 58% readiness = $0.61M vendor-adjusted TAM"
  }}
]

Return ALL {n_modules} modules listed above. Return ONLY the JSON array starting with [. No prose, no markdown.
"""



# ── Vendor Footprint (optional) ──────────────────────────────────────────────

def _vendor_footprint_prompt(company_name: str, target_vendor: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    domains_list = "\n".join(f"  - {d}" for d in AFTERMARKET_DOMAINS)
    return f"""You are a vendor competitive intelligence analyst with live Google Search.

TARGET COMPANY: {company_name}{ind}
TARGET VENDOR TO EVALUATE: {target_vendor}

STEP 1 — Identify {target_vendor}'s direct competitors in the aftermarket/field service software space.
Search: "{target_vendor}" competitors alternatives aftermarket warranty field service software

STEP 2 — Search for {target_vendor}'s AND its competitors' footprint at {company_name}.
Run ALL of the following searches:

Target vendor searches (web + social):
- "{company_name}" "{target_vendor}" implementation deployed partnership
- "{target_vendor}" case study customer "{company_name}"
- site:linkedin.com "{company_name}" "{target_vendor}" implementation
- site:linkedin.com/company "{target_vendor}" "{company_name}"
- "{target_vendor}" "{company_name}" warranty OR "field service" OR "dealer management"
- site:{target_vendor.lower().replace(" ", "")}.com "{company_name}"

Competitor searches (for each identified competitor, run similar searches):
- "{company_name}" [competitor] warranty management system
- "{company_name}" [competitor] field service dealer platform
- site:linkedin.com "{company_name}" [competitor] deployed implemented

STEP 3 — For EACH of the following aftermarket modules, return one row per vendor
({target_vendor} AND each identified competitor) covering their footprint at {company_name}:
{domains_list}

Return ONLY a JSON array — one object per domain × vendor combination:
[
  {{
    "domain": "<aftermarket module name>",
    "vendor_name": "<exact vendor name — either {target_vendor} or a competitor>",
    "is_target_vendor": "<true | false>",
    "footprint_status": "<Active Deployment | Pilot/POC | No Presence | Likely Present (inferred) | Competitor Present>",
    "evidence": "<specific evidence found: quote case study title, LinkedIn post, job posting, press release — be specific>",
    "evidence_sources": "<comma-separated list of where evidence was found: 'LinkedIn', 'Vendor case study', 'Press release', 'Job posting', 'News article'>",
    "product_deployed": "<specific product/module name if found, else '-'>",
    "opportunity_size": "<High | Medium | Low | None>",
    "opportunity_rationale": "<one sentence: opportunity for {target_vendor} given current landscape in this domain>",
    "source": "<direct URL — LinkedIn profile, case study page, press release, or '-'>"
  }}
]

CRITICAL RULES:
- Search LinkedIn specifically for job postings mentioning these vendors at {company_name}
- Search company websites and press release sites for case studies
- Return one row per domain for {target_vendor} (even if no evidence found — mark as No Presence)
- Return additional rows for each competitor found with evidence at {company_name}
- evidence field must be SPECIFIC — not generic statements

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

ALL_SECTIONS = {"capabilities", "agg_spend", "spend_deals", "spend_module", "readiness", "gaps", "competitive"}


async def run_aftermarket_deep_dive(
    company_name: str,
    domain: str,
    industry: str = "",
    competitors: str = "",
    target_vendor: str = "",
    sections_to_run: set | None = None,   # None = all sections
    existing_spend_rows: list | None = None,  # caller passes previously computed spend rows
) -> AsyncGenerator[dict, None]:
    """
    Yields: heartbeat | capability_row | gap_row | spend_module_row |
            aggregate_spend_row | readiness_row | competitor_row | complete

    sections_to_run: subset of ALL_SECTIONS to regenerate. None = run all.
    """
    run = set(sections_to_run) if sections_to_run else ALL_SECTIONS
    partial = bool(sections_to_run)  # True if only regenerating some sections

    if partial:
        yield {"type": "heartbeat", "message": f"🔄 Regenerating missing sections for {company_name}: {', '.join(sorted(run))}…"}
    else:
        yield {"type": "heartbeat", "message": f"🔍 Starting Aftermarket Deep Dive for {company_name}…"}
    await asyncio.sleep(0)

    yield {"type": "heartbeat", "message": f"🚀 Launching {len(run)} section(s) in parallel…"}
    await asyncio.sleep(0)

    loop = asyncio.get_event_loop()

    # Phase 1: Fire search-grounded calls in parallel (capabilities + aggregate spend + IT deals)
    cap_futures    = [loop.run_in_executor(None, _gemini_call_sync, _cap_prompt(company_name, dom, industry, target_vendor), True, f"cap_{dom}") for dom in AFTERMARKET_DOMAINS] if "capabilities" in run else []
    agg_future     = loop.run_in_executor(None, _gemini_call_sync, _aggregate_spend_prompt(company_name, industry), True, "agg_spend") if "agg_spend" in run else None
    deals_future   = loop.run_in_executor(None, _gemini_call_sync, _spend_deals_prompt(company_name, industry), True, "spend_deals") if "spend_deals" in run else None

    # If spend_module (but NOT readiness-only) is requested without capabilities,
    # fire a lightweight search to get context for synthesis.
    # Readiness-only skips context: it uses Google Search grounding internally and
    # serial context fetching (~105s) eats into the readiness timeout budget.
    needs_context = ("spend_module" in run) and "capabilities" not in run
    context_future = loop.run_in_executor(
        None, _gemini_call_sync,
        _cap_prompt(company_name, "Warranty Management", industry, target_vendor),
        True, "cap_context_lite"
    ) if needs_context else None

    spend_future   = None
    _ALL_MODULES = ["Warranty Management", "Service & Repair Operations", "Parts & Inventory Management",
                    "Field Service Management", "Dealer & Distribution Network",
                    "Telematics & Connected Products", "Predictive Maintenance & IoT",
                    "Analytics & Business Intelligence", "AI & Automation"]
    _BATCH_A = _ALL_MODULES[:5]
    _BATCH_B = _ALL_MODULES[5:]

    # Readiness 2-step pipeline:
    # Step 1: ONE search-grounded call collects raw evidence (vendor deployment, IT signals,
    #         exec agenda, budget signals) — focused prompt, returns raw text ~30-60s.
    # Step 2: TWO parallel no-search calls score batch A and batch B from that evidence ~10s.
    # This gives REAL evidence (no hallucination) without the multi-minute search hangs
    # that occurred when each scoring call did its own searches.
    if "readiness" in run:
        # Research prompt no longer searches for vendor deployment — that comes from cap_data
        ready_research_future = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_research_prompt(company_name, industry),
            True, "readiness_research", 4096, "gemini-2.5-flash", True,  # return_raw=True
        )
    else:
        ready_research_future = None
    ready_future_a = ready_future_b = None  # set after research completes

    yield {"type": "heartbeat", "message": "🌐 Phase 1: Researching capabilities & tech spend in parallel…"}
    await asyncio.sleep(0)

    # Collect capability rows (stream as each domain completes)
    all_cap_rows = []
    pending_caps = list(enumerate(cap_futures))
    elapsed = 0
    PARALLEL_TIMEOUT = 90  # 90s max — slow domains are dropped, pipeline moves on

    # Poll until all futures done or timeout
    completed_caps = set()
    while pending_caps and elapsed < PARALLEL_TIMEOUT:
        try:
            await asyncio.sleep(5)
            elapsed += 5
            newly_done = []
            for idx, fut in pending_caps:
                if fut.done():
                    newly_done.append(idx)
                    try:
                        rows = fut.result()
                        for row in (rows if isinstance(rows, list) else []):
                            if isinstance(row, dict):
                                all_cap_rows.append(row)
                                yield {"type": "capability_row", "row": row}
                                await asyncio.sleep(0)
                    except Exception as e:
                        logger.error(f"Cap domain {idx} error: {e}")
            pending_caps = [(i, f) for i, f in pending_caps if i not in newly_done]
            done_count = len(AFTERMARKET_DOMAINS) - len(pending_caps)
            if pending_caps:
                yield {"type": "heartbeat", "message": f"🌐 Researching… {done_count}/{len(AFTERMARKET_DOMAINS)} domains done, {len(all_cap_rows)} capabilities found ({elapsed}s)"}
                await asyncio.sleep(0)
        except Exception:
            break

    # Cancel any remaining cap futures
    for _, fut in pending_caps:
        fut.cancel()

    # If we fired a lite context search (partial regen without capabilities), collect it now
    if context_future and not all_cap_rows:
        yield {"type": "heartbeat", "message": "🔎 Collecting context for Phase 2 synthesis…"}
        await asyncio.sleep(0)
        context_rows = await _collect_future(context_future, "cap_context_lite", timeout=90)
        all_cap_rows.extend(r for r in (context_rows if isinstance(context_rows, list) else []) if isinstance(r, dict))
        yield {"type": "heartbeat", "message": f"✅ Context collected: {len(all_cap_rows)} signals for synthesis"}
        await asyncio.sleep(0)

    yield {"type": "heartbeat", "message": f"✅ Capabilities: {len(all_cap_rows)} rows — launching Phase 2 synthesis…"}
    await asyncio.sleep(0)

    # Collect aggregate spend (needed as context for spend_module synthesis)
    if agg_future:
        agg_rows = await _collect_future(agg_future, "agg_spend", timeout=90)
    elif "spend_module" in run:
        # Fetch aggregate spend as context for spend_module synthesis
        yield {"type": "heartbeat", "message": "💰 Fetching spend context for Phase 2…"}
        await asyncio.sleep(0)
        agg_rows = await _run_async(_aggregate_spend_prompt(company_name, industry), True, "agg_context", timeout=90)
    else:
        agg_rows = []
    # Only emit aggregate_spend_row events if agg_spend is explicitly requested
    # (otherwise it's just context for Phase 2 — don't add duplicate cards to frontend)
    if "agg_spend" in run:
        for row in (agg_rows if isinstance(agg_rows, list) else []):
            if isinstance(row, dict):
                yield {"type": "aggregate_spend_row", "row": row}
                await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ Aggregate spend: {len(agg_rows) if isinstance(agg_rows, list) else 0} categories — now synthesising spend by module & readiness…"}
    await asyncio.sleep(0)

    # Phase 2a: Spend by module (no search grounding — pure synthesis from cap + agg data)
    if "spend_module" in run:
        spend_future = loop.run_in_executor(
            None, _gemini_call_sync,
            _spend_module_prompt(company_name, industry, all_cap_rows, agg_rows if isinstance(agg_rows, list) else []),
            False, "spend_module"
        )

    # Collect IT deals in parallel while spend_module synthesises
    spend_deal_rows = await _collect_future(deals_future, "spend_deals", timeout=90) if deals_future else []
    for row in (spend_deal_rows if isinstance(spend_deal_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_deal_row", "row": row}
            await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ IT deals: {len(spend_deal_rows) if isinstance(spend_deal_rows, list) else 0} deals"}
    await asyncio.sleep(0)

    # Collect spend by module — must complete BEFORE launching readiness so TAM uses real spend figures
    spend_rows = await _collect_future(spend_future, "spend_module", timeout=120) if spend_future else []
    for row in (spend_rows if isinstance(spend_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_module_row", "row": row}
            await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ Spend by module: {len(spend_rows) if isinstance(spend_rows, list) else 0} rows"}
    await asyncio.sleep(0)

    # Readiness was already launched at startup — nothing to do here.

    # ── Step 4: Readiness Matrix + TAM ───────────────────────────────────────
    yield {"type": "heartbeat", "message": "🎯 Table 4: Collecting readiness matrix (2 parallel batches)…"}
    await asyncio.sleep(0)

    # ── Step 4a: Collect research evidence (search-grounded, ~30-60s) ─────────
    readiness_rows = []
    if ready_research_future:
        yield {"type": "heartbeat", "message": "🔍 Gathering real evidence for readiness signals…"}
        await asyncio.sleep(0)

        research_text = ""
        try:
            research_text = await asyncio.wait_for(ready_research_future, timeout=100)
            research_text = research_text if isinstance(research_text, str) else ""
        except asyncio.TimeoutError:
            research_text = ""
            logger.warning("readiness_research: timed out after 100s — proceeding with cap_data only")
        except Exception as e:
            research_text = ""
            logger.error(f"readiness_research failed: {e} — proceeding with cap_data only")

        # ── Step 4b: Score both batches in parallel (no search, ~10-15s each) ──
        yield {"type": "heartbeat", "message": "🎯 Scoring readiness for all 9 modules from real evidence…"}
        await asyncio.sleep(0)

        # Use spend_rows from this run; fall back to existing_spend_rows passed by caller
        # (set when regenerating readiness-only — spend_module wasn't re-run).
        effective_spend = (spend_rows if isinstance(spend_rows, list) and spend_rows
                           else (existing_spend_rows or []))
        spend_lines_for_score = "\n".join(
            f"  {r.get('domain','')}: {r.get('current_spend','')}"
            for r in effective_spend
            if r.get("domain") and r.get("current_spend")
        ) or "  (not available — use industry benchmarks)"
        agg_ref_for_score = " | ".join(
            f"{r.get('spend_type')}: {r.get('estimate','?')}"
            for r in (agg_rows if isinstance(agg_rows, list) else [])[:4]
        )

        # Search-grounded scoring: model can search for domain-specific evidence beyond research_text
        has_research = bool(research_text and "No public evidence found" not in research_text)
        score_future_a = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_score_prompt(company_name, industry, target_vendor, research_text, all_cap_rows, _BATCH_A, spend_lines_for_score, agg_ref_for_score),
            True, "readiness_score_a", 16384, "gemini-2.5-flash",
        )
        score_future_b = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_score_prompt(company_name, industry, target_vendor, research_text, all_cap_rows, _BATCH_B, spend_lines_for_score, agg_ref_for_score),
            True, "readiness_score_b", 16384, "gemini-2.5-flash",
        )

        collected: list = []
        for label_s, fut in [("readiness_score_a", score_future_a), ("readiness_score_b", score_future_b)]:
            try:
                rows = await asyncio.wait_for(fut, timeout=150)
                if isinstance(rows, list):
                    collected.extend(rows)
                    logger.info(f"{label_s} returned {len(rows)} rows")
            except asyncio.TimeoutError:
                logger.warning(f"{label_s} timed out after 90s")
            except Exception as e:
                logger.error(f"{label_s} failed: {e}")

        _MODULE_ORDER = {m: i for i, m in enumerate(_ALL_MODULES)}
        readiness_rows = sorted(collected, key=lambda r: _MODULE_ORDER.get(r.get("domain", ""), 99))

    for row in (readiness_rows if isinstance(readiness_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "readiness_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 4 complete — {len(readiness_rows) if isinstance(readiness_rows, list) else 0} modules assessed"}
    await asyncio.sleep(0)

    vendor_rows = []  # vendor footprint now integrated into capabilities

    # ── Step 5: Tech Gaps (Table 2) ───────────────────────────────────────────
    yield {"type": "heartbeat", "message": "🔎 Table 2: Identifying technology gaps…"}
    await asyncio.sleep(0)

    gap_rows = await _run_async(_gap_prompt(company_name, industry, []), True, "gaps", timeout=100) if "gaps" in run else []
    for row in (gap_rows if isinstance(gap_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "gap_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Table 2 complete — {len(gap_rows) if isinstance(gap_rows, list) else 0} gaps identified"}
    await asyncio.sleep(0)

    # ── Step 6: Competitive (optional) ────────────────────────────────────────
    comp_rows = []
    if competitors and "competitive" in run:
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
        "sections_ran": list(run),
        "capabilities": all_cap_rows,
        "gaps": gap_rows if isinstance(gap_rows, list) else [],
        "spend_modules": spend_rows if isinstance(spend_rows, list) else [],
        "aggregate_spend": agg_rows if isinstance(agg_rows, list) else [],
        "spend_deals": spend_deal_rows if isinstance(spend_deal_rows, list) else [],
        "readiness": readiness_rows if isinstance(readiness_rows, list) else [],
        "vendor_footprint": [],  # removed as separate section
        "competitors": comp_rows if isinstance(comp_rows, list) else [],
    }
