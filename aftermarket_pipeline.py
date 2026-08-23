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

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "is", "are",
    "was", "were", "at", "by", "from", "this", "that", "it", "its", "their", "as",
}


def _tokenize(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2 and w not in _STOPWORDS}


def _match_grounding_source_generic(row: dict, grounding_sources: list[tuple[str, str]]) -> str:
    """Pick the grounding-verified URL whose title best overlaps this row's content.
    Works across all Aftermarket Intelligence row shapes (capability, gap, spend, deal,
    footprint, competitor rows) by matching against every string field in the row rather
    than assuming a fixed schema. Always returns a real, working URL if grounding
    triggered (falls back to the first chunk when no title overlap), or '' if Google
    Search grounding returned nothing — so the report never shows a confidently broken link."""
    if not grounding_sources:
        return ""
    row_text = " ".join(str(v) for k, v in row.items() if k != "source" and isinstance(v, (str, int, float)))
    target_tokens = _tokenize(row_text)
    best_score, best_uri = -1, grounding_sources[0][0]
    for uri, title in grounding_sources:
        score = len(target_tokens & _tokenize(title))
        if score > best_score:
            best_score, best_uri = score, uri
    return best_uri


def _gemini_call_sync(prompt: str, use_search: bool, label: str, max_output_tokens: int = 16384, model: str = "gemini-2.5-flash", return_raw: bool = False, temperature: float = 0.15, run_id: str = ""):
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return []

    if not GOOGLE_AI_KEY:
        return []

    config_kwargs = dict(temperature=temperature, max_output_tokens=max_output_tokens)
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
    TOTAL_BUDGET = 360   # Render Pro: dedicated CPU, extend budget for deep searches
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
            from usage_logger import log_gemini_usage
            log_gemini_usage("aftermarket_intelligence", label, response, grounded=use_search, model=model, run_id=run_id)
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

    # ── Extract grounding metadata — REAL, working URLs from Google Search ──
    # The model frequently invents/misremembers "source" URLs in its JSON output,
    # which is the cause of 404s. Grounding chunks are Google's own verified
    # redirect links to the pages it actually searched, so every row's "source"
    # field is replaced with one of these below instead of trusting whatever
    # URL string the model typed into the JSON.
    grounding_sources: list[tuple[str, str]] = []
    if use_search:
        try:
            for candidate in (response.candidates or []):
                gm = getattr(candidate, "grounding_metadata", None)
                if not gm:
                    continue
                for chunk in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        grounding_sources.append((web.uri, getattr(web, "title", "") or ""))
        except Exception as g_err:
            logger.warning(f"Grounding metadata extraction failed [{label}]: {g_err}")

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

    def _apply_grounding(rows: list) -> list:
        """Replace each row's model-typed 'source' URL with a real, grounding-verified
        one (or '-' if grounding found nothing) — never trust a URL string the model
        typed directly, that's what produces 404s in the generated report."""
        if not use_search:
            return rows
        for row in rows:
            if isinstance(row, dict) and "source" in row:
                matched = _match_grounding_source_generic(row, grounding_sources)
                row["source"] = matched or "-"
        return rows

    # Try each part independently first (avoids concatenation of two full arrays)
    best: list | None = None
    for part_text in text_parts:
        result = _try_parse(part_text)
        if result:
            if best is None or len(result) > len(best):
                best = result  # keep the part with the most rows

    if best is not None:
        return _apply_grounding(best)

    # Last resort: concatenate all parts and try again
    combined = "".join(text_parts)
    result = _try_parse(combined)
    if result:
        return _apply_grounding(result)

    logger.warning(f"_gemini_call_sync [{label}]: no JSON found. Parts={len(text_parts)}, preview: {text_parts[0][:200] if text_parts else ''}")
    return []


async def _collect_future(future, label: str, timeout: int = 90) -> list:
    """Await an asyncio Future from run_in_executor with hard timeout. No asyncio.shield."""
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
        return result if isinstance(result, list) else ([] if result is None else result)
    except asyncio.TimeoutError:
        logger.warning(f"_collect_future: {label} timed out after {timeout}s")
        return []
    except Exception as e:
        logger.error(f"_collect_future: {label} error: {e}")
        return []


async def _run_async(prompt: str, use_search: bool, label: str, timeout: int = 110, run_id: str = "") -> list:
    """Run a Gemini call in executor with hard timeout. No asyncio.shield."""
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, _gemini_call_sync, prompt, use_search, label,
                                  16384, "gemini-2.5-flash", False, 0.15, run_id)
    try:
        result = await asyncio.wait_for(future, timeout=timeout)
        return result if isinstance(result, list) else []
    except asyncio.TimeoutError:
        logger.warning(f"_run_async: {label} timed out after {timeout}s")
        return []
    except Exception as e:
        logger.error(f"_run_async: {label} error: {e}")
        return []


# ── Table 1: Capability Assessment ───────────────────────────────────────────

def _cap_batch_prompt(company_name: str, domains: list[str], industry: str, target_vendor: str = "") -> str:
    """Single prompt covering multiple domains — runs rich per-domain + vendor-match searches."""
    ind = f" ({industry})" if industry else ""
    domains_list = "\n".join(f"  - {d}" for d in domains)

    # Build domain-specific search keywords
    _DOMAIN_KEYWORDS = {
        "Warranty Management": 'warranty management OR "warranty claim" OR "warranty processing" OR "dealer warranty" system software',
        "Service & Repair Operations": '"service management" OR "repair management" OR "work order" OR "service order" OR "workshop management" software',
        "Parts & Inventory Management": '"parts management" OR "spare parts" OR "parts catalog" OR "inventory" OR "parts ordering" software system',
        "Field Service Management": '"field service" OR FSM OR "technician dispatch" OR "service scheduling" OR "mobile workforce" software',
        "Technical Knowledge & Documentation": '"technical documentation" OR "service manual" OR "knowledge base" OR "technical information system" OR TIS software',
        "Dealer & Distribution Network": '"dealer management" OR DMS OR "dealer portal" OR "distribution management" OR "channel management" software',
        "Customer Service & Support": '"customer service" OR CRM OR "contact center" OR "customer portal" OR "service portal" software',
        "Telematics & Connected Products": 'telematics OR "connected equipment" OR "fleet management" OR "remote monitoring" OR IoT platform',
        "Predictive Maintenance & IoT": '"predictive maintenance" OR "condition monitoring" OR "asset health" OR "IoT platform" OR "remote diagnostics"',
        "Digital Commerce & Self-Service": '"digital commerce" OR "ecommerce parts" OR "self-service portal" OR "online parts ordering" OR "B2B portal"',
        "Analytics & Business Intelligence": '"analytics" OR "business intelligence" OR BI OR "data analytics" OR "service analytics" OR "reporting platform"',
        "AI & Automation": '"AI" OR "machine learning" OR "automation" OR "RPA" OR "generative AI" OR "AI-powered" aftermarket service',
    }

    per_domain_searches = []
    for i, d in enumerate(domains, 1):
        kw = _DOMAIN_KEYWORDS.get(d, f'"{d.lower()}" software technology')
        per_domain_searches.append(f'{i*3-2}. "{company_name}" {kw} 2022 OR 2023 OR 2024 OR 2025')
        per_domain_searches.append(f'{i*3-1}. "{company_name}" {kw} vendor implementation case study OR deployment')
        if target_vendor:
            per_domain_searches.append(f'{i*3}.  "{company_name}" "{target_vendor}" {kw} — does {target_vendor} have a deployment here?')
        else:
            per_domain_searches.append(f'{i*3}.  site:linkedin.com/jobs "{company_name}" {d.split()[0].lower()} — job titles reveal active systems')

    per_domain_block = "\n".join(per_domain_searches)

    vendor_searches = ""
    if target_vendor:
        vendor_searches = f"""
VENDOR MATCH SEARCHES — run for every domain:
- "{company_name}" "{target_vendor}" — any deployment, partnership, or pilot
- "{target_vendor}" "{company_name}" case study OR customer OR deployment OR implementation
- site:linkedin.com "{company_name}" "{target_vendor}" — employee mentions of the vendor
- "{target_vendor}" customer "{company_name}" press release OR announcement
"""

    return f"""You are an aftermarket service technology analyst with live Google Search.

COMPANY: {company_name}{ind}
{"TARGET VENDOR: " + target_vendor if target_vendor else ""}

Your task: find every technology platform, software product, and vendor deployment at {company_name} across these aftermarket domains:
{domains_list}

MANDATORY SEARCHES — run EVERY search below before answering:

BROAD COMPANY SEARCHES (run first to establish baseline):
A1. "{company_name}" aftermarket service technology platform software vendor list 2024
A2. "{company_name}" ERP OR CRM OR DMS OR FSM OR warranty OR telematics software implementation SAP OR Salesforce OR Oracle OR Microsoft OR ServiceMax OR ServiceNow
A3. "{company_name}" digital transformation aftermarket service technology investment 2022 OR 2023 OR 2024 OR 2025
A4. site:linkedin.com/jobs "{company_name}" aftermarket OR service OR warranty OR parts systems software
A5. "{company_name}" technology vendor partner case study OR implementation OR deployment
{vendor_searches}
PER-DOMAIN SEARCHES (run for each domain in scope):
{per_domain_block}

For each technology found, return ONE JSON object per domain × technology combination.

Return ONLY a JSON array — aim for MAXIMUM coverage, one row per technology per domain:
[
  {{
    "domain": "<must exactly match one of: {', '.join(domains)}>",
    "capability": "<specific sub-capability e.g. 'Claim Submission', 'Parts Ordering', 'Work Order Management', 'Fleet Telematics', 'Dealer Portal'>",
    "technology": "<exact product/vendor name — never generic e.g. 'Tavant Warranty', 'SAP S/4HANA', 'Salesforce Service Cloud', 'ServiceMax', 'Trimble'>",
    "vendor_match": "<if target_vendor is set: 'Deployed' | 'Pilot' | 'Competitor' | 'Not Found' | 'Unknown' — else '-'>",
    "use_case": "<one sentence: how {company_name} uses this technology>",
    "install_base": "<scope e.g. '~500 dealer users', 'Global', 'North America', 'Enterprise-wide'>",
    "source": "<URL to press release, case study, job posting, or LinkedIn — or '-'>"
  }}
]

Rules:
- Cover ALL {len(domains)} domains — include every technology found per domain, multiple rows if multiple tools
- Only include a technology if there is real evidence {company_name} uses it (press release, job posting, case study, LinkedIn)
- Do NOT speculate or add rows without evidence
- For vendor_match: 'Deployed' = confirmed {target_vendor + " deployment" if target_vendor else "N/A"}, 'Competitor' = competing product in same domain
- Return ONLY the raw JSON array. No prose. No markdown."""


def _cap_prompt(company_name: str, domain: str, industry: str, target_vendor: str = "") -> str:
    """Single-domain cap prompt — kept for backward compat but not used in main pipeline."""
    return _cap_batch_prompt(company_name, [domain], industry, target_vendor)


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

    return f"""You are an IT financial analyst synthesising research findings. No web search needed — use data provided.

COMPANY: {company_name}{ind}

KNOWN TECHNOLOGY STACK (from research):
{cap_summary or "  (No capability data available — use industry benchmarks)"}

VERIFIED AGGREGATE IT SPEND (use as anchor for all module estimates — totals must add up):
{agg_summary or "  (No aggregate data — apportion using industry benchmarks)"}

TASK: Estimate annual technology spend per aftermarket module.

RULES:
1. Use vendor/product pricing from the KNOWN TECHNOLOGY STACK where available
   — SaaS/subscription: typical 2024/2025 list pricing for named products
   — On-premise: licence + maintenance (typically 18–22% of licence per year)
2. Module estimates must be proportional to and consistent with AGGREGATE IT SPEND above
3. All 9 module estimates should sum to roughly the Aftermarket IT Spend figure in AGGREGATE
4. market_benchmark: use verified 2024/2025 industry benchmarks for comparable companies by revenue tier

Example spend_math: "SAP S/4HANA Warranty module: ~$3M licence + $600k annual maintenance + $800k integration support = $4.4M. Scale for company size: $6M–$9M"

Return ONLY a JSON array:
[
  {{
    "domain": "Warranty Management",
    "current_spend": "$6M–$9M",
    "spend_math": "<specific vendor pricing × scale + maintenance/support = total>",
    "market_benchmark": "<verified 2024/2025 benchmark range for comparable companies>"
  }}
]

Modules: Warranty Management, Service & Repair Operations, Parts & Inventory Management, Field Service Management, Dealer & Distribution Network, Telematics & Connected Products, Predictive Maintenance & IoT, Analytics & Business Intelligence, AI & Automation.

Return ONLY the JSON array starting with [. No prose."""


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


_INDUSTRY_BENCHMARKS = {
    # (it_pct_of_revenue, ai_pct_of_it, cloud_pct_of_it, aftermarket_pct_of_it, source_label)
    "automotive oem":          (2.5, 12, 28, 22, "Gartner 2024 Automotive"),
    "automotive":              (2.5, 12, 28, 22, "Gartner 2024 Automotive"),
    "truck":                   (2.5, 12, 28, 22, "Gartner 2024 Automotive"),
    "commercial vehicle":      (2.5, 12, 28, 22, "Gartner 2024 Automotive"),
    "industrial manufacturing":(2.8, 11, 27, 20, "Gartner 2024 Industrial Manufacturing"),
    "manufacturing":           (2.8, 11, 27, 20, "Gartner 2024 Industrial Manufacturing"),
    "aerospace":               (3.2, 13, 26, 25, "Gartner 2024 Aerospace & Defense"),
    "aerospace & defense":     (3.2, 13, 26, 25, "Gartner 2024 Aerospace & Defense"),
    "defense":                 (3.2, 13, 26, 25, "Gartner 2024 Aerospace & Defense"),
    "medical devices":         (4.5, 14, 30, 18, "Gartner 2024 Medical Devices"),
    "healthcare":              (5.0, 15, 32, 15, "Gartner 2024 Healthcare"),
    "construction equipment":  (2.3, 10, 25, 22, "Gartner 2024 Construction/Industrial"),
    "construction":            (2.3, 10, 25, 22, "Gartner 2024 Construction/Industrial"),
    "energy":                  (3.0, 12, 28, 18, "Gartner 2024 Energy & Utilities"),
    "utilities":               (3.0, 12, 28, 18, "Gartner 2024 Energy & Utilities"),
    "oil & gas":               (2.8, 11, 26, 20, "Gartner 2024 Oil & Gas"),
    "retail":                  (2.2, 13, 35, 10, "Gartner 2024 Retail"),
    "financial services":      (7.5, 18, 38, 10, "Gartner 2024 Financial Services"),
    "banking":                 (7.5, 18, 38, 10, "Gartner 2024 Banking"),
    "insurance":               (5.5, 16, 33, 12, "Gartner 2024 Insurance"),
    "technology":              (9.0, 20, 40,  8, "Gartner 2024 Technology"),
    "software":                (9.0, 20, 40,  8, "Gartner 2024 Technology"),
    "telecom":                 (4.5, 15, 32, 12, "Gartner 2024 Telecom"),
    "logistics":               (2.5, 12, 30, 18, "Gartner 2024 Transportation/Logistics"),
    "transportation":          (2.5, 12, 30, 18, "Gartner 2024 Transportation/Logistics"),
}
_DEFAULT_BENCHMARK = (2.8, 12, 30, 20, "Gartner 2024 Cross-Industry Average")


def _get_benchmark(industry: str) -> tuple:
    """Return (it_pct, ai_pct, cloud_pct, aftermarket_pct, source) for the given industry."""
    key = industry.lower().strip()
    for k, v in _INDUSTRY_BENCHMARKS.items():
        if k in key or key in k:
            return v
    return _DEFAULT_BENCHMARK


def _aggregate_spend_prompt(company_name: str, industry: str) -> str:
    ind = f" ({industry})" if industry else ""
    it_pct, ai_pct, cloud_pct, am_pct, bench_src = _get_benchmark(industry)
    return f"""You are an enterprise IT financial analyst. Use Google Search. Follow EXACTLY the steps below — do not deviate.

COMPANY: {company_name}{ind}

━━ STEP 1: FIND VERIFIED ANNUAL REVENUE ━━
Search: "{company_name}" annual revenue 2024 OR 2025 site:annualreports.com OR site:ir.*.com OR site:businesswire.com OR site:prnewswire.com
Search: "{company_name}" fiscal year 2024 earnings results revenue

Record: REVENUE = <exact figure e.g. $16.2B> from <source URL>
If you find a range (e.g. $15.8B–$16.5B), use the MIDPOINT.
If revenue is not found after searching, state clearly and estimate from industry context.

━━ STEP 2: APPLY FIXED BENCHMARK MULTIPLIERS FOR THIS INDUSTRY ━━
Source: {bench_src}
Use EXACTLY these multipliers (do not change them — consistency is critical):
  IT Spend       = REVENUE × {it_pct}%
  AI Spend       = IT Spend × {ai_pct}%
  Cloud Spend    = IT Spend × {cloud_pct}%
  Aftermarket IT = IT Spend × {am_pct}%

━━ STEP 3: SHOW THE ARITHMETIC ━━
Write out each calculation explicitly:
  IT Spend: $REVENUE × {it_pct}% = $RESULT → round to nearest $5M → "$X–$Y" (±10% range)
  AI Spend: $IT × {ai_pct}% = $RESULT → "$X–$Y"
  Cloud:    $IT × {cloud_pct}% = $RESULT → "$X–$Y"
  Aftermarket IT: $IT × {am_pct}% = $RESULT → "$X–$Y"

━━ OUTPUT ━━
Return ONLY this JSON array — 4 objects, no extras:
[
  {{
    "spend_type": "IT Spend",
    "estimate": "<$X–$Y computed above>",
    "basis": "REVENUE × {it_pct}% = <exact calc>. Revenue: <verified figure> ({bench_src})",
    "source": "<URL to revenue source>"
  }},
  {{
    "spend_type": "AI Spend",
    "estimate": "<$X–$Y>",
    "basis": "IT Spend × {ai_pct}% = <exact calc> (IDC 2025 AI within IT budget)",
    "source": "<URL to IDC or Gartner benchmark>"
  }},
  {{
    "spend_type": "Cloud Spend",
    "estimate": "<$X–$Y>",
    "basis": "IT Spend × {cloud_pct}% = <exact calc> (Flexera 2024 State of Cloud)",
    "source": "<URL to Flexera benchmark>"
  }},
  {{
    "spend_type": "Aftermarket IT Spend",
    "estimate": "<$X–$Y>",
    "basis": "IT Spend × {am_pct}% = <exact calc> (aftermarket/service operations share of IT budget)",
    "source": "<URL to revenue source>"
  }}
]
Return ONLY the JSON array. Same company = same numbers every time."""


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


def _build_vendor_rel_map(cap_data: list, vendor: str) -> dict:
    """
    Pre-compute existing relationship score per domain from cap_data.
    Returns dict: domain -> {"score": int, "evidence": str, "techs": list}
    This is computed in Python so the model cannot override it via search.
    """
    from collections import defaultdict
    vendor_lower = vendor.lower() if vendor else ""
    by_domain: dict = defaultdict(lambda: {"techs": [], "vendor_hit": False, "signals": []})
    for r in (cap_data or []):
        d = r.get("domain", "")
        if not d:
            continue
        tech = (r.get("technology") or "").strip()
        rel  = (r.get("vendor_relationship") or r.get("relationship") or "").lower()
        if tech and tech not in ("Unknown/Not Disclosed", "-", "Unknown"):
            by_domain[d]["techs"].append(tech)
        if vendor_lower and (vendor_lower in tech.lower() or vendor_lower in rel):
            by_domain[d]["vendor_hit"] = True
            by_domain[d]["signals"].append(tech or rel)

    result = {}
    for d, info in by_domain.items():
        if info["vendor_hit"]:
            sig = ", ".join(info["signals"][:2])
            result[d] = {"score": 85, "evidence": f"{vendor} DEPLOYED at {d} ({sig})", "techs": info["techs"]}
        elif info["techs"]:
            techs = ", ".join(info["techs"][:3])
            result[d] = {"score": 10, "evidence": f"{vendor} not found — current tech: {techs}", "techs": info["techs"]}
        else:
            result[d] = {"score": 5, "evidence": "No capability data available", "techs": []}
    return result


def _parse_spend_millions(spend_str: str) -> tuple[float, float] | None:
    """Parse '$6M–$9M', '$1.2B–$1.8B', '$45M' → (low_M, high_M). Returns None if unparseable."""
    if not spend_str:
        return None
    s = spend_str.replace("–", "-").replace("—", "-").replace(",", "")
    hits = re.findall(r'\$?([\d.]+)\s*([BbMm])', s)
    vals = []
    for num, unit in hits:
        v = float(num)
        if unit.upper() == "B":
            v *= 1000.0
        vals.append(v)
    if len(vals) >= 2:
        return (min(vals[0], vals[1]), max(vals[0], vals[1]))
    if len(vals) == 1:
        return (vals[0] * 0.85, vals[0] * 1.15)
    return None


def _normalize_domain(d: str) -> str:
    """Lowercase, strip punctuation/spaces for fuzzy domain matching."""
    import re as _re
    return _re.sub(r'[^a-z0-9]', '', d.lower())


def _recalculate_tam(readiness_rows: list, spend_rows: list) -> list:
    """
    Post-process readiness rows: replace total_domain_spend, vendor_adjusted_tam,
    and tam_rationale using actual spend_module figures.
    Also recalculate weighted_readiness from the 5 component scores to ensure arithmetic is correct.
    """
    # Build spend_map with both exact and normalised keys for fuzzy lookup
    spend_map_exact: dict = {}
    spend_map_norm: dict = {}
    for r in (spend_rows or []):
        d = r.get("domain", "")
        cs = r.get("current_spend", "")
        if d and cs:
            spend_map_exact[d] = cs
            spend_map_norm[_normalize_domain(d)] = cs

    def _lookup_spend(domain: str) -> str:
        return spend_map_exact.get(domain) or spend_map_norm.get(_normalize_domain(domain), "")

    out = []
    for row in readiness_rows:
        row = dict(row)
        domain = row.get("domain", "")

        # ── Recalculate weighted_readiness from component scores ──────────────
        try:
            er = float(row.get("existing_rel_score", 0))
            it = float(row.get("it_signals_score", 0))
            cs = float(row.get("company_signals_score", 0))
            es = float(row.get("exec_signals_score", 0))
            bs = float(row.get("budget_signals_score", 0))
            computed = round(er*0.30 + it*0.15 + cs*0.20 + es*0.15 + bs*0.20)
            row["weighted_readiness"] = computed
            row["displacement_opp"] = "High" if computed >= 65 else ("Medium" if computed >= 40 else "Low")
        except Exception:
            computed = int(row.get("weighted_readiness", 0))

        # ── Replace TAM using actual spend_module figure ──────────────────────
        actual_spend = _lookup_spend(domain)
        if actual_spend:
            row["total_domain_spend"] = actual_spend
            parsed = _parse_spend_millions(actual_spend)
            if parsed and computed > 0:
                low_m, high_m = parsed
                adj_low = low_m * computed / 100
                adj_high = high_m * computed / 100
                mid = (low_m + high_m) / 2
                adj_mid = mid * computed / 100

                def _fmt(v: float) -> str:
                    if v >= 1000:
                        return f"${v/1000:.2f}B"
                    if v >= 10:
                        return f"${v:.0f}M"
                    return f"${v:.1f}M"

                row["vendor_adjusted_tam"] = f"{_fmt(adj_low)}–{_fmt(adj_high)}"
                row["tam_rationale"] = (
                    f"Midpoint = ({_fmt(low_m)}+{_fmt(high_m)})/2 = {_fmt(mid)} "
                    f"× {computed}% readiness = {_fmt(adj_mid)} vendor-adjusted TAM"
                )
        out.append(row)
    return out


def _readiness_score_prompt(company_name: str, industry: str, target_vendor: str,
                             research_text: str, cap_data: list,
                             modules: list[str], spend_lines: str, agg_ref: str,
                             vendor_rel_map: dict | None = None) -> str:
    """Search-grounded scoring. existing_rel scores are PRE-COMPUTED from cap_data and locked."""
    vendor = target_vendor or "the vendor"
    n = len(modules)
    modules_list = ", ".join(modules)

    # Pre-computed relationship scores — inject as LOCKED values
    if vendor_rel_map is None:
        vendor_rel_map = _build_vendor_rel_map(cap_data, vendor)

    locked_lines = []
    for m in modules:
        info = vendor_rel_map.get(m, {"score": 5, "evidence": "No capability data", "techs": []})
        locked_lines.append(
            f"  {m}: existing_rel_score={info['score']} | evidence=\"{info['evidence']}\""
        )
    locked_rel = "\n".join(locked_lines)

    pre_research = research_text.strip() if research_text and research_text.strip() else ""

    return f"""You are a vendor displacement readiness analyst. Use Google Search to find signal evidence.

COMPANY: {company_name} ({industry})
VENDOR BEING EVALUATED: {vendor}

━━ EXISTING RELATIONSHIP SCORES — LOCKED, DO NOT CHANGE ━━
These are computed from verified capability research. Copy them exactly into your JSON output.
{locked_rel}

━━ PRE-FETCHED SIGNALS (supplement with Google Search) ━━
{pre_research or "Not available — use Google Search for each signal category below."}

━━ SEARCH INSTRUCTIONS ━━
For each module listed, search Google for REAL recent (2022–2025) evidence for {company_name}:
• IT_INVESTMENT: ERP/platform go-lives, vendor contracts, system implementations
• EXEC_AGENDA: CTO/CIO/CDO quotes, digital strategy announcements
• BUDGET_SIGNAL: IT budget announcements, RFPs, technology capex
• HIRING_SIGNAL: open roles or hiring in technology/digital for this domain

Record the URL for every finding.

━━ SCORING RULES ━━
- existing_rel_score: LOCKED — copy exactly from EXISTING RELATIONSHIP SCORES above
- existing_rel_evidence: LOCKED — copy exactly from above
- it_signals_score (0.15): evidence from IT_INVESTMENT + HIRING_SIGNAL
- company_signals_score (0.20): growth, M&A, transformation signals
- exec_signals_score (0.15): EXEC_AGENDA evidence
- budget_signals_score (0.20): BUDGET_SIGNAL evidence
- weighted_readiness = (existing_rel×0.30)+(it×0.15)+(company×0.20)+(exec×0.15)+(budget×0.20)
- displacement_opp: High if ≥65, Medium if 40–64, Low if <40
- total_domain_spend and vendor_adjusted_tam: leave as "TBD" — will be calculated server-side

MODULES: {modules_list}

Return ONLY a valid JSON array — exactly {n} objects:
[
  {{
    "domain": "<module name>",
    "current_system": "<known system or Unknown>",
    "existing_rel_score": <LOCKED value from above>,
    "existing_rel_evidence": "<LOCKED evidence from above>",
    "it_signals_score": <0-100>,
    "it_signals_evidence": "<specific finding. Source: URL>",
    "company_signals_score": <0-100>,
    "company_signals_evidence": "<specific finding. Source: URL>",
    "exec_signals_score": <0-100>,
    "exec_signals_evidence": "<specific finding. Source: URL>",
    "budget_signals_score": <0-100>,
    "budget_signals_evidence": "<specific finding. Source: URL>",
    "weighted_readiness": <computed>,
    "displacement_opp": "<High|Medium|Low>",
    "total_domain_spend": "TBD",
    "vendor_adjusted_tam": "TBD",
    "tam_rationale": "TBD"
  }}
]
If no evidence found for a signal: "No public evidence found."
Return ONLY the JSON array. No prose. No markdown."""


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
    from usage_logger import new_run_id, get_usage_by_run
    run_id = new_run_id()

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

    # Phase 1: Fire search-grounded calls in parallel
    # Capabilities: 4 batched calls (3 domains each) instead of 12 individual calls.
    # This reduces simultaneous Gemini requests from 15 to 7, preventing thread pool
    # exhaustion and rate limiting that caused "0/12 done" stalls.
    _CAP_BATCHES = [AFTERMARKET_DOMAINS[i:i+3] for i in range(0, len(AFTERMARKET_DOMAINS), 3)]
    cap_futures = [
        loop.run_in_executor(None, _gemini_call_sync,
                             _cap_batch_prompt(company_name, batch, industry, target_vendor),
                             True, f"cap_batch_{bi}", 24576, "gemini-2.5-flash", False, 0.15, run_id)
        for bi, batch in enumerate(_CAP_BATCHES)
    ] if "capabilities" in run else []
    agg_future  = loop.run_in_executor(None, _gemini_call_sync, _aggregate_spend_prompt(company_name, industry), True, "agg_spend", 4096, "gemini-2.5-flash", False, 0.0, run_id) if "agg_spend" in run else None
    deals_future = loop.run_in_executor(None, _gemini_call_sync, _spend_deals_prompt(company_name, industry), True, "spend_deals", 16384, "gemini-2.5-flash", False, 0.15, run_id) if "spend_deals" in run else None

    # If spend_module (but NOT readiness-only) is requested without capabilities,
    # fire a lightweight search to get context for synthesis.
    # Readiness-only skips context: it uses Google Search grounding internally and
    # serial context fetching (~105s) eats into the readiness timeout budget.
    needs_context = ("spend_module" in run) and "capabilities" not in run
    context_future = loop.run_in_executor(
        None, _gemini_call_sync,
        _cap_prompt(company_name, "Warranty Management", industry, target_vendor),
        True, "cap_context_lite", 16384, "gemini-2.5-flash", False, 0.15, run_id
    ) if needs_context else None

    spend_future   = None
    _ALL_MODULES = ["Warranty Management", "Service & Repair Operations", "Parts & Inventory Management",
                    "Field Service Management", "Dealer & Distribution Network",
                    "Telematics & Connected Products", "Predictive Maintenance & IoT",
                    "Analytics & Business Intelligence", "AI & Automation"]
    _BATCH_A = _ALL_MODULES[:5]
    _BATCH_B = _ALL_MODULES[5:]

    # Readiness pipeline (no-search, model-knowledge scoring):
    # use_search=False — eliminates all 503 retries and the 300s budget risk.
    # Model knowledge of established companies is accurate for scoring.
    # Inference completes in 10-20s vs 160s+ for search-grounded calls.
    # Research step still runs in background to enrich evidence text if it finishes in time.
    if "readiness" in run:
        ready_research_future = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_research_prompt(company_name, industry),
            True, "readiness_research", 4096, "gemini-2.5-flash", True, 0.15, run_id,  # return_raw=True
        )
    else:
        ready_research_future = None

    yield {"type": "heartbeat", "message": "🌐 Phase 1: Researching capabilities & tech spend in parallel…"}
    await asyncio.sleep(0)

    # Collect capability rows (stream as each batch completes)
    all_cap_rows = []
    pending_caps = list(enumerate(cap_futures))
    elapsed = 0
    N_BATCHES = len(_CAP_BATCHES) if cap_futures else 0
    PARALLEL_TIMEOUT = 160  # 4 batched calls — each now runs 30+ searches, needs more time

    while pending_caps and elapsed < PARALLEL_TIMEOUT:
        try:
            await asyncio.sleep(8)
            elapsed += 8
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
                        logger.error(f"Cap batch {idx} error: {e}")
            pending_caps = [(i, f) for i, f in pending_caps if i not in newly_done]
            done_batches = N_BATCHES - len(pending_caps)
            done_domains = done_batches * 3
            if pending_caps:
                yield {"type": "heartbeat", "message": f"🌐 Researching… {done_domains}/{len(AFTERMARKET_DOMAINS)} domains done, {len(all_cap_rows)} capabilities found ({elapsed}s)"}
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

    yield {"type": "heartbeat", "message": f"✅ Capabilities: {len(all_cap_rows)} rows — launching Phase 2…"}
    await asyncio.sleep(0)

    # ── Collect agg_spend (already running since t=0, likely done) ────────────
    if agg_future:
        agg_rows = await _collect_future(agg_future, "agg_spend", timeout=90)
    elif "spend_module" in run:
        yield {"type": "heartbeat", "message": "💰 Fetching spend context…"}
        await asyncio.sleep(0)
        agg_rows = await _run_async(_aggregate_spend_prompt(company_name, industry), True, "agg_context", timeout=90, run_id=run_id)
    else:
        agg_rows = []
    if "agg_spend" in run:
        for row in (agg_rows if isinstance(agg_rows, list) else []):
            if isinstance(row, dict):
                yield {"type": "aggregate_spend_row", "row": row}
                await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ Aggregate spend ready — launching spend synthesis + readiness in parallel…"}
    await asyncio.sleep(0)

    # ── NOW collect research (was running since t=0) ──────────────────────────
    # Budget: research started at t=0, cap loop ran ~90s, so research has had ~90s already.
    # Give it 30s more — if not done by now it's probably stuck.
    research_text = ""
    if ready_research_future and "readiness" in run:
        try:
            research_text = await asyncio.wait_for(ready_research_future, timeout=30)
            research_text = research_text if isinstance(research_text, str) else ""
            logger.info(f"readiness_research collected: {len(research_text)} chars")
        except asyncio.TimeoutError:
            research_text = ""
            logger.warning("readiness_research: still not done after 120s total — using cap_data only")
        except Exception as e:
            research_text = ""
            logger.warning(f"readiness_research failed: {e}")

    # ── Launch spend_module AND readiness scoring AT THE SAME TIME ────────────
    # Critical: do NOT wait for spend_module before starting readiness.
    # Use agg_rows as spend reference for readiness TAM (accurate enough).
    # spend_module rows stream out independently once done.
    agg_list = agg_rows if isinstance(agg_rows, list) else []
    effective_spend = existing_spend_rows or []  # caller passes these when regen-only

    if "spend_module" in run:
        spend_future = loop.run_in_executor(
            None, _gemini_call_sync,
            _spend_module_prompt(company_name, industry, all_cap_rows, agg_list),
            False, "spend_module", 8192, "gemini-2.5-flash", False, 0.0, run_id
        )

    # Build spend lines from agg_rows for readiness TAM (module-level spend not yet available)
    spend_lines_for_ready = "\n".join(
        f"  {r.get('domain','')}: {r.get('current_spend','')}"
        for r in effective_spend if r.get("domain") and r.get("current_spend")
    ) or "  (use industry benchmarks — spend_module not yet available)"
    agg_ref_for_ready = " | ".join(
        f"{r.get('spend_type')}: {r.get('estimate','?')}"
        for r in agg_list[:4]
    )

    # ── Pre-compute vendor relationship map from cap_data (Python, not LLM) ────
    vendor_rel_map = _build_vendor_rel_map(all_cap_rows, target_vendor) if "readiness" in run else {}

    # ── Launch readiness scoring NOW (search-grounded, parallel with spend_module) ──
    readiness_rows = []
    if "readiness" in run:
        yield {"type": "heartbeat", "message": "🎯 Scoring readiness with live search (parallel with spend synthesis)…"}
        await asyncio.sleep(0)

        score_a = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_score_prompt(company_name, industry, target_vendor,
                                    research_text, all_cap_rows, _BATCH_A,
                                    spend_lines_for_ready, agg_ref_for_ready,
                                    vendor_rel_map),
            True, "readiness_score_a", 16384, "gemini-2.5-flash", False, 0.15, run_id,
        )
        score_b = loop.run_in_executor(
            None, _gemini_call_sync,
            _readiness_score_prompt(company_name, industry, target_vendor,
                                    research_text, all_cap_rows, _BATCH_B,
                                    spend_lines_for_ready, agg_ref_for_ready,
                                    vendor_rel_map),
            True, "readiness_score_b", 16384, "gemini-2.5-flash", False, 0.15, run_id,
        )

    # ── Collect IT deals (already running since t=0) ──────────────────────────
    spend_deal_rows = await _collect_future(deals_future, "spend_deals", timeout=90) if deals_future else []
    for row in (spend_deal_rows if isinstance(spend_deal_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_deal_row", "row": row}
            await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ IT deals: {len(spend_deal_rows) if isinstance(spend_deal_rows, list) else 0} deals"}
    await asyncio.sleep(0)

    # ── Collect spend_module ──────────────────────────────────────────────────
    spend_rows = await _collect_future(spend_future, "spend_module", timeout=120) if spend_future else []
    for row in (spend_rows if isinstance(spend_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "spend_module_row", "row": row}
            await asyncio.sleep(0.04)
    yield {"type": "heartbeat", "message": f"✅ Spend by module: {len(spend_rows) if isinstance(spend_rows, list) else 0} rows"}
    await asyncio.sleep(0)

    # ── Collect readiness scoring (was running in parallel with spend_module) ──
    # ── Step 4: Readiness Matrix + TAM ───────────────────────────────────────
    yield {"type": "heartbeat", "message": "🎯 Collecting readiness results…"}
    await asyncio.sleep(0)

    if "readiness" in run:
        collected: list = []
        try:
            # Both batches have been running since before spend_module — should be nearly done
            remaining_budget = 180  # generous: they've had ~120s head start already
            results_ab = await asyncio.wait_for(
                asyncio.gather(score_a, score_b, return_exceptions=True),
                timeout=remaining_budget,
            )
            for i, res in enumerate(results_ab):
                label_s = f"readiness_score_{'a' if i==0 else 'b'}"
                if isinstance(res, Exception):
                    logger.error(f"{label_s} raised: {res}")
                elif isinstance(res, list) and res:
                    collected.extend(res)
                    logger.info(f"{label_s} returned {len(res)} rows")
                else:
                    logger.warning(f"{label_s} empty")
        except asyncio.TimeoutError:
            logger.warning("readiness scoring timed out after 180s")

        # ── Retry only empty batches ──────────────────────────────────────────
        got_a = any(r.get("domain","") in _BATCH_A for r in collected)
        got_b = any(r.get("domain","") in _BATCH_B for r in collected)
        if not got_a or not got_b:
            yield {"type": "heartbeat", "message": "🔄 Retrying empty readiness batch(es) with search…"}
            await asyncio.sleep(0)
            retry_futs, retry_labels = [], []
            if not got_a:
                retry_futs.append(loop.run_in_executor(
                    None, _gemini_call_sync,
                    _readiness_score_prompt(company_name, industry, target_vendor,
                                            research_text, all_cap_rows, _BATCH_A,
                                            spend_lines_for_ready, agg_ref_for_ready,
                                            vendor_rel_map),
                    True, "readiness_retry_a", 16384, "gemini-2.5-flash", False, 0.15, run_id,
                ))
                retry_labels.append("retry_a")
            if not got_b:
                retry_futs.append(loop.run_in_executor(
                    None, _gemini_call_sync,
                    _readiness_score_prompt(company_name, industry, target_vendor,
                                            research_text, all_cap_rows, _BATCH_B,
                                            spend_lines_for_ready, agg_ref_for_ready,
                                            vendor_rel_map),
                    True, "readiness_retry_b", 16384, "gemini-2.5-flash", False, 0.15, run_id,
                ))
                retry_labels.append("retry_b")
            try:
                retry_results = await asyncio.wait_for(
                    asyncio.gather(*retry_futs, return_exceptions=True), timeout=120,
                )
                for lbl, res in zip(retry_labels, retry_results):
                    if isinstance(res, list) and res:
                        collected.extend(res)
                        logger.info(f"readiness_{lbl} returned {len(res)} rows")
            except asyncio.TimeoutError:
                logger.warning("readiness retry timed out after 120s")

        _MODULE_ORDER = {m: i for i, m in enumerate(_ALL_MODULES)}
        collected_sorted = sorted(collected, key=lambda r: _MODULE_ORDER.get(r.get("domain", ""), 99))

        # ── Post-process: lock TAM from spend_module + recalculate weighted score ──
        # spend_rows is now available (collected before we awaited scoring results)
        effective_spend_final = spend_rows if isinstance(spend_rows, list) and spend_rows else (existing_spend_rows or [])
        readiness_rows = _recalculate_tam(collected_sorted, effective_spend_final)
        spend_domains = [r.get('domain') for r in effective_spend_final]
        logger.info(f"Readiness post-process: {len(readiness_rows)} rows, spend_map domains: {spend_domains}")
        for r in readiness_rows:
            if r.get("vendor_adjusted_tam") in ("TBD", "", None):
                logger.warning(f"TAM still TBD for domain='{r.get('domain')}' — no spend match in {spend_domains}")

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

    gap_rows = await _run_async(_gap_prompt(company_name, industry, []), True, "gaps", timeout=100, run_id=run_id) if "gaps" in run else []
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

        comp_rows = await _run_async(_comp_prompt(company_name, industry, competitors), True, "competitive", timeout=100, run_id=run_id)
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
        "usage": get_usage_by_run(run_id),
        "run_id": run_id,
    }
