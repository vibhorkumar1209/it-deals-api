"""Competitive Intelligence Pipeline — Gemini-powered analysis."""

import asyncio
import json
import logging
import os
import re
import time as _time
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")
CALL_TIMEOUT = 180          # seconds per Gemini call
COMPANY_SEM  = 4            # max concurrent companies
MODULE_SEM   = 3            # max concurrent modules per company
TOTAL_BUDGET = 480          # seconds for full analysis

# ── Module definitions ────────────────────────────────────────────────────────

MODULES = {
    "core":      "Company Overview",
    "portfolio": "Service & Product Portfolio",
    "technology":"Technology & Innovation",
    "customer":  "Customer Base & Retention",
    "financial": "Financial Performance",
    "gtm":       "Go-to-Market Strategy",
    "talent":    "Talent & Culture",
    "brand":     "Brand & Analyst Presence",
    "stack":     "Tech Stack & Alliances",
}

# ── Gemini helpers ────────────────────────────────────────────────────────────

def _gemini_call_sync(prompt: str, use_search: bool = True, max_tokens: int = 8192) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=GOOGLE_AI_KEY)
    config_kwargs: dict = {
        "max_output_tokens": max_tokens,
        "temperature": 0.1,
        "thinking_config": types.ThinkingConfig(thinking_budget=0),
    }
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    retries = 2
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            )
            return resp.text or ""
        except Exception as e:
            if attempt == retries:
                raise
            _time.sleep(4 * (attempt + 1))
    return ""


def _parse_json_from_text(text: str) -> dict | list | None:
    """Extract first JSON object or array from text."""
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    # Try to extract JSON block from markdown code fence
    m = re.search(r"```(?:json)?\s*([\[{].*?)\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(1)))
        except Exception:
            pass
    # Fallback: grab first { ... } or [ ... ]
    for pattern in (r"(\{.*\})", r"(\[.*\])"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(re.sub(r",\s*([\]}])", r"\1", m.group(1)))
            except Exception:
                pass
    return None


# ── Competitor Discovery ──────────────────────────────────────────────────────

def _discovery_prompt(target_company: str, target_domain: str, industry_context: str = "", technology_context: str = "") -> str:
    domain_hint = f"(domain: {target_domain})" if target_domain else ""
    has_focus = bool(industry_context or technology_context)

    if has_focus:
        focus_parts = []
        if industry_context:
            focus_parts.append(industry_context)
        if technology_context:
            focus_parts.append(technology_context)
        focus_desc = " / ".join(focus_parts)

        # Build industry-first search queries — market space is the primary axis
        search_lines = []
        if industry_context and technology_context:
            search_lines.append(f'top companies in {industry_context} {technology_context} market 2024 2025')
            search_lines.append(f'leading vendors {technology_context} for {industry_context}')
            search_lines.append(f'Gartner Forrester {industry_context} {technology_context} competitive landscape')
        elif industry_context:
            search_lines.append(f'top companies in {industry_context} market 2024 2025')
            search_lines.append(f'leading players {industry_context} industry competitive landscape')
            search_lines.append(f'Gartner Forrester {industry_context} market leaders report')
        elif technology_context:
            search_lines.append(f'top {technology_context} vendors companies 2024 2025')
            search_lines.append(f'leading {technology_context} providers competitive landscape')
            search_lines.append(f'Gartner Magic Quadrant {technology_context} market leaders')
        searches = "\n".join(f"- {l}" for l in search_lines)

        return f"""You are a competitive intelligence analyst.

Step 1 — Identify the leading companies in this market space:
{focus_desc}

Search for:
{searches}

Build a list of 8-10 companies that are genuine market participants in {focus_desc}. These are companies with real products, revenues, or customers in this space — not companies tangentially related to it.

Step 2 — For each company you found, note how they overlap or compete with {target_company} {domain_hint} in {focus_desc}.

The target company {target_company} may or may not be a major player in {focus_desc}. Your list must reflect the actual competitive landscape of {focus_desc}, NOT the general list of {target_company}'s overall competitors.

Return ONLY a JSON array (no markdown, no explanation):
[
  {{
    "name": "Company Full Name",
    "domain": "company.com",
    "descriptor": "One sentence: what they do in {focus_desc} and how they relate to {target_company}"
  }},
  ...
]

Include 8-10 companies. Order by market prominence in {focus_desc} (largest/most established first)."""

    # No focus context — general discovery
    return f"""You are a competitive intelligence analyst. Identify the 8-10 most direct overall competitors of {target_company} {domain_hint}.

Search for:
- "{target_company}" competitors rivals alternatives
- Companies with overlapping product/service offerings to {target_company}
- Recent analyst reports positioning competitors against {target_company}
- Industry benchmarking lists including {target_company}

Return ONLY a JSON array (no markdown, no explanation):
[
  {{
    "name": "Company Full Name",
    "domain": "company.com",
    "descriptor": "One sentence describing how they compete with {target_company}"
  }},
  ...
]

Include 8-10 competitors. Order by relevance (most direct competitors first)."""


async def discover_competitors(target_company: str, target_domain: str = "", industry_context: str = "", technology_context: str = "") -> list[dict]:
    """Return list of {{name, domain, descriptor}} dicts."""
    prompt = _discovery_prompt(target_company, target_domain, industry_context, technology_context)
    text = await asyncio.wait_for(
        asyncio.to_thread(_gemini_call_sync, prompt, True, 4096),
        timeout=CALL_TIMEOUT,
    )
    result = _parse_json_from_text(text)
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict) and r.get("name")]
    return []


# ── Module Prompts ────────────────────────────────────────────────────────────

def _core_prompt(company: str) -> str:
    return f"""Research the company overview for {company}.

Search for: "{company}" company overview, "{company}" annual report, "{company}" headquarters employees revenue, "{company}" business model.

Return ONLY valid JSON (no markdown):
{{
  "legal_name": "",
  "hq_location": "",
  "founded": "",
  "total_employees": "",
  "annual_revenue": "",
  "revenue_growth_yoy": "",
  "gross_margin": "",
  "operating_margin": "",
  "geographic_presence": [],
  "business_model": "",
  "public_or_private": "",
  "description": ""
}}
Use "—" for unknown fields. Be specific with numbers where available."""


def _portfolio_prompt(company: str) -> str:
    return f"""Research the service and product portfolio of {company}.

Search for: "{company}" products services portfolio, "{company}" offerings capabilities, "{company}" solutions, "{company}" product roadmap.

Return ONLY valid JSON (no markdown):
{{
  "primary_offerings": [],
  "key_products": [],
  "industry_verticals": [],
  "portfolio_type": "",
  "portfolio_breadth": "",
  "recent_launches": [],
  "discontinued_products": []
}}
Use "—" for unknown fields."""


def _technology_prompt(company: str) -> str:
    return f"""Research the technology and innovation profile of {company}.

Search for: "{company}" AI machine learning, "{company}" R&D investment, "{company}" patents technology, "{company}" digital transformation capabilities, "{company}" technology partnerships.

Return ONLY valid JSON (no markdown):
{{
  "ai_ml_capabilities": [],
  "rd_investment": "",
  "patent_count": "",
  "tech_partnerships": [],
  "cloud_strategy": "",
  "innovation_highlights": [],
  "tech_differentiators": []
}}
Use "—" for unknown fields."""


def _customer_prompt(company: str) -> str:
    return f"""Research the customer base and retention metrics of {company}.

Search for: "{company}" customers clients, "{company}" customer count enterprise, "{company}" notable clients logos, "{company}" NPS retention rate, "{company}" case studies wins.

Return ONLY valid JSON (no markdown):
{{
  "customer_count": "",
  "enterprise_clients": [],
  "key_industries_served": [],
  "customer_retention_rate": "",
  "nps_score": "",
  "notable_wins_2024_2025": [],
  "net_revenue_retention": ""
}}
Use "—" for unknown fields."""


def _financial_prompt(company: str) -> str:
    return f"""Research the financial performance of {company}. Search broadly using multiple queries.

Search for ALL of the following:
- "{company}" annual revenue 2024 fiscal year results
- "{company}" revenue growth quarterly earnings report
- "{company}" total revenue billion million 2024 2025
- "{company}" financial results investor relations
- "{company}" annual report revenue by segment
- "{company}" revenue by geography region breakdown
- "{company}" gross margin operating margin EBITDA
- "{company}" ARR recurring revenue growth rate
- "{company}" funding raised valuation IPO

For public companies like Oracle, SAP, Microsoft: use their latest fiscal year filings (10-K, earnings releases, press releases) which are public record.
For private companies: use Crunchbase, press releases, news articles, analyst estimates.

Be thorough — do not return "—" for major public companies where financials are publicly available.

Return ONLY valid JSON (no markdown):
{{
  "annual_revenue": "",
  "revenue_growth_yoy": "",
  "arr_if_saas": "",
  "revenue_by_geo": {{}},
  "revenue_by_segment": {{}},
  "gross_margin": "",
  "ebitda_margin": "",
  "net_income": "",
  "funding_raised": "",
  "valuation": "",
  "market_cap": "",
  "profitable": ""
}}
Use "—" ONLY if genuinely unavailable after searching all sources above."""


def _gtm_prompt(company: str) -> str:
    return f"""Research the go-to-market strategy of {company}.

Search for: "{company}" sales strategy, "{company}" partner ecosystem, "{company}" channel partners resellers, "{company}" marketing campaigns, "{company}" analyst relations.

Return ONLY valid JSON (no markdown):
{{
  "primary_sales_motion": "",
  "channel_partners": [],
  "key_alliances": [],
  "target_buyer_persona": "",
  "marketing_highlights": [],
  "recent_campaigns": [],
  "analyst_strategy": ""
}}
Use "—" for unknown fields."""


def _talent_prompt(company: str) -> str:
    return f"""Research the talent profile and culture of {company}.

Search for: "{company}" employees headcount growth, "{company}" Glassdoor reviews rating, "{company}" executive leadership changes, "{company}" hiring areas layoffs, "{company}" culture values.

Return ONLY valid JSON (no markdown):
{{
  "total_headcount": "",
  "headcount_yoy_change": "",
  "glassdoor_rating": "",
  "glassdoor_review_sentiment": "",
  "key_exec_changes_2024_2025": [],
  "hiring_focus_areas": [],
  "attrition_signal": ""
}}
Use "—" for unknown fields."""


def _brand_prompt(company: str) -> str:
    return f"""Research the brand and analyst positioning of {company}.

Search for: "{company}" Gartner Magic Quadrant, "{company}" Forrester Wave, "{company}" ISG Provider Lens, "{company}" Everest Group PEAK Matrix, "{company}" awards recognition 2024 2025, "{company}" LinkedIn followers.

Return ONLY valid JSON (no markdown):
{{
  "gartner_position": "",
  "forrester_position": "",
  "isg_position": "",
  "everest_position": "",
  "awards_2024_2025": [],
  "thought_leadership_score": "",
  "linkedin_followers": "",
  "brand_perception": ""
}}
Use "—" for unknown fields."""


def _stack_prompt(company: str) -> str:
    return f"""Research the internal tech stack and strategic alliances of {company}.

Search for: "{company}" cloud provider AWS Azure GCP, "{company}" CRM Salesforce HubSpot, "{company}" ERP SAP Oracle, "{company}" AI tools platform, "{company}" technology partners alliances.

Return ONLY valid JSON (no markdown):
{{
  "cloud_providers": [],
  "crm_platform": "",
  "erp_platform": "",
  "ai_tools_used": [],
  "key_tech_alliances": [],
  "development_stack": [],
  "data_platforms": []
}}
Use "—" for unknown fields."""


_MODULE_PROMPTS = {
    "core":      _core_prompt,
    "portfolio": _portfolio_prompt,
    "technology":_technology_prompt,
    "customer":  _customer_prompt,
    "financial": _financial_prompt,
    "gtm":       _gtm_prompt,
    "talent":    _talent_prompt,
    "brand":     _brand_prompt,
    "stack":     _stack_prompt,
}


# ── Confidence Scoring ────────────────────────────────────────────────────────

def _score_confidence(data: dict | list | None) -> str:
    """green = most fields filled, amber = some, grey = mostly empty."""
    if not data or not isinstance(data, dict):
        return "grey"
    vals = list(data.values())
    filled = sum(1 for v in vals if v and v != "—" and v != [] and v != {})
    pct = filled / max(len(vals), 1)
    if pct >= 0.6:
        return "green"
    if pct >= 0.3:
        return "amber"
    return "grey"


# ── Module Runner ─────────────────────────────────────────────────────────────

async def _run_module(company: str, module_id: str, sem: asyncio.Semaphore) -> dict:
    prompt_fn = _MODULE_PROMPTS.get(module_id)
    if not prompt_fn:
        return {"module": module_id, "data": {}, "confidence": "grey"}

    async with sem:
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(_gemini_call_sync, prompt_fn(company), True, 8192),
                timeout=CALL_TIMEOUT,
            )
            data = _parse_json_from_text(text) or {}
            if isinstance(data, list):
                data = {"items": data}
        except Exception as e:
            logger.warning(f"Module {module_id} failed for {company}: {e}")
            data = {}

    return {
        "module":     module_id,
        "data":       data,
        "confidence": _score_confidence(data),
    }


# ── Synthesis ─────────────────────────────────────────────────────────────────

def _synthesis_prompt(
    target: str,
    competitors: list[str],
    benchmark_focus: str,
    all_results: list[dict],
    industry_context: str = "",
    technology_context: str = "",
) -> str:
    summary_lines = []
    for r in all_results:
        company = r["company"]
        for mod in r.get("modules", []):
            d = mod.get("data", {})
            if d:
                summary_lines.append(f"[{company} / {mod['module']}] {json.dumps(d)[:600]}")

    summary = "\n".join(summary_lines[:80])
    competitors_str = ", ".join(competitors) if competitors else "no direct competitors"

    context_lines = []
    if industry_context:
        context_lines.append(f"Industry context: {industry_context}")
    if technology_context:
        context_lines.append(f"Technology focus: {technology_context}")
    context_block = ("\n" + "\n".join(context_lines) + "\n") if context_lines else ""

    return f"""You are a senior competitive strategy analyst. Based on the following competitive intelligence data, write a strategic analysis comparing {target} against its competitors ({competitors_str}).

Benchmark focus: {benchmark_focus}{context_block}

Intelligence gathered:
{summary}

Write a strategic executive summary in exactly 4 paragraphs:

Paragraph 1 — Competitive Positioning: Where does {target} stand relative to competitors? What are its core strengths and differentiation?

Paragraph 2 — Key Threats & Gaps: What competitive threats are most significant? Where are the gaps in {target}'s position?

Paragraph 3 — Market Dynamics: What trends in the competitive landscape should {target} watch? Which competitor moves are most disruptive?

Paragraph 4 — Strategic Recommendations: What 3-4 specific actions should {target} take based on this competitive intelligence?

Return ONLY the 4 paragraphs with no headers, no markdown, no bullet points. Be specific, data-driven, and concise. Total: 250-350 words."""


async def _run_synthesis(
    target: str,
    competitors: list[str],
    benchmark_focus: str,
    all_results: list[dict],
    industry_context: str = "",
    technology_context: str = "",
) -> str:
    prompt = _synthesis_prompt(target, competitors, benchmark_focus, all_results, industry_context, technology_context)
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_gemini_call_sync, prompt, False, 4096),
            timeout=120,
        )
        return text.strip()
    except Exception as e:
        logger.warning(f"Synthesis failed: {e}")
        return ""


# ── Main Analysis Runner ──────────────────────────────────────────────────────

async def run_competitive_analysis(
    target_company: str,
    target_domain: str,
    competitors: list[dict],
    enabled_modules: list[str],
    benchmark_focus: str,
    industry_context: str = "",
    technology_context: str = "",
) -> AsyncGenerator[dict, None]:
    """Yield SSE-ready dicts for each module result, then synthesis."""

    all_companies = [{"name": target_company, "domain": target_domain, "is_target": True}]
    for c in competitors:
        all_companies.append({**c, "is_target": False})

    total_calls = len(all_companies) * len(enabled_modules)
    done_calls = 0

    yield {"type": "start", "total_companies": len(all_companies), "total_modules": len(enabled_modules), "total_calls": total_calls}

    company_sem = asyncio.Semaphore(COMPANY_SEM)
    module_sem  = asyncio.Semaphore(MODULE_SEM)

    all_results: list[dict] = []

    async def _process_company(company_info: dict) -> dict:
        nonlocal done_calls
        company_name = company_info["name"]
        async with company_sem:
            tasks = [_run_module(company_name, mod, module_sem) for mod in enabled_modules]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        modules_out = []
        for mod_id, res in zip(enabled_modules, results):
            if isinstance(res, Exception):
                logger.warning(f"Module {mod_id} for {company_name}: {res}")
                modules_out.append({"module": mod_id, "data": {}, "confidence": "grey"})
            else:
                modules_out.append(res)
            done_calls += 1

        return {
            "company":   company_name,
            "domain":    company_info.get("domain", ""),
            "is_target": company_info.get("is_target", False),
            "modules":   modules_out,
        }

    # Run all companies concurrently, yield results as each completes
    tasks = [asyncio.create_task(_process_company(c)) for c in all_companies]

    start_ts = _time.time()
    pending = set(tasks)

    while pending:
        elapsed = _time.time() - start_ts
        if elapsed > TOTAL_BUDGET:
            for t in pending:
                t.cancel()
            yield {"type": "timeout", "message": "Budget exceeded; partial results below."}
            break

        done, pending = await asyncio.wait(pending, timeout=8.0, return_when=asyncio.FIRST_COMPLETED)

        for fut in done:
            try:
                company_result = fut.result()
                all_results.append(company_result)
                yield {
                    "type":          "company_result",
                    "company":       company_result["company"],
                    "domain":        company_result["domain"],
                    "is_target":     company_result["is_target"],
                    "modules":       company_result["modules"],
                    "done_calls":    done_calls,
                    "total_calls":   total_calls,
                }
            except Exception as e:
                logger.warning(f"Company task error: {e}")

        if pending:
            yield {"type": "heartbeat", "done_calls": done_calls, "total_calls": total_calls}

    # Synthesis
    if all_results:
        yield {"type": "synthesis_start"}
        competitor_names = [c["name"] for c in competitors]
        synthesis_text = await _run_synthesis(target_company, competitor_names, benchmark_focus, all_results, industry_context, technology_context)
        yield {"type": "synthesis", "text": synthesis_text}

    yield {"type": "complete", "total_companies": len(all_results)}
