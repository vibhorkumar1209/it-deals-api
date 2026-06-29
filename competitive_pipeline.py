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
    "metrics":      "Overall Company Metrics",
    "portfolio":    "Service / Product / Platform Portfolio",
    "overlap":      "Core Competitive Overlap",
    "customer":     "Customer Base",
    "brand":        "Brand & Analyst Mentions",
    "talent":       "Talent & Headcount",
    "deals":        "JV / M&A / Partnerships",
    "stack":        "Tech Stack",
    "news":         "Recent Key News",
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

def _metrics_prompt(company: str) -> str:
    return f"""Research the overall company metrics for {company}. Search broadly.

Search for:
- "{company}" annual revenue 2024 2025 fiscal year results
- "{company}" total revenue billion million earnings
- "{company}" market share market position industry
- "{company}" revenue growth YoY quarterly
- "{company}" gross margin EBITDA operating margin
- "{company}" market cap valuation funding
- "{company}" number of customers total employees
- "{company}" revenue by segment geography

For public companies use 10-K, earnings releases. For private use Crunchbase, press, analyst estimates.
Do NOT return "—" for major public companies — their financials are public record.

Return ONLY valid JSON (no markdown):
{{
  "annual_revenue": "",
  "revenue_growth_yoy": "",
  "market_cap_or_valuation": "",
  "market_share": "",
  "market_share_source": "",
  "gross_margin": "",
  "ebitda_margin": "",
  "net_income": "",
  "total_customers": "",
  "total_employees": "",
  "revenue_by_segment": {{}},
  "revenue_by_geo": {{}},
  "public_or_private": "",
  "profitable": ""
}}
Use "—" only if genuinely unavailable."""


def _portfolio_prompt(company: str) -> str:
    return f"""Research the service, product, and platform portfolio of {company}.

Search for:
- "{company}" products services solutions portfolio 2024 2025
- "{company}" platform capabilities AI integration features
- "{company}" key offerings industry solutions
- "{company}" product roadmap new launches
- "{company}" AI features generative AI integration

Return ONLY valid JSON (no markdown):
{{
  "primary_offerings": [],
  "key_products_platforms": [],
  "ai_integration_highlights": [],
  "industry_aligned_solutions": [],
  "portfolio_strengths": [],
  "recent_launches_2024_2025": [],
  "portfolio_gaps": []
}}
Use "—" for unknown fields."""


def _overlap_prompt(company: str) -> str:
    return f"""Research the core competitive positioning and overlap areas of {company} vs its rivals.

Search for:
- "{company}" competitive advantage differentiation
- "{company}" vs competitors comparison strengths weaknesses
- "{company}" win rate competitive displacement 2024 2025
- "{company}" analyst competitive assessment

Return ONLY valid JSON (no markdown):
{{
  "primary_competitive_strengths": [],
  "primary_competitive_weaknesses": [],
  "key_differentiators": [],
  "where_they_win": [],
  "where_they_lose": [],
  "competitive_moat": "",
  "pricing_positioning": ""
}}
Use "—" for unknown fields."""


def _customer_prompt(company: str) -> str:
    return f"""Research the customer base of {company} — focus on industry verticals and notable clients.

Search for:
- "{company}" customers clients case studies 2024 2025
- "{company}" enterprise clients notable logos industry
- "{company}" customer wins new accounts
- "{company}" NPS retention rate customer satisfaction
- "{company}" industries served verticals

Return ONLY valid JSON (no markdown):
{{
  "total_customers": "",
  "key_industry_verticals": [],
  "notable_enterprise_clients": [],
  "recent_customer_wins_2024_2025": [],
  "customer_retention_rate": "",
  "nps_score": "",
  "net_revenue_retention": "",
  "customer_concentration_risk": ""
}}
Use "—" for unknown fields."""


def _brand_prompt(company: str) -> str:
    return f"""Research the brand presence and analyst recognition of {company} — both overall and industry/product-specific.

Search for:
- "{company}" Gartner Magic Quadrant 2024 2025
- "{company}" Forrester Wave leader 2024 2025
- "{company}" ISG Provider Lens Everest Group PEAK Matrix
- "{company}" awards recognition industry analyst
- "{company}" LinkedIn followers brand awareness
- "{company}" thought leadership content marketing

Return ONLY valid JSON (no markdown):
{{
  "gartner_positions": [],
  "forrester_positions": [],
  "isg_everest_positions": [],
  "other_analyst_mentions": [],
  "awards_2024_2025": [],
  "linkedin_followers": "",
  "brand_perception_overall": "",
  "brand_perception_by_segment": {{}}
}}
Use "—" for unknown fields."""


def _talent_prompt(company: str) -> str:
    return f"""Research the talent profile, headcount, and key leaders of {company}.

Search for:
- "{company}" total employees headcount 2024 2025
- "{company}" key executives CEO CTO CPO leadership team
- "{company}" hiring growth layoffs workforce
- "{company}" relevant department size engineering sales
- "{company}" Glassdoor rating culture

Return ONLY valid JSON (no markdown):
{{
  "total_headcount": "",
  "headcount_yoy_change": "",
  "key_leaders": [],
  "relevant_dept_size": "",
  "hiring_focus_areas": [],
  "recent_key_hires_departures": [],
  "glassdoor_rating": "",
  "attrition_signal": ""
}}
Use "—" for unknown fields."""


def _deals_prompt(company: str) -> str:
    return f"""Research joint ventures, M&A activity, and strategic partnerships of {company} — only relevant ones.

Search for:
- "{company}" acquisition merger 2023 2024 2025
- "{company}" joint venture strategic partnership alliance
- "{company}" partnership agreement announced
- "{company}" invested in acquired divested

Return ONLY valid JSON (no markdown):
{{
  "recent_acquisitions": [],
  "recent_jv_partnerships": [],
  "key_alliances": [],
  "divestitures": [],
  "investment_activity": [],
  "strategic_rationale_summary": ""
}}
Include only deals from 2022 onwards. Use "—" for unknown fields."""


def _stack_prompt(company: str) -> str:
    return f"""Research the tech stack and technology partnerships of {company} — especially relevant products, platforms, or solutions they use or partner with to deliver offerings.

Search for:
- "{company}" cloud provider AWS Azure GCP technology
- "{company}" AI platform partner technology stack
- "{company}" CRM ERP software platform used
- "{company}" technology partnership solution integration
- "{company}" built on powered by platform

Return ONLY valid JSON (no markdown):
{{
  "cloud_infrastructure": [],
  "ai_ml_platforms": [],
  "key_software_platforms": [],
  "technology_partnerships": [],
  "homegrown_platforms": [],
  "data_and_analytics_stack": []
}}
Use "—" for unknown fields."""


def _news_prompt(company: str) -> str:
    return f"""Find the most recent and significant news about {company} from the last 12 months.

Search for:
- "{company}" news 2024 2025 latest announcements
- "{company}" press release announcement recent
- "{company}" major deal contract win 2024 2025
- "{company}" strategy change pivot leadership

Return ONLY valid JSON (no markdown) — an array of up to 8 items, newest first:
[
  {{
    "headline": "",
    "date": "",
    "category": "Deal | Partnership | Leadership | Financial | Product | Strategy | Other",
    "summary": "",
    "significance": ""
  }}
]
Only include genuinely newsworthy items from 2024–2025. Return [] if nothing significant found."""


_MODULE_PROMPTS = {
    "metrics":   _metrics_prompt,
    "portfolio": _portfolio_prompt,
    "overlap":   _overlap_prompt,
    "customer":  _customer_prompt,
    "brand":     _brand_prompt,
    "talent":    _talent_prompt,
    "deals":     _deals_prompt,
    "stack":     _stack_prompt,
    "news":      _news_prompt,
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
