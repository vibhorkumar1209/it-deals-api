"""
Signal Intelligence pipeline — Gemini 2.5 Flash + Google Search.

For each target company, runs 4 parallel signal-category searches:
  1. Executive & Leadership Shifts
  2. Corporate Expansion & Growth
  3. Financial & Corporate Structure
  4. Tech Stack & Legal Triggers

Optional: if user_company is provided, runs a 5th ranking pass per company
          to score each signal by relevance to the seller.
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

# ── Signal schema ─────────────────────────────────────────────────────────────

SIGNAL_FIELDS = [
    {"key": "company",               "label": "Company"},
    {"key": "category",              "label": "Category"},
    {"key": "signal_type",           "label": "Signal Type"},
    {"key": "signal_title",          "label": "Signal"},
    {"key": "summary",               "label": "Summary"},
    {"key": "date",                  "label": "Date"},
    {"key": "importance",            "label": "Importance"},
    {"key": "importance_rationale",  "label": "Why It Matters"},
    {"key": "source",                "label": "Source"},
]

SIGNAL_CATEGORIES = [
    "executive_leadership",
    "corporate_expansion",
    "financial_corporate",
    "tech_legal",
]

CATEGORY_LABELS = {
    "executive_leadership": "Executive & Leadership Shifts",
    "corporate_expansion":  "Corporate Expansion & Growth",
    "financial_corporate":  "Financial & Corporate Structure",
    "tech_legal":           "Tech Stack & Legal Triggers",
}


# ── Shared Gemini call (copied from aftermarket_pipeline pattern) ─────────────

def _gemini_call_sync(prompt: str, use_search: bool, label: str, max_output_tokens: int = 16384):
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

    MAX_RETRIES = 6
    TOTAL_BUDGET = 300
    call_start = _time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        if _time.time() - call_start > TOTAL_BUDGET:
            logger.warning(f"Gemini [{label}] total budget exceeded — giving up")
            return []
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
            is_retry = not is_quota and any(x in err for x in ("503", "UNAVAILABLE", "overloaded", "timeout", "429", "500", "502", "504"))
            if is_quota:
                raise RuntimeError("Gemini quota exhausted.") from e
            if is_retry and attempt < MAX_RETRIES:
                elapsed = _time.time() - call_start
                remaining = TOTAL_BUDGET - elapsed
                wait = min(20 * attempt, 60, max(0, remaining - 10))
                if wait <= 0:
                    return []
                logger.warning(f"Gemini [{label}] retry {attempt}/{MAX_RETRIES}, sleeping {wait:.0f}s: {err[:80]}")
                _time.sleep(wait)
                continue
            logger.error(f"Signal Gemini [{label}]: {e}")
            return []
    else:
        return []

    text_parts = []
    try:
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "thought", False):
                    continue
                t = getattr(part, "text", None)
                if t:
                    text_parts.append(t)
    except Exception:
        pass

    if not text_parts:
        try:
            t = response.text
            if t:
                text_parts = [t]
        except Exception:
            pass

    if not text_parts:
        return []

    def _try_parse(raw: str):
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
        return None

    best = None
    for pt in text_parts:
        result = _try_parse(pt)
        if result and (best is None or len(result) > len(best)):
            best = result

    if best is not None:
        return best

    combined = "".join(text_parts)
    result = _try_parse(combined)
    return result or []


# ── Prompt builders ───────────────────────────────────────────────────────────

def _exec_leadership_prompt(company: str, domain: str, key_triggers: str, target_tech: str) -> str:
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Using real-time web search, find all recent (last 6 months) Executive & Leadership Shift signals for {company} ({domain}).

Look for:
1. NEW DECISION-MAKER HIRES — New C-suite, VP, or Director joins in the past 30–90 days. Include: title, name, start date, previous employer.
2. INTERNAL PROMOTIONS — A manager or director stepped up to a senior decision-making role.
3. PAST CHAMPION MOVES — A power user or champion from a known IT vendor or competitor joins {company}.
4. MASS EXECUTIVE EXODUS — Multiple leadership departures from the same team within a short period.
{extra}

Search LinkedIn, company press releases, news articles, and business intelligence sources.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "new_hire" | "internal_promotion" | "champion_move" | "mass_exodus"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary of the signal and its sales implication
- person_name: name of executive(s) involved (or "Multiple" for mass exodus)
- previous_company: where they came from (or "Internal" for promotions)
- date: best estimate of event date (e.g. "May 2025" or "Q1 2025")
- source: URL or publication name

If no signals found for a type, omit it. Return [] if absolutely nothing found.
Return ONLY the JSON array, no commentary."""


def _corporate_expansion_prompt(company: str, domain: str, key_triggers: str, target_tech: str) -> str:
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Using real-time web search, find all recent (last 6 months) Corporate Expansion & Growth signals for {company} ({domain}).

Look for:
1. HEADCOUNT SURGE — Total employee count or a specific department growing by 20%+ quarter-over-quarter. Cite the numbers.
2. JOB POSTINGS — Actively publishing jobs that reveal pain points, skill gaps, or new tech initiatives. Include job titles and implied gaps.
3. NEW OFFICE OPENINGS — Physical geographic expansion or opening a new corporate division/business unit.
4. PRODUCT LINE LAUNCHES — The company entering a new market segment or launching a major new product line.
{extra}

Search LinkedIn, company newsroom, Glassdoor, job boards, and press releases.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "headcount_surge" | "job_postings" | "office_opening" | "product_launch"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- magnitude: quantitative detail where available (e.g. "+35% headcount", "15 new roles", "3 new cities")
- date: best estimate (e.g. "Q2 2025")
- source: URL or publication name

Return [] if nothing found. Return ONLY the JSON array."""


def _financial_corporate_prompt(company: str, domain: str, key_triggers: str, target_tech: str) -> str:
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Using real-time web search, find all recent (last 12 months) Financial & Corporate Structure signals for {company} ({domain}).

Look for:
1. FUNDING ROUNDS — Recent VC, PE, or debt financing (Series A/B/C, growth equity, IPO, etc.)
2. M&A ACTIVITY — Mergers, acquisitions, divestitures, or joint ventures forcing tech consolidation
3. CORPORATE RELOCATION — Moving HQ or centralizing operations to a new location
4. EARNINGS SHIFTS — Publicly reported massive profit spike or severe operational drop (>20% swing)
{extra}

Search Crunchbase, PitchBook, SEC filings, press releases, financial news.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "funding_round" | "merger_acquisition" | "relocation" | "earnings_shift"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- financial_detail: amount raised / deal value / revenue change (e.g. "$50M Series B", "Acquired Acme Corp for $200M")
- date: announcement date (e.g. "March 2025")
- source: URL or publication name

Return [] if nothing found. Return ONLY the JSON array."""


def _tech_legal_prompt(company: str, domain: str, key_triggers: str, target_tech: str) -> str:
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Using real-time web search, find all recent (last 18 months) Tech Stack & Legal Trigger signals for {company} ({domain}).

Look for:
1. CONTRACT RENEWALS — Approaching the 2- or 3-year anniversary of known software/IT contracts. Infer from deal announcement dates if available.
2. REGULATORY COMPLIANCE — Facing a new legal deadline or audit (GDPR, EU AI Act, SOC2, ISO 27001, SEC rules, HIPAA, etc.)
3. SYSTEM OUTAGE / PUBLIC FAILURE — Highly visible technical outage, data breach, or system failure revealing operational gaps
4. TECH REFRESH SIGNALS — Known legacy system end-of-life, vendor announcing sunset, or public RFP/RFI issued
{extra}

Search news archives, regulatory filings, security databases (HaveIBeenPwned, known breaches), Gartner, industry trade press.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "contract_renewal" | "regulatory_compliance" | "system_outage" | "tech_refresh"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- urgency: "Immediate" | "Within 6 months" | "Within 12 months" | "Watch"
- date: date or deadline (e.g. "Renewal due Jan 2026", "Breach: Aug 2024")
- source: URL or publication name

Return [] if nothing found. Return ONLY the JSON array."""


def _ranking_prompt(
    company: str,
    signals: list,
    user_company: str,
    user_domain: str,
    key_triggers: str,
    target_tech: str,
) -> str:
    trigger_ctx = f"Key buyer triggers to watch for: {key_triggers}" if key_triggers else ""
    tech_ctx = f"Target technology being sold: {target_tech}" if target_tech else ""
    signals_json = json.dumps(signals, indent=2)
    return f"""You are a senior sales strategist at {user_company} ({user_domain}).

You sell to companies like {company}. Rank the following buying signals by importance to YOUR sales effort.

{trigger_ctx}
{tech_ctx}

For each signal, assign:
- importance: "Critical" | "High" | "Medium" | "Low"
- importance_rationale: 1-2 sentences explaining WHY this signal matters specifically to {user_company}'s sales motion

Signals to rank:
{signals_json}

Return the SAME array with two fields added/updated per object: "importance" and "importance_rationale".
Return ONLY the JSON array, no commentary."""


# ── Per-company signal runner ─────────────────────────────────────────────────

def _run_category_sync(
    company: str,
    domain: str,
    category: str,
    key_triggers: str,
    target_tech: str,
) -> list:
    prompts = {
        "executive_leadership": _exec_leadership_prompt,
        "corporate_expansion":  _corporate_expansion_prompt,
        "financial_corporate":  _financial_corporate_prompt,
        "tech_legal":           _tech_legal_prompt,
    }
    prompt_fn = prompts[category]
    prompt = prompt_fn(company, domain, key_triggers, target_tech)
    label = f"{company[:20]}|{category}"
    rows = _gemini_call_sync(prompt, use_search=True, label=label, max_output_tokens=8192)

    # Normalize rows — inject company + category
    result = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        r["company"] = company
        r["domain"] = domain
        r["category"] = CATEGORY_LABELS[category]
        r["category_key"] = category
        # Default importance if not ranked yet
        if "importance" not in r:
            r["importance"] = "—"
        if "importance_rationale" not in r:
            r["importance_rationale"] = ""
        result.append(r)
    return result


def _rank_signals_sync(
    company: str,
    signals: list,
    user_company: str,
    user_domain: str,
    key_triggers: str,
    target_tech: str,
) -> list:
    if not signals:
        return signals
    prompt = _ranking_prompt(company, signals, user_company, user_domain, key_triggers, target_tech)
    label = f"rank|{company[:20]}"
    ranked = _gemini_call_sync(prompt, use_search=False, label=label, max_output_tokens=8192)
    if not ranked or len(ranked) != len(signals):
        return signals
    # Merge importance fields back into original signal objects
    for i, r in enumerate(ranked):
        if i < len(signals):
            signals[i]["importance"] = r.get("importance", "—")
            signals[i]["importance_rationale"] = r.get("importance_rationale", "")
    return signals


# ── Main async generator ──────────────────────────────────────────────────────

async def run_signal_intelligence(
    target_companies: list[dict],   # [{name, domain}, ...]
    user_company: str = "",
    user_domain: str = "",
    key_triggers: str = "",
    target_tech: str = "",
    max_concurrent: int = 4,
) -> AsyncGenerator[dict, None]:
    """
    Yields:
      {type: "heartbeat", message: str}
      {type: "signal_row", row: dict, company: str}
      {type: "signals_ranked", company: str, rows: list}   — only if user_company set
      {type: "complete", total: int, companies_done: int}
    """
    total_signals = 0
    companies_done = 0
    n = len(target_companies)

    yield {"type": "heartbeat", "message": f"🔍 Scanning {n} companies for buying signals…"}

    semaphore = asyncio.Semaphore(max_concurrent)

    async def _process_company(co: dict) -> list:
        nonlocal total_signals, companies_done
        name = co.get("name", "").strip()
        domain = co.get("domain", "").strip()
        if not name:
            return []

        async with semaphore:
            yield_queue = []

            # Run 4 categories in parallel via thread pool
            futures = [
                asyncio.to_thread(_run_category_sync, name, domain, cat, key_triggers, target_tech)
                for cat in SIGNAL_CATEGORIES
            ]
            cat_results = await asyncio.gather(*futures, return_exceptions=True)

            company_rows = []
            for cat, result in zip(SIGNAL_CATEGORIES, cat_results):
                if isinstance(result, Exception):
                    logger.error(f"Category {cat} error for {name}: {result}")
                    continue
                company_rows.extend(result)

            # Stream raw rows immediately
            for row in company_rows:
                yield_queue.append({"type": "signal_row", "row": row, "company": name})
                total_signals += 1

            # Rank if user company provided
            if user_company and company_rows:
                ranked = await asyncio.to_thread(
                    _rank_signals_sync,
                    name, company_rows,
                    user_company, user_domain,
                    key_triggers, target_tech,
                )
                yield_queue.append({"type": "signals_ranked", "company": name, "rows": ranked})

            companies_done += 1
            yield_queue.append({
                "type": "heartbeat",
                "message": f"✅ {name}: {len(company_rows)} signals found ({companies_done}/{n} companies done)",
            })
            return yield_queue

    # Process in batches to avoid flooding, but use semaphore for concurrency control
    batch_size = min(max_concurrent * 2, 20)
    for i in range(0, n, batch_size):
        batch = target_companies[i:i + batch_size]
        yield {"type": "heartbeat", "message": f"🔄 Processing companies {i+1}–{min(i+len(batch), n)} of {n}…"}

        tasks = [_process_company(co) for co in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Company processing error: {result}")
                continue
            for event in result:
                yield event

        # Heartbeat to prevent SSE timeout between batches
        yield {"type": "heartbeat", "message": f"📊 {total_signals} signals found so far…"}

    yield {
        "type": "complete",
        "total": total_signals,
        "companies_done": companies_done,
    }
