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

import datetime as _dt

def _date_window(lookback_days: int = 365) -> str:
    """Return a human-readable date range string, e.g. 'June 2024 – June 2025'."""
    today = _dt.date.today()
    cutoff = today - _dt.timedelta(days=lookback_days)
    fmt = "%d %B %Y"
    return f"{cutoff.strftime(fmt)} – {today.strftime(fmt)}"

def _source_instructions(domain: str) -> str:
    news_url = f"{domain.rstrip('/')}/news" if domain else ""
    press_url = f"{domain.rstrip('/')}/press" if domain else ""
    url_hint = f"  • Company newsroom / press releases: {news_url} or {press_url}\n" if domain else ""
    return f"""Search the following sources in priority order:
{url_hint}  • LinkedIn posts, announcements, and job listings for {domain or 'the company'}
  • Google News and open web search
  • Business wires: PR Newswire, Business Wire, GlobeNewswire, Businesswire
  • Industry trade press and analyst reports"""


def _exec_leadership_prompt(company: str, domain: str, key_triggers: str, target_tech: str, lookback_days: int = 365) -> str:
    window = _date_window(lookback_days)
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Find Executive & Leadership Shift signals for {company} ({domain}).

STRICT DATE FILTER: Only include events that occurred between {window}. Discard anything older.

Look for:
1. NEW DECISION-MAKER HIRES — New C-suite, VP, or Director joined within the date window. Include: title, name, start date, previous employer.
2. INTERNAL PROMOTIONS — A manager or director stepped up to a senior decision-making role within the date window.
3. PAST CHAMPION MOVES — A power user or champion from a known IT vendor or competitor joins {company} within the date window.
4. MASS EXECUTIVE EXODUS — Multiple leadership departures from the same team within a short period, within the date window.
{extra}

{_source_instructions(domain)}

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "new_hire" | "internal_promotion" | "champion_move" | "mass_exodus"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary of the signal and its sales implication
- person_name: name of executive(s) involved (or "Multiple" for mass exodus)
- previous_company: where they came from (or "Internal" for promotions)
- date: exact date or month (e.g. "May 2025") — must be within {window}
- source: MUST be a direct URL to the specific press release, news article, LinkedIn post, or filing — e.g. "https://www.reuters.com/..." — not just a publication name

Omit any signal whose date falls outside {window}. Return [] if nothing found within the window.
Return ONLY the JSON array, no commentary."""


def _corporate_expansion_prompt(company: str, domain: str, key_triggers: str, target_tech: str, lookback_days: int = 365) -> str:
    window = _date_window(lookback_days)
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Find Corporate Expansion & Growth signals for {company} ({domain}).

STRICT DATE FILTER: Only include events that occurred between {window}. Discard anything older.

Look for:
1. HEADCOUNT SURGE — Employee count or a specific department grew by 20%+ quarter-over-quarter within the date window. Cite the numbers.
2. JOB POSTINGS — Active job listings published within the date window that reveal pain points, skill gaps, or new tech initiatives.
3. NEW OFFICE OPENINGS — Physical geographic expansion or new corporate division opened within the date window.
4. PRODUCT LINE LAUNCHES — New product or market segment entered within the date window.
{extra}

{_source_instructions(domain)}
Also search: Glassdoor, Indeed, LinkedIn Jobs, Builtin, Lever/Greenhouse job boards.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "headcount_surge" | "job_postings" | "office_opening" | "product_launch"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- magnitude: quantitative detail where available (e.g. "+35% headcount", "15 new roles", "3 new cities")
- date: month or quarter — must be within {window}
- source: MUST be a direct URL to the specific press release, news article, LinkedIn post, or filing — e.g. "https://www.reuters.com/..." — not just a publication name

Omit any signal outside {window}. Return [] if nothing found. Return ONLY the JSON array."""


def _financial_corporate_prompt(company: str, domain: str, key_triggers: str, target_tech: str, lookback_days: int = 365) -> str:
    window = _date_window(lookback_days)
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Find Financial & Corporate Structure signals for {company} ({domain}).

STRICT DATE FILTER: Only include events that occurred between {window}. Discard anything older.

Look for:
1. FUNDING ROUNDS — VC, PE, or debt financing (Series A/B/C, growth equity, IPO) announced within the date window.
2. M&A ACTIVITY — Mergers, acquisitions, divestitures, or joint ventures announced within the date window.
3. CORPORATE RELOCATION — HQ move or operations consolidation announced within the date window.
4. EARNINGS SHIFTS — Publicly reported major profit spike or operational drop (>20% swing) within the date window.
{extra}

{_source_instructions(domain)}
Also search: Crunchbase, PitchBook, SEC EDGAR filings, Bloomberg, Reuters, financial news.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "funding_round" | "merger_acquisition" | "relocation" | "earnings_shift"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- financial_detail: amount raised / deal value / revenue change (e.g. "$50M Series B", "Acquired Acme Corp for $200M")
- date: announcement date — must be within {window}
- source: MUST be a direct URL to the specific press release, news article, LinkedIn post, or filing — e.g. "https://www.reuters.com/..." — not just a publication name

Omit any signal outside {window}. Return [] if nothing found. Return ONLY the JSON array."""


def _tech_legal_prompt(company: str, domain: str, key_triggers: str, target_tech: str, lookback_days: int = 365) -> str:
    window = _date_window(lookback_days)
    extra = ""
    if key_triggers:
        extra += f"\nFocus especially on triggers related to: {key_triggers}"
    if target_tech:
        extra += f"\nHighlight signals relevant to this technology: {target_tech}"
    return f"""You are a B2B sales intelligence researcher. Find Tech Stack & Legal Trigger signals for {company} ({domain}).

STRICT DATE FILTER: Only include events that occurred or were announced between {window}. Discard anything older.

Look for:
1. CONTRACT RENEWALS — IT/software contracts approaching renewal within the next 12 months, where the original contract was signed ~2–3 years ago (i.e. announced within {window} or inferrable from deal dates in the window).
2. REGULATORY COMPLIANCE — New legal deadlines or audit requirements facing the company, announced or effective within {window} (GDPR, EU AI Act, SOC2, ISO 27001, SEC rules, HIPAA, etc.)
3. SYSTEM OUTAGE / PUBLIC FAILURE — Technical outage, data breach, or system failure that became public within {window}.
4. TECH REFRESH SIGNALS — Legacy system end-of-life, vendor sunset, or public RFP/RFI issued within {window}.
{extra}

{_source_instructions(domain)}
Also search: security breach databases, Gartner, Forrester, industry trade press, regulatory filings.

Return ONLY a JSON array. Each object must have exactly these fields:
- signal_type: one of "contract_renewal" | "regulatory_compliance" | "system_outage" | "tech_refresh"
- signal_title: short headline (≤15 words)
- summary: 2–3 sentence summary including the sales implication
- urgency: "Immediate" | "Within 6 months" | "Within 12 months" | "Watch"
- date: date or deadline — must be within or triggered within {window}
- source: MUST be a direct URL to the specific press release, news article, LinkedIn post, or filing — e.g. "https://www.reuters.com/..." — not just a publication name

Omit any signal outside {window}. Return [] if nothing found. Return ONLY the JSON array."""


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


# ── Date sort helper ──────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

def _date_sort_key(date_str: str) -> tuple:
    """Return (year, month) for reverse-chronological sort. Unknown → (0, 0)."""
    if not date_str or date_str in ("-", "Unknown", ""):
        return (0, 0)
    s = date_str.strip()
    # ISO: "2024-09" or "2024-09-15"
    m = re.match(r"(\d{4})-(\d{1,2})", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # "Sep 2024" / "September 2024"
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", s)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower()[:3], 0)
        return (int(m.group(2)), mo)
    # "2024 Sep" / "2024 September"
    m = re.match(r"(\d{4})\s+([A-Za-z]+)", s)
    if m:
        mo = _MONTH_MAP.get(m.group(2).lower()[:3], 0)
        return (int(m.group(1)), mo)
    # Quarter: "Q2 2025" / "Q2 2025"
    m = re.match(r"Q([1-4])\s+(\d{4})", s, re.IGNORECASE)
    if m:
        return (int(m.group(2)), (int(m.group(1)) - 1) * 3 + 1)
    # Year only: "2024"
    m = re.match(r"(\d{4})$", s)
    if m:
        return (int(m.group(1)), 0)
    return (0, 0)

def _sort_by_date_desc(rows: list) -> list:
    return sorted(rows, key=lambda r: _date_sort_key(r.get("date", "")), reverse=True)


# ── Per-company signal runner ─────────────────────────────────────────────────

def _run_category_sync(
    company: str,
    domain: str,
    category: str,
    key_triggers: str,
    target_tech: str,
    lookback_days: int = 365,
) -> list:
    prompts = {
        "executive_leadership": _exec_leadership_prompt,
        "corporate_expansion":  _corporate_expansion_prompt,
        "financial_corporate":  _financial_corporate_prompt,
        "tech_legal":           _tech_legal_prompt,
    }
    prompt_fn = prompts[category]
    prompt = prompt_fn(company, domain, key_triggers, target_tech, lookback_days)
    label = f"{company[:20]}|{category}"
    rows = _gemini_call_sync(prompt, use_search=True, label=label, max_output_tokens=8192)

    # signal_type → category map. Each prompt asks the model for a signal_type
    # from a fixed, category-specific enum (see prompt functions above) — that
    # enum is the ground truth for classification, not incidental keywords in
    # the title/summary. A "new_hire" signal stays Executive & Leadership even
    # if its summary happens to mention "platform" or "CRM" in passing.
    _SIGNAL_TYPE_TO_CATEGORY = {
        "new_hire": "executive_leadership", "internal_promotion": "executive_leadership",
        "champion_move": "executive_leadership", "mass_exodus": "executive_leadership",
        "headcount_surge": "corporate_expansion", "job_postings": "corporate_expansion",
        "office_opening": "corporate_expansion", "product_launch": "corporate_expansion",
        "funding_round": "financial_corporate", "merger_acquisition": "financial_corporate",
        "relocation": "financial_corporate", "earnings_shift": "financial_corporate",
        "contract_renewal": "tech_legal", "regulatory_compliance": "tech_legal",
        "system_outage": "tech_legal", "tech_refresh": "tech_legal",
    }

    result = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        r["company"] = company
        r["domain"] = domain

        # Trust the category this prompt was run under, unless the model's own
        # signal_type unambiguously belongs to a different category (rare —
        # e.g. it free-associated a tech_refresh type into a non-tech prompt).
        signal_type = str(r.get("signal_type", "")).strip().lower()
        assigned_cat = _SIGNAL_TYPE_TO_CATEGORY.get(signal_type, category)

        r["category"] = CATEGORY_LABELS[assigned_cat]
        r["category_key"] = assigned_cat

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
    lookback_days: int = 365,
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
                asyncio.to_thread(_run_category_sync, name, domain, cat, key_triggers, target_tech, lookback_days)
                for cat in SIGNAL_CATEGORIES
            ]
            cat_results = await asyncio.gather(*futures, return_exceptions=True)

            company_rows = []
            for cat, result in zip(SIGNAL_CATEGORIES, cat_results):
                if isinstance(result, Exception):
                    logger.error(f"Category {cat} error for {name}: {result}")
                    continue
                company_rows.extend(result)

            # Sort by date descending before streaming — newest signals first
            company_rows = _sort_by_date_desc(company_rows)

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
