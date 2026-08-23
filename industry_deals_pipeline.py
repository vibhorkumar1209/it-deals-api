"""
Industry Deals pipeline — finds IT deals for companies in a given industry/geography.

Phase 1: generate_company_list()  → top 50 companies (sync, called via to_thread)
Phase 2: search_industry_deals()  → async generator, yields SSE events, batches of 5
"""

import asyncio
import datetime
import json
import logging
import os
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

RENEWAL_WINDOW_DAYS = {
    "1 month":  30,
    "3 months": 90,
    "6 months": 180,
    "1 year":   365,
    "3 years":  1095,
    "5 years":  1825,
}

CALL_TIMEOUT = 120  # seconds per company deal search


# ── Prompts ───────────────────────────────────────────────────────────────────

def _company_list_prompt(
    industry: str,
    geography: str,
    company_name: str = "",
    domain: str = "",
    focus_tech: str = "",
) -> str:
    geo_str = f" in {geography}" if geography else " globally"
    optional_parts = []
    if company_name or domain:
        optional_parts.append(
            f"Include '{company_name or domain}' if it operates in this industry, "
            f"and prioritise similar companies."
        )
    if focus_tech:
        optional_parts.append(
            f"Prioritise companies likely to use or procure {focus_tech} technology."
        )
    opt_block = (
        "\n\nAdditional context:\n" + "\n".join(f"- {p}" for p in optional_parts)
    ) if optional_parts else ""

    return f"""You are a market research analyst. Generate a list of the top 50 companies{geo_str} in the {industry} industry.{opt_block}

Search for:
- "top {industry} companies {geography} 2024 2025 ranking list"
- "largest {industry} enterprises {geography} by revenue"
- "leading {industry} players {geography}"
- "major {industry} companies {geography} market leaders"

Your response MUST be ONLY a valid JSON array — no prose, no headers, no markdown code fences.
Start your response with [ and end with ].

Each element must be exactly:
{{"company_name": "BNP Paribas", "domain": "bnpparibas.com", "type": "Public", "revenue_estimate": "$50B", "headquarters": "Paris, France", "employees_estimate": "190,000", "description": "France's largest bank by assets"}}

Rules:
- type must be one of: Public, Private, Government
- revenue_estimate: "$XB" or "$XM" — use public data where available
- Sort by revenue / size descending
- Include up to 50 companies
- Do NOT include [cite: ...] or any citation/footnote markers anywhere
- description must be plain text only — no brackets, no citations, no special characters
- Output ONLY the JSON array, nothing else"""


def _deals_prompt(
    company_name: str,
    domain: str,
    industry: str,
    geography: str,
    focus_tech: str = "",
) -> str:
    tech_line = (
        f'\n- "{company_name}" {focus_tech} deal contract implementation 2019 2020 2021 2022 2023 2024 2025'
        if focus_tech else ""
    )
    return f"""You are an IT deal researcher. Find all significant IT contracts, technology deals, and outsourcing agreements for {company_name}{f' ({domain})' if domain else ''} in the {industry} industry.

Run these searches:
- "{company_name}" IT contract deal signed 2019 2020 2021 2022 2023 2024 2025
- "{company_name}" outsourcing technology vendor implementation agreement
- "{company_name}" ERP CRM cloud digital transformation contract announcement
- "{company_name}" technology partnership vendor deal press release{tech_line}
- "{company_name}" annual report technology spend vendor contract commitment

For each deal return EXACTLY this JSON object:
{{
  "vendor": "vendor/supplier company name",
  "deal_category": "ERP | CRM | Cloud | Cybersecurity | Outsourcing | Digital Transformation | Analytics | Infrastructure | Core Banking | HR Tech | Networking | Security | Other",
  "deal_type": "Multi-year Contract | Outsourcing | Implementation | License | Partnership | Framework Agreement",
  "deal_value": "$XM or $XB or —",
  "announced_date": "YYYY-MM (required — best estimate if exact date unknown)",
  "duration_years": 5,
  "description": "1-2 sentence summary of the deal",
  "source_url": "https://... or ''",
  "source_label": "Press Release | News Article | Annual Report | Vendor Case Study | LinkedIn"
}}

Return ONLY a raw JSON array. Include deals from 2018 onwards. Return [] if none found."""


# ── Gemini helper ─────────────────────────────────────────────────────────────

def _gemini_sync(prompt: str, label: str = "", run_id: str = "") -> str:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai not installed")
        return ""

    if not GOOGLE_AI_KEY:
        logger.error("GOOGLE_AI_API_KEY not set")
        return ""

    import time as _t
    for attempt in range(1, 4):
        try:
            client = genai.Client(api_key=GOOGLE_AI_KEY)
            resp = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            from usage_logger import log_gemini_usage
            log_gemini_usage("it_deals_by_industry", label, resp, grounded=True, run_id=run_id)
            raw = ""
            for cand in (resp.candidates or []):
                for part in (cand.content.parts or []):
                    t = getattr(part, "text", None)
                    if t:
                        raw += t
            return raw
        except Exception as e:
            err = str(e)
            if "RESOURCE_EXHAUSTED" in err:
                raise RuntimeError("Gemini quota exhausted") from e
            if attempt < 3:
                _t.sleep(8 * attempt)
            else:
                logger.error(f"Gemini call failed after {attempt} attempts: {e}")
                return ""
    return ""


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _find_matching_bracket(text: str, start: int) -> int:
    """Return index of the closing bracket/brace that matches text[start]. -1 if not found."""
    open_c  = text[start]
    close_c = "]" if open_c == "[" else "}"
    depth = 0
    in_str = False
    esc    = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False; continue
        if c == "\\" and in_str:
            esc = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _sanitize(text: str) -> str:
    """Strip control characters that make json.loads raise InvalidControlCharacter."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def _parse_json(text: str) -> list | dict | None:
    """Extract and parse the first JSON array or object from arbitrary text."""
    if not text:
        return None

    text = _sanitize(text)

    # 1. Find first '[' and walk to its TRUE matching ']' (ignores '[cite: ...]' etc.)
    start = text.find("[")
    if start != -1:
        end = _find_matching_bracket(text, start)
        if end != -1:
            fragment = text[start: end + 1]
            try:
                return json.loads(fragment)
            except json.JSONDecodeError:
                # Truncated — cut to last complete object
                lb = fragment.rfind("}")
                if lb != -1:
                    fixed = fragment[:lb + 1].rstrip().rstrip(",") + "\n]"
                    try:
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        pass

    # 2. Try first {...} (dict-wrapped responses like {"companies": [...]})
    start = text.find("{")
    if start != -1:
        end = _find_matching_bracket(text, start)
        if end != -1:
            fragment = text[start: end + 1]
            try:
                result = json.loads(fragment)
                for key in ("companies", "results", "data", "list", "items"):
                    if isinstance(result.get(key), list):
                        return result[key]
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"_parse_json: could not extract JSON (len={len(text)}): {text[:200]!r}")
    return None


# ── Renewal date helper ────────────────────────────────────────────────────────

def _compute_renewal_date(announced_date: str, duration_years) -> datetime.date | None:
    try:
        dur = float(str(duration_years))
        if dur <= 0:
            return None
        s = str(announced_date).strip()
        if "-" in s:
            parts = s.split("-")
            year, month = int(parts[0]), int(parts[1])
        elif len(s) == 4 and s.isdigit():
            year, month = int(s), 6
        else:
            return None
        renewal_year = year + int(dur)
        return datetime.date(renewal_year, month, 1)
    except Exception:
        return None


# ── Phase 1: company list ─────────────────────────────────────────────────────

def generate_company_list(
    industry: str,
    geography: str,
    company_name: str = "",
    domain: str = "",
    focus_tech: str = "",
) -> list[dict]:
    prompt = _company_list_prompt(industry, geography, company_name, domain, focus_tech)
    text = _gemini_sync(prompt, label=f"company_list|{industry[:20]}")
    logger.info(f"Company list raw text length: {len(text)}, first 300: {text[:300]!r}")

    parsed = _parse_json(text)

    # Gemini sometimes wraps the list in {"companies": [...]}
    if isinstance(parsed, dict):
        for key in ("companies", "results", "data", "list"):
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        logger.error(f"Company list parse failed — got {type(parsed)}, raw: {text[:500]!r}")
        return []

    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        co = {
            "company_name": str(item.get("company_name", "") or item.get("name", "")).strip(),
            "domain":       str(item.get("domain", "") or item.get("website", "")).strip(),
            "type":         item.get("type", "Private"),
            "revenue_estimate": item.get("revenue_estimate", "") or item.get("revenue", "—"),
            "headquarters": item.get("headquarters", "") or item.get("hq", ""),
            "employees_estimate": item.get("employees_estimate", "") or item.get("employees", ""),
            "description":  item.get("description", "") or item.get("about", ""),
        }
        if co["company_name"]:
            out.append(co)
    logger.info(f"Company list parsed {len(out)} companies for {industry}/{geography}")
    return out[:50]


# ── Phase 2: deal search ──────────────────────────────────────────────────────

async def search_industry_deals(
    companies: list[dict],
    industry: str,
    geography: str,
    renewal_timeframe: str,
    focus_tech: str = "",
) -> AsyncGenerator[dict, None]:
    from usage_logger import new_run_id, get_usage_by_run
    run_id = new_run_id()

    renewal_days = RENEWAL_WINDOW_DAYS.get(renewal_timeframe, 365)
    today = datetime.date.today()
    window_end = today + datetime.timedelta(days=renewal_days)

    total = len(companies)
    processed = 0
    BATCH_SIZE = 5
    sem = asyncio.Semaphore(5)

    yield {"type": "heartbeat", "message": f"🔍 Searching IT deals for {total} companies…"}
    await asyncio.sleep(0)

    async def _search_one(co: dict) -> list[dict]:
        cname  = co.get("company_name", "")
        cdomain = co.get("domain", "")
        async with sem:
            try:
                text = await asyncio.wait_for(
                    asyncio.to_thread(_gemini_sync, _deals_prompt(cname, cdomain, industry, geography, focus_tech),
                                      f"deals|{cname[:20]}", run_id),
                    timeout=CALL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Deal search timed out for {cname}")
                return []
            except Exception as e:
                logger.warning(f"Deal search error for {cname}: {e}")
                return []

        parsed = _parse_json(text)
        if not isinstance(parsed, list):
            return []

        deals = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            # Always recompute renewal date server-side for consistency
            announced  = item.get("announced_date", "")
            duration   = item.get("duration_years", 0)
            renewal_dt = _compute_renewal_date(announced, duration)
            if renewal_dt:
                item["estimated_renewal_date"] = renewal_dt.strftime("%Y-%m")
                item["in_renewal_window"] = (today <= renewal_dt <= window_end)
            else:
                item["estimated_renewal_date"] = "—"
                item["in_renewal_window"] = False

            item["company_name"] = cname
            item["domain"] = cdomain
            deals.append(item)
        return deals

    # Process in batches of BATCH_SIZE
    for batch_start in range(0, total, BATCH_SIZE):
        batch = companies[batch_start: batch_start + BATCH_SIZE]
        names = [c.get("company_name", "") for c in batch]
        yield {"type": "heartbeat", "message": f"📋 Batch {batch_start // BATCH_SIZE + 1}: {', '.join(names)}…"}
        await asyncio.sleep(0)

        tasks   = [_search_one(co) for co in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for co, result in zip(batch, results):
            processed += 1
            if isinstance(result, Exception):
                logger.error(f"Batch error for {co.get('company_name')}: {result}")
                continue
            for deal in (result or []):
                yield {"type": "deal", "deal": deal}
                await asyncio.sleep(0.02)

        yield {
            "type":      "batch_done",
            "processed": processed,
            "total":     total,
            "message":   f"✓ {processed}/{total} companies searched",
        }
        await asyncio.sleep(0)

    yield {"type": "complete", "processed": processed, "total": total, "usage": get_usage_by_run(run_id), "run_id": run_id}
