"""
Aftermarket Deep Dive pipeline — Gemini 2.5 Flash + Google Search.

Analyses a company's aftermarket service operations across:
- Service maturity & capabilities
- Technology gaps vs industry benchmarks
- Competitive positioning
- Key investment signals
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

CAPABILITY_FIELDS = [
    {"key": "domain",           "label": "Domain"},
    {"key": "capability",       "label": "Capability"},
    {"key": "maturity_level",   "label": "Maturity"},   # Leading / Established / Developing / Basic / Gap
    {"key": "technology_used",  "label": "Technology Used"},
    {"key": "key_finding",      "label": "Key Finding"},
    {"key": "source",           "label": "Source"},
]

GAP_FIELDS = [
    {"key": "domain",           "label": "Domain"},
    {"key": "gap_description",  "label": "Gap / Opportunity"},
    {"key": "priority",         "label": "Priority"},   # Critical / High / Medium / Low
    {"key": "recommended_tech", "label": "Recommended Technology"},
    {"key": "benchmark",        "label": "Industry Benchmark"},
    {"key": "source",           "label": "Source"},
]

COMPETITOR_FIELDS = [
    {"key": "competitor",       "label": "Competitor"},
    {"key": "domain",           "label": "Domain"},
    {"key": "their_advantage",  "label": "Their Advantage"},
    {"key": "technology",       "label": "Technology"},
    {"key": "implication",      "label": "Implication"},
    {"key": "source",           "label": "Source"},
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


def _gemini_call_sync(prompt: str, use_search: bool, label: str) -> list[dict] | dict:
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
            logger.error(f"Aftermarket Gemini error [{label}]: {e}")
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


async def _run_async(prompt: str, use_search: bool, label: str, timeout: int = 100) -> list:
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


async def run_aftermarket_deep_dive(
    company_name: str,
    domain: str,
    industry: str = "",
    competitors: str = "",
) -> AsyncGenerator[dict, None]:
    """
    Yields: heartbeat | capability_row | gap_row | competitor_row | complete
    """
    yield {"type": "heartbeat", "message": f"🔍 Starting Aftermarket Deep Dive for {company_name}…"}
    await asyncio.sleep(0)

    industry_hint = f" ({industry})" if industry else ""
    comp_hint = f" Key competitors: {competitors}." if competitors else ""

    # ── Step 1: Capability Assessment ────────────────────────────────────────
    yield {"type": "heartbeat", "message": "📊 Assessing aftermarket service capabilities…"}
    await asyncio.sleep(0)

    cap_prompt = f"""You are an aftermarket service analyst with live Google Search.

COMPANY: {company_name}{industry_hint} | Website: {domain}

Search for information about {company_name}'s aftermarket service operations:
- "{company_name}" warranty service operations technology platform
- "{company_name}" field service management software tool
- "{company_name}" parts inventory management system
- "{company_name}" dealer network management platform
- "{company_name}" telematics connected vehicle IoT service
- "{company_name}" predictive maintenance AI analytics aftermarket
- "{company_name}" customer support service portal digital
- site:businesswire.com OR site:prnewswire.com "{company_name}" service aftermarket

Assess the company's maturity in each aftermarket domain based on what you find.

Return ONLY a JSON array:
[
  {{
    "domain": "<one of the aftermarket domains>",
    "capability": "<specific capability or function assessed>",
    "maturity_level": "<Leading | Established | Developing | Basic | Gap>",
    "technology_used": "<specific technology/platform if found, else 'Unknown'>",
    "key_finding": "<one sentence: what evidence supports this maturity rating>",
    "source": "<URL supporting this finding or '-'>"
  }}
]

Domains to cover: {', '.join(AFTERMARKET_DOMAINS)}

Maturity definitions:
- Leading: best-in-class, advanced AI/ML, fully digital
- Established: modern systems, good coverage, some automation
- Developing: partial implementation, transitioning from legacy
- Basic: manual or minimal technology
- Gap: no evidence of capability

Return ONLY the raw JSON array.
"""

    cap_rows = await _run_async(cap_prompt, True, "capability", timeout=120)
    for row in (cap_rows if isinstance(cap_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "capability_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Capabilities assessed: {len(cap_rows) if isinstance(cap_rows, list) else 0} findings"}
    await asyncio.sleep(0)

    # ── Step 2: Gap & Opportunity Analysis ───────────────────────────────────
    yield {"type": "heartbeat", "message": "🔎 Identifying technology gaps & opportunities…"}
    await asyncio.sleep(0)

    cap_summary = json.dumps(cap_rows[:20] if isinstance(cap_rows, list) else [], indent=1)
    gap_prompt = f"""You are an aftermarket technology consultant.

COMPANY: {company_name}{industry_hint}{comp_hint}

CAPABILITY ASSESSMENT DATA:
{cap_summary}

Based on the capability assessment above, identify the top technology gaps and investment
opportunities for {company_name} in aftermarket service operations.
Also search for industry benchmarks and best practices:
- aftermarket service technology trends 2024 2025 manufacturing automotive
- warranty management AI automation best practices
- predictive maintenance IoT connected service industry benchmark

Return ONLY a JSON array of the top 10-15 gaps/opportunities:
[
  {{
    "domain": "<aftermarket domain>",
    "gap_description": "<clear description of the gap or opportunity>",
    "priority": "<Critical | High | Medium | Low>",
    "recommended_tech": "<specific technology or vendor that could address this>",
    "benchmark": "<what industry leaders do in this area>",
    "source": "<URL or '-'>"
  }}
]

Return ONLY the raw JSON array.
"""

    gap_rows = await _run_async(gap_prompt, True, "gaps", timeout=120)
    for row in (gap_rows if isinstance(gap_rows, list) else []):
        if isinstance(row, dict):
            yield {"type": "gap_row", "row": row}
            await asyncio.sleep(0.04)

    yield {"type": "heartbeat", "message": f"✅ Gaps identified: {len(gap_rows) if isinstance(gap_rows, list) else 0} findings"}
    await asyncio.sleep(0)

    # ── Step 3: Competitive Positioning (if competitors provided) ─────────────
    comp_rows = []
    if competitors:
        yield {"type": "heartbeat", "message": f"🏆 Competitive benchmarking vs {competitors}…"}
        await asyncio.sleep(0)

        comp_list = [c.strip() for c in competitors.split(",") if c.strip()]
        comp_searches = "\n".join(f'- "{c}" aftermarket service technology platform capability' for c in comp_list[:4])

        comp_prompt = f"""You are a competitive intelligence analyst.

TARGET COMPANY: {company_name}{industry_hint}
COMPETITORS: {competitors}

Search for competitor aftermarket service capabilities:
{comp_searches}

Compare each competitor's aftermarket technology advantages vs {company_name}.

Return ONLY a JSON array:
[
  {{
    "competitor": "<competitor name>",
    "domain": "<aftermarket domain where they have advantage>",
    "their_advantage": "<what they do better>",
    "technology": "<technology or platform they use>",
    "implication": "<what this means for {company_name}>",
    "source": "<URL or '-'>"
  }}
]

Return ONLY the raw JSON array.
"""

        comp_rows = await _run_async(comp_prompt, True, "competitors", timeout=120)
        for row in (comp_rows if isinstance(comp_rows, list) else []):
            if isinstance(row, dict):
                yield {"type": "competitor_row", "row": row}
                await asyncio.sleep(0.04)

        yield {"type": "heartbeat", "message": f"✅ Competitive analysis: {len(comp_rows) if isinstance(comp_rows, list) else 0} findings"}
        await asyncio.sleep(0)

    yield {
        "type": "complete",
        "capabilities": cap_rows if isinstance(cap_rows, list) else [],
        "gaps": gap_rows if isinstance(gap_rows, list) else [],
        "competitors": comp_rows if isinstance(comp_rows, list) else [],
    }
