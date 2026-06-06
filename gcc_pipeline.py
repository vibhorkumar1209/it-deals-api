"""
GCC Intelligence Hub pipeline — Global Capability Centre mapping.

Searches for all GCCs (Global Capability Centres / Captive Centres / Offshore
Development Centres) of a target company across the globe.

Output per GCC:
  - Company Name
  - Location of GCC (City, Country)
  - Key Tech Projects (Languages, Cloud, Containers, Data/MLOps)
  - Key Executives (top 3)
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

# ── Output schema ─────────────────────────────────────────────────────────────
GCC_FIELDS = [
    {"key": "company_name",    "label": "Company Name"},
    {"key": "gcc_name",        "label": "GCC / Centre Name"},
    {"key": "location",        "label": "Location (City, Country)"},
    {"key": "size",            "label": "Size (Headcount est.)"},
    {"key": "established",     "label": "Established"},
    {"key": "tech_projects",   "label": "Key Tech Projects"},
    {"key": "languages",       "label": "Languages / Frameworks"},
    {"key": "cloud",           "label": "Cloud & Containers"},
    {"key": "data_mlops",      "label": "Data / MLOps"},
    {"key": "executives",      "label": "Key Executives (top 3)"},
    {"key": "source",          "label": "Source"},
]


def _gcc_search_prompt(company_name: str, domain: str, location: str) -> str:
    loc_hint = f" Focus on {location}." if location else ""
    return f"""You are a GCC (Global Capability Centre) research analyst with live Google Search.

COMPANY: {company_name}
TASK: Find ALL Global Capability Centres, Captive Centres, Offshore Development Centres,
and Technology Hubs operated by {company_name} globally.{loc_hint}

Run these searches:
- "{company_name}" GCC "Global Capability Centre" location city
- "{company_name}" captive centre offshore development centre
- "{company_name}" technology hub engineering centre India Poland Germany
- "{company_name}" GCC headcount employees engineers
- "{company_name}" GCC CEO head director managing director
- site:linkedin.com "{company_name}" GCC OR "Global Capability"
- "{company_name}" GCC technology stack cloud AI projects

For EACH GCC/centre found, extract:
- Its exact name and location (city + country)
- Estimated headcount
- Year established
- Key technology projects and initiatives
- Programming languages and frameworks used
- Cloud platforms and containerisation tools
- Data and MLOps platforms
- Top 3 executives (name + title)

Return ONLY a JSON array — one object per GCC:
[
  {{
    "company_name": "{company_name}",
    "gcc_name": "<exact name of the GCC or centre e.g. '{company_name} India GCC', '{company_name} Pune Technology Centre'>",
    "location": "<City, Country e.g. 'Pune, India' or 'Warsaw, Poland'>",
    "size": "<estimated headcount e.g. '2,000–3,000 engineers' or '500+'>",
    "established": "<year established e.g. '2018' or '-'>",
    "tech_projects": "<key technology projects and work e.g. 'Connected vehicle platform, AI warranty analytics, SAP S/4HANA rollout'>",
    "languages": "<programming languages/frameworks e.g. 'Java, Python, React, Node.js'>",
    "cloud": "<cloud and container platforms e.g. 'AWS, Azure Kubernetes Service, Docker'>",
    "data_mlops": "<data/analytics/MLOps tools e.g. 'Databricks, Apache Kafka, MLflow, Snowflake'>",
    "executives": "<top 3 executives: Name (Title), Name (Title), Name (Title) — search LinkedIn and press releases>",
    "source": "<URL to LinkedIn page, press release, or news article>"
  }}
]

IMPORTANT:
- Return one object per DISTINCT GCC location
- If a company has GCCs in Bengaluru, Pune, Warsaw, and Mexico City — return 4 separate objects
- Populate every field; use '-' only if genuinely not found after searching
- Return ONLY the raw JSON array. No prose. No markdown.
"""


def _gcc_call_sync(prompt: str, company_name: str) -> list[dict]:
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
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=8192,
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
            logger.error(f"GCC call error for {company_name}: {e}")
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

    # Parse JSON
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
    except Exception:
        pass

    try:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            text = re.sub(r",\s*([\]}])", r"\1", m.group(0))
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [r for r in parsed if isinstance(r, dict)]
    except Exception:
        # Attempt truncation recovery
        try:
            start = raw.find("[")
            if start >= 0:
                fragment = raw[start:]
                last = fragment.rfind("}")
                if last >= 0:
                    fragment = fragment[:last+1].rstrip().rstrip(",") + "\n]"
                    parsed = json.loads(fragment)
                    if isinstance(parsed, list):
                        return [r for r in parsed if isinstance(r, dict)]
        except Exception:
            pass

    return []


async def _run_with_heartbeat(
    prompt: str,
    company_name: str,
    timeout: int = 120,
) -> tuple[list[dict], bool]:
    """Run gcc_call in thread, yield heartbeats. Returns (rows, timed_out)."""
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, _gcc_call_sync, prompt, company_name)
    elapsed = 0
    while elapsed < timeout:
        try:
            rows = await asyncio.wait_for(asyncio.shield(future), timeout=10)
            return rows, False
        except asyncio.TimeoutError:
            elapsed += 10
    future.cancel()
    return [], True


# ── Main async generator ──────────────────────────────────────────────────────

async def run_gcc_intelligence(
    company_name: str,
    domain: str = "",
    location: str = "",
    target_vendor: str = "",
    focus_domains: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Searches for all GCCs of a company globally.
    Yields: heartbeat | gcc_row | complete
    """
    yield {"type": "heartbeat", "message": f"🔍 Searching for {company_name} Global Capability Centres worldwide…"}
    await asyncio.sleep(0)

    # Call 1: broad global GCC search
    prompt1 = _gcc_search_prompt(company_name, domain, location)
    yield {"type": "heartbeat", "message": f"🌐 Scanning GCC locations, tech projects & executives…"}
    await asyncio.sleep(0)

    rows1, timed_out = await _run_with_heartbeat(prompt1, company_name, timeout=120)

    if timed_out:
        yield {"type": "heartbeat", "message": "⏱ Initial search timed out — trying with focused query…"}
        await asyncio.sleep(0)
    else:
        yield {"type": "heartbeat", "message": f"✅ Found {len(rows1)} GCC locations — enriching executive data…"}
        await asyncio.sleep(0)

    seen = set()
    all_rows = []
    for row in rows1:
        key = f"{row.get('location','').lower()}|{row.get('gcc_name','').lower()}"
        if key in seen:
            continue
        seen.add(key)
        row["company_name"] = row.get("company_name") or company_name
        all_rows.append(row)
        yield {"type": "gcc_row", "row": row}
        await asyncio.sleep(0.05)

    # Call 2: executive enrichment pass if we found GCCs
    if all_rows:
        exec_prompt = f"""You are a GCC executive researcher with live Google Search.

COMPANY: {company_name}

For each of the following GCC locations, find the top 3 executives (name + title):
{json.dumps([{{"gcc_name": r.get("gcc_name",""), "location": r.get("location","")}} for r in all_rows[:10]], indent=1)}

Search LinkedIn, company websites, and news for each location:
{chr(10).join(f'- "{company_name}" GCC {r.get("location","").split(",")[0]} head director CEO managing' for r in all_rows[:6])}

Return ONLY a JSON array — one object per GCC with updated executives:
[
  {{
    "gcc_name": "<exact gcc_name from above>",
    "location": "<exact location from above>",
    "executives": "<Name (Title), Name (Title), Name (Title)>",
    "source": "<LinkedIn or news URL>"
  }}
]

Return ONLY the raw JSON array.
"""
        yield {"type": "heartbeat", "message": "👥 Enriching executive data for each GCC…"}
        await asyncio.sleep(0)

        exec_rows, _ = await _run_with_heartbeat(exec_prompt, company_name, timeout=100)

        # Merge executive data back
        exec_map = {}
        for er in (exec_rows if isinstance(exec_rows, list) else []):
            if isinstance(er, dict):
                key = er.get("gcc_name","").lower()
                exec_map[key] = er

        for row in all_rows:
            key = row.get("gcc_name","").lower()
            if key in exec_map:
                er = exec_map[key]
                if er.get("executives") and er["executives"] != "-":
                    row["executives"] = er["executives"]
                if er.get("source") and er["source"] != "-" and (not row.get("source") or row.get("source") == "-"):
                    row["source"] = er["source"]

    total = len(all_rows)
    yield {"type": "heartbeat", "message": f"✅ Done — {total} GCC{'s' if total != 1 else ''} found for {company_name}"}
    await asyncio.sleep(0)

    if total == 0:
        yield {"type": "heartbeat", "message": f"⚠️ No GCCs found — try searching for the parent company name"}

    yield {
        "type": "complete",
        "gcc_rows": all_rows,
        "total": total,
    }
