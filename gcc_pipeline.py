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
    loc_filter = f"\nLOCATION FILTER: Only return GCCs in or near: {location}" if location else \
                 "\nCOVERAGE: Return ALL GCC locations worldwide — do NOT limit to any region."
    return f"""You are a GCC (Global Capability Centre) research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}
{loc_filter}

Run ALL of the following searches to ensure complete global coverage:
- "{company_name}" "Global Capability Centre" location
- "{company_name}" GCC site:linkedin.com
- "{company_name}" captive centre offshore development centre
- "{company_name}" engineering centre technology hub India
- "{company_name}" engineering centre technology hub Poland Germany Romania
- "{company_name}" engineering centre technology hub Mexico Hungary Czech Republic
- "{company_name}" engineering centre technology hub China Singapore Philippines
- "{company_name}" GCC Bengaluru Pune Hyderabad Chennai Mumbai
- "{company_name}" GCC Warsaw Krakow Budapest Bucharest Prague
- "{company_name}" technology hub employees headcount engineers
- "{company_name}" GCC CEO head director managing director executive
- site:businesswire.com OR site:prnewswire.com "{company_name}" GCC technology centre

For EACH GCC/centre found, extract ALL of the following in detail:

tech_projects: List SPECIFIC named projects, programmes, and initiatives — not generic categories.
  Examples of GOOD detail: "Project Phoenix (SAP S/4HANA migration, 2022-2024)",
  "Connected Truck Platform v2.0 (real-time telematics)", "AI Warranty Fraud Detection (Python, TensorFlow)",
  "Dealer Portal Modernisation (React, GraphQL)", "Kubernetes migration of 120 microservices"
  Be specific: include project names, scope, technologies used per project, and year if known.

Return ONLY a JSON array — one object per DISTINCT GCC location:
[
  {{
    "company_name": "{company_name}",
    "gcc_name": "<exact centre name e.g. '{company_name} India Technology Centre, Pune'>",
    "location": "<City, Country>",
    "size": "<headcount estimate e.g. '2,000–3,000 engineers'>",
    "established": "<year e.g. '2018'>",
    "tech_projects": "<SPECIFIC named projects with scope and tech stack per project — minimum 3 projects if found. E.g.: '1) Connected Vehicle Platform: real-time OTA updates using AWS IoT, Go microservices. 2) AI Warranty Analytics: anomaly detection using Python/TensorFlow, reduced fraud by 18%. 3) SAP S/4HANA Global Rollout: finance/logistics modules, 2021-2024'>",
    "languages": "<e.g. 'Java 17, Python 3.11, TypeScript, Go, Kotlin'>",
    "cloud": "<e.g. 'AWS (primary), Azure Kubernetes Service, Terraform, Docker, Helm'>",
    "data_mlops": "<e.g. 'Databricks, Apache Kafka, dbt, MLflow, Snowflake, Airflow'>",
    "executives": "<Name (Title), Name (Title), Name (Title) — from LinkedIn or press releases>",
    "source": "<URL>"
  }}
]

RULES:
- One object per distinct city/location
- tech_projects must be specific and detailed — generic entries like 'cloud projects' are not acceptable
- Cover ALL global locations unless a location filter is specified
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

    # Pass 1: broad global GCC search
    prompt1 = _gcc_search_prompt(company_name, domain, location)
    yield {"type": "heartbeat", "message": "🌐 Pass 1: Scanning all GCC locations, tech projects & executives…"}
    await asyncio.sleep(0)

    rows1, timed_out1 = await _run_with_heartbeat(prompt1, company_name, timeout=120)
    yield {"type": "heartbeat", "message": f"✅ Pass 1: {len(rows1)} locations found — running second pass for completeness…"}
    await asyncio.sleep(0)

    # Pass 2: targeted second sweep to catch any missed locations
    if not location:  # only run second pass in global mode
        prompt2 = f"""You are a GCC location researcher with live Google Search.

COMPANY: {company_name}

Search specifically for any GCC or technology centre locations NOT commonly listed:
- "{company_name}" GCC "technology centre" OR "tech hub" 2023 OR 2024 OR 2025
- "{company_name}" offshore centre Latin America Mexico Brazil Colombia
- "{company_name}" engineering centre Eastern Europe Portugal Spain
- "{company_name}" technology hub Southeast Asia Vietnam Malaysia Thailand
- "{company_name}" new GCC announced opened launched

List any NEW locations not already in this set: {[r.get('location','') for r in rows1][:20]}

Return ONLY a JSON array of NEW GCC locations (same schema):
[{{
  "company_name": "{company_name}",
  "gcc_name": "<name>",
  "location": "<City, Country>",
  "size": "<headcount or '-'>",
  "established": "<year or '-'>",
  "tech_projects": "<specific projects with tech stack>",
  "languages": "<languages/frameworks or '-'>",
  "cloud": "<cloud/containers or '-'>",
  "data_mlops": "<data/mlops tools or '-'>",
  "executives": "<executives or '-'>",
  "source": "<URL or '-'>"
}}]

Return [] if no new locations found beyond the list above.
Return ONLY the raw JSON array.
"""
        rows2, _ = await _run_with_heartbeat(prompt2, company_name, timeout=100)
        yield {"type": "heartbeat", "message": f"✅ Pass 2: {len(rows2)} additional locations found"}
        await asyncio.sleep(0)
    else:
        rows2 = []

    all_input_rows = rows1 + rows2
    seen = set()
    all_rows = []
    for row in all_input_rows:
        loc = row.get('location','').lower().strip()
        name = row.get('gcc_name','').lower().strip()
        key = f"{loc}|{name}" if name else loc
        if key in seen or not loc:
            continue
        seen.add(key)
        row["company_name"] = row.get("company_name") or company_name
        all_rows.append(row)
        yield {"type": "gcc_row", "row": row}
        await asyncio.sleep(0.05)

    yield {"type": "heartbeat", "message": f"📍 {len(all_rows)} unique GCC locations identified — enriching executives…"}
    await asyncio.sleep(0)

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
