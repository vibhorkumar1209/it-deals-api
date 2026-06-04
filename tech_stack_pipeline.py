"""
Tech Stack Finder pipeline — Gemini 2.5 Flash + Google Search Grounding.

Constructs a technographic footprint for a target company by searching
public digital footprints: DNS, job postings, privacy disclosures, source
code fragments, and technology news.
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

TECH_STACK_FIELDS = [
    {"key": "core_tech_category",  "label": "Core Tech Category"},
    {"key": "tech_stack_category", "label": "Tech Stack Category"},
    {"key": "vendor",              "label": "Vendor"},
    {"key": "integration_partner", "label": "Integration Partner"},
    {"key": "last_detected",       "label": "Last Detected"},
    {"key": "tech_install",        "label": "Install Size (approx)"},
    {"key": "renewal_date",        "label": "Renewal (est.)"},
    {"key": "confidence_score",    "label": "Confidence"},
]

FIELD_KEYS = [f["key"] for f in TECH_STACK_FIELDS]

# ── Category taxonomy ─────────────────────────────────────────────────────────
WIDE_CATEGORIES = """
Core Enterprise Operations:
  ERP, HCM/HRMS, Finance & Accounting, Procurement & Source-to-Pay, Legal Tech, ITSM

Customer-Facing & Revenue:
  CRM, Marketing Automation, CDP, E-Commerce Platform, Customer Support / Helpdesk,
  Sales Engagement, Loyalty & Personalisation, CPQ

Infrastructure & Cloud:
  Cloud Hosting (IaaS), CDN, DNS & Domain, Email Hosting / MX,
  Identity & IAM / SSO, VPN & Network, Collaboration & Productivity,
  Video Conferencing, Device Management / MDM

Development & Engineering:
  Source Control & CI/CD, APM & Observability, Feature Flagging,
  API Gateway, Container Orchestration, Low-Code / No-Code

Data Analytics & AI:
  Data Warehouse, ETL / ELT, BI & Dashboards, AI/ML Platform,
  Data Catalogue, Real-Time Streaming, Customer Data Platform

Security & Compliance:
  Endpoint Security / EDR, SIEM, Identity Threat Detection,
  Vulnerability Management, DLP, GRC / Compliance Automation,
  Zero Trust / ZTNA
"""

CONFIDENCE_GUIDE = """
Confidence scoring:
  95-99% — visible JS pixel / MX record / DNS TXT / public API key prefix
  85-94% — named in official vendor case study, press release, or annual report
  75-84% — named explicitly in job posting (e.g. "must have Salesforce experience")
  60-74% — generic inference from industry norm or legacy stack patterns
  50-59% — indirect signal (partner listing, integration marketplace badge)
"""


def _build_tech_stack_prompt(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_categories: list[str],
    focus_vendors: list[str],
    call_num: int,
) -> str:
    linkedin_block = f" | LinkedIn: {linkedin_url}" if linkedin_url else ""

    # Determine mode
    wide_mode = not focus_categories and not focus_vendors
    mode_label = "WIDE-SPECTRUM" if wide_mode else "LASER-FOCUSED COMPETITIVE"

    if wide_mode:
        scope_block = f"""MODE: WIDE-SPECTRUM — sweep ALL tech stack categories below.
{WIDE_CATEGORIES}"""
    else:
        cat_block = ""
        if focus_categories:
            cat_block = "Focus categories: " + ", ".join(focus_categories)
        vendor_block = ""
        if focus_vendors:
            vendor_block = (
                "Focus vendors (and their direct competitors): " + ", ".join(focus_vendors)
            )
        scope_block = f"""MODE: LASER-FOCUSED — research ONLY these areas.
{cat_block}
{vendor_block}
For each listed vendor, also look for direct market competitors in the same functional bucket."""

    if call_num == 1:
        search_block = f"""SEARCHES TO RUN (Call 1 — digital footprint & public signals):
  - site:{domain} OR inurl:{domain} technology stack tools
  - "{company_name}" software tools vendors uses deployed
  - "{company_name}" job posting OR careers requires experience with
  - "{company_name}" technology partner OR case study
  - "{company_name}" privacy policy OR cookie policy third-party tools
  - site:builtwith.com OR site:similartech.com OR site:stackshare.io "{company_name}"
  - site:g2.com OR site:capterra.com "{company_name}" review uses
  - "{company_name}" ERP CRM cloud platform deployed implemented"""
    else:
        search_block = f"""SEARCHES TO RUN (Call 2 — vendor confirmations & job signals):
  - "{company_name}" SAP OR Oracle OR Microsoft OR Salesforce OR ServiceNow OR Workday
  - "{company_name}" AWS OR Azure OR Google Cloud OR hybrid cloud
  - "{company_name}" cybersecurity OR SIEM OR endpoint OR zero trust
  - "{company_name}" data warehouse OR analytics OR BI OR Snowflake OR Databricks
  - "{company_name}" DevOps OR CI/CD OR Kubernetes OR observability
  - "{company_name}" site:linkedin.com/jobs OR careers technology stack required
  - "{company_name}" annual report technology infrastructure spend
  - "{company_name}" vendor partnership OR integration"""

    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in TECH_STACK_FIELDS)
    fields_example = {f["key"]: f"<{f['label']}>" for f in TECH_STACK_FIELDS}

    return f"""You are a market intelligence technographic analyst with live Google Search.

TARGET: {company_name} | Website: {domain}{linkedin_block}

{scope_block}

{search_block}

{CONFIDENCE_GUIDE}

EXTRACTION RULES:
- One JSON object per distinct software/tool deployment
- Populate ALL 8 fields — no blanks allowed
- integration_partner: list 1-3 key systems this tool connects to (comma-separated)
- last_detected: e.g. "Active – Q2 2026" or "Detected May 2026 job posting"
- tech_install: approximate license/seat count as a range e.g. "500–1,000 seats", "10,000–50,000 users", "Enterprise-wide", "Dept-level ~50–200 seats"
- renewal_date: estimated next renewal e.g. "Est. Q1 2027" based on typical SaaS cycles
- confidence_score: percentage string e.g. "87%" with brief reason e.g. "87% – job posting"

core_tech_category must be one of:
  Core Enterprise Operations | Customer-Facing & Revenue | Infrastructure & Cloud |
  Development & Engineering | Data Analytics & AI | Security & Compliance

Return ONLY a valid JSON array — no prose, no markdown fences:
[
  {{
{fields_desc}
  }}
]

Example shape: {json.dumps(fields_example)}
"""


def _gemini_tech_stack_sync(prompt: str, company_name: str) -> list[dict]:
    """Blocking Gemini call. Runs inside run_in_executor."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai not installed")
        return []

    if not GOOGLE_AI_KEY:
        logger.error("GOOGLE_AI_API_KEY not set")
        return []

    try:
        client = genai.Client(api_key=GOOGLE_AI_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                temperature=0.1,
                max_output_tokens=16384,
            ),
        )

        # Always extract via parts — response.text raises with grounding
        raw_text = ""
        try:
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts or []):
                    t = getattr(part, "text", None)
                    if t:
                        raw_text += t
        except Exception:
            try:
                raw_text = response.text or ""
            except Exception:
                pass

        finish = "unknown"
        try:
            finish = str(response.candidates[0].finish_reason)
        except Exception:
            pass

        logger.info(f"Tech stack Gemini: {len(raw_text)} chars, finish={finish} for {company_name}")

        if not raw_text:
            return []

    except Exception as e:
        logger.error(f"Gemini tech stack error for {company_name}: {e}", exc_info=True)
        return []

    # ── Parse JSON ────────────────────────────────────────────────────────────
    try:
        clean = re.sub(r"```(?:json)?\s*", "", raw_text.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            array_match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not array_match:
                # Attempt recovery from truncated response
                start = clean.find("[")
                if start == -1:
                    return []
                fragment = clean[start:]
                last_brace = fragment.rfind("}")
                if last_brace == -1:
                    return []
                fragment = fragment[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError:
                    return []
            else:
                candidate_text = array_match.group(0)
                try:
                    parsed = json.loads(candidate_text)
                except json.JSONDecodeError:
                    candidate_text = re.sub(r",\s*([\]}])", r"\1", candidate_text)
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        last_brace = candidate_text.rfind("}")
                        if last_brace == -1:
                            return []
                        fragment = candidate_text[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                        fragment = "[" + fragment.lstrip("[")
                        try:
                            parsed = json.loads(fragment)
                        except json.JSONDecodeError:
                            return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []

        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row: dict = {}
            for key in FIELD_KEYS:
                val = item.get(key)
                row[key] = str(val) if val not in (None, "null", "None", "") else "—"
            out.append(row)

        logger.info(f"Tech stack: parsed {len(out)} tools for {company_name}")
        return out

    except Exception as e:
        logger.error(f"Tech stack parse error for {company_name}: {e}")
        return []


# ── Main async generator ──────────────────────────────────────────────────────

async def find_tech_stack(
    company_name: str,
    domain: str,
    linkedin_url: str = "",
    focus_categories: list[str] | None = None,
    focus_vendors: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Yields heartbeat + row_done events for each detected tech tool.
    Makes 2 Gemini calls (digital footprint sweep + vendor confirmation).
    """
    fc = focus_categories or []
    fv = focus_vendors or []

    mode = "wide-spectrum" if not fc and not fv else "laser-focused"
    yield {"type": "heartbeat", "message": f"🔍 Tech stack scan for {company_name} ({mode} mode)…"}
    await asyncio.sleep(0)

    seen_keys: set[str] = set()
    total = 0
    CALL_TIMEOUT = 150

    for call_num in range(1, 3):
        label = "digital footprint & public signals" if call_num == 1 else "vendor confirmations & job signals"
        yield {"type": "heartbeat", "message": f"🌐 [{call_num}/2] Scanning {company_name}: {label}…"}
        await asyncio.sleep(0)

        prompt = _build_tech_stack_prompt(company_name, domain, linkedin_url, fc, fv, call_num)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_tech_stack_sync, prompt, company_name)

        elapsed = 0
        call_tools: list[dict] = []

        while elapsed < CALL_TIMEOUT:
            try:
                call_tools = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat", "message": f"🌐 [{call_num}/2] Scanning… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Tech stack call {call_num} error for {company_name}: {e}", exc_info=True)
                yield {"type": "heartbeat", "message": f"⚠️ Call {call_num} error: {e}"}
                call_tools = []
                break
        else:
            future.cancel()
            yield {"type": "heartbeat", "message": f"⏱ Call {call_num} timed out — partial results below"}

        new_tools = 0
        for tool in call_tools:
            dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_stack_category','').lower()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok"}
            row.update(tool)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.04)
            new_tools += 1
            total += 1

        yield {"type": "heartbeat", "message": f"✅ Call {call_num} done: +{new_tools} tools (total {total})"}
        await asyncio.sleep(0)

    if total == 0:
        yield {"type": "heartbeat", "message": f"⚠️ No tech stack data found for {company_name}"}
        yield {"type": "row_done", "row": {
            "company_name": company_name, "domain": domain, "_status": "no_result",
            **{k: "—" for k in FIELD_KEYS}
        }}
