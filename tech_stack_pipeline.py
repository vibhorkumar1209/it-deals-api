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
    {"key": "integration_partner", "label": "Implementation Partner"},
    {"key": "last_detected",       "label": "Last Detected"},
    {"key": "tech_install",        "label": "Install Size (approx)"},
    {"key": "renewal_date",        "label": "Renewal"},
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
        search_block = f"""SEARCHES — Call 1: Enterprise applications & public digital signals (2021–2026)
  - site:{domain} technology stack software tools platform
  - "{company_name}" software platform deployed implemented 2023 OR 2024 OR 2025 OR 2026
  - "{company_name}" ERP HCM CRM SCM finance platform 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" SAP OR Oracle OR Workday OR ServiceNow OR Salesforce
  - "{company_name}" Microsoft 365 OR Teams OR Azure OR Dynamics 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" careers job posting software tools skills required 2023 OR 2024 OR 2025
  - site:builtwith.com OR site:stackshare.io OR site:similartech.com "{company_name}"
  - "{company_name}" privacy policy OR cookie disclosure third-party software tools
  - "{company_name}" technology partner vendor case study 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" digital transformation program technology 2023 OR 2024 OR 2025 OR 2026"""

    elif call_num == 2:
        search_block = f"""SEARCHES — Call 2: Cloud, data, security & DevOps stack (2021–2026)
  - "{company_name}" AWS OR Azure OR Google Cloud OR hybrid cloud 2023 OR 2024 OR 2025 OR 2026
  - "{company_name}" Snowflake OR Databricks OR data warehouse OR data lake 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" cybersecurity endpoint SIEM zero trust identity 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" Palo Alto OR CrowdStrike OR Microsoft Defender OR Okta OR SailPoint
  - "{company_name}" DevOps CI/CD Kubernetes containers observability APM 2022 OR 2023 OR 2024
  - "{company_name}" Splunk OR Elastic OR Dynatrace OR Datadog OR New Relic
  - "{company_name}" network infrastructure SD-WAN data center 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" collaboration productivity tools intranet 2023 OR 2024 OR 2025
  - site:linkedin.com/jobs "{company_name}" technology skills required 2024 OR 2025
  - "{company_name}" annual report technology spend infrastructure 2023 OR 2024 OR 2025"""

    else:  # call 3
        search_block = f"""SEARCHES — Call 3: Customer-facing, analytics, supply chain & sector-specific (2021–2026)
  - "{company_name}" marketing automation CRM customer data platform 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" Salesforce OR HubSpot OR Marketo OR Adobe Experience OR Adobe Campaign
  - "{company_name}" supply chain SCM procurement WMS TMS 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" BI dashboard analytics Power BI Tableau Qlik 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" AI ML platform machine learning deployment 2023 OR 2024 OR 2025 OR 2026
  - "{company_name}" ecommerce OR customer portal OR self-service platform 2022 OR 2023 OR 2024
  - "{company_name}" ITSM ticketing ServiceNow OR Jira OR Remedy OR BMC 2022 OR 2023 OR 2024 OR 2025
  - "{company_name}" low-code no-code RPA automation platform 2022 OR 2023 OR 2024 OR 2025
  - site:g2.com OR site:capterra.com "{company_name}" software review uses
  - "{company_name}" vendor relationship partnership technology 2024 OR 2025 OR 2026"""

    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in TECH_STACK_FIELDS)
    fields_example = {f["key"]: f"<{f['label']}>" for f in TECH_STACK_FIELDS}

    return f"""You are a senior market intelligence technographic analyst with live Google Search.

TARGET: {company_name} | Website: {domain}{linkedin_block}

TEMPORAL SCOPE: LAST 5 YEARS ONLY (2021–2026).
Only include tools/platforms that are active or evidenced within this period.

{scope_block}

{search_block}

{CONFIDENCE_GUIDE}

TARGET: Find as many distinct tools as possible — large enterprises typically run 80–200+ tools.
Do NOT stop early. Exhaust every category before returning results.

CATEGORIES TO COVER EXHAUSTIVELY (find tools in ALL of these):
  Core Enterprise Operations: ERP, Finance/Accounting, HCM/HRMS, Procurement, Legal Tech, ITSM
  Customer-Facing & Revenue: CRM, Marketing Automation, CDP, E-Commerce, Support/Helpdesk, CPQ, Loyalty
  Infrastructure & Cloud: IaaS, CDN, DNS, Email/MX, IAM/SSO, VPN, Collaboration, Video Conf, MDM
  Development & Engineering: Source Control, CI/CD, APM, Feature Flags, API Gateway, Containers, Low-Code
  Data Analytics & AI: Data Warehouse, ETL/ELT, BI/Dashboards, AI/ML Platform, Data Catalogue, Streaming
  Security & Compliance: EDR/Endpoint, SIEM, IAM/PAM, Vulnerability Mgmt, DLP, GRC, Zero Trust/ZTNA

EXTRACTION RULES:
- One JSON object per distinct software/tool deployment
- Populate ALL 8 fields — no blanks allowed
- integration_partner: name of the SI, consulting firm, or vendor that implemented this software
  e.g. "Accenture", "TCS", "Deloitte", "IBM", "Capgemini", "Vendor-led" — use "-" if not found
- last_detected: e.g. "Active – Q2 2026", "2024 job posting", "2023 vendor case study"
- tech_install: approximate license/seat count as a range
  e.g. "200–500 seats", "5,000–20,000 users", "50,000–100,000 users", "Enterprise-wide (100,000+)"
  Base on company headcount, department size, and typical vendor seat-to-employee ratios
- renewal_date: next estimated renewal quarter e.g. "Q1 2027", "Q3 2026" — no "Est." prefix
- confidence_score: percentage with brief reason e.g. "87% – job posting", "94% – vendor case study"

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
                max_output_tokens=65536,
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
                # Normalise empty/unknown to "-"
            clean_val = str(val).strip() if val not in (None, "null", "None", "") else ""
            if clean_val.lower() in ("", "unknown", "n/a", "na", "none", "-"):
                clean_val = "-"
            row[key] = clean_val
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
    Makes 3 Gemini calls covering enterprise apps, cloud/security/DevOps, and customer/analytics.
    """
    fc = focus_categories or []
    fv = focus_vendors or []

    mode = "wide-spectrum" if not fc and not fv else "laser-focused"
    num_calls = 3
    yield {"type": "heartbeat", "message": f"🔍 Tech stack scan for {company_name} ({mode} mode)…"}
    await asyncio.sleep(0)

    seen_keys: set[str] = set()
    total = 0
    CALL_TIMEOUT = 150

    CALL_LABELS = {
        1: "enterprise apps & digital footprint",
        2: "cloud, security & DevOps",
        3: "customer-facing, analytics & sector tools",
    }

    for call_num in range(1, num_calls + 1):
        label = CALL_LABELS[call_num]
        yield {"type": "heartbeat", "message": f"🌐 [{call_num}/{num_calls}] Scanning {company_name}: {label}…"}
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
