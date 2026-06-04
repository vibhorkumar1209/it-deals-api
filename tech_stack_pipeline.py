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
    {"key": "vendor",              "label": "Tech"},
    {"key": "integration_partner", "label": "Implementation Partner"},
    {"key": "last_detected",       "label": "Last Detected"},
    {"key": "tech_install",        "label": "Install Size (approx)"},
    {"key": "renewal_date",        "label": "Renewal"},
    {"key": "confidence_score",    "label": "Confidence"},
]

FIELD_KEYS = [f["key"] for f in TECH_STACK_FIELDS]

# ── Category taxonomy ─────────────────────────────────────────────────────────

# Canonical tech_stack_category values → must match exactly in output
TECH_STACK_CATEGORIES = {
    "Core Enterprise Operations": [
        "ERP & Finance",           # NetSuite, SAP, QuickBooks, Oracle Financials
        "HR & Payroll",            # Rippling, Gusto, Workday, Deel, ADP
        "Internal Communications", # Slack, Teams, Google Workspace, Zoom
        "Project & Knowledge Management",  # Jira, Notion, Asana, Monday, Confluence
        "Contract Lifecycle Management",   # Ironclad, DocuSign CLM, Icertis
        "Procurement & Source-to-Pay",     # Coupa, Ariba, Ivalua, Jaggaer
        "ITSM & Service Desk",             # ServiceNow, BMC Remedy, Jira Service Mgmt
    ],
    "Customer-Facing & Revenue": [
        "CRM & Account Management",  # Salesforce, HubSpot, Microsoft Dynamics
        "Customer Support & Helpdesk", # Zendesk, Intercom, Gorgias, Freshdesk
        "Marketing Automation",      # Klaviyo, Braze, Marketo, Mailchimp, Pardot
        "Sales Intelligence",        # ZoomInfo, Apollo, Outreach, Salesloft, Gong
        "Billing & Subscription",    # Stripe, Chargebee, Zuora, Recurly
        "E-Commerce Platform",       # Shopify, Magento, commercetools, SAP Commerce
        "CPQ & Configure-Price-Quote", # Salesforce CPQ, DealHub, Conga
    ],
    "Infrastructure & Cloud": [
        "Cloud Hosting",    # AWS, Azure, GCP, Vercel, Render
        "CDN & DNS",        # Cloudflare, Akamai, Fastly, AWS CloudFront
        "Databases",        # PostgreSQL, MongoDB, MySQL, Supabase, Oracle DB, Redis
        "Identity & IAM",   # Okta, Auth0, Clerk, Microsoft Entra ID, SailPoint
        "Collaboration & Productivity", # Microsoft 365, Google Workspace, Slack
        "Device Management / MDM",      # Jamf, Intune, VMware Workspace ONE
        "Network & VPN",               # Cisco, Zscaler, Palo Alto Prisma
    ],
    "Development & Engineering": [
        "Version Control",      # GitHub, GitLab, Bitbucket, Azure DevOps
        "CI/CD Automation",     # Jenkins, GitHub Actions, CircleCI, ArgoCD
        "APM & Monitoring",     # Datadog, New Relic, Sentry, Dynatrace, Elastic
        "API Management",       # MuleSoft, Apigee, Kong, AWS API Gateway
        "Container & Orchestration",  # Kubernetes, Docker, OpenShift
        "iPaaS & Integration",  # Zapier, Make, Workato, Boomi, MuleSoft
        "Low-Code / No-Code",   # OutSystems, Mendix, Power Apps, Appian
    ],
    "Data Analytics & AI": [
        "Data Warehousing",       # Snowflake, BigQuery, Databricks, Redshift
        "Data Integration & ETL", # Fivetran, Airbyte, Airflow, dbt, Informatica
        "Business Intelligence",  # Tableau, Power BI, Looker, Qlik, MicroStrategy
        "Product Analytics",      # Google Analytics, Mixpanel, Amplitude, Heap
        "AI/ML Infrastructure",   # OpenAI API, LangChain, Pinecone, Vertex AI, SageMaker
        "Data Catalogue & Governance", # Collibra, Alation, Atlan
    ],
    "Security & Compliance": [
        "Cybersecurity / EDR",    # CrowdStrike, SentinelOne, Microsoft Defender
        "SIEM & Threat Detection", # Splunk, Microsoft Sentinel, IBM QRadar
        "Vulnerability Management", # Tenable, Qualys, Rapid7
        "GRC & Compliance",       # ServiceNow GRC, MetricStream, Vanta, Drata
        "DLP & Data Security",    # Symantec DLP, Forcepoint, Nightfall
        "Zero Trust / ZTNA",      # Zscaler, Cloudflare Access, Palo Alto Prisma
    ],
    "Unclassified": [
        "Unclassified Tools",     # Any clear software not fitting above
    ],
}

# Flat list of all valid tech_stack_category values
ALL_CATEGORIES_FLAT = [cat for cats in TECH_STACK_CATEGORIES.values() for cat in cats]

WIDE_CATEGORIES = "\n".join(
    f"\n{core}:\n" + "\n".join(f"  - {cat}" for cat in cats)
    for core, cats in TECH_STACK_CATEGORIES.items()
)

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

    FOCUS_MAP = {
        1: f'enterprise software stack of {company_name}: ERP, HR, finance, ITSM, communications, project management, collaboration tools',
        2: f'technology vendors used by {company_name}: CRM, cloud infrastructure, cybersecurity, identity, databases, marketing tools',
        3: f'software tools at {company_name}: DevOps, data analytics, AI/ML, BI, API management, integration platforms, security',
    }
    focus = FOCUS_MAP.get(call_num, f'complete software tool stack of {company_name}')

    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in TECH_STACK_FIELDS)
    fields_example = {f["key"]: f"<{f['label']}>" for f in TECH_STACK_FIELDS}

    return f"""You are a technographic research analyst. Use Google Search to find the technology stack of {company_name}.

COMPANY: {company_name}{f" | {domain}" if domain else ""}{linkedin_block}

RESEARCH FOCUS: Find the {focus} (last 5 years, 2021–2026).

Search for: job postings mentioning required tools, vendor case studies, press releases, privacy policies listing third-party software, technology partnerships, and annual reports.

For each software tool found, return one JSON object with these EXACT keys:
{fields_desc}

Rules:
- core_tech_category: Core Enterprise Operations | Customer-Facing & Revenue | Infrastructure & Cloud | Development & Engineering | Data Analytics & AI | Security & Compliance | Unclassified
- tech_stack_category: use standard names like "ERP & Finance", "CRM & Account Management", "Cloud Hosting" etc., or a descriptive name if none fits
- integration_partner: SI/consulting firm that implemented it (e.g. "Accenture", "TCS") or "-"
- last_detected: e.g. "Active – 2025", "2024 job posting"
- tech_install: estimated seat/license range e.g. "1,000–5,000 seats", "Enterprise-wide"
- renewal_date: estimated quarter e.g. "Q2 2027"
- confidence_score: e.g. "85% – job posting", "94% – vendor case study"

Return ONLY a raw JSON array, no markdown:
[{json.dumps(fields_example)}]
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

    import time as _time
    MAX_RETRIES = 3
    response = None

    HTTP_TIMEOUT = 90   # hard timeout on the underlying HTTP request

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Set HTTP-level timeout so a hung Gemini server can't block forever
            client = genai.Client(
                api_key=GOOGLE_AI_KEY,
                http_options={"timeout": HTTP_TIMEOUT},
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            break   # success
        except Exception as api_err:
            err_str = str(api_err)
            is_quota = "RESOURCE_EXHAUSTED" in err_str or "free_tier" in err_str
            is_timeout = any(x in err_str.lower() for x in ("timeout", "timed out", "deadline"))
            is_retryable = not is_quota and any(x in err_str for x in ("503", "UNAVAILABLE", "overloaded")) or is_timeout
            logger.warning(f"Tech stack Gemini attempt {attempt}/{MAX_RETRIES} for {company_name}: {api_err}")
            if is_quota:
                raise RuntimeError("Gemini quota exhausted — upgrade to a paid API plan.") from api_err
            if is_retryable and attempt < MAX_RETRIES:
                _time.sleep(10 * attempt)
                continue
            logger.error(f"Gemini tech stack error for {company_name}: {api_err}", exc_info=True)
            return []

    if response is None:
        return []

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

        # Build flexible key map: handles exact keys, labels, label-as-snake_case, case-insensitive
        def _norm(s: str) -> str:
            return s.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").strip("_")

        key_map: dict[str, str] = {}
        for f in TECH_STACK_FIELDS:
            k, lbl = f["key"], f["label"]
            for variant in (k, k.lower(), lbl, lbl.lower(), _norm(lbl), lbl.replace(" ", "")):
                key_map[variant] = k
                key_map[variant.lower()] = k

        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row: dict = {}
            # Map whatever keys Gemini used → canonical FIELD_KEYS
            for gemini_key, val in item.items():
                canonical = key_map.get(gemini_key) or key_map.get(gemini_key.lower()) or key_map.get(_norm(gemini_key))
                if canonical and canonical not in row:
                    clean_val = str(val).strip() if val not in (None, "null", "None", "") else ""
                    if clean_val.lower() in ("", "unknown", "n/a", "na", "none"):
                        clean_val = "-"
                    row[canonical] = clean_val
            # Fill any missing fields with "-"
            for fk in FIELD_KEYS:
                if fk not in row:
                    row[fk] = "-"
            # Skip rows with fewer than 2 real values (likely parsing artifacts)
            real_vals = sum(1 for v in row.values() if v and v != "-")
            if real_vals < 2:
                continue
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

    # Derive a shorter "brand name" by stripping common subsidiary suffixes
    _STRIP = re.compile(
        r"\s+(north america|south america|europe|asia|apac|latam|"
        r"inc\.?|llc\.?|ltd\.?|corp\.?|corporation|group|holdings|"
        r"gmbh|ag|plc|sa|bv|nv|pty|co\.?)\s*$",
        re.IGNORECASE,
    )
    brand_name = _STRIP.sub("", company_name).strip()
    search_name = company_name  # used in queries — may be widened to brand_name on fallback

    mode = "wide-spectrum" if not fc and not fv else "laser-focused"
    num_calls = 3
    yield {"type": "heartbeat", "message": f"🔍 Tech stack scan for {company_name} ({mode} mode)…"}
    await asyncio.sleep(0)

    seen_keys: set[str] = set()
    total = 0
    CALL_TIMEOUT = 120

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
                yield {"type": "heartbeat", "message": f"🌐 [{call_num}/{num_calls}] Scanning… ({elapsed}s)"}
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

    # ── Fallback: retry with brand name if subsidiary name returned nothing ──────
    if total == 0 and brand_name.lower() != company_name.lower():
        yield {"type": "heartbeat",
               "message": f"🔄 No results for '{company_name}' — retrying as '{brand_name}'…"}
        await asyncio.sleep(0)

        fallback_prompt = _build_tech_stack_prompt(brand_name, domain, linkedin_url, fc, fv, 1)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_tech_stack_sync, fallback_prompt, brand_name)

        elapsed = 0
        fallback_tools: list[dict] = []
        while elapsed < CALL_TIMEOUT:
            try:
                fallback_tools = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat", "message": f"🌐 Fallback scan… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Fallback error for {brand_name}: {e}", exc_info=True)
                break
        else:
            future.cancel()

        for tool in fallback_tools:
            dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_stack_category','').lower()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok"}
            row.update(tool)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.04)
            total += 1

        if total > 0:
            yield {"type": "heartbeat", "message": f"✅ Fallback found {total} tools via '{brand_name}'"}
            await asyncio.sleep(0)

    if total == 0:
        yield {"type": "heartbeat", "message": f"⚠️ No tech stack data found for {company_name}"}
        yield {"type": "row_done", "row": {
            "company_name": company_name, "domain": domain, "_status": "no_result",
            **{k: "—" for k in FIELD_KEYS}
        }}
