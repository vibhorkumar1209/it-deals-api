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

# Import shared taxonomy from enrich_pipeline
try:
    from enrich_pipeline import classify_tech, TECH_L3_LIST, estimate_deal_value
except ImportError:
    def classify_tech(l3, description=""): return ("", "", l3)
    def estimate_deal_value(row): return (row.get("deal_value") or "$10M", "Y")
    TECH_L3_LIST = ""

TECH_STACK_FIELDS = [
    {"key": "tech_level1",         "label": "Level 1"},
    {"key": "tech_level2",         "label": "Level 2"},
    {"key": "tech_level3",         "label": "Level 3"},
    {"key": "vendor",              "label": "Tech"},
    {"key": "integration_partner", "label": "Implementation Partner"},
    {"key": "last_detected",       "label": "Last Detected"},
    {"key": "tech_install",        "label": "Install Size (approx)"},
    {"key": "renewal_date",        "label": "Renewal"},
    {"key": "confidence_score",    "label": "Confidence"},
    {"key": "deal_value",         "label": "TCV"},
    {"key": "deal_acv",           "label": "ACV"},
    {"key": "deal_estimated",     "label": "Est"},
    {"key": "source_info",         "label": "Source"},
]

FIELD_KEYS = [f["key"] for f in TECH_STACK_FIELDS]

# ── Category taxonomy ─────────────────────────────────────────────────────────

# Canonical tech_stack_category values → must match exactly in output
TECH_STACK_CATEGORIES = {
    "Core Technology Stack": [
        "Programming Languages & Frameworks",  # Java, Python, Node.js, .NET, Go, React, Spring
        "Data Management & Streaming",         # Kafka, Spark, Flink, Redis, Cassandra, RabbitMQ
        "Cloud & Infrastructure",              # AWS, Azure, GCP, multi-cloud, on-prem hybrid
        "DevOps & CI/CD",                      # Jenkins, GitHub Actions, ArgoCD, CircleCI, GitLab CI
        "Container & Orchestration",           # Kubernetes, Docker, OpenShift, ECS
        "Version Control",                     # GitHub, GitLab, Bitbucket, Azure DevOps
        "API Management",                      # MuleSoft, Apigee, Kong, AWS API Gateway
        "iPaaS & Integration",                 # Zapier, Make, Workato, Boomi, MuleSoft
    ],
    "Enterprise & Financial Systems": [
        "ERP & Finance",                       # SAP, Oracle Financials, NetSuite, Workday Finance
        "Financial & Enterprise Systems",      # Bloomberg, Refinitiv, Finastra, Temenos, Murex, FIS
        "Payment Infrastructure",              # Stripe, Adyen, Worldpay, SWIFT, Visa DPS, Mastercard
        "Risk & Compliance Platforms",         # Wolters Kluwer, Moody's Analytics, SAS Risk, Axiom SL
        "HR & Payroll",                        # Workday HCM, SAP SuccessFactors, ADP, Rippling
        "Procurement & Source-to-Pay",         # Coupa, Ariba, Ivalua, Jaggaer
        "Contract Lifecycle Management",       # Ironclad, DocuSign CLM, Icertis
        "ITSM & Service Desk",                 # ServiceNow, BMC Remedy, Jira Service Mgmt
    ],
    "Customer-Facing & Revenue": [
        "CRM & Account Management",            # Salesforce, HubSpot, Microsoft Dynamics
        "Customer Support & Helpdesk",         # Zendesk, Intercom, Freshdesk
        "Marketing Automation",                # Klaviyo, Braze, Marketo, Pardot
        "Sales Intelligence",                  # ZoomInfo, Apollo, Outreach, Salesloft, Gong
        "Billing & Subscription",              # Stripe, Chargebee, Zuora, Recurly
        "E-Commerce Platform",                 # Shopify, Magento, commercetools, SAP Commerce
        "CPQ & Configure-Price-Quote",         # Salesforce CPQ, DealHub, Conga
    ],
    "Infrastructure & Cloud": [
        "Cloud Hosting",                       # AWS, Azure, GCP, Vercel, Render
        "CDN & DNS",                           # Cloudflare, Akamai, Fastly
        "Databases",                           # PostgreSQL, MongoDB, MySQL, Oracle DB, Redis
        "Identity & IAM",                      # Okta, Auth0, Microsoft Entra ID, SailPoint
        "Collaboration & Productivity",        # Microsoft 365, Google Workspace, Slack
        "Device Management / MDM",             # Jamf, Intune, VMware Workspace ONE
        "Network & VPN",                       # Cisco, Zscaler, Palo Alto Prisma
    ],
    "Data Analytics & AI": [
        "Data Warehousing",                    # Snowflake, BigQuery, Databricks, Redshift
        "Data Integration & ETL",              # Fivetran, Airbyte, Airflow, dbt, Informatica
        "Business Intelligence",               # Tableau, Power BI, Looker, Qlik
        "Product Analytics",                   # Google Analytics, Mixpanel, Amplitude
        "AI/ML Infrastructure",                # OpenAI API, LangChain, Pinecone, Vertex AI
        "Data Catalogue & Governance",         # Collibra, Alation, Atlan
    ],
    "Security & Compliance": [
        "Cybersecurity / EDR",                 # CrowdStrike, SentinelOne, Microsoft Defender
        "SIEM & Threat Detection",             # Splunk, Microsoft Sentinel, IBM QRadar
        "Vulnerability Management",            # Tenable, Qualys, Rapid7
        "GRC & Compliance",                    # ServiceNow GRC, MetricStream, Vanta, Drata
        "DLP & Data Security",                 # Symantec DLP, Forcepoint, Nightfall
        "Zero Trust / ZTNA",                   # Zscaler, Cloudflare Access, Palo Alto Prisma
        "APM & Monitoring",                    # Datadog, New Relic, Sentry, Dynatrace
    ],
    "Unclassified": [
        "Unclassified Tools",
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
    wide_mode = not focus_categories and not focus_vendors

    # Wide-mode: 3 calls cover different category slices
    WIDE_SLICES = {
        1: {
            "label": "Core Technology Stack — Programming, Data, Cloud & DevOps",
            "categories": [
                "Programming Languages & Frameworks", "Data Management & Streaming",
                "Cloud & Infrastructure", "DevOps & CI/CD",
                "Container & Orchestration", "Version Control",
                "API Management", "iPaaS & Integration",
            ],
        },
        2: {
            "label": "Enterprise & Financial Systems, Security & Infrastructure",
            "categories": [
                "ERP & Finance", "Financial & Enterprise Systems",
                "Payment Infrastructure", "Risk & Compliance Platforms",
                "HR & Payroll", "Identity & IAM",
                "Cybersecurity / EDR", "SIEM & Threat Detection",
                "GRC & Compliance", "APM & Monitoring",
                "Databases", "Cloud Hosting", "Network & VPN",
            ],
        },
        3: {
            "label": "AI/Data, CRM, Sales, Marketing, Support & Productivity",
            "categories": [
                "Data Warehousing", "Data Integration & ETL", "Business Intelligence",
                "AI/ML Infrastructure", "Product Analytics",
                "CRM & Account Management", "Marketing Automation",
                "Sales Intelligence", "Customer Support & Helpdesk",
                "ITSM & Service Desk", "Project & Knowledge Management",
                "Collaboration & Productivity", "Billing & Subscription",
            ],
        },
    }

    if wide_mode:
        slice_info = WIDE_SLICES.get(call_num, WIDE_SLICES[1])
        focus_label = slice_info["label"]
        cats_to_search = slice_info["categories"]
        scope_block = f"""MODE: WIDE-SPECTRUM — sweep these tech stack categories:
{chr(10).join(f'  - {c}' for c in cats_to_search)}"""
    else:
        cats_to_search = focus_categories or []
        scope_block_parts = []
        if focus_categories:
            scope_block_parts.append(
                "Focus categories (treat as a STARTING POINT, not exhaustive — also search "
                "adjacent/related technology categories and the vendors who operate in that "
                "space, not only the literal category name):\n"
                + "\n".join(f"  - {c}" for c in focus_categories)
            )
        if focus_vendors:
            scope_block_parts.append(
                "Focus vendors (treat as a STARTING POINT, not exhaustive — for EACH vendor below, "
                "using your own knowledge also identify and search for: (a) its subsidiaries/brands, "
                "(b) its direct competitors in the same category, and (c) those competitors' "
                "subsidiaries/brands. Include but do not limit yourself to obvious names):\n"
                + "\n".join(f"  - {v}" for v in focus_vendors)
            )
        scope_block = "MODE: LASER-FOCUSED\n" + "\n".join(scope_block_parts)
        focus_label = ", ".join(focus_categories[:3] or focus_vendors[:3]) or "specified categories"

    # Build category-driven searches — open-ended, no hardcoded vendor names
    cat_searches = []
    for i, cat in enumerate(cats_to_search[:8], 1):
        cat_searches.append(f'C{i}. "{company_name}" "{cat}" software OR vendor OR platform OR tool 2022 OR 2023 OR 2024 OR 2025')
        cat_searches.append(f'C{i}b. site:linkedin.com/jobs "{company_name}" {cat.split("/")[0].strip().lower()} — extract required tools from job descriptions')

    # Vendor-specific searches if focus_vendors provided
    vendor_searches = []
    for v in (focus_vendors or [])[:6]:
        vendor_searches.append(f'V1. "{company_name}" "{v}" deployed OR implemented OR using OR "go-live" OR case study')
        vendor_searches.append(f'V2. "{company_name}" "{v}" OR competitor alternative — what is in use in the same category?')
        vendor_searches.append(f'V3. "{company_name}" "{v}" subsidiary OR brand deployed OR implemented OR using')
        vendor_searches.append(f'V4. "{company_name}" — identify direct competitors of "{v}" and their subsidiaries/brands, then check if any are deployed or implemented')

    # Implementation partner searches
    si_firms = "Accenture OR TCS OR Infosys OR Wipro OR Cognizant OR Deloitte OR IBM OR Capgemini OR HCL OR PwC OR EY OR KPMG OR BCG OR Slalom OR Atos OR CGI OR DXC"
    impl_searches = [
        f'P1. "{company_name}" implementation partner OR "system integrator" OR "SI partner" OR "consulting partner" {si_firms}',
        f'P2. "{company_name}" software "implemented by" OR "delivered by" OR "go-live" OR "rollout" partner 2022 OR 2023 OR 2024 OR 2025',
        f'P3. site:linkedin.com/jobs "{company_name}" implementation OR "system integrator" consulting 2024 OR 2025',
        f'P4. "{company_name}" ERP OR CRM OR cloud implementation "{si_firms.split(" OR ")[0]}" OR "{si_firms.split(" OR ")[1]}" case study press release',
    ]

    # Install size + renewal searches
    install_renewal_searches = [
        f'R1. "{company_name}" software "enterprise agreement" OR "multi-year contract" OR "enterprise license" 2022 OR 2023 OR 2024 OR 2025',
        f'R2. "{company_name}" employees headcount seats licenses software — to estimate install size from employee count',
        f'R3. "{company_name}" software renewal OR "contract renewal" OR "new agreement" 2024 OR 2025 OR 2026',
        f'R4. site:g2.com OR site:trustradius.com OR site:gartner.com/reviews "{company_name}" — user reviews revealing tools + scale',
        f'R5. "{company_name}" annual report OR 10-K 2024 — technology spend, vendor names, contract commitments',
    ]

    cat_block = "\n".join(cat_searches) if cat_searches else ""
    vendor_block = "\nVENDOR-SPECIFIC SEARCHES:\n" + "\n".join(vendor_searches) if vendor_searches else ""

    fields_desc = "\n".join(f'  "{f["key"]}": "{f["label"]}"' for f in TECH_STACK_FIELDS)
    fields_example = {f["key"]: f"<{f['label']}>" for f in TECH_STACK_FIELDS}

    return f"""You are a technographic research analyst. Use Google Search to build the technology stack of {company_name}.

COMPANY: {company_name}{f" | {domain}" if domain else ""}{linkedin_block}

RESEARCH FOCUS (call {call_num}): {focus_label} — 2021 to 2026.

{scope_block}

MANDATORY SEARCHES — run ALL of the following:

BROAD BASELINE (run first):
B1. "{company_name}" software technology vendors tools 2024 2025 — full stack overview
B2. "{company_name}" job postings engineer developer OR architect required skills 2024 2025 site:linkedin.com/jobs
B3. "{company_name}" privacy policy OR "cookie policy" third-party software — reveals tracking/analytics/support tools
B4. "{company_name}" technology partner OR "strategic partnership" OR "preferred vendor" announcement 2023 OR 2024 OR 2025
B5. site:builtwith.com OR site:wappalyzer.com OR site:stackshare.io "{company_name}" OR "{domain}" — public technographic data

CATEGORY-SPECIFIC SEARCHES:
{cat_block}
{vendor_block}

IMPLEMENTATION PARTNER SEARCHES:
{chr(10).join(impl_searches)}

INSTALL SIZE & RENEWAL DATE SEARCHES:
{chr(10).join(install_renewal_searches)}

For each software tool found, return ONE JSON object per tool. Aim for MAXIMUM coverage.

Fields (EXACT keys required):
{fields_desc}

Field rules:
- tech_level3: COMPULSORY — never leave blank or "-". Exact name from this taxonomy
  (pick the closest match if not a perfect fit):
{TECH_L3_LIST}
- tech_level2 / tech_level1: leave as empty string — derived automatically
- integration_partner: SI/consulting firm that IMPLEMENTED it (e.g. "Accenture", "TCS", "Deloitte") — from P1–P4 searches — or "-"
- last_detected: "Mon YYYY" if known, or just "YYYY", or "-"
- tech_install: numeric estimate of licensed users/seats based on company headcount and deployment scope (e.g. "500–2,000", "10,000–50,000", "100,000+") — or "-"
- renewal_date: estimated contract renewal quarter (e.g. "Q2 2027") — infer from: contract age (typical 3-5yr enterprise), press release dates, or R1/R3 search results — or "-"
- confidence_score: "87%" etc — 95-99% for DNS/pixel/public API key; 85-94% for vendor case study/press release; 75-84% for job posting; 60-74% for industry inference
- deal_value: TCV of this technology contract. NUMERIC $ ONLY — "$XM" or "$XB". Use public figure
  if stated; otherwise estimate based on tech_install size and {company_name}'s likely company
  size/IT budget — do NOT default to a generic enterprise benchmark for a mid-size or smaller
  company. As a guide: <500 seats/light deployment → $0.1M–$1M; 500–5,000 seats or mid-scope
  platform → $1M–$5M; 5,000–50,000 seats or company-wide platform → $5M–$25M; 50,000+ seats or
  Fortune 500-scale rollout → $25M+. NO other text.
- deal_acv: Annual contract value. NUMERIC $ ONLY (e.g. "$2M/yr → "$2M"). Empty string if single TCV or not derivable.
- deal_estimated: "Y" if deal_value was estimated from benchmarks. Empty string "" if from a confirmed public source.
- source_info: "Job posting", "Vendor case study", "Press release", "Annual report", "Privacy policy", "LinkedIn jobs", "BuiltWith", "G2 review", etc.

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


    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Set HTTP-level timeout so a hung Gemini server can't block forever
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
            # Enforce taxonomy: derive L1/L2 from L3 — always run, even if L3 is blank/"-",
            # so every row is fully categorized (classify_tech guarantees a non-empty triple).
            # Falls back to mapping the vendor/tool name itself when L3 is unmatched.
            l3_raw = row.get("tech_level3", "")
            l1, l2, l3_canon = classify_tech("" if l3_raw == "-" else l3_raw, row.get("vendor", ""))
            row["tech_level1"] = l1
            row["tech_level2"] = l2
            row["tech_level3"] = l3_canon
            # Enforce deal value: never blank/"-" — estimate from category benchmark
            dv, est = estimate_deal_value(row)
            row["deal_value"] = dv
            row["deal_estimated"] = est
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
    has_focus = bool(fc or fv)

    # Derive a shorter "brand name" by stripping common subsidiary suffixes
    _STRIP = re.compile(
        r"\s+(north america|south america|europe|asia|apac|latam|"
        r"inc\.?|llc\.?|ltd\.?|corp\.?|corporation|group|holdings|"
        r"gmbh|ag|plc|sa|bv|nv|pty|co\.?)\s*$",
        re.IGNORECASE,
    )
    brand_name = _STRIP.sub("", company_name).strip()

    # Always run a 3-call wide general scan regardless of focus
    # If focus is provided, run 2 additional focused calls on top
    CALL_TIMEOUT = 240

    # Phase 1: wide general scan (always, no focus filters)
    WIDE_LABELS = {
        1: "core tech stack — languages, data, cloud & DevOps",
        2: "enterprise & financial systems, security & infrastructure",
        3: "AI/data, CRM, sales, marketing & productivity",
    }
    total_phases = 2 if has_focus else 1
    phase1_calls = 3

    mode_desc = "wide general + focused" if has_focus else "wide general"
    yield {"type": "heartbeat", "message": f"🔍 Tech stack scan for {company_name} ({mode_desc})…"}
    await asyncio.sleep(0)

    seen_keys: set[str] = set()
    total = 0

    async def _run_call(prompt_company: str, prompt_domain: str, call_fc: list, call_fv: list,
                        call_num: int, phase_label: str) -> int:
        """Run one Gemini call, yield events, return count of new tools added."""
        nonlocal total
        yield_count = 0
        prompt = _build_tech_stack_prompt(prompt_company, prompt_domain, linkedin_url, call_fc, call_fv, call_num)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_tech_stack_sync, prompt, prompt_company)

        elapsed = 0
        call_tools: list[dict] = []
        while elapsed < CALL_TIMEOUT:
            await asyncio.sleep(10)
            elapsed += 10
            if future.done():
                try:
                    call_tools = future.result() or []
                except Exception as e:
                    logger.error(f"Tech stack call error for {prompt_company}: {e}", exc_info=True)
                    return 0
                break
            # no per-tick heartbeat — outer loop emits progress
        else:
            future.cancel()

        for tool in call_tools:
            dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_level3','').lower()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok"}
            row.update(tool)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.04)
            yield_count += 1
            total += 1
        return yield_count

    # ── Phase 1: wide scan (no focus) ────────────────────────────────────────────
    yield {"type": "heartbeat", "message": f"🌐 Phase 1/{'2' if has_focus else '1'}: General wide-spectrum scan…"}
    await asyncio.sleep(0)

    for call_num in range(1, phase1_calls + 1):
        label = WIDE_LABELS[call_num]
        yield {"type": "heartbeat", "message": f"  [{call_num}/{phase1_calls}] {label}…"}
        await asyncio.sleep(0)

        # Wide scan: pass empty focus lists so prompt uses wide-spectrum slices
        prompt = _build_tech_stack_prompt(company_name, domain, linkedin_url, [], [], call_num)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_tech_stack_sync, prompt, company_name)

        elapsed = 0
        call_tools: list[dict] = []
        while elapsed < CALL_TIMEOUT:
            await asyncio.sleep(10)
            elapsed += 10
            if future.done():
                try:
                    call_tools = future.result() or []
                except Exception as e:
                    logger.error(f"Tech stack call {call_num} error for {company_name}: {e}", exc_info=True)
                    yield {"type": "heartbeat", "message": f"⚠️ Call {call_num} error: {e}"}
                break
            yield {"type": "heartbeat", "message": f"  [{call_num}/{phase1_calls}] Scanning… ({elapsed}s)"}
            await asyncio.sleep(0)
        else:
            future.cancel()
            yield {"type": "heartbeat", "message": f"⏱ Call {call_num} timed out — partial results below"}

        new_tools = 0
        for tool in call_tools:
            dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_level3','').lower()}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok"}
            row.update(tool)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.04)
            new_tools += 1
            total += 1

        yield {"type": "heartbeat", "message": f"  ✓ Call {call_num} done: +{new_tools} tools (running total: {total})"}
        await asyncio.sleep(0)

    # ── Phase 2: focused scan (only if focus_categories or focus_vendors given) ──
    if has_focus:
        focus_summary = ", ".join((fc + fv)[:4])
        yield {"type": "heartbeat", "message": f"🎯 Phase 2/2: Focused scan — {focus_summary}…"}
        await asyncio.sleep(0)

        for call_num in range(1, 3):  # 2 focused calls
            yield {"type": "heartbeat", "message": f"  [F{call_num}/2] Focused scan: {focus_summary}…"}
            await asyncio.sleep(0)

            prompt = _build_tech_stack_prompt(company_name, domain, linkedin_url, fc, fv, call_num)
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, _gemini_tech_stack_sync, prompt, company_name)

            elapsed = 0
            call_tools = []
            while elapsed < CALL_TIMEOUT:
                await asyncio.sleep(10)
                elapsed += 10
                if future.done():
                    try:
                        call_tools = future.result() or []
                    except Exception as e:
                        logger.error(f"Focused call {call_num} error for {company_name}: {e}", exc_info=True)
                        yield {"type": "heartbeat", "message": f"⚠️ Focused call {call_num} error: {e}"}
                    break
                yield {"type": "heartbeat", "message": f"  [F{call_num}/2] Scanning… ({elapsed}s)"}
                await asyncio.sleep(0)
            else:
                future.cancel()
                yield {"type": "heartbeat", "message": f"⏱ Focused call {call_num} timed out — partial results"}

            new_tools = 0
            for tool in call_tools:
                dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_level3','').lower()}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                row = {"company_name": company_name, "domain": domain, "_status": "ok"}
                row.update(tool)
                yield {"type": "row_done", "row": row}
                await asyncio.sleep(0.04)
                new_tools += 1
                total += 1

            yield {"type": "heartbeat", "message": f"  ✓ Focused call {call_num} done: +{new_tools} new tools (total: {total})"}
            await asyncio.sleep(0)

    # ── Fallback: retry with brand name if subsidiary name returned nothing ──────
    if total == 0 and brand_name.lower() != company_name.lower():
        yield {"type": "heartbeat",
               "message": f"🔄 No results for '{company_name}' — retrying as '{brand_name}'…"}
        await asyncio.sleep(0)

        fallback_prompt = _build_tech_stack_prompt(brand_name, domain, linkedin_url, [], [], 1)
        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_tech_stack_sync, fallback_prompt, brand_name)

        elapsed = 0
        fallback_tools: list[dict] = []
        while elapsed < CALL_TIMEOUT:
            await asyncio.sleep(10)
            elapsed += 10
            if future.done():
                try:
                    fallback_tools = future.result() or []
                except Exception as e:
                    logger.error(f"Fallback error for {brand_name}: {e}", exc_info=True)
                break
            yield {"type": "heartbeat", "message": f"🌐 Fallback scan… ({elapsed}s)"}
            await asyncio.sleep(0)
        else:
            future.cancel()

        for tool in fallback_tools:
            dedup_key = f"{tool.get('vendor','').lower()}|{tool.get('tech_level3','').lower()}"
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
