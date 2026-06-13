"""
IT Deal enrichment pipeline — Gemini 2.5 Flash + Google Search Grounding.

Single-call architecture:
  1. Build a research prompt: company × industry × vendors × keywords (2010–2026)
  2. Call gemini-2.5-flash with googleSearch tool so it searches live web
  3. Parse JSON array from response → yield one row_done per deal
"""

import asyncio
import json
import logging
import os
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)

GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_API_KEY", "")

# ── Tech Taxonomy (from Tech Categorization.xlsx) ────────────────────────────
# Maps Level 3 → (Level 1, Level 2). Used for post-processing classification.
TECH_TAXONOMY: dict[str, tuple[str, str]] = {
    # Communications
    "Fixed Data": ("Communications", "Fixed Data"),
    "Fixed Voice": ("Communications", "Fixed Voice"),
    "Mobile Services": ("Communications", "Mobile Services"),
    "Satellite Communications Services": ("Communications", "Satellite Communications Services"),
    # Hardware – Client Computing
    "Components": ("Hardware", "Client Computing"),
    "Computing Device": ("Hardware", "Client Computing"),
    "Peripherals": ("Hardware", "Client Computing"),
    # Hardware – Network Infrastructure
    "Customer Premise Equipment": ("Hardware", "Network Infrastructure"),
    "Telecommunication Equipment": ("Hardware", "Network Infrastructure"),
    # Hardware – Security
    "Content Filtering and Anti-spam Appliances": ("Hardware", "Security"),
    "Encryption and SSL Accelerators": ("Hardware", "Security"),
    "Firewall Appliance": ("Hardware", "Security"),
    "Physical Security Devices": ("Hardware", "Security"),
    # Hardware – Server Computing
    "High End Servers": ("Hardware", "Server Computing"),
    "Mid Range Servers": ("Hardware", "Server Computing"),
    # Hardware – Storage
    "Direct Attached Storage": ("Hardware", "Storage"),
    "Network Attached Storage": ("Hardware", "Storage"),
    "Storage Area Networks": ("Hardware", "Storage"),
    "Tape Systems": ("Hardware", "Storage"),
    "Unclassified Storage": ("Hardware", "Storage"),
    # Services – Application-Led Outsourcing
    "Application Development and Maintenance": ("Services", "Application-Led Outsourcing"),
    "Application Hosting": ("Services", "Application-Led Outsourcing"),
    "Application Management": ("Services", "Application-Led Outsourcing"),
    "Application Testing / Quality Management": ("Services", "Application-Led Outsourcing"),
    # Services – Business Process Outsourcing
    "Analytics / Business Intelligence": ("Services", "Business Process Outsourcing"),
    "Customer Relationship Management (CRM) BPO": ("Services", "Business Process Outsourcing"),
    "Finance and Accounting (F&A) BPO": ("Services", "Business Process Outsourcing"),
    "Human Resources (HR) BPO": ("Services", "Business Process Outsourcing"),
    "Industry-specific BPO": ("Services", "Business Process Outsourcing"),
    "KPO": ("Services", "Business Process Outsourcing"),
    "Legal Services Outsourcing": ("Services", "Business Process Outsourcing"),
    "Processing Services": ("Services", "Business Process Outsourcing"),
    "Procurement BPO": ("Services", "Business Process Outsourcing"),
    "Robotic Process Automation (RPA)": ("Services", "Business Process Outsourcing"),
    # Services – Cloud Services
    "Business-Process-as-a-Service (BPaaS)": ("Services", "Cloud Services"),
    "Desktop-as-a-Service (DaaS) / Workplace-as-a-Service (WaaS)": ("Services", "Cloud Services"),
    "Infrastructure-as-a-Service (IaaS)": ("Services", "Cloud Services"),
    "Platform-as-a-Service (PaaS)": ("Services", "Cloud Services"),
    "Software-as-a-Service (SaaS)": ("Services", "Cloud Services"),
    # Services – Consulting & System Integration
    "Automation Advisory": ("Services", "Consulting & System Integration"),
    "Business Consulting": ("Services", "Consulting & System Integration"),
    "IT Consulting": ("Services", "Consulting & System Integration"),
    "Sourcing Advisory": ("Services", "Consulting & System Integration"),
    "System Integration": ("Services", "Consulting & System Integration"),
    # Services – Digital Enterprise
    "Blockchain": ("Services", "Digital Enterprise"),
    "Digital Transformation": ("Services", "Digital Enterprise"),
    "Enterprise Mobility": ("Services", "Digital Enterprise"),
    "Social Collaboration / Enterprise Social Networking / Enterprise 2.0": ("Services", "Digital Enterprise"),
    "Unified Communications and Collaboration": ("Services", "Digital Enterprise"),
    "Content Services": ("Services", "Digital Enterprise"),
    # Services – End User Services
    "Desktop Management": ("Services", "End User Services"),
    "Helpdesk Management": ("Services", "End User Services"),
    "Managed Print Services": ("Services", "End User Services"),
    "Managed Security Services": ("Services", "End User Services"),
    "Professional Services": ("Services", "End User Services"),
    "Service Desk": ("Services", "End User Services"),
    # Services – Infrastructure-Led Outsourcing
    "Business Continuity / Disaster Recovery": ("Services", "Infrastructure-Led Outsourcing"),
    "Colocation Services": ("Services", "Infrastructure-Led Outsourcing"),
    "Data Center Outsourcing": ("Services", "Infrastructure-Led Outsourcing"),
    "Hardware Integration": ("Services", "Infrastructure-Led Outsourcing"),
    "Infrastructure Hosting": ("Services", "Infrastructure-Led Outsourcing"),
    "Infrastructure Management": ("Services", "Infrastructure-Led Outsourcing"),
    "Infrastructure Testing": ("Services", "Infrastructure-Led Outsourcing"),
    "Server Management": ("Services", "Infrastructure-Led Outsourcing"),
    "Storage Services": ("Services", "Infrastructure-Led Outsourcing"),
    # Services – IoT
    "Industry Specific IoT Applications": ("Services", "Internet of Things (IoT)"),
    "Smart Energy (Energy Management Application)": ("Services", "Internet of Things (IoT)"),
    "Smart Factory / Industry 4.0": ("Services", "Internet of Things (IoT)"),
    "Smart Health": ("Services", "Internet of Things (IoT)"),
    "Smart Transportation / Car IT": ("Services", "Internet of Things (IoT)"),
    # Services – Network Services
    "Network Consulting": ("Services", "Network Services"),
    "Network Infrastructure": ("Services", "Network Services"),
    "Network Integration": ("Services", "Network Services"),
    "Network Management": ("Services", "Network Services"),
    # Software – Application Development & Deployment
    "Application Development Software": ("Software", "Application Development & Deployment"),
    "Application Infrastructure Middleware": ("Software", "Application Development & Deployment"),
    "Application Life Cycle Management": ("Software", "Application Development & Deployment"),
    "Business Process Management": ("Software", "Application Development & Deployment"),
    "Database Management System": ("Software", "Application Development & Deployment"),
    "Managed File Transfer": ("Software", "Application Development & Deployment"),
    # Software – Enterprise Applications
    "Business Intelligence and Analytics Tools": ("Software", "Enterprise Applications"),
    "Business Risk and Compliance": ("Software", "Enterprise Applications"),
    "Collaboration": ("Software", "Enterprise Applications"),
    "Commerce Applications": ("Software", "Enterprise Applications"),
    "Content Management and Creation": ("Software", "Enterprise Applications"),
    "Engineering Applications": ("Software", "Enterprise Applications"),
    "Enterprise Asset Management": ("Software", "Enterprise Applications"),
    "ERP Financials": ("Software", "Enterprise Applications"),
    "ERP Human Capital Management": ("Software", "Enterprise Applications"),
    "ERP Procurement": ("Software", "Enterprise Applications"),
    "ERP Order Management": ("Software", "Enterprise Applications"),
    "ERP Manufacturing": ("Software", "Enterprise Applications"),
    "ERP Supply Chain Planning": ("Software", "Enterprise Applications"),
    "ERP Projects / PSA": ("Software", "Enterprise Applications"),
    "ERP GRC": ("Software", "Enterprise Applications"),
    "Industry Specific Applications": ("Software", "Enterprise Applications"),
    "Office Productivity Applications and Suites": ("Software", "Enterprise Applications"),
    "Performance Management and Analytic Applications": ("Software", "Enterprise Applications"),
    "Sales Automation": ("Software", "Enterprise Applications"),
    "Marketing Automation": ("Software", "Enterprise Applications"),
    # Software – Software Infrastructure
    "Master Data Management (MDM)": ("Software", "Software Infrastructure"),
    "API Management": ("Software", "Software Infrastructure"),
    "Information Management": ("Software", "Software Infrastructure"),
    "Operating Systems": ("Software", "Software Infrastructure"),
    "Operations Management": ("Software", "Software Infrastructure"),
    "Security": ("Software", "Software Infrastructure"),
    "Storage Management": ("Software", "Software Infrastructure"),
    "Virtualisation": ("Software", "Software Infrastructure"),
}

# Build normalised lookup (lowercase → canonical) for fuzzy matching
_TAXONOMY_LOWER: dict[str, str] = {k.lower(): k for k in TECH_TAXONOMY}

TECH_L3_LIST = "\n".join(
    f"  [{l1} > {l2}] {l3}"
    for l3, (l1, l2) in TECH_TAXONOMY.items()
)


def classify_tech(level3_raw: str) -> tuple[str, str, str]:
    """Return (level1, level2, canonical_level3) from a raw model output string."""
    if not level3_raw:
        return ("", "", "")
    key = level3_raw.strip().lower()
    # Exact match
    canonical = _TAXONOMY_LOWER.get(key)
    if not canonical:
        # Partial match — find best substring hit
        for lk, lv in _TAXONOMY_LOWER.items():
            if key in lk or lk in key:
                canonical = lv
                break
    if canonical and canonical in TECH_TAXONOMY:
        l1, l2 = TECH_TAXONOMY[canonical]
        return (l1, l2, canonical)
    # No match — return raw value with blanks for L1/L2
    return ("", "", level3_raw)

# Fixed output schema — matches the IT Deal Details preset in the frontend
SCHEMA_FIELDS = [
    {"key": "vendor",          "label": "Vendor Name",        "type": "string",
     "description": "Technology vendor or service provider name"},
    {"key": "tech_level3",     "label": "Level 3 Category",   "type": "string",
     "description": "Most specific tech category (e.g. Warranty Management, Core Banking, CRM, ERP, Cloud IaaS)"},
    {"key": "tech_level2",     "label": "Level 2 Category",   "type": "string",
     "description": "Mid-level tech category derived from Level 3 (e.g. Aftermarket Tech, Enterprise Applications, Infrastructure)"},
    {"key": "tech_level1",     "label": "Level 1 Category",   "type": "string",
     "description": "Top-level tech category (e.g. Operations Technology, Business Applications, Infrastructure & Cloud)"},
    {"key": "deal_value",      "label": "TCV",                "type": "string",
     "description": "Total contract value — numeric $ only (e.g. '$50M', '$2.5B'). Public figure if stated; else estimate from benchmarks. NO other text."},
    {"key": "deal_acv",       "label": "ACV",                "type": "string",
     "description": "Annual contract value / annual fee — numeric $ only (e.g. '$10M'). Empty string if single TCV payment or not derivable."},
    {"key": "deal_estimated", "label": "Est",                "type": "string",
     "description": "'Y' if deal_value was estimated from benchmarks (not public). Empty string if value came from a confirmed public source."},
    {"key": "start_date",      "label": "Start Date",         "type": "date",
     "description": "Contract start or go-live date (YYYY-MM-DD or YYYY-MM or YYYY)"},
    {"key": "end_date",        "label": "End Date",           "type": "date",
     "description": "Contract end or renewal date if known"},
    {"key": "duration_months", "label": "Duration (months)",  "type": "string",
     "description": "Contract duration in months (e.g. '36'); derive from start+end dates if not stated"},
    {"key": "last_detected",   "label": "Last Detected",      "type": "date",
     "description": "Date of press release or announcement (YYYY-MM-DD or YYYY-MM or YYYY)"},
    {"key": "deal_focus",      "label": "Deal Focus",         "type": "string",
     "description": "1-3 primary technology focus tags e.g. AI, Cloud, ERP"},
    {"key": "description",     "label": "Deal Description",   "type": "string",
     "description": "One sentence describing what was agreed"},
    {"key": "source",          "label": "Source",             "type": "string",
     "description": "URL of the press release, news article, or filing"},
]

FIXED_GOAL = (
    "Find every IT and technology deal, contract, outsourcing agreement, and digital "
    "transformation initiative involving this company."
)

# ── Industry-aware vendor + tech-area maps ────────────────────────────────────
# Gemini will auto-detect industry; these give it richer context per sector.

INDUSTRY_CONTEXT = {
    "banking_finance": {
        "tech_areas": ["core banking", "digital banking", "payments", "risk & compliance",
                       "anti-money laundering", "trade finance", "wealth management platform",
                       "open banking", "cloud migration", "cybersecurity"],
        "vendors": ["Temenos", "Finacle", "FIS", "Fiserv", "Finastra", "Mambu",
                    "Thought Machine", "Intellect Design", "TCS BaNCS", "Newgen",
                    "Oracle FLEXCUBE", "SAP Banking", "IBM", "Accenture", "Infosys",
                    "Wipro", "TCS", "HCLTech", "Cognizant", "AWS", "Microsoft Azure"],
    },
    "insurance": {
        "tech_areas": ["policy administration", "claims management", "underwriting automation",
                       "insurtech", "telematics", "digital distribution", "cloud migration"],
        "vendors": ["Guidewire", "Duck Creek", "Majesco", "EIS", "SAP Insurance",
                    "Oracle Insurance", "Accenture", "Capgemini", "TCS", "Infosys"],
    },
    "manufacturing": {
        "tech_areas": ["ERP", "MES", "supply chain", "IoT", "industry 4.0",
                       "digital twin", "PLM", "warehouse management", "procurement"],
        "vendors": ["SAP", "Oracle", "Siemens", "PTC", "Dassault Systèmes", "IBM",
                    "Accenture", "Wipro", "TCS", "HCLTech", "Infor", "IFS"],
    },
    "retail_ecommerce": {
        "tech_areas": ["e-commerce platform", "POS", "loyalty programme", "supply chain",
                       "omnichannel", "demand forecasting", "CRM", "personalisation"],
        "vendors": ["Salesforce", "SAP", "Oracle", "Manhattan Associates", "Blue Yonder",
                    "Shopify", "commercetools", "Infosys", "TCS", "Cognizant"],
    },
    "telecom": {
        "tech_areas": ["BSS/OSS", "5G", "network virtualisation", "billing", "CRM",
                       "network managed services", "cloud transformation"],
        "vendors": ["Ericsson", "Nokia", "Amdocs", "Netcracker", "Huawei", "IBM",
                    "Accenture", "TCS", "Infosys", "Wipro"],
    },
    "healthcare": {
        "tech_areas": ["EMR/EHR", "hospital information system", "revenue cycle",
                       "telemedicine", "AI diagnostics", "pharmacy management", "claims"],
        "vendors": ["Epic", "Cerner", "Meditech", "Allscripts", "Philips", "Siemens Healthineers",
                    "Oracle Health", "TCS", "Cognizant", "Accenture"],
    },
    "default": {
        "tech_areas": ["ERP", "CRM", "cloud migration", "managed services", "cybersecurity",
                       "digital transformation", "analytics", "AI/ML", "outsourcing",
                       "infrastructure", "SaaS implementation"],
        "vendors": ["SAP", "Oracle", "Microsoft", "AWS", "Google Cloud", "Salesforce",
                    "ServiceNow", "Workday", "IBM", "Accenture", "TCS", "Infosys",
                    "Wipro", "HCLTech", "Cognizant", "Capgemini", "DXC Technology",
                    "Palo Alto Networks", "CrowdStrike", "Snowflake"],
    },
}


def _make_prompt(company_name: str, domain: str, linkedin_block: str,
                 search_focus: str, known_vendors: str,
                 extra_searches: str, fields_desc: str, fields_json_keys: dict,
                 sector_block: str = "") -> str:
    return f"""You are an enterprise IT deal research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}{linkedin_block}

TASK: Find IT, technology, and strategic tech deals for {company_name}.

DEAL CATEGORIES TO CAPTURE (find ALL of these):
1. IT Outsourcing & Managed Services — SI contracts, BPO, ITO, infrastructure managed services
2. Cloud & Digital Transformation — cloud migration, SaaS rollouts, digital programmes
3. ERP / CRM / HCM / SCM — enterprise application implementations and upgrades
4. IT Acquisitions — tech company acquisitions, acqui-hires, asset purchases with IT angle
5. Strategic Joint Ventures & Corporate Venture — JVs with tech firms, CVC investments in tech cos
6. Enterprise Operations Partnerships — long-term IT ops partnerships, co-innovation agreements
7. Technology Disinvestments — IT asset sales, carve-outs, spin-offs, divestitures of tech units
8. Cybersecurity & Compliance — security platform contracts, SOC outsourcing, compliance tools
9. Analytics, AI & Data — AI/ML platform deals, data lake, BI, advanced analytics contracts
{sector_block}

SEARCHES TO RUN:
{search_focus}
Vendor/partner searches (pair each with company name):
{known_vendors}
{extra_searches}

EXTRACTION RULES:
- Read full articles, not just headlines
- One JSON object per distinct deal — never merge two deals
- Capture deals across ALL years available, not just recent ones
- Include press releases, news, vendor announcements, IR filings, annual reports
- PAY SPECIAL ATTENTION to: post spin-off / demerger IT separation programmes,
  nine-figure or billion-dollar IT outsourcing restructurings, IT carve-out programmes
  that replace systems from a former parent company, large infrastructure consolidation deals

Return ONLY a valid JSON array:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- vendor: exact name (e.g. "Infosys", "SAP S/4HANA", "Microsoft Azure", "Trimble")
- tech_level3: pick the BEST matching Level 3 from this taxonomy (use exact name):
{TECH_L3_LIST}
- tech_level2: leave empty string — derived automatically from taxonomy
- tech_level1: leave empty string — derived automatically from taxonomy
- deal_value: TCV (total contract value). NUMERIC $ ONLY — "$XM" or "$XB" (e.g. "$50M", "$2.5B"). ALWAYS provide a value: use public figure if stated, else ESTIMATE from benchmarks:
  * Large IT outsourcing (5+ yr, major vendor): $100M–$2B → e.g. "$500M"
  * Mid-size ERP/platform implementation: $10M–$100M → e.g. "$40M"
  * SaaS subscription (enterprise): $1M–$20M/yr → e.g. "$5M"
  * Cloud migration programme: $20M–$200M → e.g. "$80M"
  * Managed services (3–5 yr): $30M–$300M → e.g. "$120M"
  * Cybersecurity contract: $5M–$50M → e.g. "$20M"
  NO other text — output exactly like "$50M" or "$2.5B"
- deal_acv: ACV (annual contract value). NUMERIC $ ONLY (e.g. "$10M"). Empty string if it is a one-off TCV with no annual breakdown or if not derivable.
- deal_estimated: "Y" if deal_value was estimated from benchmarks (not a stated public figure). Empty string "" if value came from a confirmed public source.
- start_date: contract start or go-live date if mentioned
- end_date: contract expiry or renewal date if mentioned
- duration_months: contract length in months — derive from start+end if not stated explicitly; typical outsourcing=60, SaaS=12 or 36
- last_detected: date of the press release or news article (YYYY-MM-DD or YYYY-MM or YYYY)
- deal_focus: 1-3 tags from: AI | ML | Cloud | Big Data | Analytics | Cybersecurity | IoT |
  Automation | ERP | Digital Transformation | Blockchain | Edge Computing | 5G | Robotics |
  Autonomous | Payments | Open Banking | DevOps | Data Platform | Other
- description: one concise sentence — what was agreed and why it matters
- source: direct URL to press release, article, or filing

Return ONLY the raw JSON array. No prose. No markdown fences.
Example shape: {json.dumps(fields_json_keys)}
"""


# ── Sector detection keywords → extra search topics ──────────────────────────
_SECTOR_KEYWORDS = {
    "agriculture_agtech": {
        "triggers": ["agco", "deere", "cnh", "claas", "kubota", "trimble agriculture",
                     "precision ag", "agtech", "farm", "agriculture", "agricultural",
                     "crop", "harvest", "livestock", "grain"],
        "deal_types": [
            "10. AgTech Acquisitions — precision ag, farm management software, autonomous machinery startups",
            "11. Precision Agriculture Platform — FMIS, guidance, telematics, yield monitoring deals",
            "12. Autonomous Farming Tech — autonomous tractor, robotics, AI harvesting JVs",
            "13. Farm Data & Analytics — farm data platforms, satellite/sensor analytics deals",
        ],
        "extra_vendors": (
            "Trimble, Climate Corporation, Monsanto, Bayer Crop Science, Raven Industries, "
            "Precision Planting, 640 Labs, 84.51°, Ag Leader Technology, CNH Industrial, "
            "AGCO Fendt, Hexagon Agriculture, Farmers Edge, Granular, Proagrica"
        ),
        "extra_searches": [
            "precision agriculture technology acquisition",
            "farm management software deal acquisition",
            "autonomous farming startup acquisition",
            "agtech acquisition startup buy",
            "precision ag platform partnership",
            "farm data analytics deal",
            "autonomous tractor robotics technology",
        ],
    },
    "automotive_manufacturing": {
        "triggers": ["truck", "automotive", "vehicle", "motor", "daimler",
                     "caterpillar", "volvo", "ford", "gm", "stellantis",
                     "manufacturer", "manufacturing", "industrial"],
        "deal_types": [
            "10. Telematics & Fleet Tech — OEM telematics, connected vehicle platforms, fleet SaaS",
            "11. Autonomous & ADAS Tech — autonomous driving JVs, ADAS software, sensor partnerships",
            "12. Heavy Hardware & AI SaaS — AI-powered machinery, precision agriculture/construction tech",
            "13. Manufacturing Execution & IoT — MES, IIoT platforms, Industry 4.0, digital twin deals",
        ],
        "extra_vendors": (
            "Trimble, Hexagon, Geotab, Samsara, Mobileye, NVIDIA, Qualcomm, HERE Technologies, "
            "Aptiv, Verizon Connect, PTC, Dassault Systèmes, Siemens Digital Industries, "
            "Bosch Connected Devices, Continental, ZF Friedrichshafen"
        ),
        "extra_searches": [
            "telematics fleet management deal",
            "autonomous vehicle technology partnership",
            "connected truck platform agreement",
            "precision agriculture technology deal",
            "AI SaaS manufacturing contract",
            "digital twin industrial IoT agreement",
            "spin-off IT separation carve-out independent infrastructure",
            "post spin-off IT systems standalone architecture",
            "demerger IT outsourcing restructuring billion",
        ],
    },
    "banking_finance": {
        "triggers": ["bank", "financial", "insurance", "hdfc", "icici", "fintech",
                     "payment", "lending", "asset management"],
        "deal_types": [
            "10. Core Banking & Payments — core banking platform migrations, payment rails, open banking",
            "11. RegTech & Compliance — AML, KYC, risk platform contracts",
            "12. Digital Banking — mobile/internet banking platform deals, neobank JVs",
        ],
        "extra_vendors": (
            "Temenos, Finastra, FIS, Fiserv, Mambu, Thought Machine, Backbase, "
            "TCS BaNCS, Oracle FLEXCUBE, Intellect Design, Newgen, Finacle"
        ),
        "extra_searches": [
            "core banking platform deal",
            "digital banking transformation",
            "payment technology partnership",
            "fintech investment acquisition",
        ],
    },
    "telecom": {
        "triggers": ["telecom", "telco", "wireless", "network", "5g", "broadband",
                     "spectrum", "operator"],
        "deal_types": [
            "10. Network & BSS/OSS — 5G rollout contracts, BSS/OSS transformation, network managed services",
            "11. Digital Services Platform — content, cloud, enterprise ICT deals",
        ],
        "extra_vendors": (
            "Ericsson, Nokia, Amdocs, Netcracker, Huawei, Mavenir, Rakuten Symphony"
        ),
        "extra_searches": [
            "5G network contract deal",
            "BSS OSS transformation",
            "network managed services agreement",
        ],
    },
}


def _detect_sector_block(company_name: str) -> str:
    """Return sector-specific deal categories and searches if company matches a sector."""
    name_lower = company_name.lower()
    for sector, cfg in _SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in cfg["triggers"]):
            lines = cfg["deal_types"]
            return "\n".join(lines)
    return ""


def _detect_sector_extra(company_name: str) -> tuple[str, str]:
    """Return (extra_vendors, extra_searches_block) for detected sector."""
    name_lower = company_name.lower()
    for sector, cfg in _SECTOR_KEYWORDS.items():
        if any(kw in name_lower for kw in cfg["triggers"]):
            searches = "\n".join(
                f'  - "{company_name}" {s}' for s in cfg["extra_searches"]
            )
            return cfg["extra_vendors"], searches
    return "", ""


def _build_prompts(
    company_name: str,
    domain: str,
    linkedin_url: str,
    focus_tech: list[str],
    focus_vendor: list[str],
) -> list[str]:
    """
    Return 2 focused prompts.
    Call 1: broad IT + acquisitions/JVs/disinvestments + top SI vendors.
    Call 2: year sweep + sector-specific + user focus.
    Each targets ~60s completion.
    """
    linkedin_block = f" | LinkedIn: {linkedin_url}" if linkedin_url else ""
    fields_json_keys = {f["key"]: f"<{f['type']}>" for f in SCHEMA_FIELDS}
    fields_desc = "\n".join(f'  "{f["key"]}": "{f["description"]}"' for f in SCHEMA_FIELDS)

    sector_block = _detect_sector_block(company_name)
    sector_vendors, sector_searches = _detect_sector_extra(company_name)

    # ── Prompt 1: broad sweep + acquisitions/JVs/disinvestments + top vendors ─
    p1_searches = f"""  - "{company_name}" IT outsourcing contract deal signed
  - "{company_name}" technology acquisition acqui-hire
  - "{company_name}" joint venture technology partner
  - "{company_name}" corporate venture fund investment tech startup
  - "{company_name}" IT disinvestment carve-out spin-off asset sale
  - "{company_name}" digital transformation program cloud migration
  - "{company_name}" managed services ERP SAP Oracle implementation
  - "{company_name}" cybersecurity infrastructure data center contract
  - "{company_name}" IT separation spin-off independent infrastructure
  - "{company_name}" post spin-off IT systems carve-out restructuring
  - "{company_name}" IT outsourcing restructuring nine-figure billion
  - "{company_name}" independent digital architecture enterprise systems
  - "{company_name}" demerger IT separation standalone systems
  - "{company_name}" acquires technology company startup
  - "{company_name}" acquisition technology software hardware
  - "{company_name}" bought acquired tech firm stake joint venture
  - site:businesswire.com OR site:prnewswire.com "{company_name}" acquires
  - site:businesswire.com OR site:prnewswire.com "{company_name}" technology deal"""

    p1_vendors = (
        "Accenture, Infosys, TCS, Wipro, HCLTech, Cognizant, Capgemini, DXC Technology, "
        "IBM, SAP, Oracle, Microsoft, AWS, Google Cloud, ServiceNow, Salesforce, Workday, "
        "Baker McKenzie, EY, Deloitte, KPMG, PwC"   # advisory/legal for large IT restructurings
    )

    prompt1 = _make_prompt(company_name, domain, linkedin_block,
                           p1_searches, p1_vendors, "",
                           fields_desc, fields_json_keys, sector_block)

    # ── Prompt 2: year sweep + sector vendors + user focus ────────────────────
    # Combine IT deal + acquisition into one query per year to halve search count
    year_searches = "\n".join(
        f'  - "{company_name}" IT deal OR technology acquisition OR tech contract {y}' for y in range(2016, 2026)
    )

    p2_vendors = "Dell Technologies, HPE, Atos, NTT DATA, Unisys, Fujitsu, T-Systems, CGI, Snowflake, CrowdStrike, Palo Alto Networks"
    if sector_vendors:
        p2_vendors += f", {sector_vendors}"

    extra_parts = []
    if sector_searches:
        extra_parts.append(f"Sector-specific searches:\n{sector_searches}")
    if focus_tech or focus_vendor:
        lines = ["User-specified focus searches:"]
        for t in focus_tech:
            lines.append(f'  - "{company_name}" {t} deal contract')
        for v in focus_vendor:
            lines.append(f'  - "{company_name}" {v} deal partnership acquisition')
        extra_parts.append("\n".join(lines))
    extra = "\n\n".join(extra_parts)

    prompt2 = _make_prompt(company_name, domain, linkedin_block,
                           year_searches, p2_vendors, extra,
                           fields_desc, fields_json_keys, sector_block)

    return [prompt1, prompt2]


def _gemini_extract_deals_sync(
    prompt: str,
    company_name: str,
) -> list[dict]:
    """
    Blocking Gemini 2.5 Flash call with Google Search grounding.
    Accepts a pre-built prompt. Runs inside run_in_executor.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai not installed. Run: pip install google-genai")
        return []

    if not GOOGLE_AI_KEY:
        logger.error("GOOGLE_AI_API_KEY env var not set")
        return []

    field_keys = [f["key"] for f in SCHEMA_FIELDS]

    import time as _time
    MAX_RETRIES = 3


    response = None
    for attempt in range(1, MAX_RETRIES + 1):
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
            break   # success
        except Exception as api_err:
            err_str = str(api_err)
            is_quota = "RESOURCE_EXHAUSTED" in err_str or "free_tier" in err_str
            is_timeout = any(x in err_str.lower() for x in ("timeout", "timed out", "deadline"))
            is_retryable = not is_quota and (any(x in err_str for x in ("503", "UNAVAILABLE", "overloaded")) or is_timeout)
            logger.warning(f"Gemini attempt {attempt}/{MAX_RETRIES} failed for {company_name}: {api_err}")
            if is_quota:
                # Quota exhausted — not retryable, raise so callers can surface to user
                raise RuntimeError(f"Gemini quota exhausted (free tier limit reached). "
                                   f"Please upgrade your Google AI API key to a paid plan.") from api_err
            if is_retryable and attempt < MAX_RETRIES:
                _time.sleep(15 * attempt)
                continue
            logger.error(f"Gemini API error for {company_name}: {api_err}", exc_info=True)
            return []

    if response is None:
        return []

    try:

        # ── Extract raw text — ALWAYS use parts iteration, never rely on .text ──
        raw_text = ""
        try:
            for candidate in (response.candidates or []):
                for part in (candidate.content.parts or []):
                    t = getattr(part, "text", None)
                    if t:
                        raw_text += t
        except Exception as parts_err:
            logger.warning(f"Parts extraction failed for {company_name}: {parts_err}")
            # Last-ditch: try .text property
            try:
                raw_text = response.text or ""
            except Exception:
                pass

        finish_reason = "unknown"
        try:
            finish_reason = str(response.candidates[0].finish_reason)
        except Exception:
            pass

        logger.info(f"Gemini response: {len(raw_text)} chars, finish={finish_reason} for {company_name}")

        if not raw_text:
            logger.warning(f"Empty response for {company_name}. finish_reason={finish_reason}")
            return []

    except Exception as e:
        logger.error(f"Gemini API error for {company_name}: {e}", exc_info=True)
        return []

    # ── Parse JSON array from response ───────────────────────────────────────
    try:
        # Strip markdown fences
        clean = re.sub(r"```(?:json)?\s*", "", raw_text.strip())
        clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

        # Try direct parse first
        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            # Find the outermost [ ... ] array
            array_match = re.search(r"\[.*\]", clean, re.DOTALL)
            if not array_match:
                # Response might be truncated — try to recover partial objects
                logger.warning(f"No complete JSON array for {company_name}. Attempting recovery. Preview: {clean[:400]}")
                # Find start of array, close it after last complete object
                start = clean.find("[")
                if start == -1:
                    return []
                fragment = clean[start:]
                # Find last complete '}' and close array there
                last_brace = fragment.rfind("}")
                if last_brace == -1:
                    return []
                fragment = fragment[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                try:
                    parsed = json.loads(fragment)
                except json.JSONDecodeError:
                    logger.error(f"Recovery failed for {company_name}")
                    return []
            else:
                candidate_text = array_match.group(0)
                try:
                    parsed = json.loads(candidate_text)
                except json.JSONDecodeError:
                    # Fix trailing commas + try again
                    candidate_text = re.sub(r",\s*([\]}])", r"\1", candidate_text)
                    try:
                        parsed = json.loads(candidate_text)
                    except json.JSONDecodeError:
                        # Truncated: recover last complete object
                        last_brace = candidate_text.rfind("}")
                        if last_brace == -1:
                            return []
                        fragment = candidate_text[:last_brace + 1].rstrip().rstrip(",") + "\n]"
                        fragment = "[" + fragment.lstrip("[")
                        parsed = json.loads(fragment)

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []

        out = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            row: dict = {}
            for key in field_keys:
                val = item.get(key)
                row[key] = str(val) if val not in (None, "null", "None", "") else ""
            # Enforce taxonomy: derive L1/L2 from L3
            l3_raw = row.get("tech_level3", "")
            l1, l2, l3_canon = classify_tech(l3_raw)
            row["tech_level1"] = l1
            row["tech_level2"] = l2
            row["tech_level3"] = l3_canon
            out.append(row)

        logger.info(f"Parsed {len(out)} deals for {company_name}")
        return out

    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {company_name}: {e} | raw: {raw_text[:400]}")
        return []
    except Exception as e:
        logger.error(f"Parse error for {company_name}: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def enrich_company(
    company_name: str,
    domain: str,
    goal: str = FIXED_GOAL,           # kept for API compat, not used in prompt
    schema_fields: list[dict] | None = None,   # kept for API compat, not used
    year_range: tuple[int, int] = (2010, 2026),
    max_urls: int = 20,               # kept for API compat, unused
    linkedin_url: str = "",
    focus_tech: list[str] | None = None,
    focus_vendor: list[str] | None = None,
    # legacy aliases accepted but ignored
    extra_vendors: list[str] | None = None,
    extra_sources: list[str] | None = None,
    extra_keywords: list[str] | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Find IT deals for one company (2010–2026) via Gemini + Google Search Grounding.
    Yields heartbeat events then row_done events (one per deal).
    """
    ft = focus_tech or []
    fv = focus_vendor or []

    # Strip common subsidiary suffixes to derive a broader search name for fallback
    _STRIP = re.compile(
        r"\s+(north america|south america|europe|asia|apac|latam|"
        r"inc\.?|llc\.?|ltd\.?|corp\.?|corporation|group|holdings|"
        r"gmbh|ag|plc|sa|bv|nv|pty|co\.?)\s*$",
        re.IGNORECASE,
    )
    brand_name = _STRIP.sub("", company_name).strip()

    prompts = _build_prompts(company_name, domain, linkedin_url, ft, fv)
    seen_keys: set[str] = set()   # deduplicate across the two calls
    total_deals = 0
    CALL_TIMEOUT = 150            # 2.5 min per call

    for call_idx, prompt in enumerate(prompts, 1):
        label = "broad IT & cloud deals" if call_idx == 1 else "year-by-year + vendor sweep"
        yield {"type": "heartbeat",
               "message": f"🔍 [{call_idx}/{len(prompts)}] Searching {company_name}: {label}…"}
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()
        future = loop.run_in_executor(None, _gemini_extract_deals_sync, prompt, company_name)

        elapsed = 0
        call_deals: list[dict] = []

        while elapsed < CALL_TIMEOUT:
            try:
                call_deals = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat",
                       "message": f"🌐 [{call_idx}/{len(prompts)}] Gemini searching… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Gemini call {call_idx} error for {company_name}: {e}", exc_info=True)
                yield {"type": "heartbeat", "message": f"⚠️ Call {call_idx} error: {e}"}
                call_deals = []
                break
        else:
            future.cancel()
            logger.warning(f"Call {call_idx} timed out for {company_name}")
            yield {"type": "heartbeat", "message": f"⏱ Call {call_idx} timed out — partial results below"}

        # Deduplicate and stream new deals immediately
        new_deals = 0
        for deal in call_deals:
            dedup_key = f"{deal.get('vendor','').lower()}|{deal.get('date_signed','')}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)
            row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": 1}
            row.update(deal)
            yield {"type": "row_done", "row": row}
            await asyncio.sleep(0.05)
            new_deals += 1
            total_deals += 1

        yield {"type": "heartbeat",
               "message": f"✅ Call {call_idx} done: +{new_deals} deals (total {total_deals})"}
        await asyncio.sleep(0)

    # ── Fallback: retry with brand name if subsidiary name returned nothing ──────
    if total_deals == 0 and brand_name.lower() != company_name.lower():
        yield {"type": "heartbeat",
               "message": f"🔄 No deals for '{company_name}' — retrying as '{brand_name}'…"}
        await asyncio.sleep(0)

        fallback_prompts = _build_prompts(brand_name, domain, linkedin_url, ft, fv)
        for call_idx, prompt in enumerate(fallback_prompts, 1):
            yield {"type": "heartbeat",
                   "message": f"🔍 Fallback [{call_idx}/{len(fallback_prompts)}] Searching '{brand_name}'…"}
            await asyncio.sleep(0)

            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, _gemini_extract_deals_sync, prompt, brand_name)
            elapsed = 0
            call_deals: list[dict] = []

            while elapsed < CALL_TIMEOUT:
                try:
                    call_deals = await asyncio.wait_for(asyncio.shield(future), timeout=10)
                    break
                except asyncio.TimeoutError:
                    elapsed += 10
                    yield {"type": "heartbeat", "message": f"🌐 Fallback searching… ({elapsed}s)"}
                    await asyncio.sleep(0)
                except Exception as e:
                    logger.error(f"Fallback error for {brand_name}: {e}", exc_info=True)
                    call_deals = []
                    break
            else:
                future.cancel()

            for deal in call_deals:
                dedup_key = f"{deal.get('vendor','').lower()}|{deal.get('date_signed','')}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                row = {"company_name": company_name, "domain": domain,
                       "_status": "ok", "_sources": 1}
                row.update(deal)
                yield {"type": "row_done", "row": row}
                await asyncio.sleep(0.05)
                total_deals += 1

        if total_deals > 0:
            yield {"type": "heartbeat",
                   "message": f"✅ Fallback found {total_deals} deals via '{brand_name}'"}

    # ── IT-partner fallback: if still 0 deals, search for the company's IT partners ──
    if total_deals == 0:
        yield {"type": "heartbeat",
               "message": f"🔄 No deals found — searching IT partner ecosystem for {company_name}…"}
        await asyncio.sleep(0)

        partner_prompt = f"""You are an IT partnership research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}

TASK: Find all known IT partners, technology alliances, reseller agreements, OEM deals,
and strategic technology partnerships that {company_name} has with other companies.

Run these searches:
- "{company_name}" IT partner alliance technology partnership agreement announcement
- "{company_name}" partner ecosystem reseller integration partner program
- "{company_name}" strategic alliance OEM technology vendor 2022 OR 2023 OR 2024 OR 2025
- "{company_name}" partnership program certified partner implementation partner
- site:businesswire.com OR site:prnewswire.com "{company_name}" partnership alliance

For each IT partner relationship found, return one JSON object with ALL fields populated:
[
  {{
    "vendor": "<exact name of the partner company>",
    "deal_type": "<one of: Technology Alliance | Reseller Partnership | Implementation Partner | OEM Agreement | Integration Partner | Co-sell Agreement | Strategic Partner | Managed Services Partner | Cloud Partner>",
    "deal_value": "<deal or partnership value if publicly stated, else empty string>",
    "date_signed": "<announcement date as YYYY-MM-DD or YYYY-MM or YYYY — search press releases to find this>",
    "deal_focus": "<1-3 specific technology areas covered by this partnership from: AI | ML | Cloud | CRM | ERP | Big Data | Analytics | Cybersecurity | IoT | Automation | DevOps | Digital Transformation | Payments | Infrastructure | SaaS>",
    "description": "<one clear sentence: what {company_name} and the partner do together and why the partnership exists>",
    "source": "<direct URL to the press release, partner page, or announcement — must be a real URL>"
  }}
]

FIELD RULES:
- vendor: exact company name, not a product name
- deal_type: must be one of the options listed above
- date_signed: always populate if findable — search "[company] [partner] partnership announced"
- deal_focus: specific technologies e.g. "AI, Cloud" or "CRM, Analytics" — not generic terms
- description: explain what each company contributes and what customers get
- source: real URL — press release preferred over generic partner page

Return ONLY the raw JSON array. No prose. No markdown fences.
"""

        loop = asyncio.get_event_loop()
        partner_future = loop.run_in_executor(None, _gemini_extract_deals_sync, partner_prompt, company_name)
        elapsed = 0
        partner_deals: list[dict] = []

        while elapsed < CALL_TIMEOUT:
            try:
                partner_deals = await asyncio.wait_for(asyncio.shield(partner_future), timeout=10)
                break
            except asyncio.TimeoutError:
                elapsed += 10
                yield {"type": "heartbeat", "message": f"🌐 Searching IT partner ecosystem… ({elapsed}s)"}
                await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"IT partner fallback error for {company_name}: {e}", exc_info=True)
                partner_deals = []
                break
        else:
            partner_future.cancel()

        if partner_deals:
            yield {"type": "heartbeat",
                   "message": f"✅ Found {len(partner_deals)} IT partners for {company_name}"}
            for deal in partner_deals:
                dedup_key = f"{deal.get('vendor','').lower()}|partner"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)
                row = {"company_name": company_name, "domain": domain, "_status": "ok", "_sources": 1}
                row.update(deal)
                yield {"type": "row_done", "row": row}
                await asyncio.sleep(0.05)
                total_deals += 1
        else:
            yield {"type": "heartbeat", "message": f"⚠️ No deals or IT partners found for {company_name}"}
            row = {"company_name": company_name, "domain": domain,
                   "_status": "no_result", "_sources": 0}
            for f in SCHEMA_FIELDS:
                row[f["key"]] = ""
            yield {"type": "row_done", "row": row}
