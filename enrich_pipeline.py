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


# Guaranteed fallback bucket — used only when Level 3 is blank or has zero
# token overlap with any taxonomy entry, so L1/L2/L3 are never left empty.
_FALLBACK_L3 = "Industry Specific Applications"

_STOPWORDS = {"and", "or", "the", "for", "of", "a", "an", "to", "in", "on", "&"}


def _tokenize(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if w and w not in _STOPWORDS}


# Words too generic to count as a meaningful taxonomy match on their own — a single
# shared hit on one of these (e.g. "data", "customer") must never be enough to win a
# category, since it caused real misclassifications (a cloud/consulting deal mapped
# to "Fixed Data"/"Customer Premise Equipment" purely from one incidental shared word).
_GENERIC_MATCH_WORDS = {
    "data", "service", "services", "system", "systems", "management", "platform",
    "solution", "solutions", "technology", "application", "applications", "digital",
    "enterprise", "business", "software", "infrastructure", "network", "cloud",
    "customer", "customers", "company", "operations", "operation", "process",
    "processes", "support", "team", "teams", "project", "projects", "client",
}

# Description-text matching is noisier than matching the model's own concise
# tech_level3 choice (long free text has far more chances of incidental word overlap)
# so it requires a higher bar — at least 2 meaningful shared terms, not 1.
_MIN_FUZZY_SCORE_KEY = 1
_MIN_FUZZY_SCORE_DESC = 2

# Strong signal words that indicate an IT services / consulting / SI engagement —
# used as a smarter fallback than the generic catch-all bucket when nothing else matches.
_CONSULTING_SIGNAL_WORDS = (
    "migrat", "consult", "implement", "integrat", "moderni", "transform",
    "deploy", "rollout", "onboard", "advisory",
)

# Curated, deterministic phrase → Level 3 mappings for the deal types this tool
# actually encounters most (cloud, data/analytics, AI, ERP, security, outsourcing).
# Checked BEFORE noisy token-overlap fuzzy matching because a specific product/
# platform name (e.g. "BigQuery", "Compute Engine") is a far more reliable signal
# than generic word overlap against a 108-entry taxonomy.
_STRONG_SIGNALS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("bigquery", "looker", "tableau", "power bi", "qlik", "data warehouse",
      "data analytics platform", "business intelligence", "data analysis"),
     "Business Intelligence and Analytics Tools"),
    (("speech-to-text", "speech api", "text-to-speech", "vision api",
      "natural language api", "generative ai", "large language model", " llm "),
     "Industry Specific Applications"),
    (("compute engine", "amazon ec2", " ec2 ", "virtual machine", "vm instance",
      "infrastructure-as-a-service", " iaas"),
     "Infrastructure-as-a-Service (IaaS)"),
    (("app engine", "cloud run", "elastic beanstalk", "platform-as-a-service", " paas"),
     "Platform-as-a-Service (PaaS)"),
    (("software-as-a-service", " saas "),
     "Software-as-a-Service (SaaS)"),
    (("s/4hana", "sap erp", "oracle erp", "erp implementation"),
     "ERP Financials"),
    (("workday", "successfactors", "human capital management", " hcm "),
     "ERP Human Capital Management"),
    (("salesforce", "crm platform", "customer relationship management platform"),
     "Sales Automation"),
    (("robotic process automation", " rpa ", "uipath", "automation anywhere"),
     "Robotic Process Automation (RPA)"),
    (("managed security service", "mssp", "siem", "threat detection", "soc as a service"),
     "Managed Security Services"),
    (("data center outsourcing", "colocation"),
     "Data Center Outsourcing"),
    (("help desk", "service desk", "itsm"),
     "Service Desk"),
    (("digital transformation programme", "digital transformation program"),
     "Digital Transformation"),
    (("data migration", "cloud migration", "migrating its data", "migrated its",
      "system integration", "systems integrator", "implementation partner"),
     "System Integration"),
)


def _fuzzy_score(a_tokens: set[str], b_tokens: set[str]) -> int:
    """Token overlap score that ignores matches made up entirely of generic words —
    e.g. {"data"} ∩ {"fixed","data"} scores 0, not 1, since "data" alone is meaningless."""
    shared = a_tokens & b_tokens
    meaningful = shared - _GENERIC_MATCH_WORDS
    return len(meaningful) if meaningful else 0


def _detect_strong_signal(description: str) -> str | None:
    """Deterministic phrase match against curated, high-confidence signals. Returns
    the first matching canonical Level 3 name, or None if nothing matches."""
    if not description:
        return None
    d = f" {description.lower()} "
    for phrases, canonical in _STRONG_SIGNALS:
        if any(p in d for p in phrases):
            return canonical
    return None


def classify_tech(level3_raw: str, description: str = "") -> tuple[str, str, str]:
    """Return (level1, level2, canonical_level3) — mandatory, non-empty triple.

    Matching order (most to least reliable):
      1. Curated strong-signal phrase match on the DESCRIPTION (e.g. "BigQuery" ->
         Business Intelligence and Analytics Tools) — deterministic and high-confidence
      2. Exact / substring match on the model's own tech_level3 choice — but only
         trusted if it isn't contradicted by a strong signal (see override below)
      3. Token-overlap fuzzy match on tech_level3 (handles paraphrasing)
      4. Token-overlap fuzzy match on the deal DESCRIPTION text — requires a HIGHER
         overlap bar than step 3 since free text has far more incidental word overlap
      5. Consulting/SI signal fallback — implementation/migration language defaults
         to "System Integration" rather than an unrelated category
      6. Guaranteed fallback bucket — never leaves Level 1/2/3 blank.
    """
    key = (level3_raw or "").strip().lower()
    strong_signal = _detect_strong_signal(description)

    canonical = _TAXONOMY_LOWER.get(key) if key else None

    if not canonical and key:
        # Substring match
        for lk, lv in _TAXONOMY_LOWER.items():
            if key in lk or lk in key:
                canonical = lv
                break

    # Sanity-check: the model can pick a literal taxonomy name that's actually
    # unrelated to the deal (e.g. "Customer Premise Equipment" for a cloud/data
    # deal just because "customer" appears in the description). If a strong,
    # specific signal from the description disagrees with an implausible
    # Hardware/Communications pick, prefer the strong signal.
    if canonical:
        l1_check, _ = TECH_TAXONOMY[canonical]
        if l1_check in ("Communications", "Hardware") and canonical != strong_signal:
            if strong_signal:
                canonical = strong_signal
            elif description and any(sig in description.lower() for sig in _CONSULTING_SIGNAL_WORDS):
                canonical = "System Integration"

    if not canonical and key:
        # Token-overlap fuzzy match on the model's tech_level3 choice
        key_tokens = _tokenize(key)
        if key_tokens:
            best_score, best_lv = 0, None
            for lk, lv in _TAXONOMY_LOWER.items():
                score = _fuzzy_score(key_tokens, _tokenize(lk))
                if score > best_score:
                    best_score, best_lv = score, lv
            if best_score >= _MIN_FUZZY_SCORE_KEY:
                canonical = best_lv

    if not canonical and strong_signal:
        canonical = strong_signal

    if not canonical and description:
        # Fall back to mapping the deal DESCRIPTION itself against the taxonomy —
        # stricter threshold since this is the noisiest signal
        desc_tokens = _tokenize(description)
        if desc_tokens:
            best_score, best_lv = 0, None
            for lk, lv in _TAXONOMY_LOWER.items():
                score = _fuzzy_score(desc_tokens, _tokenize(lk))
                if score > best_score:
                    best_score, best_lv = score, lv
            if best_score >= _MIN_FUZZY_SCORE_DESC:
                canonical = best_lv

    if not canonical and description:
        desc_lower = description.lower()
        if any(sig in desc_lower for sig in _CONSULTING_SIGNAL_WORDS):
            canonical = "System Integration"

    if not canonical:
        canonical = _FALLBACK_L3

    l1, l2 = TECH_TAXONOMY[canonical]
    return (l1, l2, canonical)


# ── Deal-value estimation model ───────────────────────────────────────────────
# Three independent signals are combined: deal type, company size, technical
# scale tier. Each axis narrows the plausible $ range so estimates vary
# meaningfully across deals rather than collapsing to a flat category midpoint.

# 1. Deal-type patterns (checked in priority order: outsourcing > saas >
#    consulting > implementation).  "implementation" is the catch-all default.
_DEAL_TYPE_RE: dict[str, list[re.Pattern]] = {
    "outsourcing": [re.compile(p, re.I) for p in [
        r"outsourc", r"managed.?service", r"\bbpo\b", r"\bilo\b", r"\bito\b",
        r"facilities.management", r"run.the.bank", r"end.to.end.operations",
        r"strategic.partnership.*service", r"multi.year.*service",
    ]],
    "saas": [re.compile(p, re.I) for p in [
        r"\bsaas\b", r"software.as.a.service", r"\bsubscription\b",
        r"annual.license", r"per.seat", r"per.user", r"cloud.license",
        r"perpetual.license",
    ]],
    "consulting": [re.compile(p, re.I) for p in [
        r"\bconsult", r"\badvisory\b", r"strategy.engagement",
        r"\bassessment\b", r"\baudit\b", r"\broadmap\b", r"\bfeasibility\b",
    ]],
    "implementation": [re.compile(p, re.I) for p in [
        r"implement", r"migrat", r"deploy", r"rollout", r"\bintegrat",
        r"go.live", r"digital.transform", r"upgrade", r"replac", r"moderniz",
        r"overhaul", r"development",
    ]],
}

# 2. Company-size detection from deal description text.
#    enterprise > large > small; "mid" is the default when nothing matches.
_CO_SIZE_RE: dict[str, list[re.Pattern]] = {
    "enterprise": [re.compile(p, re.I) for p in [
        r"fortune.?(?:500|\d+)", r"s&p\s*500",
        r"\$[\d,]+\s*b(?:illion)?\s*(?:in\s*)?(?:revenue|turnover|assets)",
        r"(?:revenue|turnover).{0,30}\$[\d,]+\s*b",
        r"\d{3},\d{3}[\s+]*employee",             # 100,000+ employees
        r"global\s+(?:bank|insurer|conglomerate|manufacturer|retailer|group)",
        r"multinational", r"listed.on.(?:nyse|nasdaq|lse|bse|nse)",
        r"tier.?1\s+bank", r"central\s+bank",
    ]],
    "large": [re.compile(p, re.I) for p in [
        r"\$[\d,.]+\s*(?:m|mn|million)[\s,]+(?:in\s*)?(?:revenue|turnover)",
        r"(?:revenue|turnover).{0,30}\$[\d,.]+\s*(?:m|mn|million)",
        r"\d{1,2},\d{3}[\s+]*employee",            # 1,000–19,999 employees
        r"national\s+(?:bank|insurer|company|airline)",
        r"state.owned", r"public\s+sector\s+(?:bank|company|enterprise)",
        r"leading\s+(?:bank|company|provider|insurer|telecom)",
        r"(?:regional|domestic)\s+(?:bank|lender)",
    ]],
    "small": [re.compile(p, re.I) for p in [
        r"\bstartup\b", r"\bstart.up\b", r"\bscale.up\b",
        r"early.stage", r"\bsmb\b", r"\bsme\b",
        r"seed.funded", r"series\s+[a-c]\s+funded",
    ]],
}

# 3. TCV ranges (low_M, high_M) indexed by (deal_type, company_size).
#    SaaS ranges are annual (ACV); the function multiplies by contract years.
#    Outsourcing ranges assume a ~5-year base; shorter/longer contracts scale.
_VALUE_MODEL: dict[tuple[str, str], tuple[float, float]] = {
    # Outsourcing / managed services
    ("outsourcing", "enterprise"): (150.0, 2000.0),
    ("outsourcing", "large"):       (40.0,  400.0),
    ("outsourcing", "mid"):         (10.0,  100.0),
    ("outsourcing", "small"):        (3.0,   25.0),
    # SaaS / subscription (ACV — multiplied by years below)
    ("saas",        "enterprise"):  (10.0,  200.0),
    ("saas",        "large"):        (3.0,   50.0),
    ("saas",        "mid"):          (0.5,   15.0),
    ("saas",        "small"):        (0.1,    3.0),
    # Implementation / migration / SI
    ("implementation", "enterprise"): (15.0, 200.0),
    ("implementation", "large"):       (5.0,  60.0),
    ("implementation", "mid"):         (1.5,  20.0),
    ("implementation", "small"):       (0.3,   5.0),
    # Consulting / advisory
    ("consulting",  "enterprise"):    (2.0,  25.0),
    ("consulting",  "large"):         (0.8,  12.0),
    ("consulting",  "mid"):           (0.2,   5.0),
    ("consulting",  "small"):         (0.1,   2.0),
}
_DEFAULT_MODEL_RANGE = (2.0, 30.0)   # mid + unknown type

# Scale-signal tier → percentile within (low, high)
_TIER_PCT = {0: 0.30, 1: 0.45, 2: 0.60, 3: 0.75, 4: 0.90}


def _detect_deal_type(description: str, tech_level2: str) -> str:
    """Classify deal as outsourcing | saas | consulting | implementation."""
    text = f"{description} {tech_level2}"
    for dtype in ("outsourcing", "saas", "consulting", "implementation"):
        for pat in _DEAL_TYPE_RE[dtype]:
            if pat.search(text):
                return dtype
    l2 = tech_level2.lower()
    if any(x in l2 for x in ("outsourc", "bpo", "managed")):
        return "outsourcing"
    if any(x in l2 for x in ("saas", "software infra", "license")):
        return "saas"
    if "consult" in l2:
        return "consulting"
    return "implementation"


def _detect_company_size(description: str) -> str:
    """Return enterprise | large | mid | small based on description signals."""
    for size in ("enterprise", "large", "small"):
        for pat in _CO_SIZE_RE[size]:
            if pat.search(description):
                return size
    return "mid"


def _fmt_m(amount_m: float) -> str:
    """Format a $-million amount into a compact string like '$45M' or '$1.2B'."""
    if amount_m >= 1000:
        b = amount_m / 1000
        return f"${b:.1f}B" if b % 1 else f"${b:.0f}B"
    if amount_m >= 100:
        return f"${round(amount_m / 10) * 10:.0f}M"
    if amount_m >= 10:
        return f"${round(amount_m / 5) * 5:.0f}M"
    if amount_m >= 1:
        return f"${round(amount_m):.0f}M"
    return f"${round(amount_m * 1000):.0f}K"


def _match_grounding_source(vendor: str, description: str,
                             grounding_sources: list[tuple[str, str]]) -> str:
    """Pick the grounding-verified URL whose title best overlaps the deal's vendor/
    description. Always returns a real, working URL if grounding triggered (falls
    back to the first chunk when no title overlap), or '' if Google Search grounding
    returned nothing at all — so the frontend never shows a confidently broken link."""
    if not grounding_sources:
        return ""
    target_tokens = _tokenize(f"{vendor} {description}")
    best_score, best_uri = -1, grounding_sources[0][0]
    for uri, title in grounding_sources:
        score = len(target_tokens & _tokenize(title))
        if score > best_score:
            best_score, best_uri = score, uri
    return best_uri


def _detect_scale_signal(description: str) -> int:
    """Scan the deal description for concrete technical-scale numbers (data volume,
    systems/tables/databases migrated) and return a size tier 1-4, or 0 if no signal
    found. Calibrated so a typical SI case-study migration (e.g. 60-70TB, 5,000+
    tables, 20+ databases) lands in tier 2 ($1.5M-$4.5M) — matching real-world
    market pricing for that scale rather than a flat enterprise-category default."""
    if not description:
        return 0
    d = description.lower()

    data_tb = None
    pb_match = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:pb|petabytes?)', d)
    tb_match = re.search(r'(\d+(?:\.\d+)?)\s*\+?\s*(?:tb|terabytes?)', d)
    if pb_match:
        data_tb = float(pb_match.group(1)) * 1000
    elif tb_match:
        data_tb = float(tb_match.group(1))

    count_match = re.search(
        r'(\d{1,3}(?:,\d{3})*)\s*\+?\s*(?:legacy\s+)?(?:tables?|databases?|applications?|systems?|servers?)', d
    )
    system_count = int(count_match.group(1).replace(",", "")) if count_match else None

    score = 0
    if data_tb is not None:
        if data_tb >= 1000: score = max(score, 4)
        elif data_tb >= 100: score = max(score, 3)
        elif data_tb >= 10: score = max(score, 2)
        else: score = max(score, 1)
    if system_count is not None:
        if system_count > 20000: score = max(score, 4)
        elif system_count > 5000: score = max(score, 3)
        elif system_count >= 50: score = max(score, 2)
        else: score = max(score, 1)
    return score




def estimate_deal_value(row: dict) -> tuple[str, str]:
    """Return (deal_value, deal_estimated).

    Three-axis model: deal_type × company_size × scale_tier.
    Only called when the LLM didn't return a confirmed value.
    """
    existing = (row.get("deal_value") or "").strip()
    if existing and existing not in ("-", "—"):
        return existing, row.get("deal_estimated", "")

    description = row.get("description", "")
    tech_l2     = row.get("tech_level2", "")

    deal_type  = _detect_deal_type(description, tech_l2)
    co_size    = _detect_company_size(description)
    scale_tier = _detect_scale_signal(description)   # 0–4

    low, high = _VALUE_MODEL.get((deal_type, co_size), _DEFAULT_MODEL_RANGE)
    amount_m  = low + _TIER_PCT[scale_tier] * (high - low)

    # SaaS: amount_m is ACV — multiply by contract years for TCV
    if deal_type == "saas":
        try:
            months = int(row.get("duration_months") or 0)
        except (ValueError, TypeError):
            months = 0
        years   = months / 12 if months > 0 else 3.0   # default 3-yr SaaS
        amount_m = amount_m * years

    # Outsourcing: base range assumes ~5 years; scale for actual duration
    if deal_type == "outsourcing":
        try:
            months = int(row.get("duration_months") or 0)
        except (ValueError, TypeError):
            months = 0
        if months > 0:
            years    = months / 12
            amount_m = amount_m * (years / 5.0) ** 0.85  # sub-linear scaling

    return _fmt_m(amount_m), "Y"

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
     "description": "Detailed description of what was agreed — up to 100-200 words when "
                     "enough information is available (scope, technical detail, scale, business "
                     "rationale); a short one-sentence summary only if little is known"},
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


def _tcv_acv_rules(company_name: str) -> str:
    return f"""- deal_value (TCV): NUMERIC $ ONLY — "$XM" or "$XB". Use the publicly stated figure if
  available; otherwise ESTIMATE using ALL of the following signals in order:

  SIGNAL 1 — Deal type baseline:
  • Multi-year outsourcing / managed services (BPO/ITO/MSP): $50M–$2B depending on scope & company size
  • SaaS subscription / perpetual license: estimate ACV first, then multiply by contract years for TCV
  • SI / implementation / migration project: $1M–$200M based on technical scope below
  • Consulting / advisory engagement: $0.2M–$25M
  • If value is "undisclosed" or "undisclosed enterprise account": label as
    "Estimated Enterprise Account", "Volume-Based Pricing", or "Managed Services Framework" AND still
    provide a numeric estimate in the deal_value field.

  SIGNAL 2 — Company scale (use your knowledge of {company_name}'s revenue & headcount):
  • Annual IT spend ≈ 1–4% of revenue. A single project TCV rarely exceeds 10–20% of annual IT budget
    unless it is a flagship multi-year mega-deal.
  • Fortune 500 / global enterprise ($B revenue): enterprise-scale budgets apply.
  • Mid-market ($100M–$1B revenue): scale estimates down by 3–10×.
  • Startup / SMB (<$100M revenue): scale estimates down by 10–50×.

  SIGNAL 3 — Technical scope (extract concrete numbers from source):
  • <10 systems / <10TB / single team / <6 months → $0.3M–$1.5M
  • 10–5,000 systems / 10–100TB / one quarter → $1.5M–$10M
  • 5,000–20,000 systems / 100TB–1PB / 6–18 months / dedicated programme → $10M–$50M
  • 20,000+ systems / 1PB+ / multi-year / large org-wide programme → $50M–$200M
  • Full enterprise outsourcing (Fortune 500, 5–10 yr, global scope) → $200M–$2B

  SIGNAL 4 — Duration multiplier (for outsourcing & SaaS):
  • If duration_months is known: TCV = ACV × (duration_months / 12). Apply to both SaaS and
    recurring managed-services deals. Use this to cross-check your estimate.

  SIGNAL 5 — Sanity check: Final TCV must satisfy BOTH company scale and technical scope signals.
  Whichever gives the lower value wins.

  Output format: exactly "$50M" or "$2.5B" — NO other text.

- deal_acv (ACV): NUMERIC $ ONLY. ALWAYS derive if possible:
  • If TCV and duration are known: ACV = TCV / (duration_months / 12)
  • If deal is a subscription/SaaS: ACV = annual license/subscription fee
  • If deal is recurring managed services: ACV = annual run-rate
  • If truly one-off with no annual recurring component (e.g. a one-time migration): empty string
  Output format: "$10M" — NO other text. Empty string ONLY if genuinely non-recurring."""


def _make_prompt(company_name: str, domain: str, linkedin_block: str,
                 search_focus: str, known_vendors: str,
                 extra_searches: str, fields_desc: str, fields_json_keys: dict,
                 sector_block: str = "") -> str:
    return f"""[Role]: You are an elite IT Market Intelligence Analyst and Deal Finder with live Google Search access.

COMPANY: {company_name} | Website: {domain}{linkedin_block}

DEAL CATEGORIES TO CAPTURE (find ALL of these):
1. IT Outsourcing & Managed Services — SI contracts, BPO, ITO, MSP, infrastructure managed services
2. Cloud & Digital Transformation — cloud migration, SaaS rollouts, digital programmes
3. ERP / CRM / HCM / SCM — enterprise application implementations and upgrades
4. IT Acquisitions — tech company acquisitions, acqui-hires, asset purchases with IT angle
   (acquirer must actually integrate/operate the technology — not a passive financial stake)
5. Strategic Joint Ventures & Technology Partnerships — JVs with tech firms to jointly build
   or operate a technology platform (NOT capital/equity investments — see exclusion below)
6. Enterprise Operations Partnerships — long-term IT ops partnerships, co-innovation agreements
7. Technology Disinvestments — IT asset sales, carve-outs, spin-offs, divestitures of tech units
8. Cybersecurity & Compliance — security platform contracts, SOC outsourcing, compliance tools
9. Analytics, AI & Data — AI/ML platform deals, data lake, BI, advanced analytics contracts
{sector_block}

CAPTIVE / GCC AWARENESS: If {company_name} operates internal Captive/GCC centres or has been
insourcing IT work that was previously outsourced, document it explicitly — internal GCC footprints
are market-competing entities. Note any "lift and shift" from vendor to captive.

EXCLUDE (do NOT return):
- Funding rounds, equity investments, CVC, seed/Series A-Z, or purely financial transactions
- IPOs, SPAC mergers, minority stakes without operational IT delivery component

SEARCHES TO RUN:
{search_focus}
Vendor/partner searches (pair each with company name):
{known_vendors}
{extra_searches}

EXTRACTION RULES:
- Do NOT omit a deal because the value is "Undisclosed" — label its size as
  "Estimated Enterprise Account", "Volume-Based Pricing", or "Managed Services Framework"
  and still provide a numeric TCV estimate in deal_value.
- Read full articles, not just headlines. One JSON object per distinct deal.
- Capture deals across ALL years available (2010 onwards).
- Include press releases, news, vendor announcements, IR filings, annual reports,
  AND LinkedIn posts/case studies from implementation partners ("proud to announce", etc.).
- When a deal involves both an SI partner (Accenture, TCS…) AND an underlying platform
  (Google Cloud, AWS…), the SI partner is "vendor" — platform goes in description.

Return ONLY a valid JSON array:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- vendor: exact name. If a subsidiary of a parent, write "Subsidiary (Parent)" e.g.
  "Niveus Solutions (NTT DATA)". Do not add brackets for standalone vendors.
- tech_level3: COMPULSORY — pick the BEST match from this taxonomy (exact name, no paraphrasing):
{TECH_L3_LIST}
- tech_level2: leave empty string — auto-mapped from tech_level3
- tech_level1: leave empty string — auto-mapped from tech_level3
{_tcv_acv_rules(company_name)}
- deal_estimated: "Y" if deal_value is estimated (not a stated public figure). "" if confirmed public.
- start_date: contract start or go-live date (YYYY-MM-DD or YYYY-MM or YYYY)
- end_date: contract expiry or renewal date if known
- duration_months: contract length in months; derive from start+end if not stated; outsourcing≈60, SaaS≈36
- last_detected: date of press release / article (YYYY-MM-DD or YYYY-MM or YYYY)
- deal_focus: 1-3 tags from: AI | ML | Cloud | Big Data | Analytics | Cybersecurity | IoT |
  Automation | ERP | Digital Transformation | Payments | Open Banking | DevOps | Data Platform | Other
- description: UP TO 100-200 WORDS — scope, platforms, data volumes, team size, business rationale.
  Short sentence only if source has genuinely little detail.
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
            "fintech acquisition technology integration",
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


# ── Tech/Segment → Operational Taxonomy Overdrive ─────────────────────────────
# Maps a user's free-text tech/segment input to precise contract types and
# search metrics instead of generic keyword search.
_TECH_TAXONOMY: list[tuple[tuple[str, ...], dict]] = [
    (("customer service", "contact center", "call center", "cx", "cx outsourcing",
      "customer experience", "customer support"), {
        "label": "Customer Service / Contact Center / CX Outsourcing",
        "contracts": ["Omnichannel Contact Center as a Service (CCaaS)",
                      "Tier-1/Tier-2 Inbound Support", "Multilingual Technical Troubleshooting",
                      "Customer Retention & Churn Mitigation", "Social Media Triage & Content Moderation",
                      "Inbound/Outbound Tele-Sales"],
        "metrics": ["per-minute or per-interaction pricing", "agent headcount/seat volume",
                    "First Contact Resolution (FCR) bonuses", "NPS-linked SLAs",
                    "onshore vs. offshore seat blend"],
    }),
    (("bpo", "bps", "shared services", "horizontal bpo", "f&a", "finance and accounting",
      "hro", "human resources outsourcing", "procurement bpo"), {
        "label": "BPO / BPS / Horizontal Shared Services",
        "contracts": ["Finance & Accounting (F&A) incl. AP/AR, Global Treasury Management, Invoice Consolidation",
                      "Human Resources Outsourcing (HRO) incl. International Payroll, Benefits Management, Contract Staff Augmentation",
                      "Procurement & Strategic Sourcing BPO"],
        "metrics": ["transaction-volume pricing", "FTE rate models",
                    "transformation-linked cost savings clauses"],
    }),
    (("mortgage", "kyc", "aml", "pharmacovigilance", "utility field service",
      "medical billing", "claims adjudication", "travel reservations", "vertical bpo",
      "industry-specific bpo"), {
        "label": "Industry-Specific Vertical BPO",
        "contracts": ["Mortgage Loan Processing & Underwriting", "KYC/AML Fraud Alert Triage",
                      "Pharmacovigilance & Adverse Event Reporting", "Utility Field-Service Dispatch Routing",
                      "Medical Billing/Claims Adjudication", "Flight/Travel Inventory Reservations"],
        "metrics": ["per-loan/per-claim pricing", "cyclical volume scale-up/scale-down flexibility",
                    "regulatory compliance failure penalties"],
    }),
    (("ai", "genai", "generative ai", "artificial intelligence", "automation", "llm",
      "rag", "agentic", "rpa", "computer vision", "machine learning", "ml"), {
        "label": "AI / Generative AI / Automation",
        "contracts": ["LLM Fine-Tuning Labs", "RAG (Retrieval-Augmented Generation) Architecture",
                      "AI-Native Core Engineering", "Agentic Workflow Orchestration",
                      "Computer Vision Inspection", "RPA-to-Agentic migrations"],
        "metrics": ["token-based pricing frameworks", "outcome-based automation SLAs",
                    "compute-hosting co-location agreements", "MLOps pipeline maintenance"],
    }),
    (("cloud", "infrastructure", "modernization", "migration", "mainframe",
      "kubernetes", "containerization", "data center", "finops", "edge compute"), {
        "label": "Cloud / Infrastructure / Modernization",
        "contracts": ["Multi-Cloud Migrations (AWS/Azure/GCP)", "Mainframe Decommissioning",
                      "Application Containerization (Kubernetes/Docker)",
                      "Hybrid Infrastructure Outsourcing", "FinOps Cost Optimization", "Edge Compute Nodes"],
        "metrics": ["data center exit timelines", "cloud consumption commitments",
                    "managed infrastructure service levels (99.99% uptime)",
                    "refactoring vs. lift-and-shift scope"],
    }),
    (("cybersecurity", "security", "identity", "risk", "mssp", "soc", "zero trust",
      "iam", "sase", "vulnerability"), {
        "label": "Cybersecurity / Identity / Risk",
        "contracts": ["Managed Security Service Providers (MSSP)", "SOC as a Service",
                      "Zero-Trust Implementations", "IAM (Identity & Access Management)",
                      "SASE Deployments", "Vulnerability Remediation pipelines"],
        "metrics": ["incident response retainer structures", "continuous compliance monitoring",
                    "sovereign data center localization mandates"],
    }),
    (("data", "analytics", "fabric", "data mesh", "data lake", "snowflake",
      "databricks", "mdm", "master data", "cdp"), {
        "label": "Data / Analytics / Fabric",
        "contracts": ["Enterprise Data Mesh/Fabric deployment", "Snowflake/Databricks data lake migrations",
                      "Real-time telemetry analytics", "Master Data Management (MDM) cleanups",
                      "Customer Data Platform (CDP) integrations"],
        "metrics": ["terabyte/petabyte ingest pricing", "pipeline architecture overhaul SOWs",
                    "downstream data monetization ventures"],
    }),
    (("erp", "crm", "hcm", "business applications", "sap", "salesforce", "workday",
      "servicenow"), {
        "label": "Business Applications (ERP/CRM/HCM)",
        "contracts": ["SAP S/4HANA migrations", "Salesforce Core/Industry Cloud updates",
                      "Workday global rollouts", "ServiceNow workflow integrations",
                      "legacy ERP customizations cleanup"],
        "metrics": ["seat-licensing vs. implementation SOW values", "SI phase-gates",
                    "multi-region rollout timelines"],
    }),
]


def _match_tech_taxonomy(tech_input: str) -> dict | None:
    t = tech_input.lower()
    for triggers, tax in _TECH_TAXONOMY:
        if any(kw in t for kw in triggers):
            return tax
    return None


# ── 37-Industry Matrix → Macro-Sector fallback (both tech & vendor blank) ─────
_INDUSTRY_MATRIX: list[tuple[tuple[str, ...], str, str]] = [
    (("bank", "financial", "insurance", "asset management", "wealth", "lending",
      "payments", "fintech", "capital markets"), "BFS",
     "Core Banking Platforms, RegTech/AML Compliance BPO, Digital Banking Customer Care, Fraud/Cyber Risk Security"),
    (("hospital", "health", "pharma", "biotech", "medical", "clinical", "life sciences"), "Healthcare",
     "EMR/EHR Core Platforms, Revenue Cycle Management BPO, Patient Engagement Customer Care, Pharmacovigilance ER&D"),
    (("telecom", "media", "broadcast", "entertainment", "streaming", "publishing"), "TMT",
     "BSS/OSS Core Platforms, Network Engineering R&D, Subscriber Customer Care, Content Security & DRM"),
    (("manufactur", "industrial", "automotive", "aerospace", "defense", "machinery"), "Manufacturing",
     "MES/PLM Core Platforms, Product Engineering R&D, Supply Chain BPO, OT/ICS Security"),
    (("retail", "e-commerce", "consumer goods", "apparel", "grocery"), "Retail",
     "POS/OMS Core Platforms, Merchandising Engineering, Customer Care & Loyalty BPO, Payment Security"),
    (("energy", "oil", "gas", "utility", "power", "renewable"), "Energy",
     "SCADA/Grid Core Platforms, Field Engineering R&D, Field-Service Dispatch BPO, Critical Infrastructure Security"),
    (("logistics", "shipping", "freight", "supply chain", "transportation", "airline", "rail"), "Logistics",
     "TMS/WMS Core Platforms, Fleet Engineering R&D, Dispatch & Tracking BPO, Cargo Security"),
    (("government", "public sector", "defense", "federal", "municipal", "gcc"), "Government",
     "Citizen Services Core Platforms, Systems Engineering R&D, Citizen Support BPO, National Security/Compliance"),
]


def _match_industry_matrix(company_name: str) -> tuple[str, str] | None:
    """Best-effort industry guess from company name keywords only (no external call —
    the model itself will verify/correct using its own knowledge of the company)."""
    name_lower = company_name.lower()
    for triggers, macro_sector, contract_lines in _INDUSTRY_MATRIX:
        if any(kw in name_lower for kw in triggers):
            return macro_sector, contract_lines
    return None


def _conditional_search_block(company_name: str, focus_tech: list[str], focus_vendor: list[str]) -> str:
    """Implements the 3-mode conditional search logic:
    1. Tech/segment provided → Tech & Operational Taxonomy Overdrive
    2. Vendor provided → Competitive Swarm Logic (Steps A/B/C)
    3. Both blank → 37-Industry Matrix Fallback
    Modes combine when both tech and vendor are provided.
    """
    lines: list[str] = []

    if focus_tech:
        lines.append("[MODE: TECH/SEGMENT — Operational Taxonomy Overdrive]")
        lines.append("Do NOT search generic keywords. Use the precise operational/delivery layers below.")
        for t in focus_tech:
            tax = _match_tech_taxonomy(t)
            if tax:
                lines.append(f'\n  Input "{t}" → {tax["label"]}:')
                lines.append(f"  Target Contracts: {', '.join(tax['contracts'])}")
                lines.append(f"  Search Metrics: {', '.join(tax['metrics'])}")
                for c in tax["contracts"][:4]:
                    lines.append(f'    - "{company_name}" {c} contract OR deal OR agreement')
            else:
                lines.append(f'\n  Input "{t}" (no taxonomy match — search literally):')
                lines.append(f'    - "{company_name}" {t} contract OR pilot OR MSA OR implementation')
                lines.append(f'    - "{company_name}" {t} deal OR agreement OR rollout')

    if focus_vendor:
        lines.append("\n[MODE: VENDOR — Competitive Swarm Logic]")
        for v in focus_vendor:
            lines.append(f"  STEP A — Isolate all active contracts, SOWs, and Business Unit Specific")
            lines.append(f"    Agreements (BUSAs) held by {v} within {company_name}:")
            lines.append(f'    - "{company_name}" "{v}" contract OR SOW OR BUSA OR agreement OR deal')
            lines.append(f'    - "{company_name}" "{v}" managed services OR implementation OR outsourcing')
            lines.append(f"  STEP B — Automatically identify the top 5-10 direct industry competitors of {v}")
            lines.append(f"    (Tier-1 Indian IT peers, Global System Integrators, or boutique tech firms):")
            lines.append(f'    - "{company_name}" [competitor name] deal OR contract — for EACH top competitor')
            lines.append(f"  STEP C — Extract all active deals held by those competitors within {company_name}")
            lines.append(f"    to map market-share context. Also check for internal Captive/GCC insourcing")
            lines.append(f"    displacing {v} or its competitors.")

    if not focus_tech and not focus_vendor:
        matrix = _match_industry_matrix(company_name)
        lines.append("[MODE: 37-INDUSTRY MATRIX FALLBACK — both tech and vendor blank]")
        lines.append(f"  STEP A — Identify which of the 37 standard industry verticals {company_name}")
        lines.append(f"    belongs to (use your own knowledge of the company, do not guess blindly).")
        if matrix:
            macro_sector, contract_lines = matrix
            lines.append(f"  STEP B — Best-guess Macro-Sector: {macro_sector}. Map to high-value contract")
            lines.append(f"    lines: {contract_lines}. VERIFY this against your own knowledge of the company")
            lines.append(f"    and correct if wrong — dynamically extract deals targeting the ACTUAL vertical's")
            lines.append(f"    contract lines (ER&D, Core Platforms, Industry BPO, Customer Care, Security).")
        else:
            lines.append(f"  STEP B — Map {company_name}'s vertical to its Macro-Sector (BFS, Healthcare, TMT,")
            lines.append(f"    Manufacturing, Retail, Energy, Logistics, or Government), then dynamically")
            lines.append(f"    extract deals targeting that vertical's high-value contract lines: ER&D,")
            lines.append(f"    Core Platforms, Industry BPO, Customer Care, Security.")
        lines.append(f'    - "{company_name}" IT outsourcing vendor third-party 2020 2021 2022 2023 2024 2025')
        lines.append(f'    - "{company_name}" captive GCC internal technology centre insourcing')
        lines.append(f'    - "{company_name}" managed services BPO ITO infrastructure 2020 2021 2022 2023 2024')
        lines.append(f'    - "{company_name}" digital transformation ERP CRM cloud AI deal 2020 2021 2022 2023 2024')
        lines.append(f'    - "{company_name}" technology partnership joint venture co-innovation')

    lines.append("\n[STRICT INTELLIGENCE RULES]")
    lines.append('  - Do NOT omit a deal because the value is "Undisclosed". Label its size via an')
    lines.append('    alternative metric: "Estimated Enterprise Account", "Volume-Based Pricing", or')
    lines.append('    "Managed Services Framework" — and still provide a numeric TCV estimate.')
    lines.append(f"  - Treat internal Captive/GCC footprints at {company_name} as market-competing")
    lines.append(f"    entities. If {company_name} is pulling work away from third-party vendors to")
    lines.append(f"    insource it, explicitly document it as a deal row (vendor = \"Internal Captive/GCC\").")

    return "\n".join(lines)


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
    # LinkedIn/SI-partner searches are listed FIRST — these surface deals that never get
    # a press release (implementation partner case studies, employee announcement posts)
    # and are otherwise the most likely to be skipped if buried at the end of a long list.
    p1_searches = f"""  - site:linkedin.com/posts "{company_name}" partnered cloud migration OR data modernization
  - site:linkedin.com "{company_name}" case study cloud migration OR data platform
  - site:linkedin.com "{company_name}" "proud to" OR "excited to announce" implementation
  - site:linkedin.com "{company_name}" successfully migrated OR transformed onto
  - "{company_name}" implementation partner case study cloud OR data OR AI
  - "{company_name}" Google Cloud OR AWS OR Azure migration "implementation partner"
  - "{company_name}" IT outsourcing contract deal signed
  - "{company_name}" technology acquisition acqui-hire
  - "{company_name}" joint venture technology partner
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
  - site:businesswire.com OR site:prnewswire.com "{company_name}" technology deal

IMPORTANT: Run ALL of the searches above — do not stop early just because you found a few
deals. The LinkedIn and implementation-partner searches at the top often surface deals that
have NO press release anywhere else, so skipping them means missing real deals."""

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

    extra_parts.append(_conditional_search_block(company_name, focus_tech, focus_vendor))
    extra = "\n\n".join(extra_parts)

    prompt2 = _make_prompt(company_name, domain, linkedin_block,
                           year_searches, p2_vendors, extra,
                           fields_desc, fields_json_keys, sector_block)

    # ── Prompt 3: dedicated implementation-partner / case-study sweep ─────────
    # Isolated from the other ~20 search topics in Prompt 1 so the model's search
    # budget and attention go entirely toward LinkedIn case studies and SI/
    # implementation-partner announcements — these never get a press release and
    # were getting skipped when buried among unrelated queries.
    prompt3 = f"""You are an enterprise IT deal research analyst with live Google Search.

COMPANY: {company_name} | Website: {domain}{linkedin_block}

TASK: Find IT implementation-partner and systems-integrator deals for {company_name} that
were announced ONLY via LinkedIn posts, case studies, or partner blog posts — NOT via a
formal press release. These are commonly the technology/cloud/data implementation partner
(e.g. an NTT DATA subsidiary, a regional SI, a boutique consultancy) who executed a project
on behalf of a bigger-name platform vendor (Google Cloud, AWS, Azure, Salesforce, SAP, etc).

Run ALL of these searches before answering:
  - site:linkedin.com/posts "{company_name}" migrated OR migration OR modernization
  - site:linkedin.com "{company_name}" "case study"
  - site:linkedin.com "{company_name}" "proud to announce" OR "excited to share" OR "thrilled to partner"
  - "{company_name}" implementation partner cloud OR data OR AI case study
  - "{company_name}" "in partnership with" technology data cloud
  - "{company_name}" systems integrator OR SI partner project
  - "{company_name}" Google Cloud partner case study
  - "{company_name}" AWS partner case study
  - "{company_name}" Microsoft Azure partner case study
  - "{company_name}" NTT DATA OR Wipro OR Infosys OR TCS OR Capgemini OR Accenture case study

For each deal found, return one JSON object:
[
  {{
{fields_desc}
  }}
]

FIELD RULES:
- vendor: the SI/IMPLEMENTATION PARTNER who executed the work, NOT the platform
  (Google Cloud, AWS, etc.) — put platform in description. Subsidiary: "Sub (Parent)".
- tech_level3: COMPULSORY — pick the best match from this taxonomy:
{TECH_L3_LIST}
{_tcv_acv_rules(company_name)}
- deal_estimated: "Y" if deal_value is estimated. "" if confirmed public figure.
- description: UP TO 100-200 WORDS — what was migrated/modernized, onto which platform,
  technical scope (data volume, systems/tables/databases, team size, timeline), and business
  rationale. Short sentence only if source genuinely has little detail.
- source: leave as empty string — the real source link is attached automatically from
  search grounding; do not fabricate a URL.

Return ONLY the raw JSON array. If nothing found after all searches, return [].
No prose. No markdown fences."""

    return [prompt1, prompt2, prompt3]


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
                    max_output_tokens=24576,  # raised for longer (100-200 word) deal descriptions
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

        # ── Extract grounding metadata — REAL, working URLs from Google Search ──
        # The model frequently invents/misremembers "source" URLs in its JSON output,
        # which is the cause of 404s. Grounding chunks are Google's own verified
        # redirect links to the pages it actually searched, so we use these instead
        # of trusting whatever URL string the model typed into the JSON.
        grounding_sources: list[tuple[str, str]] = []  # (uri, title)
        try:
            for candidate in (response.candidates or []):
                gm = getattr(candidate, "grounding_metadata", None)
                if not gm:
                    continue
                for chunk in (getattr(gm, "grounding_chunks", None) or []):
                    web = getattr(chunk, "web", None)
                    if web and getattr(web, "uri", None):
                        grounding_sources.append((web.uri, getattr(web, "title", "") or ""))
        except Exception as g_err:
            logger.warning(f"Grounding metadata extraction failed for {company_name}: {g_err}")

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
            # Enforce taxonomy: derive L1/L2 from L3 (mandatory — falls back to
            # mapping the deal description itself if tech_level3 is unmatched)
            l1, l2, l3_canon = classify_tech(row.get("tech_level3", ""), row.get("description", ""))
            row["tech_level1"] = l1
            row["tech_level2"] = l2
            row["tech_level3"] = l3_canon
            # Enforce deal value: never blank — estimate from category benchmark
            # and flag as estimated ("A" superscript shown by the frontend)
            dv, est = estimate_deal_value(row)
            row["deal_value"] = dv
            row["deal_estimated"] = est
            # Replace the model's invented source URL with a Google-Search-verified
            # grounding link (real, working URL) — prevents 404s entirely
            row["source"] = _match_grounding_source(
                row.get("vendor", ""), row.get("description", ""), grounding_sources
            )
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

    def _vendor_core(v: str) -> str:
        """Strip any '(Parent Company)' bracket so subsidiary-with-parent naming
        doesn't break vendor-identity comparisons."""
        return re.sub(r"\s*\([^)]*\)\s*$", "", (v or "").strip().lower())

    emitted_deals: list[dict] = []  # all deals already yielded, for fuzzy dup check

    def _is_fuzzy_duplicate(deal: dict) -> bool:
        """Catches the SAME real-world deal being phrased differently across the
        2-3 separate Gemini calls (different snippet wording, missing/different
        dates) — the exact-match _dedup_key alone misses these, causing the same
        deal to appear 2-3 times. A duplicate is: same vendor (ignoring parent
        bracket) AND (same grounding source URL OR heavy description overlap)."""
        vendor = _vendor_core(deal.get("vendor", ""))
        if not vendor:
            return False
        source = (deal.get("source") or "").strip()
        desc_tokens = _tokenize(deal.get("description", ""))
        for existing in emitted_deals:
            if _vendor_core(existing.get("vendor", "")) != vendor:
                continue
            if source and source == (existing.get("source") or "").strip():
                return True
            existing_tokens = _tokenize(existing.get("description", ""))
            if not desc_tokens or not existing_tokens:
                continue
            smaller = min(len(desc_tokens), len(existing_tokens))
            if smaller and len(desc_tokens & existing_tokens) / smaller >= 0.5:
                return True
        return False

    def _dedup_key(deal: dict, suffix: str = "") -> str:
        """Vendor + date + description snippet — NOT vendor alone, so a vendor
        with multiple distinct deals (common) isn't collapsed into one row."""
        vendor = (deal.get("vendor") or "").strip().lower()
        date_part = (deal.get("start_date") or deal.get("last_detected")
                     or deal.get("end_date") or deal.get("date_signed") or "").strip()
        desc_part = (deal.get("description") or "")[:60].strip().lower()
        return f"{vendor}|{date_part}|{desc_part}|{suffix}"

    prompts = _build_prompts(company_name, domain, linkedin_url, ft, fv)
    seen_keys: set[str] = set()   # deduplicate across the two calls
    total_deals = 0
    CALL_TIMEOUT = 150            # 2.5 min per call

    for call_idx, prompt in enumerate(prompts, 1):
        label = {1: "broad IT & cloud deals", 2: "year-by-year + vendor sweep",
                  3: "LinkedIn / implementation-partner sweep"}.get(call_idx, "additional sweep")
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
            dedup_key = _dedup_key(deal)
            if dedup_key in seen_keys or _is_fuzzy_duplicate(deal):
                continue
            seen_keys.add(dedup_key)
            emitted_deals.append(deal)
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
                dedup_key = _dedup_key(deal)
                if dedup_key in seen_keys or _is_fuzzy_duplicate(deal):
                    continue
                seen_keys.add(dedup_key)
                emitted_deals.append(deal)
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
- vendor: exact company name, not a product name. If the partner is a subsidiary/brand of a
  larger parent company, name the subsidiary first with the parent in brackets:
  "Niveus Solutions (NTT DATA)". Only add the bracket when a real parent-subsidiary
  relationship is involved in THIS deal.
- deal_type: must be one of the options listed above
- date_signed: always populate if findable — search "[company] [partner] partnership announced"
- deal_focus: specific technologies e.g. "AI, Cloud" or "CRM, Analytics" — not generic terms
- description: write UP TO 100-200 WORDS when the source material supports it — explain what
  each company contributes, what customers get, technical/commercial scope, and why the
  partnership exists. Only stay short if the source genuinely has little detail.
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
                dedup_key = _dedup_key(deal, suffix="partner")
                if dedup_key in seen_keys or _is_fuzzy_duplicate(deal):
                    continue
                seen_keys.add(dedup_key)
                emitted_deals.append(deal)
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
