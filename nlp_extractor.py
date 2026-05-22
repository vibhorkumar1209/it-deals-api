"""NLP extraction engine: vendor/SI detection, deal value, dates, scope."""

import logging
import os
import re
from typing import Any

import dateparser
import regex

logger = logging.getLogger(__name__)

# ─── MASTER LISTS ────────────────────────────────────────────────────────────

VENDOR_MASTER: dict[str, list[str]] = {
    "ERP": [
        "SAP S/4HANA", "SAP ECC", "SAP RISE", "SAP Business One", "SAP",
        "Oracle Fusion Cloud", "Oracle Fusion", "Oracle E-Business Suite",
        "Oracle NetSuite", "NetSuite", "Oracle ERP", "Oracle",
        "Dynamics 365", "Dynamics AX", "Dynamics NAV", "Dynamics GP",
        "Microsoft Dynamics",
        "Infor CloudSuite", "Infor LN", "Infor M3", "Infor ERP", "Infor",
        "IFS Cloud", "IFS Applications", "IFS",
        "Epicor Kinetic", "Epicor ERP", "Epicor",
        "Sage X3", "Sage Intacct", "Sage 300", "Sage",
        "Unit4 ERP", "Unit4 Business World", "Unit4",
        "QAD ERP", "QAD",
        "Workday Financials",
        "Deltek", "Acumatica", "Syspro", "Rootstock", "Cetec ERP",
    ],
    "CRM": [
        "Sales Cloud", "Service Cloud", "Marketing Cloud", "Salesforce Platform",
        "Salesforce CRM", "Salesforce",
        "Dynamics 365 Sales", "Dynamics 365 Customer Engagement",
        "Microsoft Dynamics CRM",
        "HubSpot CRM", "HubSpot",
        "Oracle CX", "Oracle Siebel", "Siebel CRM",
        "SAP C/4HANA", "SAP Customer Experience", "SAP CX",
        "Zendesk Sell", "Zendesk CRM", "Zendesk",
        "ServiceNow CSM", "ServiceNow CRM",
        "Freshsales", "SugarCRM", "Zoho CRM", "Pipedrive", "Copper CRM", "Creatio",
    ],
    "HCM_HR": [
        "Workday HCM", "Workday Human Capital Management", "Workday",
        "SAP SuccessFactors", "SuccessFactors", "SAP HCM",
        "Oracle Fusion HCM", "Oracle Cloud HCM", "PeopleSoft HCM", "Oracle HCM",
        "ADP Workforce Now", "ADP Vantage HCM", "ADP TotalSource", "ADP",
        "Ceridian Dayforce", "Dayforce", "Ceridian",
        "UKG Pro", "UKG Ready", "Kronos", "Ultimate Software", "UKG",
        "Cornerstone OnDemand", "Cornerstone",
        "BambooHR", "Personio", "Rippling", "Lattice", "Paylocity", "Paychex",
        "Infor HCM", "Infor Workforce Management",
    ],
    "SCM_PROCUREMENT": [
        "SAP Ariba", "Ariba", "SAP Integrated Business Planning", "SAP IBP", "SAP SCM",
        "Oracle Transportation Management", "OTM", "Oracle Supply Chain", "Oracle SCM",
        "Blue Yonder WMS", "Blue Yonder", "JDA Software",
        "Kinaxis RapidResponse", "Kinaxis Maestro", "Kinaxis",
        "o9 Solutions", "o9",
        "Coupa Business Spend Management", "Coupa Procurement", "Coupa",
        "Jaggaer", "GEP Smart", "GEP NEXXE", "GEP",
        "Manhattan Active", "Manhattan SCALE", "Manhattan Associates",
        "E2open", "Infor Nexus", "Llamasoft", "Logility", "Relex Solutions",
        "Anaplan", "Ivalua", "Zycus", "Basware",
    ],
    "CLOUD": [
        "Amazon Web Services", "Amazon EC2", "Amazon S3", "AWS",
        "Microsoft Azure", "Azure Cloud", "Azure",
        "Google Cloud Platform", "Google Workspace", "Google Cloud", "GCP",
        "IBM Cloud Infrastructure", "IBM Cloud",
        "Oracle Cloud Infrastructure", "Oracle Cloud", "OCI",
        "Alibaba Cloud", "Aliyun",
        "VMware Cloud", "VMware vSphere", "VMware Horizon", "VMware",
        "Nutanix", "HPE GreenLake", "Hewlett Packard Enterprise", "HPE",
        "Dell Technologies", "Dell EMC", "Dell",
        "Rackspace", "OVHcloud", "DigitalOcean", "Equinix", "Zayo",
        "Lumen Technologies", "Cloudflare", "Akamai",
    ],
    "CYBER": [
        "Palo Alto Networks", "Prisma Cloud", "Cortex XDR", "Palo Alto",
        "CrowdStrike Falcon", "Falcon Platform", "CrowdStrike",
        "FortiGate", "FortiAnalyzer", "Fortinet",
        "Cisco SecureX", "Cisco Umbrella", "Cisco Security", "Cisco",
        "Check Point Harmony", "Check Point Software", "Check Point",
        "Zscaler Internet Access", "Zscaler Private Access", "Zscaler",
        "Darktrace",
        "SentinelOne Singularity", "SentinelOne",
        "Microsoft Sentinel", "Microsoft Defender", "Azure Sentinel",
        "Splunk SIEM", "Splunk SOAR", "Splunk",
        "IBM QRadar", "QRadar", "IBM Security",
        "Tenable.io", "Nessus", "Tenable",
        "Rapid7", "Qualys", "CyberArk", "BeyondTrust",
        "Okta", "Ping Identity", "ForgeRock", "SailPoint",
        "Secureworks", "Mandiant", "Arctic Wolf", "Trellix",
        "Broadcom Security", "Symantec", "Cybereason", "Illumio", "Vectra AI",
    ],
    "ANALYTICS": [
        "Palantir Foundry", "Palantir AIP", "Palantir",
        "Snowflake Data Cloud", "Snowflake",
        "Databricks Lakehouse", "Databricks",
        "MicroStrategy ONE", "MicroStrategy",
        "Qlik Sense", "QlikView", "Qlik",
        "Tableau Cloud", "Tableau Software", "Tableau",
        "Microsoft Power BI", "Power BI",
        "C3.ai", "C3 AI",
        "DataRobot AI Cloud", "DataRobot",
        "SAS Analytics", "SAS Viya", "SAS",
        "IBM watsonx", "IBM Watson", "Watson",
        "Google BigQuery", "BigQuery",
        "Alteryx Analytics", "Alteryx",
        "ThoughtSpot", "Domo", "Looker",
        "Teradata", "Informatica", "IICS",
        "Talend", "MuleSoft", "Dell Boomi", "Boomi", "Azure Data Factory",
    ],
    "ITSM": [
        "ServiceNow ITSM", "ServiceNow ITOM", "ServiceNow HR", "ServiceNow",
        "BMC Helix", "BMC Remedy", "BMC",
        "Jira Service Management", "Atlassian Jira", "Jira",
        "Freshservice", "Freshdesk",
        "Ivanti", "Cherwell",
        "Dynatrace", "New Relic", "Datadog", "AppDynamics",
        "OpenText ITSM", "OpenText", "Micro Focus",
        "Axios Systems", "EasyVista", "ManageEngine", "Manage Engine",
    ],
    "OUTSOURCING": [
        "HCL Technologies", "HCLTech", "HCL",
        "Unisys", "DXC Technology", "DXC",
        "NTT DATA", "NTT", "Fujitsu", "CGI Group", "CGI",
        "Atos SE", "Atos Syntel", "Sopra Steria", "Atos",
        "T-Systems", "Deutsche Telekom IT",
        "Conduent", "Xerox", "Stefanini",
        "Hexaware", "Mphasis", "Zensar", "Birlasoft", "Coforge",
    ],
}

SI_PARTNER_MASTER: dict[str, list[str]] = {
    "BIG_4_CONSULTING": [
        "Deloitte Digital", "Deloitte Consulting", "Deloitte Technology", "Deloitte",
        "EY-Parthenon", "Ernst & Young", "EY",
        "PwC Digital", "PricewaterhouseCoopers", "PwC",
        "KPMG Consulting", "KPMG",
    ],
    "TIER_1_GLOBAL_SI": [
        "Accenture Technology", "Accenture Federal", "Accenture",
        "IBM Global Business Services", "IBM GBS", "IBM Consulting", "IBM",
        "Capgemini Engineering", "Capgemini Invent", "Capgemini",
        "Cognizant Technology Solutions", "Cognizant",
        "Infosys BPM", "Infosys Consulting", "Infosys",
        "Tata Consultancy Services", "TCS",
        "Wipro Technologies", "Wipro Digital", "Wipro",
        "HCL Technologies", "HCLTech",
        "Tech Mahindra", "Tech M",
    ],
    "TIER_2_SPECIALIST_SI": [
        "Avanade", "Slalom Consulting", "Slalom",
        "Publicis Sapient", "Genpact",
        "LTIMindtree", "LTI", "Mindtree",
        "Mphasis", "Hexaware", "NIIT Technologies",
        "Mastech Digital", "Syntel",
        "Persistent Systems", "Coforge",
        "Stefanini", "ScienceSoft", "Ness Digital",
    ],
    "VENDOR_OWN_CONSULTING": [
        "SAP Premium Engagement", "SAP Services", "SAP Consulting",
        "Oracle Advanced Customer Services", "Oracle Consulting",
        "Microsoft Consulting Services", "MCS",
        "Salesforce Professional Services",
        "AWS Professional Services",
        "Google Cloud Professional Services",
        "Workday Consulting", "ServiceNow Elite Partner",
    ],
    "BOUTIQUE_REGIONAL_SI": [
        "Rizing", "Seidor", "SNP Transformation Backbone", "SNP Group",
        "All for One Group", "Plexus Systems", "Velocity Technology Solutions",
        "Rimini Street", "Amdocs", "Netcracker",
        "Unison", "Clarkston Consulting",
        "Sunrise Technologies", "Arbela Technologies",
    ],
}

VENDOR_ALIASES: dict[str, str] = {
    "SAP SE": "SAP", "SAP AG": "SAP", "SAP America": "SAP", "SAP Inc": "SAP",
    "Oracle Corporation": "Oracle", "Oracle Corp": "Oracle", "ORCL": "Oracle",
    "Microsoft Corp": "Microsoft", "Microsoft Corporation": "Microsoft", "MSFT": "Microsoft",
    "Amazon Web Services": "AWS", "Amazon AWS": "AWS", "Amazon.com": "AWS",
    "Google Cloud Platform": "Google Cloud", "GCP": "Google Cloud", "Alphabet Inc": "Google Cloud",
    "International Business Machines": "IBM", "IBM Corp": "IBM",
    "Tata Consultancy Services": "TCS", "TCS Ltd": "TCS", "TCS Limited": "TCS",
    "HCL Technologies": "HCLTech", "HCL Tech": "HCLTech", "HCL Ltd": "HCLTech",
    "Palo Alto": "Palo Alto Networks", "PANW": "Palo Alto Networks",
    "CrowdStrike Inc": "CrowdStrike", "CRWD": "CrowdStrike",
    "Salesforce Inc": "Salesforce", "Salesforce.com": "Salesforce", "SFDC": "Salesforce",
    "Workday Inc": "Workday", "Workday Ltd": "Workday", "WDAY": "Workday",
    "Ernst & Young": "EY", "Ernst and Young": "EY",
    "PricewaterhouseCoopers": "PwC", "PricewaterhouseCoopers LLP": "PwC",
    "Deloitte LLP": "Deloitte", "Deloitte Touche": "Deloitte", "DTT": "Deloitte",
    "Capgemini SE": "Capgemini", "Cap Gemini": "Capgemini",
    "Cognizant Technology Solutions": "Cognizant", "CTS": "Cognizant",
    "Blue Yonder Group": "Blue Yonder", "JDA": "Blue Yonder",
    "DXC Technology Company": "DXC",
}

VENDOR_SIGNALS = {"platform", "software", "license", "cloud", "solution", "suite", "product"}
SI_SIGNALS = {"implement", "deploy", "integrate", "partner", "systems integrator",
              "consulting", "awarded to", "services partner", "delivered by"}


def normalize_name(raw: str) -> str:
    return VENDOR_ALIASES.get(raw.strip(), raw.strip())


def _find_matches_in_text(text: str, master: dict[str, list[str]]) -> list[tuple[str, str, int]]:
    """Returns list of (name, category, char_position) sorted by match length desc."""
    text_lower = text.lower()
    matches = []
    for category, names in master.items():
        for name in names:
            idx = text_lower.find(name.lower())
            if idx != -1:
                matches.append((name, category, idx))
    # Longest match wins
    matches.sort(key=lambda x: len(x[0]), reverse=True)
    # Deduplicate: remove shorter matches subsumed by longer ones
    seen_positions = []
    unique = []
    for name, cat, pos in matches:
        overlaps = any(abs(pos - sp) < len(name) for sp in seen_positions)
        if not overlaps:
            unique.append((name, cat, pos))
            seen_positions.append(pos)
    return unique


def extract_vendor(text: str, company_names: list[str] | None = None) -> dict[str, str]:
    matches = _find_matches_in_text(text, VENDOR_MASTER)
    if matches:
        name, category, _ = matches[0]
        return {"vendor": normalize_name(name), "vendor_category": category}

    # spaCy NER fallback
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:5000])
        orgs = [e.text for e in doc.ents if e.label_ == "ORG"]
        if company_names:
            orgs = [o for o in orgs if o not in company_names]
        if orgs:
            return {"vendor": normalize_name(orgs[0]), "vendor_category": "OTHER",
                    "_confidence_override": "Low"}
    except Exception:
        pass

    return {"vendor": "", "vendor_category": "OTHER"}


def extract_si_partner(text: str) -> str | None:
    matches = _find_matches_in_text(text, SI_PARTNER_MASTER)
    if not matches:
        return None

    text_lower = text.lower()
    for name, _cat, pos in matches:
        # Check surrounding context to determine role
        window = text_lower[max(0, pos - 150):pos + 150]
        vendor_score = sum(1 for s in VENDOR_SIGNALS if s in window)
        si_score = sum(1 for s in SI_SIGNALS if s in window)
        if si_score >= vendor_score:
            return normalize_name(name)

    return normalize_name(matches[0][0])


def extract_deal_value(text: str) -> float | None:
    patterns = [
        r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(million|billion|bn|mn|m\b)',
        r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s?(million|billion)\s?(dollar|USD)',
        r'USD\s?(\d+(?:\.\d+)?)\s?(M|B|bn|mn)',
        r'(?:deal|contract|agreement)\s+(?:worth|valued at|of)\s+\$?(\d+(?:\.\d+)?)\s*(million|billion|bn|mn|M|B)?',
    ]
    for pat in patterns:
        m = regex.search(pat, text, regex.IGNORECASE)
        if m:
            groups = [g for g in m.groups() if g]
            try:
                amount = float(groups[0].replace(",", ""))
                unit = groups[1].lower() if len(groups) > 1 else "million"
                if unit in ("billion", "bn", "b"):
                    amount *= 1000
                if amount > 50_000:
                    return None  # sanity check
                return amount
            except (ValueError, IndexError):
                continue
    return None


def extract_deal_duration(text: str) -> str | None:
    patterns = [
        (r'(\d+)[\-\s]year\s+(?:deal|contract|agreement|term)', lambda m: f"{m.group(1)} years"),
        (r'(multi[\-\s]year|multiyear)', lambda m: "multi-year"),
        (r'through\s+(20\d{2})', lambda m: f"through {m.group(1)}"),
        (r'until\s+(20\d{2})', lambda m: f"until {m.group(1)}"),
        (r'(\d+)[\-\s]month\s+(?:contract|agreement)', lambda m: f"{m.group(1)} months"),
        (r'from\s+(20\d{2})\s+(?:to|through|until)\s+(20\d{2})',
         lambda m: f"{int(m.group(2)) - int(m.group(1))} years ({m.group(1)}–{m.group(2)})"),
    ]
    for pat, fmt in patterns:
        m = regex.search(pat, text, regex.IGNORECASE)
        if m:
            return fmt(m)
    return None


def extract_announcement_date(text: str, url: str, soup=None) -> str | None:
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc)

    def _parse(raw: str | None) -> str | None:
        if not raw:
            return None
        try:
            dt = dateparser.parse(raw, settings={"RETURN_AS_TIMEZONE_AWARE": True})
            if dt and datetime(2018, 1, 1, tzinfo=timezone.utc) <= dt <= today:
                return dt.strftime("%Y-%m-%d")
        except Exception:
            pass
        return None

    if soup:
        # 1. article:published_time meta
        meta = soup.find("meta", property="article:published_time")
        if meta:
            r = _parse(meta.get("content"))
            if r:
                return r

        # 2. JSON-LD datePublished
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                import json
                data = json.loads(tag.string or "")
                if isinstance(data, dict) and "datePublished" in data:
                    r = _parse(data["datePublished"])
                    if r:
                        return r
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "datePublished" in item:
                            r = _parse(item["datePublished"])
                            if r:
                                return r
            except Exception:
                pass

        # 3. og:article:published_time
        meta2 = soup.find("meta", property="og:article:published_time")
        if meta2:
            r = _parse(meta2.get("content"))
            if r:
                return r

        # 4. <time> tag
        time_tag = soup.find("time")
        if time_tag:
            r = _parse(time_tag.get("datetime") or time_tag.get_text(strip=True))
            if r:
                return r

        # 5. class-based date elements
        date_el = soup.find(class_=re.compile(r"date|pubdate|timestamp", re.I))
        if date_el:
            r = _parse(date_el.get_text(strip=True))
            if r:
                return r

    # 6. URL pattern
    m = re.search(r'/(20\d{2})[/-](\d{2})[/-](\d{2})/', url)
    if m:
        candidate = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        r = _parse(candidate)
        if r:
            return r

    # 7. Text date patterns
    date_patterns = [
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}\b',
        r'\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}\b',
        r'\b20\d{2}-\d{2}-\d{2}\b',
    ]
    for pat in date_patterns:
        m = re.search(pat, text[:3000])
        if m:
            r = _parse(m.group(0))
            if r:
                return r

    return None


SCOPE_KEYWORDS: dict[str, list[str]] = {
    "ERP": ["finance", "procurement", "supply chain", "module", "go-live", "implementation",
            "country", "global", "rollout"],
    "CRM": ["sales", "customer", "pipeline", "leads", "service", "marketing", "contacts"],
    "CLOUD": ["migration", "workloads", "datacenter", "data center", "infrastructure",
              "compute", "storage"],
    "CYBER": ["security", "threat", "endpoint", "SOC", "vulnerability", "identity"],
    "ANALYTICS": ["data", "analytics", "reporting", "BI", "intelligence", "dashboard"],
    "HCM_HR": ["HR", "payroll", "talent", "workforce", "employees", "human resources"],
}


def extract_scope_of_service(text: str, vendor: str, vendor_category: str) -> str:
    if not vendor:
        return ""

    idx = text.lower().find(vendor.lower())
    if idx == -1:
        return ""

    # Extract 3-sentence window
    sentences = re.split(r'(?<=[.!?])\s+', text)
    vendor_sent_idx = None
    char_count = 0
    for i, s in enumerate(sentences):
        char_count += len(s)
        if char_count >= idx:
            vendor_sent_idx = i
            break

    if vendor_sent_idx is None:
        window = text[max(0, idx - 300): idx + 300]
    else:
        start = max(0, vendor_sent_idx - 1)
        end = min(len(sentences), vendor_sent_idx + 3)
        window = " ".join(sentences[start:end])

    # Return the most relevant sentence window — no LLM call (avoids hallucination)
    return window[:300].strip()


def is_deal_relevant(text: str, company_names: list[str], focus_deal_types: list[str]) -> bool:
    """
    Returns True only when the text both mentions the company AND contains strong,
    explicit IT deal/partnership/implementation language within 1500 chars of
    that company mention.  Generic words (cloud, platform, program, AI) alone
    no longer qualify — they must appear alongside an action verb or named vendor.
    """
    text_lower = text.lower()

    # Find the first occurrence of any company name
    company_pos = -1
    for cn in company_names:
        idx = text_lower.find(cn.lower())
        if idx != -1 and (company_pos == -1 or idx < company_pos):
            company_pos = idx
    if company_pos == -1:
        return False

    # Search window: 1500 chars either side of company mention
    window = text_lower[max(0, company_pos - 1500): company_pos + 1500]

    # Tier-1: explicit deal / action language — any single hit qualifies
    tier1 = [
        "signed a contract", "awarded a contract", "contract awarded",
        "outsourcing agreement", "outsourcing deal", "managed services agreement",
        "technology agreement", "it agreement", "service agreement",
        "selects ", "selected ", "chooses ", "chosen ", "adopts ", "adopted ",
        "implements ", "implementation of", "go-live", "goes live", "went live",
        "rolled out", "deploying ", "deployment of",
        "partners with", "partnership with", "strategic alliance",
        "teams with", "joined forces with",
        "rfp ", "rfq ", " bid ", "tender ",
        "digital transformation", "cloud migration", "cloud adoption",
        "managed service", "outsourc", "systems integrator", "si partner",
    ]
    if any(s in window for s in tier1):
        return True

    # Tier-2: named vendor/SI must appear alongside an action word
    vendors = [
        "sap", "oracle", "salesforce", "servicenow", "workday", "microsoft dynamics",
        "azure", "amazon web services", " aws ", "google cloud", "snowflake",
        "databricks", "palo alto networks", "crowdstrike", "fortinet", "zscaler",
        "splunk", "power bi", "successfactors", "ariba", "dynamics 365",
        "infosys", "tcs", "wipro", "accenture", "capgemini", "cognizant",
        "deloitte", "ibm consulting", "hcltech", "dxc", "atos",
    ]
    actions = [
        "implement", "deploy", "select", "choose", "adopt", "partner",
        "contract", "agreement", "outsourc", "migrat", "rollout",
        "go-live", "integrat", "award",
    ]
    vendor_in_window = any(v in window for v in vendors)
    action_in_window = any(a in window for a in actions)
    return vendor_in_window and action_in_window


def _vendor_near_company(text: str, company_names: list[str], vendor: str, window: int = 800) -> bool:
    """Return True if vendor appears within `window` chars of a company mention."""
    if not vendor:
        return False
    tl = text.lower()
    vl = vendor.lower()
    for cn in company_names:
        ci = tl.find(cn.lower())
        if ci == -1:
            continue
        region = tl[max(0, ci - window): ci + window]
        if vl in region:
            return True
    return False


def build_deal_record(
    text: str,
    url: str,
    source_type: str,
    company_name: str,
    company_names: list[str],
    soup=None,
) -> dict[str, Any] | None:
    if not is_deal_relevant(text, company_names, []):
        return None

    vendor_info = extract_vendor(text, company_names)
    vendor = vendor_info.get("vendor", "")

    # Reject if vendor found but not co-located with company (prevents false associations)
    if vendor and not _vendor_near_company(text, company_names, vendor):
        vendor_info = {"vendor": "", "vendor_category": "OTHER"}
        vendor = ""

    si = extract_si_partner(text)
    value = extract_deal_value(text)
    duration = extract_deal_duration(text)
    date = extract_announcement_date(text, url, soup)
    cat = vendor_info.get("vendor_category", "")
    scope = extract_scope_of_service(text, vendor, cat)

    # Drop record if nothing concrete was found (prevents phantom records)
    # Must have at least a vendor OR an explicit deal value OR a duration
    has_evidence = bool(vendor) or value is not None or bool(duration)
    if not has_evidence:
        return None

    # Classify record type from text signals
    tl = text.lower()
    if any(w in tl for w in ["partners with", "partnership", "alliance", "collaboration", "teams with"]):
        record_type = "partnership"
    elif any(w in tl for w in ["goes live", "go-live", "went live", "rolled out", "deployed", "launched"]):
        record_type = "implementation"
    elif any(w in tl for w in ["contract", "deal", "agreement", "awarded", "tender", "outsourc"]):
        record_type = "contract"
    elif any(w in tl for w in ["selects", "chooses", "chosen", "adopts", "selected"]):
        record_type = "vendor_selection"
    elif any(w in tl for w in ["transformation", "modernisation", "modernization", "initiative", "programme", "program"]):
        record_type = "initiative"
    else:
        record_type = "technology_announcement"

    # Build contextual title
    action_map = {
        "partnership":            f"{company_name} — {vendor} technology partnership" if vendor else f"{company_name} technology partnership",
        "implementation":         f"{company_name} implements {vendor}" if vendor else f"{company_name} technology implementation",
        "contract":               f"{company_name} — {vendor} contract" if vendor else f"{company_name} IT contract",
        "vendor_selection":       f"{company_name} selects {vendor}" if vendor else f"{company_name} vendor selection",
        "initiative":             f"{company_name} — {cat} digital initiative" if cat else f"{company_name} digital initiative",
        "technology_announcement": f"{company_name} — {vendor} {cat} announcement" if vendor else f"{company_name} technology announcement",
    }
    title = action_map.get(record_type, f"{company_name} technology announcement")[:120]

    # Confidence
    is_list_matched = vendor_info.get("_confidence_override") != "Low" and bool(vendor)
    is_primary = source_type in ("press_release", "sec_filing", "company_ir")
    if is_list_matched and date and value is not None and is_primary:
        confidence = "High"
    elif is_list_matched and date and scope:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Summary — only include facts actually extracted from the source
    type_label = record_type.replace("_", " ").title()
    summary_parts = []
    if scope:
        # scope is a direct text window — use it as the primary summary
        summary_parts.append(scope[:300])
    if vendor and date:
        summary_parts.append(f"[{company_name} / {vendor} — {date}]")
    elif vendor:
        summary_parts.append(f"[{company_name} / {vendor}]")
    elif date:
        summary_parts.append(f"[{company_name} — {date}]")
    if value is not None:
        summary_parts.append(f"Value: ${value}M.")
    if duration:
        summary_parts.append(f"Duration: {duration}.")
    summary = " ".join(summary_parts)[:500]
    if not summary:
        summary = scope[:300] if scope else ""

    return {
        "company_name": company_name,
        "deal_title": title,
        "record_type": record_type,          # contract | partnership | implementation | vendor_selection | initiative | technology_announcement
        "vendor": vendor,
        "vendor_category": cat,
        "si_partner": si,
        "scope_of_service": scope,
        "deal_value_usd": value,
        "deal_duration": duration,
        "announcement_date": date or "",
        "source_url": url,
        "all_source_urls": [url],
        "source_type": source_type,
        "confidence_level": confidence,
        "summary": summary,
    }
