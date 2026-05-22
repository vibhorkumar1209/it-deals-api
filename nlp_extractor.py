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
    "FRAMEWORK_LISTED": [
        "1E",
        "1Kosmos",
        "1NCE",
        "3 pillar",
        "3 pillar global",
        "3Infotech",
        "3i infotech",
        "42Q",
        "4L Data Intelligence",
        "6Connex",
        "6sense",
        "8x8",
        "A.P. Moller - Maersk",
        "ABB",
        "ABBYY",
        "ACL Digital",
        "ACL digital",
        "AGS Health",
        "AIMMS",
        "ANSR",
        "ARCON",
        "AT&T",
        "ATSG",
        "AU10TIX",
        "AVEVA",
        "Ab Initio Software",
        "Abnormal",
        "Absyss",
        "Access IT Automation",
        "Acoustic",
        "Acquia",
        "Act-On Software",
        "Act-on Software",
        "Actian",
        "Action IQ",
        "Adaptive clinical System",
        "Adexa",
        "Adobe",
        "Advantage club AI",
        "Aegis Software",
        "Aeries Technology",
        "Aeris Communications",
        "Aerospike",
        "Agiloft",
        "AiRISTA",
        "Aidaxis",
        "Airbase",
        "Airtable",
        "Aisera",
        "Aiwozo",
        "Akamai Technologies",
        "Akinon",
        "Akkodis",
        "Alaffia Health",
        "Alation",
        "Alchemer",
        "Alex Solutions",
        "Algolia",
        "Algonomy",
        "Alida",
        "Alivia Analytics",
        "Alkym",
        "Alkymi",
        "Allied Telesis",
        "Allmed healthcare Management",
        "Almaden",
        "Alorica",
        "Alpega Group",
        "Alpha Lifescience",
        "Altair",
        "Alten",
        "Alten",
        "Altilia",
        "Altimetrik",
        "Amalgamated Medical care management",
        "Amelia",
        "Amperity",
        "Anaconda",
        "Andesa Services",
        "Anglepoint",
        "Anjana Data",
        "Anodot",
        "Anomalo",
        "Anteriad",
        "Antworks",
        "Anunta",
        "Apexon",
        "Apollo.io",
        "Aporia",
        "AppDirect",
        "AppLearn",
        "AppNavi",
        "Apparound",
        "Appian",
        "Apporto",
        "Appsheet",
        "Aprimo",
        "Apromore",
        "Aptean",
        "Apty",
        "Aquanima",
        "Arcgate",
        "Arcserve",
        "Aria Systems",
        "Arista Networks",
        "Arkieva",
        "Armis",
        "Asana",
        "Ascentys",
        "Ashling partners",
        "AsiaInfo",
        "Aspire system",
        "Ataccama",
        "Athenahealth",
        "Atlan",
        "Augeo",
        "Automation Anywhere",
        "Avaloq",
        "Avarni",
        "Avature",
        "Aveva",
        "AvidXchange",
        "Awardco",
        "Awaya",
        "Axiom Real time metrics",
        "Axonius",
        "Axway",
        "BCG (Boston Consulting Group)",
        "BI worldwide",
        "BIPO",
        "BIS",
        "BMC Software",
        "BT",
        "BT Group",
        "BUSINESSNEXT",
        "Bain & Company",
        "Bandwidth",
        "BaramundiSoftware",
        "Barracuda",
        "Beamery",
        "BearingPoint",
        "Belkins",
        "Benchmark Gensuite",
        "Bespin Global",
        "Beta Systems Software",
        "BetterCloud",
        "Betty Blocks",
        "BigCommerce",
        "Bigcommerce",
        "BillingPlatform",
        "Billtrust",
        "Billwerk+",
        "Bit2win",
        "Bitdefender",
        "Bitsight",
        "Bizzabo",
        "BlackKite",
        "BlackLine",
        "Blackberry",
        "Bloomreach",
        "Blue Ridge",
        "Blue Triangle",
        "BlueConic",
        "BlueRock TMS",
        "Blueconic",
        "Blueiot",
        "Blueshift",
        "Bluevoyant",
        "Blume Global",
        "Bolloré Logistics",
        "Bonusly",
        "Bosch SDS",
        "Boston Consulting Group",
        "Boston Consulting Group",
        "Botcity",
        "Box",
        "Braincube",
        "Brandlive",
        "Bravura Solutions",
        "Braze",
        "Bridgenext",
        "Brightest",
        "Brillio",
        "Bristlecone",
        "BriteCore",
        "British telecom",
        "Broadbridge",
        "Broadcom (Symantec)",
        "Broadcom (VMware)",
        "Broadcom (Vmware)",
        "Broadridge",
        "Bryq",
        "Bubty",
        "Builder.io",
        "Buildkite",
        "C.H. Robinson (TMC)",
        "CBTW",
        "CData",
        "CEVA Logistics",
        "CI&T",
        "CICT Mobile",
        "CM.com",
        "COdoxo",
        "CSG",
        "CSS corps",
        "CXC",
        "Calero",
        "Cambium Networks",
        "Canonical",
        "Capegemini",
        "Carecloud",
        "Carelon",
        "Carenet",
        "Caspio",
        "Catchpoint",
        "Cato Networks",
        "Cegid",
        "Celigo",
        "Celonis",
        "CenTrak",
        "Cencora Pharma Lex",
        "Certero",
        "Certinia",
        "Chain IQ",
        "Chargebee",
        "Check Point Software Technologies",
        "CheckMarx",
        "Chetu",
        "Chronosphere",
        "ChurnZero",
        "Cience technology",
        "Cigniti",
        "Cigniti Technologies",
        "CircleCI",
        "Cisco Systems",
        "Citius Tech",
        "Citrix",
        "Clarishealth",
        "Claritev",
        "Clearbit",
        "CleverTap",
        "Clevertap",
        "ClickUp",
        "ClientSuccess",
        "Cloud 4C",
        "Cloud4C",
        "CloudBees",
        "CloudBolt",
        "CloudEagle",
        "CloudZero",
        "Cloudera",
        "Cloudfare",
        "Cluedin",
        "CoSchedule",
        "CobbleStone Software",
        "Cobblestone Software",
        "Cobrainer",
        "Cockroach Labs",
        "Cockroach labs",
        "Codeium",
        "CodiumAI",
        "Codoxo",
        "Cogito Tech",
        "Cognigy",
        "Cognism",
        "Cognite",
        "Cohere health",
        "Cohesity",
        "Collibra",
        "Color tokens",
        "Colt Technology Services",
        "Comagine Health",
        "Comba Telecom",
        "Comcast Business",
        "CommScope (RUCKUS)",
        "Commerce Layer",
        "Commerce layer",
        "Commercetools",
        "Commvault",
        "Computacenter",
        "Comviva",
        "Concentrix",
        "Conectys",
        "Confluent",
        "Conga",
        "Conga",
        "Conifer Health solutions",
        "Constructor",
        "Contemi",
        "Content Guru",
        "Contentful",
        "Contently",
        "Contentstack",
        "Contentstack(Lytics)",
        "ContractPodAi",
        "ContractPodAi",
        "ContractpodAI",
        "Contrast Security",
        "ControlUp",
        "Cora Systems",
        "Corbus",
        "Corcentric",
        "Cordial",
        "CoreMedia",
        "CoreStack",
        "Cority",
        "Cornerstone Guide",
        "Cotiviti",
        "Couchbase",
        "Covalen",
        "Coveo",
        "Crayon",
        "Credo AI",
        "Critical Manufacturing",
        "Crownpeak",
        "Crunchr",
        "Cubic Telecom",
        "Custify",
        "Cvent",
        "Cybage",
        "CyberProof",
        "Cyberproof",
        "Cyclone Robotics",
        "Cycognito",
        "Cyient",
        "DB Schenker",
        "DHL",
        "DP World",
        "DQLabs",
        "DSV",
        "DXC Technologies",
        "Darwinbox",
        "Dassault Systèmes",
        "Data Axle",
        "Data Theorem",
        "DataGalaxy",
        "Dataart",
        "Datactics",
        "Dataiku",
        "Datamatics",
        "Davra",
        "Deel",
        "Degreed",
        "Delaware",
        "Delinea",
        "Demandbase",
        "Demandscience",
        "Dematic",
        "Denodo",
        "Deposco",
        "Devo",
        "Dexian IT Solutions",
        "Dialpad",
        "Diligent",
        "Dizzion",
        "DocuSign",
        "DocuWare",
        "Docusign",
        "Domino Data Lab",
        "Dragon sourcing",
        "Dropbox (Business)",
        "Druva",
        "Duck Creek Technologies",
        "Dun & Bradstreet",
        "Dun and Bradstreet",
        "EDB",
        "EOS Software",
        "EPAM",
        "ESET",
        "EXL",
        "EXL Service",
        "EY (Ernst & Young)",
        "Eclerx",
        "Edetek",
        "EdgeVerve",
        "Edgeverve",
        "Edligo",
        "Egnyte",
        "Egress, a KnowBe4 company",
        "Ehrhardt Partner Group (EPG)",
        "Eightfold",
        "Elastic",
        "Elastic Path",
        "Elasticpath",
        "Electroneek",
        "Eleks",
        "Elisity",
        "Emagia",
        "Emids",
        "Emitwise",
        "Emmes",
        "Emporix",
        "EmpowerID",
        "Encora",
        "Endava",
        "Engage2Excel",
        "Entrust",
        "Epiance",
        "Epic Systems",
        "Epicor Software",
        "Epicore",
        "Epsilon",
        "Equiniti",
        "Equisoft",
        "Ericsson",
        "Eseye",
        "Esker",
        "Espressive",
        "Eurotech",
        "EventMobi",
        "Evicore",
        "Eviden (Atos subsidiary)",
        "Eviden (Atos)",
        "Evisort",
        "Evolent Health",
        "Evonsys",
        "Exabeam",
        "Exela Technologies",
        "Exosite",
        "Exotel",
        "Expandi Group",
        "Expeditors",
        "Experian",
        "Experian health",
        "Expert.ai",
        "Extreme Networks",
        "FNZ",
        "FOrtra",
        "FPT Softwares",
        "FactFinder",
        "Fairly AI",
        "Fiddler AI",
        "Fieldnation",
        "Filemaker",
        "FinQuery",
        "Finthrive",
        "Firmbee",
        "Firsthive",
        "Firstsource",
        "Fischer International Identity",
        "Five9",
        "Fiverr Pro",
        "Fivetran",
        "Flatworld solutions",
        "Flexera",
        "Flexera (Snow Software)",
        "Flexport",
        "Flexxible",
        "Flytxt",
        "Foiwe",
        "Forcepoint",
        "Fortra",
        "Fortrea",
        "Foundever",
        "FourKites",
        "Frends",
        "Freshworks",
        "Fuel50",
        "G-P",
        "GAINSystems",
        "GAVS",
        "GAVS Technologies",
        "GB Group",
        "GBST",
        "GE Vernova",
        "GEODIS",
        "GFT",
        "GFT technologies",
        "GS Lab",
        "GS Labs",
        "GS labs",
        "GTT Communications",
        "GaVS",
        "Gainsight",
        "Gainwell Technologies",
        "Gatekeeper",
        "Gear Inc",
        "Gemini People analytics",
        "Generix Group",
        "Genesys",
        "Getronics",
        "GitHub",
        "GitLab",
        "Glider AI",
        "Gloat",
        "Global Data Excellence",
        "Globallogic",
        "Globant",
        "Go Global",
        "Go Integro",
        "GoTo",
        "Gofigr",
        "GoodData",
        "Google (Workspace)",
        "Google Cloud (Apigee)",
        "Google LLC",
        "Gotransverse",
        "Grafana Labs",
        "Grant Thornton India",
        "Graphite",
        "Gratifi",
        "Gravitee.io",
        "Gravity Climate",
        "Grazitti",
        "Grazitti Interactive",
        "Greenlight",
        "GridDynamic",
        "Gridgain",
        "GroupBy",
        "Gsense",
        "Guidewire (InsuranceNow)",
        "Guidewire (InsuranceSuite)",
        "Gurucul",
        "Guusto",
        "Gyde",
        "H20.ai",
        "H3C",
        "HALO Recognition",
        "HCL Software",
        "HCLSoftware",
        "HGS",
        "HID Global",
        "HP Inc.",
        "HPE (Aruba)",
        "HSO",
        "HTC Global Services",
        "HTC Global services",
        "HYCU",
        "Halo Service Solutions",
        "Happiest minds",
        "Harman",
        "Harman DTS",
        "Harman Digital Transformation Solutions",
        "Harness",
        "HawkSearch",
        "Health Catalyst",
        "Healthedge",
        "Hellmann Worldwide Logistics",
        "HelpHero",
        "Helpware",
        "Hewlett Packard Enterprises",
        "Hexaware Technologies",
        "Hibob",
        "HighRadius",
        "Hillstone Networks",
        "HintEd",
        "Hireart",
        "Hireroad",
        "Hitachi Digital Services",
        "Hitachi Digitalservices",
        "Holistic AI",
        "Honeycomb",
        "Honico Systems",
        "Hornetsecurity",
        "Huawei",
        "Huawei Cloud",
        "Huawei Technologies",
        "Hughes",
        "Huron",
        "Hyland",
        "Hypatos",
        "Hyperscience",
        "Hyperscience Edge Verve Systems",
        "ICON PLC",
        "ICUC Social",
        "IKS Health",
        "IQM",
        "IQVIA",
        "IRONSCALES",
        "ISS corporatesolution",
        "ITA Group",
        "ITAM solutions",
        "ITC Infotech",
        "ITRS",
        "Icertis",
        "Ignitarium",
        "Illumifin",
        "Imprivata",
        "Improved Apps",
        "InMoment",
        "Incode Technologies",
        "Incorta",
        "Indico Data",
        "Indium",
        "Indium Software",
        "Infinite Computer solutions",
        "Infinx",
        "Infobip",
        "Infogain",
        "Infostretch",
        "Infosys Equinox",
        "Infosys Finacle",
        "Infrrd",
        "Inline Manual",
        "Innominds",
        "Innova Solutions",
        "Innova solution",
        "Innova solutions",
        "Innovacer",
        "Innover digital",
        "Inovoo",
        "Inpixon",
        "Insider",
        "Insight",
        "Inspira",
        "Inspirus",
        "Insuresoft",
        "IntelAgree",
        "IntelliTrans",
        "Intelligence indeed",
        "InterSystems",
        "Intershop",
        "Intersystems",
        "Invenio LSI",
        "Inveon",
        "Investcloud",
        "Invicti",
        "IonQ",
        "Irion",
        "Iris Software",
        "IronOrbit",
        "Ironclad",
        "Ironclad",
        "Iterable",
        "Itron",
        "JFrog",
        "JIFFY.ai",
        "Jade Global",
        "Jedox",
        "JetBrains",
        "Jiffy.ai",
        "Jio Platforms",
        "Jitterbit",
        "John Galt Solutions",
        "Josys",
        "Jumio",
        "Juniper",
        "Juniper Networks",
        "K2view",
        "KINEXON",
        "KNIME",
        "KORE",
        "Kaar technologies",
        "Kaltura",
        "Kameleoon",
        "Kaspersky",
        "Kentico",
        "Kepion",
        "Kerry Logistics",
        "KeyedIn",
        "Kibo",
        "Kin + Carta",
        "Kintone",
        "Kissflow",
        "Klevu",
        "Knoetic",
        "Knowledge Lake",
        "Knowmore",
        "Kodak Alaris",
        "Kong",
        "Kontark.io",
        "Korber",
        "Kore.AI",
        "Kroll",
        "Kudelski Security",
        "Kudos",
        "Kuehne+Nagel",
        "Kyndryl",
        "Kyndryl",
        "Körber",
        "L&T technology Services",
        "LIDP",
        "LTI Mindtree",
        "LTTS",
        "LYTIQS",
        "Laiye",
        "Lakeside Software",
        "Lansweeper",
        "Laserfiche",
        "LeadSquared",
        "Leaddesk",
        "Leadspace",
        "League",
        "Lemnisk",
        "Lemon Learning",
        "Lenovo",
        "LevelBlue",
        "LexisNexis",
        "Liferay",
        "Lightcast",
        "LinkSquares",
        "Links International",
        "Linksquares",
        "Liquidware",
        "Litmus",
        "Litum",
        "Liveperson",
        "Livingstone Group",
        "LogRhythm",
        "LogiSense",
        "LogicMonitor",
        "Logicsource",
        "Logpoint",
        "Logz.io",
        "Lokavant",
        "Lookout",
        "Losant",
        "Lucidworks",
        "Luma Health",
        "Lumos",
        "Lusha",
        "Lyric",
        "M-Files",
        "MDI",
        "MEHRWERK",
        "MIOsoft",
        "MRIOA",
        "MTI",
        "Machinify",
        "Made4net",
        "Madison Recognition",
        "Magnolia",
        "Majesco",
        "Majesco (P&C CoreConnect)",
        "Majesco (P&C Intelligent Core Suite)",
        "Malt",
        "Malwarebytes",
        "Mangoapps",
        "Mantis",
        "Maprecruit",
        "Marlabs",
        "Martal Group",
        "Mastek",
        "Mastercard Dynamic Yield",
        "MathWorks",
        "Matillion",
        "Matrix42",
        "Mauve Group",
        "Mavenir",
        "Maverick systems",
        "Maxio",
        "Maxis IT",
        "McKinsey & Company",
        "McKinsey",
        "Mckinsey and Company",
        "Mecalux",
        "Medallia",
        "Medidata Solutions",
        "Meditech",
        "Medpace",
        "Medreview",
        "Meduit",
        "Meiro",
        "Mendix",
        "Mercado Eletronico",
        "Mercans",
        "MercuryGate",
        "MessageGears",
        "MetTel",
        "Mia‑Platform",
        "Microland",
        "Microsoft (Business Central)",
        "Microsoft (Dynamics 365)",
        "Midmark",
        "Milestone Technologies",
        "Mimecast",
        "Mineral Tree",
        "Mirafra",
        "Mirantis",
        "Mirketa",
        "Mitek Systems",
        "Mitto",
        "MoEngage",
        "ModelOP",
        "Modsquad",
        "Modulos",
        "Monetate",
        "MongoDB",
        "Monitaur",
        "MorganFranklin Consulting",
        "Motivosity",
        "Movate",
        "Moveworks",
        "Muchskills",
        "Multiplier",
        "Myridius",
        "N-iX",
        "N.Rich",
        "NEC",
        "NICE",
        "NTT  Data",
        "NTT Application Security",
        "NX Group",
        "Nacelle",
        "Nakisa",
        "Nanoheal",
        "Nectar",
        "Neeyamo",
        "Neo4j",
        "Neobrain",
        "Neocrm",
        "Ness Digital engineering",
        "Net0",
        "NetApp",
        "NetWitness",
        "Netcore Cloud",
        "Netcore Unbxd",
        "Netlify​",
        "Netomi",
        "Netskope",
        "Netwitness",
        "Netwrix",
        "Newired",
        "Nexdigm",
        "Next Trail AI",
        "Nextgen Healthcare",
        "Nexthink",
        "Nexthink Adopt",
        "Nice",
        "Nintex",
        "Nividous",
        "Nokia",
        "North Highland (UMT360)",
        "Nosto",
        "Nous Infosystem",
        "Noventiq",
        "Nusummit",
        "OC tanners",
        "ONIX",
        "Odyssey",
        "Oliver Wyman",
        "Omada",
        "Omega Healthcare",
        "Omilia",
        "Omnipresent",
        "Omnissa",
        "Onapsis",
        "One Identity",
        "One Model",
        "One Network Enterprises",
        "OneBill Software",
        "OneReach.ai",
        "OneShield (Enterprise)",
        "OneShield (OMS)",
        "OneStream",
        "Onit",
        "Onward Technologies",
        "Onward technologies",
        "Oomnitza",
        "OpenIAM",
        "OpenIT",
        "Openbots",
        "Openforce",
        "Optimizely",
        "Optimove",
        "Optiv",
        "Optum",
        "Oracle (Fusion Cloud ERP)",
        "Oracle (NetSuite)",
        "Oracle (Netsuite)",
        "Oracle Cerner",
        "Oracle Guided Learning",
        "Orange",
        "Orange Business",
        "Orange Cyberdefence",
        "Orange cyberdefence",
        "Ordr",
        "Origami Risk",
        "Orion Innovation",
        "Oro",
        "Ottimate",
        "OutSystems",
        "OvalEdge",
        "Overhaul",
        "Oyster",
        "PASQAL",
        "PG Forsta",
        "PPD thermo fisher Scientific",
        "PROS",
        "PTC",
        "Panalyt",
        "Panorays",
        "Papaya Global",
        "Parakar Group",
        "Parallels",
        "Parascript",
        "Parashift",
        "Parexel",
        "Parsec",
        "Partner hero",
        "Pega",
        "Pegasystems",
        "Pendo",
        "Penn River",
        "Penstock Group",
        "Pentafon",
        "PeopleIX",
        "Perception Point",
        "Perficient",
        "Performant healthcare Inc",
        "Perkbox",
        "Persefoni",
        "Persona",
        "Phenom",
        "Phreesia",
        "Pigment",
        "Pimcore",
        "PingCAP",
        "Pisano",
        "Planforge",
        "Planful",
        "Planhat",
        "Planisware",
        "Planit",
        "Planview",
        "Platform.sh",
        "Plex, by Rockwell Automation",
        "Postman",
        "Praisidio",
        "Precisely",
        "Precision for Medicine",
        "Premier research",
        "Presidio",
        "Press Ganey",
        "Prevelant",
        "Prime Vigilance",
        "Priority Software",
        "Prismforce",
        "Pro pharma group",
        "ProSymmetry",
        "Probe group",
        "Productiv",
        "Profinda",
        "Proggio",
        "Progress",
        "Progressive Infotech",
        "Prohance",
        "ProjectManager.com",
        "Proofpoint",
        "Prophix",
        "Proxverse",
        "Pulsora",
        "Purple",
        "Pyramid Analytics",
        "Python RPA",
        "QAX",
        "QPR Software",
        "Quadient",
        "Qualitest",
        "Qualitrics",
        "Qualtrics",
        "Qualzeal",
        "Quantinuum",
        "Quantiphi",
        "Quest Global",
        "QuestionPro",
        "Quickbase",
        "Quinnox",
        "R Systems",
        "R1 RCM",
        "RIEDEL Networks",
        "ROOTCLOUD",
        "RRD Go creative",
        "RSA",
        "RSA Security",
        "Rackspace Technology",
        "RainFocus",
        "Raindrop",
        "Rakuten Symphony",
        "Ramp",
        "Randstad Digital",
        "RecVue",
        "Recorded Future",
        "Recurly",
        "Red Canary",
        "Red Hat",
        "Red Hat​",
        "Red River",
        "Redis",
        "Redpoint Global",
        "Redwood Software",
        "Reejig",
        "Refact.ai",
        "Reliaquest",
        "Remarkable Commerce",
        "Remarkable commerce",
        "Remofirst",
        "Remote",
        "Remundo",
        "Render",
        "Reply",
        "Resolve Systems",
        "Resolve tech solutions",
        "Retool",
        "Retrain.ai",
        "Reward Gateway",
        "Ricoh",
        "Rigetti Computing",
        "Ring Central",
        "RingCentral",
        "Rippling Horizons",
        "RiskRecon",
        "Riverbed",
        "Roboyo",
        "Rocket Software",
        "Rocketreach",
        "Rockwell Automation",
        "Rockwell Software",
        "Rokt (mparticle)",
        "RollWorks",
        "Rossum",
        "Routable",
        "Route Mobile",
        "Rubrik",
        "SAIO",
        "SAP (Business ByDesign)",
        "SAP (Emarsys)",
        "SAP (S/4HANA Cloud Public Edition)",
        "SAP (S/4HANA Cloud)",
        "SAP Enable Now",
        "SAP Signavio",
        "SCAYLE",
        "SD Worx",
        "SER Group",
        "SETS",
        "SLK Software",
        "SMA Technologies",
        "SMX",
        "SS&C",
        "SS&C Blue Prism",
        "SSI SCHAEFER",
        "SUSE",
        "Saama",
        "Safe Software",
        "Safeguard Global",
        "Sagility",
        "Salesforce (Heroku)",
        "Salesforce (MuleSoft)",
        "Salesforce (Mulesoft)",
        "Salesforce (Tableau)",
        "Salesforce Automation Edge",
        "Samsung",
        "Samsung SDS",
        "Sana Commerce",
        "Sangfor Technologies",
        "Sangoma",
        "Sapience Analytics",
        "Sapiens",
        "Sasken",
        "Saviynt",
        "Scayle",
        "Sciforma",
        "Scopeland",
        "Search Inc",
        "Searchspring",
        "Securitas",
        "Security scorecard",
        "Securonix",
        "Semos cloud",
        "Sensedia",
        "Sepasoft",
        "Sequeretek",
        "Serviceaide",
        "ShareFile (Citrix)",
        "Shearwater Health",
        "Shibumi",
        "Shippeo",
        "Shipsy",
        "Shipwell",
        "Shopify",
        "Shopware",
        "Shortways",
        "Sidetrade",
        "Siemens",
        "Sify Technologies",
        "Signant health",
        "Simply Get result",
        "Sinch",
        "SingleStore",
        "Singlestore",
        "Sirion",
        "Sirion",
        "Sisense",
        "Sitecore",
        "Skan",
        "Skuad",
        "Skygen",
        "Skyhigh Security",
        "Skyhive by cornerstone",
        "Skyword",
        "Slimstock",
        "SmartBear",
        "Smarte",
        "Smartsheet",
        "SnapLogic",
        "Snyk",
        "Socotra",
        "Socure",
        "Soffid",
        "SoftServe",
        "Softdel",
        "Softek",
        "Softeon",
        "Softserve",
        "Softtek",
        "Software AG",
        "SoftwareOne",
        "SolarWinds",
        "Solidatus",
        "Solo.io",
        "Sonata Software",
        "SonicWall",
        "Sophos",
        "Sourceday",
        "Sourcegraph",
        "Southworks",
        "Spekit",
        "Spendesk",
        "Sphera",
        "Splash",
        "Splash BI",
        "Spotfire",
        "Spotted Zebra",
        "Sprinklr",
        "Spryker",
        "Squiz",
        "Stampli",
        "StereoLOGIC",
        "Stonebranch",
        "Storyteq",
        "Stova",
        "Strata",
        "Stripe",
        "Sumo Logic",
        "Sumsub",
        "Sutherland",
        "Sweep",
        "SymphonyAI",
        "SymphonyAI Industrial",
        "Synechron",
        "Syneos Health",
        "Synergy Logistics",
        "Synopsys",
        "Systal",
        "Systems limited",
        "TCS BaNCS",
        "TDCX",
        "TELUS Digital",
        "TIBCO",
        "TIBCO Software",
        "TMJ",
        "TP-Link",
        "TSP - The Silicon Patners",
        "TTEC",
        "Tabnine",
        "Tacton",
        "Talent desk",
        "Talentguard",
        "Talkdesk",
        "Tanium",
        "Tanla",
        "Taskus",
        "Tata Communications",
        "Tata Elxsi",
        "Tealium",
        "Tebra",
        "Tech rules",
        "Techmobius",
        "Techwolf",
        "Tecnotree",
        "Tecsys",
        "Telefonica",
        "Telefónica",
        "Telenor Group",
        "Teleperformance",
        "Telit Cinterion",
        "Tellius",
        "Telstra",
        "Telus Digital",
        "Tencent Cloud",
        "Tencent cloud",
        "Terminus",
        "Terryberry",
        "Tessolve",
        "TestingXperts",
        "Thales",
        "The SSI Group",
        "Thoughtsphere",
        "Thoughtworks",
        "TibcoSoftware",
        "Tietovery",
        "Tigermed",
        "To the New",
        "ToolsGroup",
        "Toonimo",
        "Torii",
        "Totango",
        "TrackVia",
        "Transcosmos",
        "Transformify",
        "Tray.io",
        "Treasure Data",
        "Trelica",
        "Trend Micro",
        "Trigent",
        "Trio Mobil",
        "Trucker Tools",
        "Truefort",
        "Trustarc",
        "Truyo",
        "Tuebora",
        "Tulip",
        "Tulip",
        "Tungsten Automation",
        "Twilio",
        "Twilio Segment",
        "Tyk",
        "UBC",
        "UPS Supply Chain Solutions",
        "USU",
        "Uber Freight",
        "Ubisense",
        "UiPath",
        "Uipath",
        "Uniform",
        "Uniphore(Action IQ)",
        "Unitrends",
        "Univers",
        "UpFlux",
        "UpGuard",
        "Upland",
        "Uppwise",
        "Upwork Enterprise",
        "Userlane",
        "VMware Carbon Black",
        "VTEX",
        "VVDN technologies",
        "Vaco",
        "Vantage Circle",
        "Veeam",
        "Veeva",
        "Vega HR",
        "Velaris",
        "Velocity Global",
        "Velocity procurement",
        "Vemo workforce",
        "Vena",
        "Vendavo",
        "Venustech",
        "Veracode",
        "Veradigm",
        "Vercel​",
        "VeriFone",
        "Verifone",
        "Verinite Technologies",
        "Verint",
        "Verisk",
        "Veritas",
        "Verizon",
        "Versa Networks",
        "Versapay",
        "Vic.ai",
        "Vinculum",
        "Virto Commerce",
        "Virtusa",
        "Visier",
        "Visionet",
        "Visionnet Systems INC",
        "Visonnet",
        "Vitally",
        "Vodafone",
        "Vonage",
        "Vtex",
        "Vtiger",
        "WALLIX",
        "WNS",
        "WNS Global Services",
        "WNS Procurement",
        "WSO2",
        "WalkMe",
        "WalkMe",
        "WatchGuard",
        "Waterlabs ai",
        "Watershed",
        "Waystar",
        "Webpurify",
        "Wellspring (Sopheon)",
        "Whale Cloud Technology",
        "Whatfix",
        "Wildix",
        "Wiliot",
        "Wireless Logic",
        "WithSecure",
        "Wizeline",
        "Wolters Kluwer",
        "WorkOtter",
        "Workato",
        "Workera",
        "Workfusion",
        "Workhuman",
        "Workhuman Achievers",
        "Workiva",
        "Workmarket",
        "Worksome",
        "Workspot",
        "Workstars",
        "Worksuite",
        "Worktango",
        "Worldwide clinical trials",
        "Wrike",
        "Writer business services",
        "XCMG HANYUN",
        "Xanadau",
        "Xebia",
        "Xoriant",
        "Xoxoday",
        "Yagna iQ",
        "Yash technologies",
        "Yellow.ai",
        "Yext",
        "Yonyou",
        "Yooz",
        "Yunojuno",
        "Yusen Logistics",
        "ZOLOZ",
        "ZTE",
        "Zebra Technologies",
        "Zelis",
        "Zelta",
        "Zensar technologies",
        "Zeotap",
        "Zero Networks",
        "ZeroedIn",
        "Zeta Global",
        "ZetaGlobal",
        "Zilla Security",
        "Zilliant",
        "Zinnia",
        "Ziplyne",
        "Zluri",
        "Zones",
        "Zoom",
        "ZoomInfo",
        "Zoominfo",
        "Zoovu",
        "Zudy",
        "Zuora",
        "Zylo",
        "Zyter",
        "akaBot",
        "anch AI",
        "b.well",
        "commercetools",
        "data world",
        "eClinical Solutions",
        "eGain",
        "eInfochips",
        "eSentire",
        "ebidtopay",
        "erwin by Quest",
        "fabric",
        "iBASE-t",
        "iCIMS",
        "iMagnum Healthcare",
        "iMocha",
        "iboss",
        "insight software",
        "ip-label",
        "isolved",
        "keylight",
        "mParticle",
        "mPulse",
        "mindzie",
        "monday.com",
        "myMeta Software",
        "myMeta Software",
        "oneclick",
        "project44",
        "qbotica",
        "servicePath",
        "tts GmbH",
        "vFairs",
        "watercooler",
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
        "Microsoft Consulting Services",
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
    "MANAGED_SERVICES_INDIA": [
        "CMS Info Systems", "CMS Infosystems",
        "Newgen Software", "Nucleus Software",
        "Intellect Design Arena", "Intellect",
        "FSS Technologies", "Financial Software and Systems",
        "AGS Transact Technologies", "AGS Transact",
        "Hitachi Payment Services",
        "NCR Corporation", "NCR Atleos",
        "Diebold Nixdorf",
        "FIS Global", "FIS",
        "Fiserv",
        "Temenos",
        "Finastra",
        "Mambu",
        "i-exceed Technology",
        "Subex",
        "Aurionpro Solutions",
        "SBI Cards", "In-Solutions Global",
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
    """Returns list of (name, category, char_position) sorted by match length desc.
    Uses word-boundary matching to avoid substrings (e.g. 'MCS' inside 'CMS').
    """
    matches = []
    for category, names in master.items():
        for name in names:
            # Word-boundary regex: \b on both sides if name starts/ends with word char
            pat = r'(?<![A-Za-z0-9])' + re.escape(name) + r'(?![A-Za-z0-9])'
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                matches.append((name, category, m.start()))
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
    """Match against VENDOR_MASTER only — no NER fallback (NER invents phantom vendors)."""
    matches = _find_matches_in_text(text, VENDOR_MASTER)
    if matches:
        name, category, _ = matches[0]
        return {"vendor": normalize_name(name), "vendor_category": category}
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


def extract_deal_description(text: str, company_names: list[str], vendor: str) -> str:
    """Extract 1-3 sentences from the article that best describe the deal.
    Looks for sentences containing both the company and the vendor/deal keywords,
    then returns the most informative window — no fabrication, only source text.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    deal_keywords = [
        "selects", "selected", "signs", "signed", "contract", "agreement",
        "implements", "deploys", "partners", "partnership", "outsourc",
        "go-live", "migration", "awarded", "chooses", "adopts", "deal",
        "managed service", "transformation",
    ]
    company_lower = [c.lower() for c in company_names]
    vendor_lower = vendor.lower() if vendor else ""

    scored = []
    for i, sent in enumerate(sentences):
        sl = sent.lower()
        score = 0
        if any(c in sl for c in company_lower):
            score += 2
        if vendor_lower and vendor_lower in sl:
            score += 2
        if any(k in sl for k in deal_keywords):
            score += 1
        if score >= 3:
            scored.append((score, i, sent))

    if scored:
        scored.sort(key=lambda x: -x[0])
        best_idx = scored[0][1]
        # Return best sentence + next sentence for context
        window_sents = sentences[best_idx: best_idx + 2]
        return " ".join(window_sents)[:400].strip()

    # Fallback: first 400 chars around first company mention
    tl = text.lower()
    for cn in company_lower:
        idx = tl.find(cn)
        if idx != -1:
            return text[max(0, idx - 50): idx + 350].strip()[:400]
    return ""


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


# ── Phrases that disqualify a page as a deal article ─────────────────────────
# These indicate analyst reports, award lists, job posts, stock news — not deals.
NON_DEAL_DISQUALIFIERS = [
    "magic quadrant", "gartner peer insights", "forrester wave",
    "idc marketscape", "named a leader", "named a visionary",
    "named a challenger", "positioned in the", "recognition award",
    "best place to work", "employer of the year", "ranked #", "ranked no.",
    "analyst report", "market report", "market research", "market size",
    "press release issued by", "stock price", "share price", "quarterly results",
    "earnings call", "q1 results", "q2 results", "q3 results", "q4 results",
    "job opening", "we are hiring", "careers page", "apply now",
    "ceo interview", "cto interview", "opinion:", "commentary:",
]

# ── Phrases that MUST appear near company for a true deal ─────────────────────
DEAL_ACTION_PHRASES = [
    # Contracts / awards
    "signed a contract", "signs a contract", "awarded a contract",
    "contract awarded", "contract signed", "multi-year contract",
    "outsourcing contract", "managed services contract",
    "outsourcing agreement", "outsourcing deal",
    "managed services agreement", "service level agreement",
    # Selection / adoption
    "selects ", "selected ", "has selected", "chooses ", "chosen ",
    "adopts ", "adopted ", "has adopted", "standardises on", "standardizes on",
    # Implementation / go-live
    "goes live", "go-live", "went live", "has gone live",
    "rolled out", "successfully deployed", "implementation complete",
    "deployment of", "migrated to", "migration to",
    # Partnership (only specific/bilateral)
    "partners with", "has partnered with", "entered into a partnership",
    "strategic partnership with", "strategic alliance with",
    "signed a memorandum", "signed an mou",
    # Outsourcing
    "outsourced to", "outsourcing to", "handed over to",
    "managed by ", "managed services provided by",
    # Procurement
    "rfp awarded", "tender awarded", "bid awarded", "bid won",
    "purchase order", "framework agreement signed",
]


def is_deal_relevant(text: str, company_names: list[str], focus_deal_types: list[str]) -> bool:
    """
    Strict two-step check:
    1. Reject pages that are awards / analyst reports / stock news / job posts.
    2. Require an explicit deal-action phrase within 800 chars of the company mention.
       Generic terms (digital transformation, cloud adoption) no longer qualify.
    """
    text_lower = text.lower()

    # Step 1 — hard reject non-deal page types
    if any(d in text_lower for d in NON_DEAL_DISQUALIFIERS):
        return False

    # Step 2 — company must appear in text
    company_pos = -1
    for cn in company_names:
        idx = text_lower.find(cn.lower())
        if idx != -1 and (company_pos == -1 or idx < company_pos):
            company_pos = idx
    if company_pos == -1:
        return False

    # Step 3 — a deal-action phrase must appear within 800 chars of the company
    window = text_lower[max(0, company_pos - 800): company_pos + 800]
    return any(phrase in window for phrase in DEAL_ACTION_PHRASES)


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

    # Detect vendor-centric mode: company being searched IS itself a vendor
    # e.g. searching for "AWS" → articles say "Company X selects AWS"
    company_is_vendor = any(
        cn.lower() in {
            "aws", "amazon web services", "microsoft azure", "azure", "google cloud",
            "sap", "oracle", "salesforce", "servicenow", "workday", "ibm",
            "accenture", "infosys", "tcs", "wipro", "capgemini", "cognizant",
            "deloitte", "pwc", "kpmg", "ey", "hcltech", "dxc", "palo alto networks",
            "crowdstrike", "fortinet", "zscaler", "splunk", "snowflake", "databricks",
        }
        for cn in company_names
    )

    if company_is_vendor:
        # In vendor-centric mode, the company IS the vendor — find the customer
        # Look for org names near the company mention that aren't the company itself
        vendor = company_name   # company is the vendor/product
        cat = vendor_info.get("vendor_category", "CLOUD")  # use detected cat or default

        # Try to find the customer company via NER or context
        customer = ""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text[:3000])
            orgs = [e.text for e in doc.ents if e.label_ == "ORG"
                    and e.text.lower() not in {cn.lower() for cn in company_names}]
            if orgs:
                customer = orgs[0]
        except Exception:
            pass

        # Rewrite company_name to the customer if found, keep vendor as searched company
        if customer:
            company_name = customer
    else:
        # Reject if vendor found but not co-located with company
        if vendor and not _vendor_near_company(text, company_names, vendor):
            vendor_info = {"vendor": "", "vendor_category": "OTHER"}
            vendor = ""
        cat = vendor_info.get("vendor_category", "")

    si = extract_si_partner(text)
    value = extract_deal_value(text)
    duration = extract_deal_duration(text)
    date = extract_announcement_date(text, url, soup)
    scope = extract_scope_of_service(text, vendor, cat)
    description = extract_deal_description(text, company_names, vendor)

    # Require a known vendor from master list — no vendor = no deal record
    # (description + date alone are insufficient; they appear in news without a deal)
    has_evidence = bool(vendor)
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
        # ── Core fields (always populated where possible) ──────────────────────
        "company_name":     company_name,           # Customer name
        "vendor":           vendor,                 # Vendor name
        "deal_description": description,            # Extracted sentences from article
        "announcement_date": date or "",            # Date of the article / announcement

        # ── Secondary fields (filled when available) ───────────────────────────
        "deal_title":       title,
        "record_type":      record_type,            # contract | partnership | implementation | vendor_selection | initiative | technology_announcement
        "vendor_category":  cat,
        "si_partner":       si,
        "scope_of_service": scope,
        "deal_value_usd":   value,
        "deal_duration":    duration,
        "source_url":       url,
        "all_source_urls":  [url],
        "source_type":      source_type,
        "confidence_level": confidence,
        "summary":          summary,
    }
