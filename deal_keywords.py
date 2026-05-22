"""
Deal domain keywords extracted from Frameworks and Company List_14052025.xlsx — Vendor List sheet.
These are used to enrich search queries for any company scan.
NOT tied to any specific vendor.
"""

# Business process areas — used to find deals by domain
PROCESS_KEYWORDS: list[str] = ['ATM Management', 'Account Payable', 'Account Receivable', 'Benefits Administration', 'Billing Services', 'Budgeting', 'Business Process Management', 'Card Processing', 'Case Management', 'Claims', 'Contact Center', 'Contract Management', 'Core Banking', 'Core HR', 'Customer Experience (CX)', 'Customer Relationship Management', 'Customer Service', 'Digital Marketing', 'Electronic Health Record', 'Employee Self-Service', 'Enterprise Asset Management', 'Finance & Accounting', 'Fleet Scheduling/Planning', 'Fund Administration', 'General Ledger', 'Human Resource Management', 'Inventory Management', 'Learning', 'Loans (Banking)', 'Manufacturing', 'Mortgage Processing', 'Online banking', 'Order Management', 'Order-to-Cash', 'Payment Processing', 'Payroll', 'Point of Sale', 'Policy Administration', 'Procure to Pay', 'Procurement', 'Product Life-Cycle Management', 'Project Management', 'Record to Report', 'Recruitment', 'Risk Management', 'Sales and Marketing', 'Sourcing', 'Supply Chain Planning', 'Talent Management', 'Tax Management', 'Transportation Management Solution', 'Underwriting', 'Workforce Management', 'e-commerce']

# Technology categories — used to find deals by technology type  
TECHNOLOGY_KEYWORDS: list[str] = ['Anti-Fraud', 'Anti-Money Laundering', 'Application Development', 'Application Performance Management', 'Artificial Intelligence', 'Automation', 'Big Data', 'Biometrics', 'Blockchain', 'Business Analytics', 'Business Intelligence', 'Chatbot', 'Cloud (Hybrid)', 'Cloud (Private)', 'Cloud (Public)', 'Compliance', 'Content Management', 'Cybersecurity', 'Data Center Migration', 'Data Integration', 'Data Warehousing', 'Digital Transformation', 'Disaster Recovery', 'Document Management', 'ERP', 'Endpoint Security', 'Fraud Detection', 'IT Asset Management', 'IT Service Management', 'IoT', 'Learning Management System', 'Managed Data Service', 'Managed Security', 'Mobile Application Development', 'Mobile Payments', 'Multi-cloud', 'Natural Language Processing', 'Network Security', 'Network Upgrade', 'Predictive Analytics', 'RPA', 'SaaS', 'Unified Communications', 'Virtualization']

# Specific products mentioned in the keyword data
PRODUCT_KEYWORDS: list[str] = ['AWS', 'Azure', 'Dynamics 365', 'Google Apps', 'Guidewire', 'IBM Watson', 'Infor', 'NetSuite', 'Office 365', 'Oracle', 'Oracle Exadata', 'PeopleSoft', 'SAP', 'SAP HANA', 'Salesforce.com', 'ServiceNow', 'Sharepoint', 'SuccessFactor', 'Workday']

# Combined flat list for general search enrichment
ALL_DEAL_KEYWORDS: list[str] = list(dict.fromkeys(
    PROCESS_KEYWORDS + TECHNOLOGY_KEYWORDS + PRODUCT_KEYWORDS
))
