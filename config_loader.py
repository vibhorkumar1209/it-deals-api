"""Input config validator and loader."""

import json
import os
from dataclasses import dataclass, field
from typing import Any


REQUIRED_FIELDS = [
    "company_name", "domain", "linkedin_url",
    "search_year_range", "known_sources", "focus_deal_types", "output_dir"
]


@dataclass
class ScraperConfig:
    company_name: str
    company_aliases: list[str]
    domain: str
    secondary_domains: list[str]
    linkedin_url: str
    stock_ticker: str | None
    exchange: str | None
    industry_sector: str
    hq_country: str
    hq_city: str
    search_year_range: dict[str, int]
    known_sources: list[str]
    known_deals_to_skip: list[str]
    focus_deal_types: list[str]
    min_deal_value_usd_million: float | None
    output_dir: str
    run_linkedin: bool
    run_pdf_extraction: bool
    use_proxies: bool
    proxy_pool: list[str]
    notify_email: str | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def all_company_names(self) -> list[str]:
        names = [self.company_name] + self.company_aliases
        if self.stock_ticker:
            names.append(self.stock_ticker)
        return list(dict.fromkeys(names))

    @property
    def all_domains(self) -> list[str]:
        return [self.domain] + self.secondary_domains


def load_config(path: str) -> ScraperConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    yr = data["search_year_range"]
    if not isinstance(yr, dict) or "start" not in yr or "end" not in yr:
        raise ValueError("search_year_range must have 'start' and 'end' keys")
    if yr["start"] > yr["end"]:
        raise ValueError("search_year_range start must be <= end")

    os.makedirs(data["output_dir"], exist_ok=True)

    return ScraperConfig(
        company_name=data["company_name"],
        company_aliases=data.get("company_aliases", []),
        domain=data["domain"],
        secondary_domains=data.get("secondary_domains", []),
        linkedin_url=data["linkedin_url"],
        stock_ticker=data.get("stock_ticker"),
        exchange=data.get("exchange"),
        industry_sector=data.get("industry_sector", ""),
        hq_country=data.get("hq_country", ""),
        hq_city=data.get("hq_city", ""),
        search_year_range=yr,
        known_sources=data["known_sources"],
        known_deals_to_skip=data.get("known_deals_to_skip", []),
        focus_deal_types=data["focus_deal_types"],
        min_deal_value_usd_million=data.get("min_deal_value_usd_million"),
        output_dir=data["output_dir"],
        run_linkedin=data.get("run_linkedin", True),
        run_pdf_extraction=data.get("run_pdf_extraction", True),
        use_proxies=data.get("use_proxies", False),
        proxy_pool=data.get("proxy_pool", []),
        notify_email=data.get("notify_email"),
        raw=data,
    )
