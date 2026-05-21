"""Hash-based deduplication and merge logic."""

import hashlib
from typing import Any


CONFIDENCE_SCORE = {"High": 3, "Medium": 2, "Low": 1}


def _deal_hash(deal: dict[str, Any]) -> str:
    scope = (deal.get("scope_of_service") or "")[:50]
    key = "|".join([
        deal.get("company_name", "").lower(),
        deal.get("vendor", "").lower(),
        deal.get("announcement_date", "")[:7],  # year-month level
        scope.lower(),
    ])
    return hashlib.sha256(key.encode()).hexdigest()


def _count_populated(deal: dict[str, Any]) -> int:
    return sum(1 for v in deal.values() if v not in (None, "", [], 0, 0.0))


def _merge_deals(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    score_a = CONFIDENCE_SCORE.get(a.get("confidence_level", "Low"), 1)
    score_b = CONFIDENCE_SCORE.get(b.get("confidence_level", "Low"), 1)

    # Keep whichever has more populated fields; break ties by confidence
    if _count_populated(b) > _count_populated(a) or score_b > score_a:
        primary, secondary = b, a
    else:
        primary, secondary = a, b

    merged = dict(primary)

    # Merge source URLs
    urls_a = a.get("all_source_urls") or [a.get("source_url", "")]
    urls_b = b.get("all_source_urls") or [b.get("source_url", "")]
    combined = list(dict.fromkeys(urls_a + urls_b))
    merged["all_source_urls"] = combined

    # Fill missing fields from secondary
    for field in ["deal_value_usd", "deal_duration", "announcement_date",
                  "si_partner", "scope_of_service"]:
        if not merged.get(field) and secondary.get(field):
            merged[field] = secondary[field]

    return merged


def deduplicate(deals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for deal in deals:
        h = _deal_hash(deal)
        if h in seen:
            seen[h] = _merge_deals(seen[h], deal)
        else:
            seen[h] = deal
    return list(seen.values())


def _recency_weight(date_str: str) -> float:
    if not date_str:
        return 0.3
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        months_ago = (now - dt).days / 30
        if months_ago <= 12:
            return 1.0
        elif months_ago <= 24:
            return 0.85
        elif months_ago <= 36:
            return 0.70
        else:
            return 0.50
    except ValueError:
        return 0.3


def sort_by_recency_confidence(deals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(d: dict) -> float:
        w = _recency_weight(d.get("announcement_date", ""))
        c = CONFIDENCE_SCORE.get(d.get("confidence_level", "Low"), 1)
        return w * c

    return sorted(deals, key=sort_key, reverse=True)
