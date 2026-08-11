"""
Shared Gemini API usage logger — used by every pipeline module.

Extracts real token counts from each Gemini response's `usage_metadata` (not
estimated from prompt length) and logs a structured line + keeps an in-memory
ring buffer so `/api/usage` can report measured, not modeled, cost per report.
"""

import logging
import time
from collections import deque

logger = logging.getLogger("usage")

# In-memory ring buffer — resets on deploy/restart, which is fine: this is a
# rolling window for spot-checking recent activity, not a billing system of
# record. Swap for a DB/file sink later if durable history is needed.
_RECENT: deque = deque(maxlen=5000)

# Gemini 2.5 Flash pricing (Aug 2026) — see cost ledger for sourcing. Used only
# to attach a derived cost estimate to each logged call; the token counts
# themselves are real, read directly off the API response.
_IN_RATE = 0.30 / 1_000_000
_OUT_RATE = 2.50 / 1_000_000
_GROUND_RATE = 35 / 1000  # per grounded request, beyond the free daily allotment


def log_gemini_usage(module: str, label: str, response, grounded: bool = True,
                      model: str = "gemini-2.5-flash") -> dict | None:
    """Extract real usage_metadata from a Gemini response and record it.

    Call this right after every `client.models.generate_content(...)` call,
    across every pipeline. Never raises — a logging failure must never break
    the pipeline that's actually serving the request.
    """
    try:
        um = getattr(response, "usage_metadata", None)
        if not um:
            return None
        in_tok = getattr(um, "prompt_token_count", None) or 0
        out_tok = getattr(um, "candidates_token_count", None) or 0
        total_tok = getattr(um, "total_token_count", None) or (in_tok + out_tok)
        cost = in_tok * _IN_RATE + out_tok * _OUT_RATE + (_GROUND_RATE if grounded else 0)

        entry = {
            "ts": time.time(),
            "module": module,
            "label": label,
            "model": model,
            "grounded": grounded,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": total_tok,
            "cost_usd": round(cost, 6),
        }
        _RECENT.append(entry)
        logger.info(
            f"[usage] {module}/{label} model={model} grounded={grounded} "
            f"in={in_tok} out={out_tok} total={total_tok} cost=${cost:.4f}"
        )
        return entry
    except Exception as e:
        logger.warning(f"[usage] extraction failed for {module}/{label}: {e}")
        return None


def get_recent_usage(limit: int = 200) -> list[dict]:
    """Most recent N logged calls, newest last."""
    return list(_RECENT)[-limit:]


def get_usage_summary() -> dict:
    """Aggregate the in-memory buffer by module — calls, grounded calls, tokens, cost."""
    entries = list(_RECENT)
    if not entries:
        return {"count": 0, "by_module": {}}

    by_module: dict[str, dict] = {}
    total_cost = 0.0
    for e in entries:
        m = by_module.setdefault(e["module"], {
            "calls": 0, "grounded_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
        })
        m["calls"] += 1
        m["grounded_calls"] += 1 if e["grounded"] else 0
        m["input_tokens"] += e["input_tokens"]
        m["output_tokens"] += e["output_tokens"]
        m["cost_usd"] += e["cost_usd"]
        total_cost += e["cost_usd"]

    for m in by_module.values():
        m["cost_usd"] = round(m["cost_usd"], 4)

    return {
        "count": len(entries),
        "total_cost_usd": round(total_cost, 4),
        "window_start": entries[0]["ts"],
        "window_end": entries[-1]["ts"],
        "by_module": by_module,
    }
