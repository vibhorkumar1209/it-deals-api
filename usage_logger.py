"""
Shared Gemini API usage logger — used by every pipeline module.

Extracts real token counts from each Gemini response's `usage_metadata` (not
estimated from prompt length) and records it, so `/api/usage` can report
measured, not modeled, cost per report.

Backed by SQLite rather than an in-memory buffer: the app runs as 4 separate
Uvicorn worker processes (see render.yaml `--workers 4`), each with its own
Python interpreter — an in-memory deque would silently fragment usage data
across 4 invisible-to-each-other buffers, which is why an earlier version of
this file under-reported real usage. SQLite's file is shared by all workers
in the same container, and WAL mode lets 4 processes write concurrently
without corrupting each other's writes.

Storage is still ephemeral — Render's local disk doesn't survive a redeploy —
so this remains a rolling window, not a permanent billing record.
"""

import logging
import os
import sqlite3
import time

logger = logging.getLogger("usage")

_DB_PATH = os.path.join(os.getenv("USAGE_DB_DIR", "/tmp"), "gemini_usage.db")

# Gemini 2.5 Flash pricing (Aug 2026) — see cost ledger for sourcing. Used only
# to attach a derived cost estimate to each logged call; the token counts
# themselves are real, read directly off the API response.
_IN_RATE = 0.30 / 1_000_000
_OUT_RATE = 2.50 / 1_000_000
_GROUND_RATE = 35 / 1000  # per grounded request, beyond the free daily allotment


def _retrying(fn, *, attempts: int = 8, base_delay: float = 0.05):
    """Run fn() with retries on 'database is locked' — SQLite raises this even
    with busy_timeout set when several processes hit the very first write
    (WAL-mode initialization or table creation) at the exact same instant on
    cold start, e.g. 4 Uvicorn workers all handling their first Gemini call
    right after a deploy. busy_timeout covers lock waits during a single
    statement; it does NOT cover the gap between our own retry attempts, so
    we still need this wrapper on top."""
    last_err = None
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last_err = e
            if "locked" not in str(e).lower():
                raise
            time.sleep(base_delay * (attempt + 1))
    raise last_err


def _get_conn() -> sqlite3.Connection:
    """New connection per call — cheap for SQLite, avoids sharing a connection
    object across threads/processes in ways sqlite3 doesn't like."""
    conn = sqlite3.connect(_DB_PATH, timeout=10)

    def _setup():
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                module TEXT NOT NULL,
                label TEXT,
                model TEXT,
                grounded INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL
            )
        """)
        conn.commit()

    _retrying(_setup)
    return conn


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

        conn = _get_conn()
        try:
            def _insert():
                conn.execute(
                    "INSERT INTO usage (ts, module, label, model, grounded, input_tokens, "
                    "output_tokens, total_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (entry["ts"], module, label, model, int(grounded), in_tok, out_tok,
                     total_tok, entry["cost_usd"]),
                )
                conn.commit()
            _retrying(_insert)
        finally:
            conn.close()

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
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT ts, module, label, model, grounded, input_tokens, output_tokens, "
            "total_tokens, cost_usd FROM usage ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()

    cols = ["ts", "module", "label", "model", "grounded", "input_tokens",
            "output_tokens", "total_tokens", "cost_usd"]
    entries = [dict(zip(cols, row)) for row in rows]
    for e in entries:
        e["grounded"] = bool(e["grounded"])
    entries.reverse()  # oldest-first within the returned window, newest last
    return entries


def get_usage_summary() -> dict:
    """Aggregate all logged usage by module — calls, grounded calls, tokens, cost."""
    conn = _get_conn()
    try:
        total_row = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), MIN(ts), MAX(ts) FROM usage"
        ).fetchone()
        count, total_cost, window_start, window_end = total_row

        rows = conn.execute("""
            SELECT module,
                   COUNT(*) AS calls,
                   SUM(grounded) AS grounded_calls,
                   COALESCE(SUM(input_tokens),0) AS input_tokens,
                   COALESCE(SUM(output_tokens),0) AS output_tokens,
                   COALESCE(SUM(cost_usd),0) AS cost_usd
            FROM usage
            GROUP BY module
            ORDER BY cost_usd DESC
        """).fetchall()
    finally:
        conn.close()

    if not count:
        return {"count": 0, "by_module": {}}

    by_module = {
        module: {
            "calls": calls,
            "grounded_calls": grounded_calls or 0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost_usd, 4),
        }
        for module, calls, grounded_calls, input_tokens, output_tokens, cost_usd in rows
    }

    return {
        "count": count,
        "total_cost_usd": round(total_cost, 4),
        "window_start": window_start,
        "window_end": window_end,
        "by_module": by_module,
    }
