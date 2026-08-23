"""
Server-side report ledger — records that a report was generated, by which
module, and what it cost, regardless of which client triggered it (the app's
own frontend, or a direct API call/curl/automation/another client).

Stores the FULL result data alongside metadata, so any report — including
ones a browser never streamed itself — can be opened for full detail from
`GET /api/reports/{run_id}`, not just listed with a cost figure.
`list_reports()` deliberately excludes the data blob to keep the summary
listing cheap; `get_report()` is the only call that pays for it.

Shares the same SQLite file as usage_logger.py — same WAL/multi-worker
rationale applies, and storage is ephemeral (Render's local disk doesn't
survive a redeploy), so this is a rolling window, not a permanent record.
"""

import json
import logging
import os
import sqlite3
import time

logger = logging.getLogger("reports")

_DB_PATH = os.path.join(os.getenv("USAGE_DB_DIR", "/tmp"), "gemini_usage.db")


def _retrying(fn, *, attempts: int = 8, base_delay: float = 0.05):
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
    conn = sqlite3.connect(_DB_PATH, timeout=10)

    def _setup():
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                run_id TEXT PRIMARY KEY,
                ts REAL NOT NULL,
                module TEXT NOT NULL,
                target TEXT,
                summary TEXT,
                usage_json TEXT,
                data_json TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_module ON reports(module)")
        conn.commit()

    _retrying(_setup)
    return conn


def save_report(run_id: str, module: str, target: str, summary: str,
                 usage: dict | None, data: dict | None = None) -> None:
    """Record one completed report, including its full result data so it can
    later be opened for detail regardless of which client generated it.
    Never raises — a ledger write failure must never break the response the
    caller is waiting on."""
    if not run_id:
        return
    try:
        conn = _get_conn()
        try:
            def _insert():
                conn.execute(
                    "INSERT OR REPLACE INTO reports "
                    "(run_id, ts, module, target, summary, usage_json, data_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (run_id, time.time(), module, target or "", summary or "",
                     json.dumps(usage or {}), json.dumps(data if data is not None else {})),
                )
                conn.commit()
            _retrying(_insert)
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"save_report failed for {module}/{run_id}: {e}")


def list_reports(module: str = "", limit: int = 100) -> list[dict]:
    """Summary listing only — no data_json, keeps this cheap for the History panel."""
    conn = _get_conn()
    try:
        if module:
            rows = conn.execute(
                "SELECT run_id, ts, module, target, summary, usage_json FROM reports "
                "WHERE module = ? ORDER BY ts DESC LIMIT ?",
                (module, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT run_id, ts, module, target, summary, usage_json FROM reports "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
    finally:
        conn.close()

    out = []
    for run_id, ts, mod, target, summary, usage_json in rows:
        try:
            usage = json.loads(usage_json) if usage_json else {}
        except Exception:
            usage = {}
        out.append({"run_id": run_id, "ts": ts, "module": mod, "target": target,
                     "summary": summary, "usage": usage})
    return out


def get_report(run_id: str) -> dict | None:
    """Full record including result data — for opening an entry's detail view."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT run_id, ts, module, target, summary, usage_json, data_json "
            "FROM reports WHERE run_id = ?",
            (run_id,),
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    run_id, ts, mod, target, summary, usage_json, data_json = row
    try:
        usage = json.loads(usage_json) if usage_json else {}
    except Exception:
        usage = {}
    try:
        data = json.loads(data_json) if data_json else {}
    except Exception:
        data = {}
    return {"run_id": run_id, "ts": ts, "module": mod, "target": target,
            "summary": summary, "usage": usage, "data": data}


def delete_report(run_id: str) -> None:
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM reports WHERE run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()


def clear_reports(module: str = "") -> None:
    conn = _get_conn()
    try:
        if module:
            conn.execute("DELETE FROM reports WHERE module = ?", (module,))
        else:
            conn.execute("DELETE FROM reports")
        conn.commit()
    finally:
        conn.close()
