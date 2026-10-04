"""Intent Map — account scoring & dynamic stakeholder mapping.

Per account:
  cache    SQLite, reuse runs < 90 days old (research cached separately from
           scoring, so changing weights/buyers re-runs only Claude).
  Gemini   2.5 Flash + Google Search grounding → raw web signals + evidence.
  Claude   Sonnet 5 → matches signals to the user's weighted terms, infers
           department themes from signal text, routes buyers to them.
  score    Deterministic, in Python: each term counts once at its freshest
           evidence, weight × 0.5^(age_days / half_life); score = found / max × 100.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

from usage_logger import get_usage_by_run, log_claude_usage, log_gemini_usage, new_run_id

logger = logging.getLogger("intent_map")

GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-5"
MODULE = "intent_map"
CACHE_DAYS = 90
MAX_CONCURRENT = 4
ACCOUNT_TIMEOUT = 600
UNDATED_DECAY = 0.5
_DB_PATH = os.path.join(os.getenv("USAGE_DB_DIR", "/tmp"), "intent_map_cache.db")


# ── Cache ────────────────────────────────────────────────────────────────────
class ResearchStorageLayer:
    def __init__(self, path: str = _DB_PATH):
        self.path = path
        with self._conn() as c:
            c.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS raw_research (
                domain TEXT, terms_hash TEXT, created_at TEXT, payload TEXT,
                PRIMARY KEY (domain, terms_hash));
            CREATE TABLE IF NOT EXISTS processed_companies (
                domain TEXT, config_hash TEXT, company_name TEXT, created_at TEXT,
                composite_score REAL, result TEXT,
                PRIMARY KEY (domain, config_hash));
            CREATE TABLE IF NOT EXISTS enriched_executives (
                domain TEXT, config_hash TEXT, buyer_key TEXT, full_name TEXT,
                job_title TEXT, linkedin_url TEXT, department TEXT, fit TEXT,
                rationale TEXT, created_at TEXT,
                PRIMARY KEY (domain, config_hash, buyer_key));
            """)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA busy_timeout=10000")
        return c

    @staticmethod
    def _fresh(created_at: str) -> bool:
        return datetime.fromisoformat(created_at) > datetime.now(timezone.utc) - timedelta(days=CACHE_DAYS)

    def get_raw(self, domain, terms_hash):
        with self._conn() as c:
            row = c.execute("SELECT created_at, payload FROM raw_research WHERE domain=? AND terms_hash=?",
                            (domain, terms_hash)).fetchone()
        return json.loads(row[1]) if row and self._fresh(row[0]) else None

    def put_raw(self, domain, terms_hash, payload):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO raw_research VALUES (?,?,?,?)",
                      (domain, terms_hash, _now(), json.dumps(payload)))

    def get_processed(self, domain, config_hash):
        with self._conn() as c:
            row = c.execute("SELECT created_at, result FROM processed_companies WHERE domain=? AND config_hash=?",
                            (domain, config_hash)).fetchone()
            if not row or not self._fresh(row[0]):
                return None
            execs = c.execute("SELECT buyer_key, department, fit, rationale FROM enriched_executives "
                              "WHERE domain=? AND config_hash=?", (domain, config_hash)).fetchall()
        result = json.loads(row[1])
        by_key = {k: (d, f, r) for k, d, f, r in execs}
        for s in result["stakeholders"]:
            if s["buyer_key"] in by_key:
                s["department"], s["fit"], s["rationale"] = by_key[s["buyer_key"]]
        result["cached_at"] = row[0]
        return result

    def put_processed(self, domain, config_hash, result):
        now = _now()
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO processed_companies VALUES (?,?,?,?,?,?)",
                      (domain, config_hash, result["company_name"], now,
                       result["composite_score"], json.dumps(result)))
            c.executemany("INSERT OR REPLACE INTO enriched_executives VALUES (?,?,?,?,?,?,?,?,?,?)", [
                (domain, config_hash, s["buyer_key"], s["full_name"], s["job_title"], s["linkedin_url"],
                 s["department"], s["fit"], s["rationale"], now) for s in result["stakeholders"]])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def norm_domain(d: str) -> str:
    d = re.sub(r"^https?://", "", (d or "").strip().lower())
    d = d.split("/")[0]
    return d[4:] if d.startswith("www.") else d


def _buyer_key(b: dict) -> str:
    return (b.get("linkedin_url") or f'{b.get("full_name", "")}|{b.get("job_title", "")}').strip().lower()


# ── Gemini research ──────────────────────────────────────────────────────────
def _gemini_prompt(account: dict, terms: list[str], paraphrase: bool) -> str:
    today = date.today().isoformat()
    or_terms = " OR ".join(f'"{t}"' for t in terms)
    evidence_rule = ("one-sentence factual summary of what the source says, in your own words, "
                     "keeping names, numbers and technologies exact") if paraphrase else \
                    "a SHORT verbatim excerpt (max 20 words) from the source page that shows the signal"
    return f"""Today is {today}. Research the company below on the live web and return raw buying signals.

Company: {account['company_name']}
Domain: {account['domain']}
LinkedIn: {account.get('company_linkedin_url') or '-'}

Run these searches first, then broaden:
  - "{account['company_name']}" ({or_terms})
  - site:{account['domain']} ({or_terms})
  - "{account['company_name']}" hiring OR "job opening" engineer OR data OR cloud
  - "{account['company_name']}" funding OR acquisition OR expansion OR partnership 2025 OR 2026
  - "{account['company_name']}" appoints OR "new CTO" OR "new CIO" OR "chief data officer"

Collect up to 25 distinct signals: technologies in use, hiring, news, funding, leadership
changes, expansion, partnerships, product launches. Prefer the last 24 months.

Return ONLY a JSON object, no prose, no code fences:
{{"signals": [{{
  "id": "S1",
  "type": "technology | hiring | news | funding | leadership | expansion | partnership | product | other",
  "title": "short label",
  "evidence": "{evidence_rule}",
  "source_url": "page URL",
  "date": "YYYY-MM-DD, YYYY-MM, or empty string if the page shows no date"
}}]}}
Never invent a signal, quote, URL or date. Fewer real signals beat more fabricated ones."""


def _parse_json_object(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Gemini returned no JSON object")
    return json.loads(re.sub(r",\s*([\]}])", r"\1", text[start:end + 1]))


def gemini_research(account: dict, terms: list[str], run_id: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"],
                          http_options={"timeout": 180_000})  # milliseconds
    # Asking for page quotes can trip Gemini's recitation filter (finish_reason
    # RECITATION, empty text). Retry once asking for a paraphrase instead.
    for paraphrase in (False, True):
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_gemini_prompt(account, terms, paraphrase),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=16384,
            ),
        )
        log_gemini_usage(MODULE, f"research|{account['company_name'][:20]}", resp, run_id=run_id)
        if resp.text and "{" in resp.text:
            break
    else:
        reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
        raise RuntimeError(f"Gemini returned no usable output ({reason})")

    payload = _parse_json_object(resp.text)
    signals = [s for s in payload.get("signals", []) if isinstance(s, dict) and s.get("evidence")]
    for i, s in enumerate(signals, 1):
        s["id"] = f"S{i}"
        s["evidence_kind"] = "paraphrase" if paraphrase else "quote"
    sources = []
    try:
        for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if ch.web:
                sources.append({"title": ch.web.title, "uri": ch.web.uri})
    except (AttributeError, IndexError, TypeError):
        pass
    return {"signals": signals, "grounding_sources": sources, "researched_at": _now()}


# ── Claude analysis ──────────────────────────────────────────────────────────
def _claude_schema(terms: list[str]) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["matches", "departments", "stakeholders", "account_summary"],
        "properties": {
            "account_summary": {"type": "string"},
            "matches": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["signal_id", "term", "rationale"],
                "properties": {"signal_id": {"type": "string"},
                               "term": {"type": "string", "enum": terms},
                               "rationale": {"type": "string"}}}},
            "departments": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "theme", "signal_ids"],
                "properties": {"name": {"type": "string"}, "theme": {"type": "string"},
                               "signal_ids": {"type": "array", "items": {"type": "string"}}}}},
            "stakeholders": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["buyer_index", "department", "fit", "rationale"],
                "properties": {"buyer_index": {"type": "integer"},
                               "department": {"type": "string"},
                               "fit": {"type": "string", "enum": ["High", "Medium", "Low", "None"]},
                               "rationale": {"type": "string"}}}},
        },
    }


_CLAUDE_SYSTEM = """You are a B2B account-intelligence analyst. You receive raw web signals about one
target account, a list of weighted intent terms configured by the seller, and the seller's list of
prospective buyers at that account.

1. MATCHES — link a signal to an intent term only when the signal's evidence text genuinely shows
   that intent (semantic match, not keyword overlap). One signal may match several terms. Use the
   term text verbatim. Skip signals that match nothing.
2. DEPARTMENTS — read the matched signals and infer the internal department themes they point to,
   named the way this company would name them (e.g. "Data Platform Engineering", "Clinical
   Operations"). Derive them purely from the signal text; do not use a fixed list. Every matched
   signal belongs to exactly one department. 1–6 departments.
3. STAKEHOLDERS — for EVERY buyer (by buyer_index), judge from their title what they own, and map
   them to the one department whose theme they most plausibly buy for. If none fits, department
   "Unmapped" and fit "None". Rationale: one or two sentences naming the signal(s) that make this
   person relevant.
4. account_summary — two sentences on why this account is or is not showing intent."""


def claude_analyze(account: dict, research: dict, weights: dict[str, float], run_id: str) -> dict:
    import anthropic

    buyers = account["prospective_buyers"]
    user = json.dumps({
        "today": date.today().isoformat(),
        "account": {k: account[k] for k in ("company_name", "domain")},
        "intent_terms": [{"term": t, "weight": w} for t, w in weights.items()],
        "signals": [{k: s.get(k, "") for k in ("id", "type", "title", "evidence", "date")}
                    for s in research["signals"]],
        "buyers": [{"buyer_index": i, "full_name": b["full_name"], "job_title": b["job_title"]}
                   for i, b in enumerate(buyers)],
    }, indent=1)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300.0)
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=16000, system=_CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _claude_schema(list(weights))}},
    )
    log_claude_usage(MODULE, f"analyze|{account['company_name'][:20]}", resp, run_id=run_id)
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this account")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude output was cut off (max_tokens)")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


# ── Scoring + assembly ───────────────────────────────────────────────────────
def _parse_date(s: str):
    s = (s or "").strip()
    for fmt, n in (("%Y-%m-%d", 10), ("%Y-%m", 7), ("%Y", 4)):
        try:
            return datetime.strptime(s[:n], fmt).date()
        except ValueError:
            continue
    return None


def _decay(d, half_life: float):
    if d is None:
        return UNDATED_DECAY, None
    age = max((date.today() - d).days, 0)
    return 0.5 ** (age / half_life), age


def assemble(account: dict, research: dict, analysis: dict, weights: dict[str, float],
             half_life: float) -> dict:
    sig_by_id = {s["id"]: s for s in research["signals"]}
    matches = []
    for m in analysis["matches"]:
        s = sig_by_id.get(m["signal_id"])
        if not s or m["term"] not in weights:
            continue
        decay, age = _decay(_parse_date(s.get("date", "")), half_life)
        matches.append({
            "signal_id": s["id"], "term": m["term"], "weight": weights[m["term"]],
            "decay": round(decay, 3), "age_days": age, "date": s.get("date", ""),
            "contribution": round(weights[m["term"]] * decay, 3),
            "title": s.get("title", ""), "evidence": s["evidence"],
            "evidence_kind": s.get("evidence_kind", "quote"),
            "source_url": s.get("source_url", ""), "type": s.get("type", ""),
            "rationale": m["rationale"],
        })

    # Each term counts once, at its freshest evidence — repeats can't inflate the score.
    best: dict[str, dict] = {}
    for m in matches:
        if m["term"] not in best or m["contribution"] > best[m["term"]]["contribution"]:
            best[m["term"]] = m
    for m in matches:
        m["counted"] = best.get(m["term"]) is m
    max_possible = sum(w for w in weights.values() if w > 0) or 1
    composite = round(100 * sum(m["contribution"] for m in best.values()) / max_possible, 1)

    matched_ids = {m["signal_id"] for m in matches}
    dept_names, sig_to_dept, departments = [], {}, []
    for d in analysis["departments"]:
        ids = [i for i in d["signal_ids"] if i in matched_ids and i not in sig_to_dept]
        for i in ids:
            sig_to_dept[i] = d["name"]
        dept_names.append(d["name"])
        departments.append({"name": d["name"], "theme": d["theme"], "signal_ids": ids,
                            "contribution": round(sum(m["contribution"] for m in matches
                                                      if m["counted"] and m["signal_id"] in ids), 3)})
    for m in matches:
        m["department"] = sig_to_dept.get(m["signal_id"], "Unassigned")

    routed = {s["buyer_index"]: s for s in analysis["stakeholders"]}
    stakeholders = []
    for i, b in enumerate(account["prospective_buyers"]):
        r = routed.get(i, {})
        dept = r.get("department", "Unmapped")
        if dept not in dept_names:
            dept = "Unmapped"
        stakeholders.append({
            "buyer_key": _buyer_key(b), "full_name": b["full_name"], "job_title": b["job_title"],
            "linkedin_url": b.get("linkedin_url", ""), "department": dept,
            "fit": r.get("fit", "None") if dept != "Unmapped" else "None",
            "rationale": r.get("rationale", "Not routed by the model."),
        })

    fit_rank = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
    dept_rank = {d["name"]: d["contribution"] for d in departments}
    primary = max(stakeholders, default=None,
                  key=lambda s: (fit_rank[s["fit"]], dept_rank.get(s["department"], 0)))
    return {
        "company_name": account["company_name"], "domain": account["domain"],
        "company_linkedin_url": account.get("company_linkedin_url", ""),
        "composite_score": composite, "summary": analysis.get("account_summary", ""),
        "matches": sorted(matches, key=lambda m: -m["contribution"]),
        "departments": sorted(departments, key=lambda d: -d["contribution"]),
        "stakeholders": stakeholders,
        "primary_route": (f'{primary["full_name"]} ({primary["job_title"]}) → {primary["department"]}'
                          if primary and primary["fit"] != "None" else "No fit"),
        "grounding_sources": research.get("grounding_sources", []),
        "signals_found": len(research["signals"]),
    }


# ── Runner ───────────────────────────────────────────────────────────────────
def _process_account_sync(account, weights, half_life, force_refresh, run_id, store, status_cb):
    domain = account["domain"]
    terms_hash = _hash(sorted(weights))
    config_hash = _hash({"w": weights, "hl": half_life,
                         "b": sorted(_buyer_key(b) for b in account["prospective_buyers"])})
    if not force_refresh and (hit := store.get_processed(domain, config_hash)):
        return hit
    research = None if force_refresh else store.get_raw(domain, terms_hash)
    if research is None:
        status_cb("researching")
        research = gemini_research(account, list(weights), run_id)
        store.put_raw(domain, terms_hash, research)
    status_cb("scoring")
    if research["signals"]:
        analysis = claude_analyze(account, research, weights, run_id)
    else:
        analysis = {"matches": [], "departments": [], "stakeholders": [],
                    "account_summary": "No web signals found."}
    result = assemble(account, research, analysis, weights, half_life)
    store.put_processed(domain, config_hash, result)
    return result


async def run_intent_map(accounts: list[dict], weights: dict[str, float], half_life: float = 180,
                         force_refresh: bool = False):
    run_id = new_run_id()
    store = await asyncio.to_thread(ResearchStorageLayer)
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    _DONE = object()

    yield {"type": "run_started", "run_id": run_id, "total": len(accounts)}

    async def _pump(account):
        name = account["company_name"]

        def status_cb(status):
            loop.call_soon_threadsafe(queue.put_nowait,
                                      {"type": "account_status", "company": name, "status": status})
        try:
            async with sem:
                result = await asyncio.wait_for(asyncio.to_thread(
                    _process_account_sync, account, weights, half_life, force_refresh,
                    run_id, store, status_cb), timeout=ACCOUNT_TIMEOUT)
            await queue.put({"type": "account_result", "company": name, "result": result})
        except Exception as e:
            logger.error(f"Intent Map {name} failed: {e!r}")
            msg = "timed out" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}: {e}"
            await queue.put({"type": "account_error", "company": name, "message": msg})
        finally:
            await queue.put(_DONE)

    tasks = [asyncio.ensure_future(_pump(a)) for a in accounts]
    remaining, t0, results = len(tasks), time.time(), []
    while remaining:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=20)
        except asyncio.TimeoutError:
            yield {"type": "heartbeat", "message": f"⏳ Researching accounts… ({int(time.time() - t0)}s)"}
            continue
        if item is _DONE:
            remaining -= 1
            continue
        if item["type"] == "account_result":
            results.append(item["result"])
        yield item

    usage = await asyncio.to_thread(get_usage_by_run, run_id)
    yield {"type": "complete", "run_id": run_id, "usage": usage, "total": len(results),
           "results": sorted(results, key=lambda r: -r["composite_score"])}
