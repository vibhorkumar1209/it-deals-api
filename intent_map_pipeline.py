"""Intent Map — parameter-based intent scoring & dynamic stakeholder mapping.

Scoring follows the analyst framework in "Intent Intelligence.xlsx":
  Each parameter (default five: investment, strategic initiatives,
  regulatory/operational urgency, hiring, IT decision-maker change) is rated
  High=3 / Medium=2 / Low=1 / None=0 against a fixed evidence rubric.
  score = Σ weight × level/3 × 100 (weights normalised to 100%)
  band  = High ≥75, Medium 45–74, Low <45  — also the "Fit for <seller>" label.

Per account:
  cache    SQLite, reuse runs < 90 days old (research cached separately from
           rating, so changing weights/buyers re-runs only Claude).
  Gemini   2.5 Flash + Google Search grounding → evidence per parameter.
  Claude   Sonnet 5 → rates each parameter, infers departments, routes buyers.
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

from json_salvage import parse_lenient
from usage_logger import get_usage_by_run, log_claude_usage, log_gemini_usage, new_run_id

logger = logging.getLogger("intent_map")

GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-5"
MODULE = "intent_map"
CACHE_DAYS = 90
MAX_CONCURRENT = 4
ACCOUNT_TIMEOUT = 600
_DB_PATH = os.path.join(os.getenv("USAGE_DB_DIR", "/tmp"), "intent_map_cache.db")

LEVEL_SCORE = {"High": 3, "Medium": 2, "Low": 1, "None": 0}
BAND_HIGH, BAND_MEDIUM = 75, 45

DEFAULT_PARAMETERS = [
    {"name": "Investment & IT spend increase", "weight": 25,
     "description": "Rising investment or IT spend in {focus}: funded programmes, licence/platform rollouts, budget statements."},
    {"name": "Strategic initiatives", "weight": 30,
     "description": "Named strategies, roadmaps, transformation programmes, pilots or certifications in {focus}."},
    {"name": "Regulatory / operational urgency", "weight": 20,
     "description": "Regulation, deadlines, resilience, security or third-party requirements that create {focus} demand."},
    {"name": "Hiring signal", "weight": 15,
     "description": "Open roles or team build-out in {focus} and the IT functions that run it."},
    {"name": "Decision-maker change (IT)", "weight": 10,
     "description": "A new CIO, CTO, CDO, CISO, Head of {focus} or other IT leader appointed in the last 12 months."},
]

RUBRIC = """LEVELS — rate every parameter with exactly one:
  High   (Confirmed) — direct, recent, credible evidence: a stated programme, tender, contract end-date,
                       budget, or executive commitment.
  Medium (Likely)    — two or more relevant indicators, but no direct procurement or contract confirmation.
  Low    (Emerging)  — one indirect or early-stage indicator only.
  None               — no relevant evidence found."""


# Concrete scope + search vocabulary for known focus areas; others are expanded by the model.
FOCUS_PROFILES = {
    "workplace": {
        "scope": ("employee-facing technology: laptops/devices and endpoint management (Intune, Autopilot), "
                  "Microsoft 365 / Copilot / Teams / SharePoint, collaboration, VDI / Azure Virtual Desktop / "
                  "Citrix, IT service desk and ITSM (e.g. ServiceNow), end-user computing (EUC), hybrid-working "
                  "technology, employee experience, and workplace/endpoint security"),
        "keywords": ["Copilot", "Microsoft 365", "Intune", "endpoint", "virtual desktop", "service desk",
                     "ServiceNow", "end user computing", "hybrid working", "laptop refresh"],
    },
}


def focus_profile(focus: str) -> dict:
    f = focus.lower()
    for key, prof in FOCUS_PROFILES.items():
        if key in f:
            return prof
    return {"scope": "", "keywords": []}


def band_for(score: float) -> str:
    return "High" if score >= BAND_HIGH else "Medium" if score >= BAND_MEDIUM else "Low"


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
def _param_searches(param: dict, name: str, domain: str, focus: str, kw: str, region: str) -> list[str]:
    n = param["name"].lower()
    if "invest" in n or "spend" in n:
        return [f'"{name}" ({kw}) licences OR rollout OR deployment OR investment',
                f'"{name}" annual report 2025 OR 2026 technology investment {focus}',
                f'site:linkedin.com/posts "{name}" ({kw})']
    if "strateg" in n or "initiative" in n:
        return [f'"{name}" {focus} programme OR strategy OR roadmap OR transformation OR pilot',
                f'"{name}" ({kw}) project OR programme 2025 OR 2026',
                f'site:{domain} {focus} OR ({kw})']
    if "regulat" in n or "urgency" in n or "operational" in n:
        return [f'"{name}" operational resilience OR "third-party risk" OR "material third party" {region}',
                f'"{name}" cyber OR endpoint security OR outage OR regulatory deadline {region}',
                f'"{name}" annual report principal risks technology {region}']
    if "hiring" in n or "hire" in n or "talent" in n:
        return [f'site:linkedin.com/jobs "{name}" ({kw})',
                f'site:myworkdayjobs.com OR site:greenhouse.io OR site:lever.co "{name}" ({kw}) OR "{focus}"',
                f'"{name}" careers {focus} OR "service desk" OR "end user" {region}']
    if "decision" in n or "leader" in n or "appoint" in n or "executive" in n:
        return [f'"{name}" appoints OR appointed "Chief Information Officer" OR CIO OR CTO OR CISO 2025 OR 2026',
                f'"{name}" "Chief Digital Officer" OR "Chief Data Officer" OR "Head of {focus}" joins OR appointed',
                f'site:linkedin.com "{name}" "new role" CIO OR CTO OR "Head of IT"']
    return [f'"{name}" {param["name"]} {focus}', f'"{name}" ({kw}) {param["name"]}']


def _gemini_prompt(account: dict, param: dict, idx: int, focus: str, region: str, paraphrase: bool) -> str:
    today = date.today().isoformat()
    name = account["company_name"]
    evidence_rule = ("one-sentence factual summary of what the source says, in your own words, "
                     "keeping names, numbers, dates and technologies exact") if paraphrase else \
                    "a SHORT verbatim excerpt (max 25 words) from the source page that shows the signal"
    region_line = f"\nMarket / region of interest: {region}" if region else ""
    prof = focus_profile(focus)
    scope_line = f"\n{focus} here means: {prof['scope']}." if prof["scope"] else (
        f"\nFirst expand '{focus}' into the concrete technologies, platforms and roles it covers, "
        f"and search for those specific terms.")
    kw = " OR ".join(f'"{k}"' for k in prof["keywords"]) or f'"{focus}"'
    searches = "\n".join(f"  - {q}" for q in _param_searches(param, name, account["domain"], focus, kw, region))
    return f"""Today is {today}. Research the company below on the live web for ONE kind of buying-intent
evidence in the area of {focus}.{scope_line}{region_line}

Company: {name}
Domain: {account['domain']}
LinkedIn: {account.get('company_linkedin_url') or '-'}

EVIDENCE TO FIND — {param['name']}: {param['description']}

Run these searches, then broaden with your own:
{searches}

Collect up to 10 distinct signals for this evidence type only, preferring the last 24 months.
Include the publication date whenever the page shows one.

Return ONLY a JSON object, no prose, no code fences:
{{"signals": [{{
  "title": "short label",
  "evidence": "{evidence_rule}",
  "source_url": "page URL",
  "source_type": "annual report | press release | news | job posting | LinkedIn post | company website | other",
  "date": "YYYY-MM-DD, YYYY-MM, or empty string if the page shows no date"
}}]}}
Never invent a signal, quote, URL or date. Return {{"signals": []}} if nothing relevant exists."""


def _research_one(account: dict, param: dict, idx: int, focus: str, region: str, run_id: str):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"],
                          http_options={"timeout": 180_000})  # milliseconds
    # Asking for page quotes can trip Gemini's recitation filter (finish_reason
    # RECITATION, empty text). Retry once asking for a paraphrase instead.
    for paraphrase in (False, True):
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_gemini_prompt(account, param, idx, focus, region, paraphrase),
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
                max_output_tokens=16384,
            ),
        )
        log_gemini_usage(MODULE, f"research|P{idx}|{account['company_name'][:20]}", resp, run_id=run_id)
        if resp.text and "{" in resp.text:
            break
    else:
        return [], []
    payload = parse_lenient(resp.text)
    raw = payload.get("signals", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    signals = [dict(s, parameter=f"P{idx}", evidence_kind="paraphrase" if paraphrase else "quote")
               for s in raw if isinstance(s, dict) and s.get("evidence")]
    sources = []
    try:
        for ch in resp.candidates[0].grounding_metadata.grounding_chunks or []:
            if ch.web:
                sources.append({"title": ch.web.title, "uri": ch.web.uri})
    except (AttributeError, IndexError, TypeError):
        pass
    return signals, sources


def gemini_research(account: dict, params: list[dict], focus: str, region: str, run_id: str) -> dict:
    """One focused grounded search per parameter, in parallel — a single broad
    call missed key evidence (e.g. Copilot licence counts) for many accounts."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(params)) as pool:
        futures = [pool.submit(_research_one, account, p, i, focus, region, run_id)
                   for i, p in enumerate(params)]
        outcomes = []
        for f in futures:
            try:
                outcomes.append(f.result())
            except Exception as e:
                logger.warning(f"Intent Map research call failed for {account['company_name']}: {e!r}")
                outcomes.append(None)
    if all(o is None for o in outcomes):
        raise RuntimeError("All Gemini research calls failed")

    signals, sources, seen = [], [], set()
    for o in outcomes:
        if not o:
            continue
        sigs, srcs = o
        sources += srcs
        for sg in sigs:
            key = (sg.get("source_url", "").split("?")[0].rstrip("/"), sg["evidence"][:80].lower())
            if key in seen:
                continue
            seen.add(key)
            signals.append(sg)
    for i, sg in enumerate(signals, 1):
        sg["id"] = f"S{i}"
    return {"signals": signals, "grounding_sources": sources, "researched_at": _now()}


def _claude_schema(n_params: int) -> dict:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["sector", "account_summary", "parameters", "departments", "stakeholders"],
        "properties": {
            "sector": {"type": "string"},
            "account_summary": {"type": "string"},
            "parameters": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["param_index", "level", "finding", "signal_ids"],
                "properties": {"param_index": {"type": "integer"},
                               "level": {"type": "string", "enum": list(LEVEL_SCORE)},
                               "finding": {"type": "string"},
                               "signal_ids": {"type": "array", "items": {"type": "string"}}}}},
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


def _claude_system(focus: str, seller: str, region: str) -> str:
    seller_line = f" for {seller}" if seller else ""
    prof = focus_profile(focus)
    scope = f" — i.e. {prof['scope']}" if prof["scope"] else ""
    region_rule = (f"\n   REGION: the market of interest is {region}. Evidence about operations elsewhere (e.g. a CIO "
                   f"appointed for another country's office) counts at most Low, unless it is group-wide.") if region else ""
    return f"""You are a B2B account-intelligence analyst assessing buying intent in {focus}{seller_line}.
You receive raw web signals about one target account, the intent parameters to rate, and the
seller's prospective buyers at that account.

1. PARAMETERS — rate EVERY parameter (by param_index) using only the signals provided.
{RUBRIC}
   Recency matters: evidence older than ~24 months cannot be High; a decision-maker change must be
   within the last 12 months. Undated evidence can be at most Medium. A job posting on its own is
   hiring evidence — for any other parameter it can support at most Medium.
   RELEVANCE: only evidence specifically about {focus}{scope} counts. General IT or cloud
   spend, customer-facing digital products, core-system or data programmes, and money the company
   invests INTO other businesses, funds or infrastructure assets (as an investor) do NOT count —
   rate those None. Regulatory / operational urgency may rely on regulation that plausibly creates
   {focus} demand (resilience, third-party risk, endpoint/access security).{region_rule}
   finding: 1–2 factual sentences in analyst style naming the specific
   programme, technology, date, regulation or person (e.g. "Approximately 1,200 Copilot licences
   deployed by Dec-2025, >25% of colleagues; rollout continuing through 2026."). For None, finding
   is "None". signal_ids: the signals that support the rating (empty for None).
2. sector — the account's industry in 1–3 words (e.g. BFSI, Retail, Healthcare).
3. DEPARTMENTS — infer the internal department themes the evidence points to, named the way this
   company would name them (e.g. "Digital Workplace Services", "Operational Resilience"). Derive
   them from the signal text; no fixed list. 1–6 departments.
4. STAKEHOLDERS — for EVERY buyer (by buyer_index), judge from their title what they own and map
   them to the department they most plausibly buy for. If none fits, department "Unmapped" and fit
   "None". Rationale: one or two sentences naming the evidence that makes this person relevant.
5. account_summary — two sentences on why this account is or is not showing {focus} intent."""


def claude_analyze(account: dict, research: dict, params: list[dict], focus: str, seller: str,
                   region: str, run_id: str) -> dict:
    import anthropic

    buyers = account["prospective_buyers"]
    user = json.dumps({
        "today": date.today().isoformat(),
        "account": {k: account[k] for k in ("company_name", "domain")},
        "parameters": [{"param_index": i, "name": p["name"], "what_counts": p["description"]}
                       for i, p in enumerate(params)],
        "signals": [{k: s.get(k, "") for k in ("id", "parameter", "title", "evidence", "source_type", "date")}
                    for s in research["signals"]],
        "buyers": [{"buyer_index": i, "full_name": b["full_name"], "job_title": b["job_title"]}
                   for i, b in enumerate(buyers)],
    }, indent=1)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300.0)
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=16000, system=_claude_system(focus, seller, region),
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _claude_schema(len(params))}},
    )
    log_claude_usage(MODULE, f"analyze|{account['company_name'][:20]}", resp, run_id=run_id)
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this account")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Claude output was cut off (max_tokens)")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


# ── Scoring + assembly (deterministic) ───────────────────────────────────────
def assemble(account: dict, research: dict, analysis: dict, params: list[dict], seller: str) -> dict:
    sig_by_id = {s["id"]: s for s in research["signals"]}
    total_w = sum(p["weight"] for p in params) or 1
    rated = {r["param_index"]: r for r in analysis.get("parameters", [])
             if 0 <= r.get("param_index", -1) < len(params)}

    parameters, score = [], 0.0
    for i, p in enumerate(params):
        r = rated.get(i, {})
        level = r.get("level", "None") if r.get("level") in LEVEL_SCORE else "None"
        evidence = [sig_by_id[sid] for sid in r.get("signal_ids", []) if sid in sig_by_id]
        weight_pct = 100 * p["weight"] / total_w
        points = weight_pct * LEVEL_SCORE[level] / 3
        score += points
        parameters.append({
            "name": p["name"], "weight": round(weight_pct, 1), "level": level,
            "level_score": LEVEL_SCORE[level], "points": round(points, 1),
            "finding": r.get("finding") or "None",
            "evidence": [{k: s.get(k, "") for k in ("id", "title", "evidence", "evidence_kind", "source_url", "date")}
                         for s in evidence],
        })
    score = round(score, 1)

    cited = {e["id"] for p in parameters for e in p["evidence"]}
    dept_names, sig_to_dept, departments = [], {}, []
    for d in analysis.get("departments", []):
        ids = [i for i in d["signal_ids"] if i in sig_by_id and i not in sig_to_dept]
        for i in ids:
            sig_to_dept[i] = d["name"]
        dept_names.append(d["name"])
        departments.append({"name": d["name"], "theme": d["theme"], "signal_ids": ids,
                            "contribution": len([i for i in ids if i in cited])})
    for p in parameters:
        for e in p["evidence"]:
            e["department"] = sig_to_dept.get(e["id"], "Unassigned")

    routed = {s["buyer_index"]: s for s in analysis.get("stakeholders", [])}
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
    sources = list(dict.fromkeys(e["source_url"] for p in parameters for e in p["evidence"] if e["source_url"]))
    return {
        "company_name": account["company_name"], "domain": account["domain"],
        "company_linkedin_url": account.get("company_linkedin_url", ""),
        "sector": analysis.get("sector", ""), "seller": seller,
        "composite_score": score, "band": band_for(score),
        "summary": analysis.get("account_summary", ""),
        "parameters": parameters,
        "departments": sorted(departments, key=lambda d: -d["contribution"]),
        "stakeholders": stakeholders,
        "primary_route": (f'{primary["full_name"]} ({primary["job_title"]}) → {primary["department"]}'
                          if primary and primary["fit"] != "None" else "No fit"),
        "sources": sources,
        "grounding_sources": research.get("grounding_sources", []),
        "signals_found": len(research["signals"]),
    }


# ── Runner ───────────────────────────────────────────────────────────────────
def _process_account_sync(account, params, focus, region, seller, force_refresh, run_id, store, status_cb):
    domain = account["domain"]
    research_hash = _hash({"p": [[p["name"], p["description"]] for p in params], "f": focus, "r": region})
    config_hash = _hash({"p": params, "f": focus, "r": region, "s": seller,
                         "b": sorted(_buyer_key(b) for b in account["prospective_buyers"])})
    if not force_refresh and (hit := store.get_processed(domain, config_hash)):
        return hit
    research = None if force_refresh else store.get_raw(domain, research_hash)
    if research is None:
        status_cb("researching")
        research = gemini_research(account, params, focus, region, run_id)
        store.put_raw(domain, research_hash, research)
    status_cb("scoring")
    if research["signals"]:
        analysis = claude_analyze(account, research, params, focus, seller, region, run_id)
    else:
        analysis = {"sector": "", "parameters": [], "departments": [], "stakeholders": [],
                    "account_summary": "No web signals found."}
    result = assemble(account, research, analysis, params, seller)
    store.put_processed(domain, config_hash, result)
    return result


def resolve_parameters(params: list[dict] | None, focus: str) -> list[dict]:
    base = params or DEFAULT_PARAMETERS
    return [{"name": p["name"].strip(), "weight": float(p["weight"]),
             "description": (p.get("description") or "").replace("{focus}", focus).strip()}
            for p in base if p.get("name", "").strip() and float(p.get("weight", 0)) > 0]


async def run_intent_map(accounts: list[dict], params: list[dict], focus: str, region: str = "",
                         seller: str = "", force_refresh: bool = False):
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
                    _process_account_sync, account, params, focus, region, seller, force_refresh,
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
