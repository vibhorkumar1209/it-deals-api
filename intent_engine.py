"""Intent Engine — Top-5 verifiable buying triggers, in two modes.

COMPANY_BASED  account triggers → 1–3 real executives per trigger + outreach hook.
EXEC_BASED     signals ranked by how directly they hit one executive's KPIs
               (Tier 1 the exec's own words/moves, Tier 2 their dept & tech,
               Tier 3 macro triggers tied to their domain).

Pipeline per target:
  Gemini   2.5 Flash + Google Search, one call per channel in parallel
           (exec footprint, talent, technographic, corporate/financial, and in
           COMPANY_BASED a leadership lookup).
  Gates    code drops signals with no source URL, or older than WINDOW_DAYS
           unless marked as an active programme / open posting.
  Claude   Sonnet 5 ranks the survivors. It may only cite signal ids and
           person ids from the research, so it cannot invent a fact, date,
           URL or person; code re-checks every citation afterwards.

Output follows the spec's Schema A / Schema B exactly, plus `campaign_leads`
in the outlook-campaigns lead shape (core fields + a `data` dict that its
templates read as {{ intent_hook }}, {{ intent_trigger }}, …).
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
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

from intent_map_pipeline import _REDIRECT_HOST, _resolve_redirect, norm_domain
from json_salvage import parse_lenient
from usage_logger import get_usage_by_run, log_claude_usage, log_gemini_usage, new_run_id

logger = logging.getLogger("intent_engine")

GEMINI_MODEL = "gemini-2.5-flash"
CLAUDE_MODEL = "claude-sonnet-5"
MODULE = "intent_map"
WINDOW_DAYS = 180          # company-level signals: past 3–6 months
EXEC_WINDOW_DAYS = 90      # executive-level intent: past 1–3 months only
EMAILS_PER_LEAD = 3        # email 1 from the most recent trigger, follow-ups from older ones
CACHE_DAYS = 30
MAX_CONCURRENT = 3
TARGET_TIMEOUT = 600
TOP_N = 5
_DB_PATH = os.path.join(os.getenv("USAGE_DB_DIR", "/tmp"), "intent_engine_cache.db")

MODES = ("COMPANY_BASED", "EXEC_BASED")
DEFAULT_INTENTS = ["Leadership Change", "Tech Stack Adoption", "Hiring Surge", "Cloud Migration",
                   "M&A", "Budget Expansion"]
CUSTOM_INTENT = "Custom Intent"
COMPANY_CATEGORIES = ["Executive Activity", "Hiring Surge", "Tech Stack Shift", "Strategic/Financial",
                      "Product Launch"]
EXEC_TIERS = ["Tier 1: Direct Exec Activity", "Tier 2: Dept & Tech Shift", "Tier 3: Macro Trigger"]
SOURCE_TYPES = {
    "COMPANY_BASED": ["LinkedIn Post", "Social Post", "Earnings Call", "Careers Page", "News/PR", "Engineering Blog"],
    "EXEC_BASED": ["LinkedIn Post", "Social Post", "Earnings Call", "Careers Page", "News/PR", "Podcast/Interview"],
}


# ── Input ────────────────────────────────────────────────────────────────────
def _as_list(v) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[,;\n]", v)
    return [x.strip() for x in (v or []) if str(x).strip()]


_SOCIAL_TEXT = ("text", "content", "post", "body", "caption", "message", "description", "title", "summary")
_SOCIAL_DATE = ("date", "posted_at", "created_at", "published_at", "timestamp", "time", "posted_on")
_SOCIAL_URL = ("url", "link", "post_url", "permalink", "source_url")


def normalize_social(v) -> list[dict]:
    """Optional social-media activity (past 3–6 months) as any JSON: a list of
    posts, {"posts": [...]}, or a JSON string of either. Unknown shapes are kept
    as text so nothing the user pasted is silently dropped."""
    if v in (None, "", [], {}):
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            raise ValueError("social_activity must be valid JSON")
    if isinstance(v, dict):
        lists = [x for x in v.values() if isinstance(x, list)]
        v = [i for l in lists for i in l] if lists else [v]
    if not isinstance(v, list):
        v = [v]
    out = []
    for item in v[:100]:
        if isinstance(item, dict):
            pick = lambda keys: next((str(item[k]).strip() for k in keys if item.get(k)), "")
            text = pick(_SOCIAL_TEXT) or json.dumps(item, ensure_ascii=False)
            kind = " ".join(str(item.get(k, "")) for k in ("type", "action", "activity_type")).strip()
            out.append({"text": text[:1500], "date": pick(_SOCIAL_DATE)[:10], "url": pick(_SOCIAL_URL),
                        "platform": str(item.get("platform") or item.get("network") or "").strip(),
                        "kind": kind})
        elif str(item).strip():
            out.append({"text": str(item)[:1500], "date": "", "url": "", "platform": "", "kind": ""})
    return out


def normalize_target(raw: dict) -> dict:
    """Spec variable names in, validated dict out. Raises ValueError on bad input."""
    mode = str(raw.get("intent_mode", "COMPANY_BASED")).strip().upper()
    if mode not in MODES:
        raise ValueError('intent_mode must be "COMPANY_BASED" or "EXEC_BASED"')
    t = {
        "intent_mode": mode,
        "company_name": str(raw.get("company_name", "")).strip(),
        "company_domain": norm_domain(raw.get("company_domain", "")),
        "company_linkedin_url": str(raw.get("company_linkedin_url", "")).strip(),
        "focus_department": ", ".join(_as_list(raw.get("focus_department"))),
        "focus_technologies": _as_list(raw.get("focus_technologies")),
        "predefined_intents": _as_list(raw.get("predefined_intents")) or list(DEFAULT_INTENTS),
        "custom_intent_notes": str(raw.get("custom_intent_notes", "")).strip(),
        "executive_name": str(raw.get("executive_name", "")).strip(),
        "executive_designation": str(raw.get("executive_designation", "")).strip(),
        "executive_linkedin_url": str(raw.get("executive_linkedin_url", "")).strip(),
        # Positioning: outreach pitches only this seller and these offerings.
        "seller_company": str(raw.get("seller_company", "")).strip(),
        "seller_offerings": _as_list(raw.get("seller_offerings")),
        # Former names / parent the caller already knows; the engine adds what it finds.
        "alternate_names": _as_list(raw.get("alternate_names")),
        "social_activity": normalize_social(raw.get("social_activity")),
    }
    if not t["company_name"] or not t["company_domain"]:
        raise ValueError("company_name and company_domain are required")
    if not t["focus_department"]:
        raise ValueError("focus_department is required")
    if mode == "EXEC_BASED" and not (t["executive_name"] and t["executive_designation"]):
        raise ValueError("EXEC_BASED needs executive_name and executive_designation")
    return t


# ── Cache ────────────────────────────────────────────────────────────────────
class _Cache:
    def __init__(self, path: str = _DB_PATH):
        self.path = path
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, created_at TEXT, result TEXT)")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA busy_timeout=10000")
        return c

    def get(self, key):
        with self._conn() as c:
            row = c.execute("SELECT created_at, result FROM results WHERE key=?", (key,)).fetchone()
        if not row or datetime.fromisoformat(row[0]) < datetime.now(timezone.utc) - timedelta(days=CACHE_DAYS):
            return None
        return json.loads(row[1])

    def put(self, key, result):
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)", (key, _now(), json.dumps(result)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_key(t: dict) -> str:
    return hashlib.sha256(json.dumps({"v": 3, **t}, sort_keys=True).encode()).hexdigest()[:24]


# ── Gemini research ──────────────────────────────────────────────────────────
def _gemini_json(prompt: str, tag: str, run_id: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"], http_options={"timeout": 180_000})
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt,
        config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())],
                                           max_output_tokens=4096))
    log_gemini_usage(MODULE, tag, resp, run_id=run_id)
    payload = parse_lenient(resp.text or "") if resp.text and "{" in resp.text else {}
    return payload if isinstance(payload, dict) else {}


def resolve_identity(t: dict, run_id: str) -> dict:
    """Acquisitions and renames (esp. the last year): older signals and
    leaders sit under the former name or the parent, so research covers them too."""
    since = (date.today() - timedelta(days=540)).isoformat()
    prompt = f"""Today is {date.today().isoformat()}. Check the live web: since {since}, has the company
"{t['company_name']}" ({t['company_domain']}) been acquired, merged, or renamed / rebranded?

Return ONLY JSON, no prose:
{{"event": "acquired | merged | renamed | none",
  "event_date": "YYYY-MM or empty",
  "former_names": ["previous names of this same company, if renamed"],
  "current_name": "new name if this company now trades under another name, else empty",
  "parent_company": "acquirer / new parent, if acquired or merged, else empty",
  "parent_domain": "parent's web domain, else empty",
  "source_url": "page URL that states it"}}
Only report an event a source states. Never guess. If nothing happened, return event "none"."""
    try:
        d = _gemini_json(prompt, f"engine|identity|{t['company_name'][:20]}", run_id)
    except Exception as e:
        logger.warning(f"Intent Engine identity check failed for {t['company_name']}: {e!r}")
        d = {}
    found = str(d.get("event", "none")).lower() in ("acquired", "merged", "renamed") and \
        str(d.get("source_url", "")).startswith("http")
    own = t["company_name"].lower()
    names = [n for n in _as_list((d.get("former_names") or []) + [d.get("current_name") or ""]) if found] + t["alternate_names"]
    names = list(dict.fromkeys(n for n in names if n.lower() != own))
    parent = str(d.get("parent_company") or "").strip() if found else ""
    return {"event": str(d.get("event")).lower() if found else "none", "event_date": str(d.get("event_date") or "") if found else "",
            "alternate_names": [n for n in names if n.lower() != parent.lower()], "parent_company": parent,
            "parent_domain": norm_domain(d.get("parent_domain") or "") if parent else "",
            "source_url": d.get("source_url", "") if found else ""}


def _names(t: dict) -> str:
    ids = t.get("_identity") or {}
    all_names = [t["company_name"], *ids.get("alternate_names", []), *([ids["parent_company"]] if ids.get("parent_company") else [])]
    return "(" + " OR ".join(f'"{n}"' for n in all_names) + ")" if len(all_names) > 1 else f'"{t["company_name"]}"'


def _channels(t: dict) -> list[dict]:
    name, dom, dept = _names(t), t["company_domain"], t["focus_department"]
    tech = f'"{dept}"'
    intents = ", ".join(t["predefined_intents"])
    chans = []
    if t["intent_mode"] == "EXEC_BASED":
        ex = f'"{t["executive_name"]}"'
        chans.append({"key": "exec", "label": "Executive footprint",
                      "what": f'Things {t["executive_name"]} ({t["executive_designation"]}) personally said or did: '
                              "LinkedIn posts/comments, podcasts, conference keynotes, media interviews, "
                              "promotions or role changes. Only this exact person at this company.",
                      "searches": [f'{ex} {name}', f'site:linkedin.com/posts {ex}',
                                   f'{ex} {name} interview OR podcast OR keynote OR panel',
                                   f'{ex} appointed OR joins OR promoted {name}']})
    else:
        chans.append({"key": "exec", "label": "Executive footprint",
                      "what": f"Posts, interviews, keynotes, appointments or promotions by {dept} leaders "
                              "and the executive team.",
                      "searches": [f'{name} appoints OR appointed OR "joins as" {dept}',
                                   f'site:linkedin.com/posts {name} {dept}',
                                   f'{name} CIO OR CTO OR "Head of" interview OR keynote OR podcast']})
    chans += [
        {"key": "talent", "label": "Department & talent",
         "what": f"Open or recent job postings and team build-outs inside {dept}"
                 + (f" or these initiatives: {t['custom_intent_notes']}" if t["custom_intent_notes"] else "")
                 + ". Note team build-outs and hiring surges.",
         "searches": [f'site:linkedin.com/jobs {name} ({tech})',
                      f'site:myworkdayjobs.com OR site:greenhouse.io OR site:lever.co {name} ({tech})',
                      f'{name} careers {dept} ({tech})']},
        {"key": "tech", "label": "Technographic & product",
         "what": "Vendor selections or switches, platform migrations, cloud moves, engineering-blog posts, "
                 f"RFPs/tenders or product launches affecting {dept}.",
         "searches": [f'{name} {tech} selects OR migrates OR deploys OR partnership OR rollout',
                      f'{name} tender OR RFP OR contract {tech}',
                      f'site:{dom} {tech}']},
        {"key": "corp", "label": "Corporate & financial",
         "what": "Earnings-call priorities, annual report / 10-K / 10-Q technology priorities, M&A, "
                 f"funding, budget expansion or cost-cutting mandates that affect {dept}.",
         "searches": [f'{name} earnings call technology priorities {dept}',
                      f'{name} annual report OR 10-K strategy investment {tech}',
                      f'{name} acquisition OR funding OR "cost reduction" OR "budget" 2026']},
    ]
    if t["intent_mode"] == "COMPANY_BASED":
        chans.append({"key": "people", "label": "Leadership", "people": True,
                      "what": f"The CURRENT executives at {t['company_name']} who own budget, architecture or KPIs "
                              f"in {dept} (C-level, VP, Head of, Director) plus the relevant executive team.",
                      "searches": [f'site:linkedin.com/in {name} {dept}',
                                   f'{name} leadership team', f'{name} "Head of" OR "VP" OR "Director" {dept}']})
    if t["focus_technologies"]:
        techs = " OR ".join(f'"{x}"' for x in t["focus_technologies"])
        chans.append({"key": "usage", "label": "Technology usage check", "usage": True,
                      "what": f"Evidence that {t['company_name']} USES any of these technologies today: "
                              f"{', '.join(t['focus_technologies'])}. Good evidence: job postings requiring it, "
                              "vendor case studies / customer lists, engineering blogs, tech-stack disclosures, "
                              "staff profiles describing their work with it at this company.",
                      "searches": [f'{name} ({techs})', f'site:linkedin.com/jobs {name} ({techs})',
                                   f'({techs}) customer OR "case study" {name}']})
    for c in chans:
        c["intents"] = intents
    return chans


def _gemini_prompt(t: dict, ch: dict, paraphrase: bool) -> str:
    today = date.today()
    since = (today - timedelta(days=_window(t, ch["key"]))).isoformat()
    ev = ("one-sentence factual summary in your own words, keeping names, numbers, dates and technologies exact"
          if paraphrase else "a SHORT verbatim excerpt (max 25 words) from the page")
    anchor = (f"Company: {t['company_name']}\nDomain: {t['company_domain']} (primary anchor)\n"
              f"LinkedIn: {t['company_linkedin_url'] or '-'}\n")
    ids = t.get("_identity") or {}
    if ids.get("alternate_names"):
        anchor += f"Also known as / former name: {', '.join(ids['alternate_names'])} — items under these names count.\n"
    if ids.get("parent_company"):
        anchor += (f"Parent / acquirer ({ids.get('event') or 'acquired'} {ids.get('event_date') or ''}): {ids['parent_company']}"
                   f" {('(' + ids['parent_domain'] + ')') if ids.get('parent_domain') else ''} — include parent items that "
                   "concern this business or group-wide decisions covering it"
                   + (", and parent-level leaders who now own this function for it" if ch.get("people") else "") + ".\n")
    if t["intent_mode"] == "EXEC_BASED":
        anchor += (f"Executive: {t['executive_name']}, {t['executive_designation']}\n"
                   f"Executive LinkedIn: {t['executive_linkedin_url'] or '-'}\n")
    searches = "\n".join(f"  - {q}" for q in ch["searches"])
    if ch.get("usage"):
        shape = f'''{{"usage": [{{
  "technology": "one of the listed technologies, exactly as written",
  "evidence": "{ev} that shows the company uses it",
  "source_url": "page URL",
  "date": "YYYY-MM-DD or YYYY-MM; empty if none shown"
}}]}}'''
        rules = "At most 2 items per technology. Omit a technology with no evidence. A vendor merely pitching it is not evidence."
    elif ch.get("people"):
        shape = f'''{{"people": [{{
  "name": "full name",
  "title": "exact current title",
  "linkedin_url": "https://www.linkedin.com/in/... exactly as found, or empty string",
  "evidence": "{ev} showing they hold this role now",
  "source_url": "page URL"
}}]}}'''
        rules = ("Up to 12 people. Only people who hold the role NOW (not former). Never guess a LinkedIn URL; "
                 "leave it empty unless the URL itself appeared in your search results.")
    else:
        shape = f'''{{"signals": [{{
  "title": "short label",
  "evidence": "{ev}",
  "source_url": "page URL",
  "source_type": "LinkedIn Post | Earnings Call | Careers Page | News/PR | Engineering Blog | Podcast/Interview | Filing",
  "date": "YYYY-MM-DD or YYYY-MM; empty if the page shows no date",
  "ongoing": "true only for an explicitly active multi-year programme or a currently open job posting, else false",
  "people": ["names of company staff named in the source"]
}}]}}'''
        rules = (f"Up to 10 signals from {since} to {today.isoformat()}. Older items only if \"ongoing\" is true.")
    return f"""Today is {today.isoformat()}. Research ONE channel of B2B buying signals on the live web.

{anchor}
ENTITY RULES: use the domain and LinkedIn as anchors. Discard anything about a similarly named company,
an out-of-scope subsidiary, or a different person who shares the executive's name.

CHANNEL — {ch['label']}: {ch['what']}
Focus department(s): {t['focus_department']}
Intent triggers of interest: {ch['intents']}
Custom intent criteria: {t['custom_intent_notes'] or '-'}

Run these searches, then broaden with your own:
{searches}

{rules}
Return ONLY a JSON object, no prose, no code fences:
{shape}
Never invent a fact, quote, date, title, person or URL. Return an empty list if nothing qualifies."""


def _research_channel(t: dict, ch: dict, run_id: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_AI_API_KEY"], http_options={"timeout": 180_000})
    # Page quotes can trip the RECITATION filter (empty text); retry asking for a paraphrase.
    for paraphrase in (False, True):
        resp = client.models.generate_content(
            model=GEMINI_MODEL, contents=_gemini_prompt(t, ch, paraphrase),
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())],
                                               max_output_tokens=16384))
        log_gemini_usage(MODULE, f"engine|{ch['key']}|{t['company_name'][:20]}", resp, run_id=run_id)
        if resp.text and "{" in resp.text:
            break
    else:
        return {}
    payload = parse_lenient(resp.text)
    return payload if isinstance(payload, dict) else {}


def research(t: dict, run_id: str) -> tuple[list[dict], list[dict]]:
    chans = _channels(t)
    with ThreadPoolExecutor(max_workers=len(chans)) as pool:
        futures = [(ch, pool.submit(_research_channel, t, ch, run_id)) for ch in chans]
        outcomes = []
        for ch, f in futures:
            try:
                outcomes.append((ch, f.result()))
            except Exception as e:
                logger.warning(f"Intent Engine {ch['key']} failed for {t['company_name']}: {e!r}")
                outcomes.append((ch, None))
    if all(o is None for _, o in outcomes):
        raise RuntimeError("All research calls failed")

    signals, people, usage, seen = [], [], [], set()
    for ch, o in outcomes:
        for s in (o or {}).get("signals", []) or []:
            if not isinstance(s, dict) or not s.get("evidence"):
                continue
            key = ((s.get("source_url") or "").split("?")[0].rstrip("/"), s["evidence"][:80].lower())
            if key not in seen:
                seen.add(key)
                signals.append(dict(s, channel=ch["key"]))
        for p in (o or {}).get("people", []) or []:
            if isinstance(p, dict) and p.get("name") and p.get("title"):
                people.append(p)
        for u in (o or {}).get("usage", []) or []:
            if isinstance(u, dict):
                usage.append(u)
    return signals, people, usage


# ── Gate 2 (deterministic): source + recency ─────────────────────────────────
def _parse_date(s: str) -> date | None:
    s = (s or "").strip()
    for fmt, n in (("%Y-%m-%d", 10), ("%Y-%m", 7)):
        try:
            return datetime.strptime(s[:n], fmt).date()
        except ValueError:
            continue
    return None


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() == "true"


def _window(t: dict, channel: str) -> int:
    """Executive intent goes stale fast: the exec's own activity (and everything
    in EXEC_BASED mode) must be from the last 90 days; company signals 180."""
    return EXEC_WINDOW_DAYS if t["intent_mode"] == "EXEC_BASED" or channel == "exec" else WINDOW_DAYS


def gate_signals(signals: list[dict], t: dict) -> list[dict]:
    kept = []
    for s in signals:
        if s.get("user_provided"):
            kept.append(s)
            continue
        if not str(s.get("source_url", "")).startswith("http"):
            continue
        d = _parse_date(s.get("date", ""))
        ongoing = _truthy(s.get("ongoing"))
        if d and d > date.today() + timedelta(days=1):
            continue  # a future date is a misread
        cutoff = date.today() - timedelta(days=_window(t, s.get("channel", "")))
        if not ongoing and (d is None or d < cutoff):
            continue
        kept.append(dict(s, _date=d, ongoing=ongoing))
    for i, s in enumerate(kept, 1):
        s["id"] = f"S{i}"
    return kept


def social_signals(t: dict) -> list[dict]:
    """User-supplied social activity, as signals. No URL needed (the user is the
    source); dated items still have to fall inside the window."""
    out, cutoff = [], date.today() - timedelta(days=_window(t, "exec" if t["intent_mode"] == "EXEC_BASED" else "social"))
    who = t["executive_name"] if t["intent_mode"] == "EXEC_BASED" else t["company_name"]
    for a in t["social_activity"]:
        d = _parse_date(a["date"])
        if d and (d < cutoff or d > date.today() + timedelta(days=1)):
            continue
        plat = a["platform"] or "social media"
        out.append({"title": f"{who} {a['kind'] or 'activity'} on {plat}".strip(), "evidence": a["text"],
                    "source_url": a["url"] if a["url"].startswith("http") else "",
                    "source_type": "LinkedIn Post" if "linkedin" in plat.lower() else "Social Post",
                    "date": a["date"], "_date": d, "ongoing": False, "people": [who], "channel": "social",
                    "user_provided": True})
    return out


def gate_usage(t: dict, usage: list[dict]) -> list[dict]:
    """One row per focus technology: evidence the target uses it, or none found.
    Evidence must come with a source and actually name the technology."""
    rows = []
    for tech in t["focus_technologies"]:
        hit = next((u for u in usage
                    if str(u.get("source_url", "")).startswith("http")
                    and tech.lower() in f"{u.get('technology', '')} {u.get('evidence', '')}".lower()
                    and tech.lower() in str(u.get("evidence", "")).lower()), None)
        rows.append({"technology": tech, "evidence_found": bool(hit),
                     "evidence": (hit or {}).get("evidence", ""), "source_url": (hit or {}).get("source_url", ""),
                     "date": (hit or {}).get("date", "")})
    return rows


def gate_people(people: list[dict]) -> list[dict]:
    out, seen = [], set()
    for p in people:
        name = re.sub(r"\s+", " ", p["name"]).strip()
        if not str(p.get("source_url", "")).startswith("http") or name.lower() in seen or len(name.split()) < 2:
            continue
        seen.add(name.lower())
        li = str(p.get("linkedin_url") or "").strip()
        out.append({"id": f"P{len(out) + 1}", "name": name, "title": p["title"].strip(),
                    "linkedin_url": li if re.match(r"https?://([a-z]{2,3}\.)?linkedin\.com/in/[\w%-]+/?$", li) else "N/A",
                    "evidence": p.get("evidence", ""), "source_url": p["source_url"]})
    return out


# ── Claude ranking (Gates 1 & 3) ─────────────────────────────────────────────
def _schema(t: dict) -> dict:
    company = t["intent_mode"] == "COMPANY_BASED"
    intents = t["predefined_intents"] + ([CUSTOM_INTENT] if t["custom_intent_notes"] else [])
    trigger = {
        "type": "object", "additionalProperties": False,
        "required": ["signal_ids", "signal_title", "matched_intent_category", "category", "confidence_score",
                     "evidence_bullets", "source_type", "executives", "outreach_angle"],
        "properties": {
            "signal_ids": {"type": "array", "items": {"type": "string"}},
            "signal_title": {"type": "string"},
            "matched_intent_category": {"type": "string", "enum": intents},
            "category": {"type": "string", "enum": COMPANY_CATEGORIES if company else EXEC_TIERS},
            "confidence_score": {"type": "string", "enum": ["High", "Medium"]},
            "evidence_bullets": {"type": "array", "items": {"type": "string"}},
            "source_type": {"type": "string", "enum": SOURCE_TYPES[t["intent_mode"]]},
            "executives": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["person_id", "ownership_rationale", "outreach_angle"],
                "properties": {"person_id": {"type": "string"}, "ownership_rationale": {"type": "string"},
                               "outreach_angle": {"type": "string"}}}},
            "outreach_angle": {"type": "string"},
        },
    }
    return {"type": "object", "additionalProperties": False,
            "required": ["overall_intent_score", "strategic_narrative", "triggers"],
            "properties": {"overall_intent_score": {"type": "string", "enum": ["High", "Medium", "Low"]},
                           "strategic_narrative": {"type": "string"},
                           "triggers": {"type": "array", "items": trigger}}}


def _offer(t: dict) -> str:
    seller = t["seller_company"] or "the seller"
    return f"{seller}'s {', '.join(t['seller_offerings'])}" if t["seller_offerings"] else f"{seller}'s offering"


def _system(t: dict) -> str:
    tech = _offer(t)
    ids = t.get("_identity") or {}
    alias = ""
    if ids.get("alternate_names") or ids.get("parent_company"):
        alias = (f"\nThe company is also known as / was formerly {', '.join(ids.get('alternate_names') or []) or '-'}; "
                 f"parent/acquirer: {ids.get('parent_company') or '-'}. Signals under those names count for this account.")
    common = f"""You are a B2B sales-intelligence analyst. You receive pre-verified, dated, sourced signals
(ids S1..) about {t['company_name']} ({t['company_domain']}). Rank the Top {TOP_N} buying triggers, best first.{alias}

POSITIONING: every outreach angle positions ONLY {tech}. Do not pitch any other product, vendor or
technology, and do not invent capabilities beyond the offering names given. The "technology_usage" list says
which focus technologies the target is known to use; use it only as context (e.g. "since you run X"),
never as the thing being sold.
SOCIAL ACTIVITY: signals with channel "social" are the target's own recent social-media activity, supplied
by the user. Weigh every one of them; when one is relevant, prefer it as evidence, and use it to sharpen
the outreach angle.

GATE 1 — RELEVANCE: keep a signal only if it directly affects {t['focus_department']}, creates a need that
{tech} addresses, or satisfies a listed intent trigger or the custom criteria. Drop awards, generic PR,
unrelated regional news.
GATE 2 — FACTS: cite only the given signal ids (1–3 per trigger; merge signals about the same event).
Every evidence bullet must be supported by the cited signals: never add a number, date, name or claim
that is not in them. Fewer than {TOP_N} good triggers is fine; return only those. Zero is fine.
confidence_score: High = direct and specific (named programme, deal, posting, statement); Medium = indirect.
matched_intent_category: the single best-matching listed trigger{", or 'Custom Intent' when it matches the custom criteria" if t['custom_intent_notes'] else ""}.
evidence_bullets: 2 bullets — what happened (with metrics, roles, context), and why it matters for
{t['focus_department']} / {tech}.
overall_intent_score: High = several strong, recent, on-focus triggers; Medium = some; Low = few or weak.
strategic_narrative: exactly 2 sentences."""
    if t["intent_mode"] == "COMPANY_BASED":
        return common + f"""

GATE 3 — COMPANY_BASED: rank by urgency, budget impact and fit with {tech} and the custom criteria.
category: Executive Activity | Hiring Surge | Tech Stack Shift | Strategic/Financial | Product Launch.
executives: 1–3 per trigger, chosen ONLY from the given people (person ids P1..) who own budget,
architecture or KPIs for that trigger. If no listed person fits, return an empty list — never invent one.
ownership_rationale: 1 sentence on why that person owns it. outreach_angle per executive: 1 sentence
positioning {tech}, tied to the trigger. Leave the trigger-level outreach_angle as an empty string.
strategic_narrative: the account's main technical and business priorities right now."""
    return common + f"""

GATE 3 — EXEC_BASED: judge every signal by "does this directly affect {t['executive_name']}'s KPIs as
{t['executive_designation']}?" Drop it if not. Rank strictly by tier, then by impact:
  Tier 1: Direct Exec Activity — {t['executive_name']}'s own words, posts, interviews or role change
          (only when the cited signal names {t['executive_name']}, or is their own social activity).
  Tier 2: Dept & Tech Shift — hiring, tech-stack shifts or projects in {t['focus_department']}.
  Tier 3: Macro Trigger — company-level priorities (earnings, M&A, cost mandates) explicitly tied to their domain.
outreach_angle: 1 hyper-personalised sentence showing {t['executive_name']} how {tech} solves the priority
in that trigger. Second evidence bullet: the direct impact on {t['executive_name']}'s KPIs.
Leave executives as an empty list.
strategic_narrative: what {t['executive_name']} is prioritising and measured on right now."""


def rank(t: dict, signals: list[dict], people: list[dict], usage: list[dict], run_id: str) -> dict:
    import anthropic

    user = json.dumps({
        "today": date.today().isoformat(),
        "inputs": {k: t[k] for k in ("intent_mode", "company_name", "company_domain", "focus_department",
                                     "focus_technologies", "predefined_intents", "custom_intent_notes",
                                     "executive_name", "executive_designation", "seller_company",
                                     "seller_offerings")},
        "company_identity": t.get("_identity") or {},
        "technology_usage": [{k: u[k] for k in ("technology", "evidence_found", "evidence")} for u in usage],
        "signals": [{"id": s["id"], "channel": s["channel"], "title": s.get("title", ""), "evidence": s["evidence"],
                     "source_type": s.get("source_type", ""), "date": s.get("date", ""), "ongoing": s["ongoing"],
                     "people": s.get("people", [])} for s in signals],
        "people": [{k: p[k] for k in ("id", "name", "title", "evidence")} for p in people],
    }, indent=1)
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=300.0)
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=12000, system=_system(t),
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": _schema(t)}},
    )
    log_claude_usage(MODULE, f"engine|{t['company_name'][:20]}", resp, run_id=run_id)
    if resp.stop_reason == "refusal":
        raise RuntimeError("The ranking model declined this target")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Ranking output was cut off")
    return json.loads(next(b.text for b in resp.content if b.type == "text"))


# ── Assembly (deterministic re-checks) ───────────────────────────────────────
def _timeframe(cited: list[dict]) -> str:
    dated = sorted((s["_date"] for s in cited if s["_date"]), reverse=True)
    # An active programme that started before the window reads as live, not stale.
    if dated and dated[0] >= date.today() - timedelta(days=WINDOW_DAYS):
        return dated[0].strftime("%Y-%m")
    return "Ongoing"


def assemble(t: dict, signals: list[dict], people: list[dict], ranked: dict, usage: list[dict] | None = None) -> dict:
    company = t["intent_mode"] == "COMPANY_BASED"
    by_sig = {s["id"]: s for s in signals}
    by_person = {p["id"]: p for p in people}
    last = t["executive_name"].split()[-1].lower() if t["executive_name"] else ""

    out = []
    for tr in ranked.get("triggers", []):
        cited = [by_sig[i] for i in dict.fromkeys(tr.get("signal_ids", [])) if i in by_sig]
        if not cited:
            continue  # no verified source behind it
        if company and tr["category"] == "Executive Activity" and not any(
                s["ongoing"] or (s["_date"] and s["_date"] >= date.today() - timedelta(days=EXEC_WINDOW_DAYS)) for s in cited):
            continue  # executive intent older than 3 months is stale
        item = {
            "rank": 0,
            "signal_title": tr["signal_title"],
            "matched_intent_category": tr["matched_intent_category"],
            ("signal_category" if company else "signal_tier"): tr["category"],
            "date_or_timeframe": _timeframe(cited),
            "confidence_score": tr["confidence_score"],
            "evidence_bullets": [b for b in tr.get("evidence_bullets", []) if b.strip()][:3],
            "source_type": tr["source_type"],
            "source_url": next((s["source_url"] for s in cited if s["source_url"]), ""),
        }
        if any(s.get("user_provided") for s in cited):
            item["from_social_activity"] = True
        if company:
            execs, seen = [], set()
            for e in tr.get("executives", []):
                p = by_person.get(e.get("person_id"))
                if not p or p["id"] in seen:
                    continue  # unknown id = invented person
                seen.add(p["id"])
                execs.append({"executive_name": p["name"], "designation": p["title"], "linkedin_url": p["linkedin_url"],
                              "ownership_rationale": e["ownership_rationale"], "outreach_angle": e["outreach_angle"]})
            item["recommended_executives"] = execs[:3]
        else:
            # Tier 1 needs a cited source that actually names the executive.
            if tr["category"] == EXEC_TIERS[0] and not any(
                    s.get("user_provided") or last and last in json.dumps([s.get("evidence"), s.get("title"), s.get("people")]).lower()
                    for s in cited):
                item["signal_tier"] = EXEC_TIERS[1]
            item["outreach_angle"] = tr.get("outreach_angle", "")
        out.append(item)

    if not company:  # strict tier order, model order within a tier
        out.sort(key=lambda x: EXEC_TIERS.index(x["signal_tier"]))
    out = out[:TOP_N]
    for i, x in enumerate(out, 1):
        x["rank"] = i

    score = ranked.get("overall_intent_score", "Low") if out else "Low"
    narrative = ranked.get("strategic_narrative", "") if out else \
        "No verified, recent signals matched the focus department, technologies or intent criteria."
    if company:
        result = {"intent_mode": "COMPANY_BASED",
                  "account_summary": {"company_name": t["company_name"], "company_domain": t["company_domain"],
                                      "focus_department": t["focus_department"], "overall_intent_score": score,
                                      "strategic_narrative": narrative},
                  "top_5_triggers": out}
        summary = result["account_summary"]
    else:
        result = {"intent_mode": "EXEC_BASED",
                  "executive_summary": {"company_name": t["company_name"], "company_domain": t["company_domain"],
                                        "executive_name": t["executive_name"],
                                        "executive_designation": t["executive_designation"],
                                        "overall_intent_score": score, "strategic_narrative": narrative},
                  "top_5_mapped_signals": out}
        summary = result["executive_summary"]
    ids = t.get("_identity") or {}
    summary["positioning"] = {"seller_company": t["seller_company"], "seller_offerings": t["seller_offerings"]}
    summary["company_identity"] = {k: ids.get(k, "") for k in ("event", "event_date", "alternate_names", "parent_company",
                                                              "parent_domain", "source_url")} if ids else {}
    summary["technology_usage"] = usage or []
    summary["social_activity_items"] = len(t["social_activity"])
    return result


def _resolve_links(result: dict) -> dict:
    items = result.get("top_5_triggers") or result.get("top_5_mapped_signals") or []
    tech = (result.get("account_summary") or result.get("executive_summary") or {}).get("technology_usage") or []
    urls = {x["source_url"] for x in [*items, *tech] if _REDIRECT_HOST in (x.get("source_url") or "")}
    if urls:
        with ThreadPoolExecutor(max_workers=8) as pool:
            resolved = dict(zip(urls, pool.map(_resolve_redirect, urls)))
        for x in [*items, *tech]:
            x["source_url"] = resolved.get(x["source_url"], x["source_url"])
    return result


# ── Campaign handoff (outlook-campaigns lead shape) ──────────────────────────
def _split_name(full: str) -> tuple[str, str]:
    parts = full.split()
    return (parts[0], " ".join(parts[1:])) if parts else ("", "")


def _lead_data(item: dict, score: str, hook: str) -> dict:
    return {"intent_trigger": item["signal_title"], "intent_hook": hook,
            "intent_evidence": (item["evidence_bullets"] or [""])[0], "intent_category": item["matched_intent_category"],
            "intent_date": item["date_or_timeframe"], "intent_rank": item["rank"], "intent_score": score}


def campaign_leads(t: dict, result: dict) -> list[dict]:
    """One lead per executive, carrying their best trigger. Email is left blank:
    the engine never guesses addresses; the campaign app fills or matches it."""
    if result["intent_mode"] == "COMPANY_BASED":
        score = result["account_summary"]["overall_intent_score"]
        leads, seen = [], set()
        for item in result["top_5_triggers"]:
            for e in item["recommended_executives"]:
                if e["executive_name"].lower() in seen:
                    continue
                seen.add(e["executive_name"].lower())
                first, last = _split_name(e["executive_name"])
                leads.append({"email": "", "first_name": first, "last_name": last, "company": t["company_name"],
                              "title": e["designation"],
                              "linkedin_url": "" if e["linkedin_url"] == "N/A" else e["linkedin_url"],
                              "data": _lead_data(item, score, e["outreach_angle"])})
        return leads
    items = result["top_5_mapped_signals"]
    if not items:
        return []
    first, last = _split_name(t["executive_name"])
    data = _lead_data(items[0], result["executive_summary"]["overall_intent_score"], items[0]["outreach_angle"])
    data["intent_signals"] = " | ".join(x["signal_title"] for x in items)
    return [{"email": "", "first_name": first, "last_name": last, "company": t["company_name"],
             "title": t["executive_designation"], "linkedin_url": t["executive_linkedin_url"], "data": data}]


# ── Email sequences ──────────────────────────────────────────────────────────
def _recency(item: dict) -> tuple:
    """Newest first. A live programme counts as recent but after anything dated
    in the last two months; rank breaks ties."""
    d = _parse_date(item["date_or_timeframe"])
    if d is None:
        d = date.today() - timedelta(days=60)
    return (-d.toordinal(), item["rank"])


def _lead_triggers(result: dict, lead: dict) -> list[dict]:
    items = result.get("top_5_triggers") or result.get("top_5_mapped_signals") or []
    name = f"{lead['first_name']} {lead['last_name']}".strip().lower()
    if result["intent_mode"] == "COMPANY_BASED":
        own = [x for x in items if any(e["executive_name"].lower() == name for e in x["recommended_executives"])]
        # Their own triggers first (newest first), then other account triggers as extra follow-ups.
        return (sorted(own, key=_recency) + sorted([x for x in items if x not in own], key=_recency))[:EMAILS_PER_LEAD]
    return sorted(items, key=_recency)[:EMAILS_PER_LEAD]


def _angle(item: dict, name: str) -> str:
    for e in item.get("recommended_executives") or []:
        if e["executive_name"].lower() == name:
            return e["outreach_angle"]
    return item.get("outreach_angle") or (item.get("evidence_bullets") or [""])[0]


def _fallback_email(lead: dict, item: dict | None, k: int, t: dict) -> dict:
    first = lead["first_name"] or "there"
    offer = _offer(t)
    if item is None:
        body = (f"Hi {first},\n\nLast note from me. If {offer} isn't a priority for {t['company_name']} right now, "
                "just reply \"not now\" and I'll close the loop.")
        return {"subject": "", "body": body, "trigger": ""}
    angle = _angle(item, f"{lead['first_name']} {lead['last_name']}".strip().lower())
    lead_in = "I noticed" if k == 1 else "Following up — I also saw"
    body = f"Hi {first},\n\n{lead_in}: {item['signal_title']}.\n\n{angle}\n\nWorth a 15-minute call?"
    return {"subject": item["signal_title"][:70] if k == 1 else "", "body": body, "trigger": item["signal_title"]}


def add_email_sequences(t: dict, result: dict, leads: list[dict], run_id: str) -> list[dict]:
    """Writes EMAILS_PER_LEAD emails per lead. Email 1 is built on the lead's most
    recent trigger; each follow-up on the next older one; with fewer triggers
    the last email is a short close-the-loop note. Stored in lead.data as
    intent_email_{k}_subject / _body / _trigger for the campaign templates."""
    import anthropic

    plans = []
    for i, lead in enumerate(leads):
        trig = _lead_triggers(result, lead)
        name = f"{lead['first_name']} {lead['last_name']}".strip().lower()
        plans.append({"lead_id": f"L{i + 1}", "first_name": lead["first_name"], "title": lead["title"],
                      "emails": [{"email": k + 1, "trigger": x["signal_title"], "date": x["date_or_timeframe"],
                                  "evidence": x["evidence_bullets"], "angle": _angle(x, name)}
                                 if k < len(trig) else {"email": k + 1, "trigger": None}
                                 for k, x in enumerate(trig + [None] * (EMAILS_PER_LEAD - len(trig)))],
                      "_trig": trig})
    system = f"""You write cold B2B email sequences for {_offer(t)}, to people at {t['company_name']}.
For each lead write exactly {EMAILS_PER_LEAD} emails, in order, each built ONLY on the trigger given for that
email number (email 1 = the most recent trigger; follow-ups = older triggers). When an email's trigger is
null, write a 2–3 sentence polite close-the-loop note that references nothing new.
Rules: plain text; start "Hi <first_name>," on its own line, then a blank line; 2–3 short paragraphs separated by blank lines; 50–110 words; one concrete reference to the trigger using only
facts in its evidence; position only {_offer(t)}; one soft call to action; no signature, no placeholders,
no links, no invented facts, numbers or names. Follow-ups are replies in the same thread: don't re-introduce
yourself, open with a fresh angle. subject: only for email 1 (max 7 words, no clickbait); empty for others."""
    schema = {"type": "object", "additionalProperties": False, "required": ["leads"], "properties": {"leads": {
        "type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["lead_id", "emails"],
                                   "properties": {"lead_id": {"type": "string"}, "emails": {"type": "array", "items": {
                                       "type": "object", "additionalProperties": False, "required": ["subject", "body"],
                                       "properties": {"subject": {"type": "string"}, "body": {"type": "string"}}}}}}}}}
    written = {}
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=180.0)
        resp = client.messages.create(
            model=CLAUDE_MODEL, max_tokens=8000, system=system,
            messages=[{"role": "user", "content": json.dumps([{k: v for k, v in p.items() if k != "_trig"} for p in plans], indent=1)}],
            output_config={"format": {"type": "json_schema", "schema": schema}})
        log_claude_usage(MODULE, f"engine|emails|{t['company_name'][:20]}", resp, run_id=run_id)
        if resp.stop_reason == "end_turn":
            payload = json.loads(next(b.text for b in resp.content if b.type == "text"))
            written = {x["lead_id"]: x["emails"] for x in payload.get("leads", [])}
    except Exception as e:
        logger.warning(f"Intent Engine email writing failed for {t['company_name']}: {e!r}")

    for p, lead in zip(plans, leads):
        trig = p["_trig"]
        got = written.get(p["lead_id"]) or []
        seq = []
        for k in range(EMAILS_PER_LEAD):
            item = trig[k] if k < len(trig) else None
            fb = _fallback_email(lead, item, k + 1, t)
            w = got[k] if k < len(got) and got[k].get("body", "").strip() else None
            if w:  # greeting on its own line, even if the model ran it into the first sentence
                w = {**w, "body": re.sub(r"^(Hi [^,\n]{1,40},)[ \t]+", r"\1\n\n", w["body"].strip())}
            subject = ((w or {}).get("subject") or "").strip() or fb["subject"] if k == 0 else ""
            seq.append({"subject": subject or f"{t['company_name']} — quick idea", "body": (w or fb)["body"].strip(),
                        "trigger": fb["trigger"]} if k == 0 else
                       {"subject": "", "body": (w or fb)["body"].strip(), "trigger": fb["trigger"]})
        lead["email_sequence"] = seq
        for k, e in enumerate(seq, 1):
            lead["data"][f"intent_email_{k}_body"] = e["body"]
            lead["data"][f"intent_email_{k}_trigger"] = e["trigger"]
        lead["data"]["intent_email_1_subject"] = seq[0]["subject"]
    return leads


# ── Runner ───────────────────────────────────────────────────────────────────
def run_target_sync(t: dict, run_id: str, force_refresh: bool = False, status_cb=lambda s: None) -> dict:
    cache = _Cache()
    key = _cache_key(t)
    if not force_refresh and (hit := cache.get(key)):
        return hit
    status_cb("checking identity")
    t = {**t, "_identity": resolve_identity(t, run_id)}
    status_cb("researching")
    raw_signals, raw_people, raw_usage = research(t, run_id)
    signals, people = gate_signals(raw_signals + social_signals(t), t), gate_people(raw_people)
    usage = gate_usage(t, raw_usage)
    status_cb("ranking")
    ranked = rank(t, signals, people, usage, run_id) if signals else {}
    result = _resolve_links(assemble(t, signals, people, ranked, usage))
    leads = campaign_leads(t, result)
    if leads:
        status_cb("writing emails")
        leads = add_email_sequences(t, result, leads, run_id)
    result["campaign_leads"] = leads
    result["generated_at"] = _now()
    cache.put(key, result)
    return result


async def run_targets(targets: list[dict], force_refresh: bool = False):
    run_id = new_run_id()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    _DONE = object()

    yield {"type": "run_started", "run_id": run_id, "total": len(targets)}

    def label(t):
        return f'{t["company_name"]} · {t["executive_name"]}' if t["intent_mode"] == "EXEC_BASED" else t["company_name"]

    async def _pump(t):
        name = label(t)

        def status_cb(status):
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "target_status", "target": name, "status": status})
        try:
            async with sem:
                result = await asyncio.wait_for(asyncio.to_thread(run_target_sync, t, run_id, force_refresh, status_cb),
                                                timeout=TARGET_TIMEOUT)
            await queue.put({"type": "target_result", "target": name, "result": result})
        except Exception as e:
            logger.error(f"Intent Engine {name} failed: {e!r}")
            msg = "timed out" if isinstance(e, asyncio.TimeoutError) else f"{type(e).__name__}: {e}"
            await queue.put({"type": "target_error", "target": name, "message": msg})
        finally:
            await queue.put(_DONE)

    tasks = [asyncio.ensure_future(_pump(t)) for t in targets]
    remaining, t0, results = len(tasks), time.time(), []
    while remaining:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=20)
        except asyncio.TimeoutError:
            yield {"type": "heartbeat", "message": f"⏳ Researching… ({int(time.time() - t0)}s)"}
            continue
        if item is _DONE:
            remaining -= 1
            continue
        if item["type"] == "target_result":
            results.append(item["result"])
        yield item

    usage = await asyncio.to_thread(get_usage_by_run, run_id)
    yield {"type": "complete", "run_id": run_id, "usage": usage, "total": len(results)}
