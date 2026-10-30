"""Outside callers (the Claude MCP, scripts, other servers) get the output only.

No sources, citations, grounding metadata, search queries, scoring rationale,
model or provider names, token usage, and no diagnostic endpoints. The app
itself still gets the full payload, because its pages show source links.

"The app" means a request whose Origin or Referer is the it-deals frontend
(browser fetches), or one carrying the INTERNAL_API_TOKEN in X-Internal-Token
(the frontend's server-side calls, ops scripts). Origin can be forged by a
caller that knows to; it keeps casual API and MCP use output-only.
"""
from __future__ import annotations

import json
import os
import re
from urllib.parse import urlparse

_APP_HOST = re.compile(r"^(localhost|127\.0\.0\.1|it-deals-frontend[a-z0-9-]*\.vercel\.app)$", re.I)

# Diagnostic and spend endpoints: raw prompts, model usage, cached source lists.
_INTERNAL_ONLY = re.compile(r"^/api/(usage|debug|fetch-test|cached-urls)(/|-|$)")

# Fields that describe how an insight was built rather than the insight itself.
_REDACT_KEYS = {
    "source", "sources", "_sources", "source_url", "source_urls", "all_source_urls", "source_type",
    "source_label", "source_info", "sources_attempted", "known_sources", "evidence_sources",
    "evidence_kind", "market_share_source", "url", "urls", "url_results", "urls_discovered",
    "urls_attempted", "url_count", "citations", "references", "grounding_sources",
    "grounding_metadata", "grounding_chunks", "query", "queries", "search_queries", "method",
    "methodology", "model", "provider", "usage", "usage_metadata", "finish_reason", "raw_preview",
    "raw_chars", "reasoning", "rationale", "importance_rationale", "tam_rationale",
    "opportunity_rationale", "_confidence_override", "paraphrased", "weights", "weight", "rubric", "points", "contribution", "level_score",
    "signal_ids",
}

_INLINE_CITATIONS = [
    re.compile(r"\s*\[(?:\d+(?:,\s*\d+)*)\]"),
    re.compile(r"\s*\(\s*(?:source|sources|via|per|according to)\b[^()]*\)", re.I),
    re.compile(r"\s*\(?https?://[^\s)]+\)?"),
]

_PROVIDERS = re.compile(
    r"\s*\b(?:(?:via|from|on|using|with|per|by)\s+)?(?:Gemini(?: 2\.5)?(?: Flash| Pro)?|Claude(?: Sonnet| Opus| Haiku)?(?: \d+(?:\.\d+)?)?"
    r"|Sonnet(?: \d+(?:\.\d+)?)?|Anthropic|Google Search(?: grounding)?|Parallel(?:\.ai)?|scrape\.do|OpenAI)(?!\w)",
    re.I,
)


def _host(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return urlparse(value).hostname
    except ValueError:
        return None


def is_internal(headers: dict[str, str]) -> bool:
    token = os.getenv("INTERNAL_API_TOKEN")
    if token and headers.get("x-internal-token") == token:
        return True
    extra = [_host(o.strip()) for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
    for h in (headers.get("origin"), headers.get("referer")):
        host = _host(h)
        if host and (_APP_HOST.match(host) or host in extra):
            return True
    return False


def _mcp_key_ok(headers: dict[str, str]) -> bool:
    keys = {k.strip() for k in os.getenv("MCP_API_KEYS", "").split(",") if k.strip()}
    sent = headers.get("x-api-key") or headers.get("authorization", "").removeprefix("Bearer ").strip()
    return bool(keys) and sent in keys


def clean_text(text: str) -> str:
    for rx in _INLINE_CITATIONS:
        text = rx.sub("", text)
    text = _PROVIDERS.sub("", text)
    return re.sub(r" {2,}", " ", re.sub(r" +([.,;:])", r"\1", text)).strip() if text.strip() else text


def redact(node):
    if isinstance(node, str):
        return clean_text(node)
    if isinstance(node, list):
        return [redact(v) for v in node]
    if not isinstance(node, dict):
        return node
    out = {}
    for k, v in node.items():
        if k in _REDACT_KEYS or k.endswith("_evidence_urls"):
            continue
        if k in ("run_id", "id", "domain", "website") or k.endswith("linkedin_url"):
            out[k] = v
            continue
        if k == "message" and node.get("type") == "error":
            out[k] = "The request could not be completed. Run it again in some time."
            continue
        out[k] = redact(v)
    return out


def _redact_sse(text: str) -> str:
    def sub(m):
        try:
            return "data: " + json.dumps(redact(json.loads(m.group(1))))
        except ValueError:
            return m.group(0)
    return re.sub(r"^data: (.*)$", sub, text, flags=re.M)


async def _send_json(send, status: int, body: bytes):
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


class ExternalRedactionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode(errors="ignore") for k, v in scope.get("headers", [])}
        if is_internal(headers):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        # Outside callers (MCP, scripts) need a key from MCP_API_KEYS; /health stays open
        # for Render's checks. Unset MCP_API_KEYS keeps the API open (output-only).
        if path != "/health" and os.getenv("MCP_API_KEYS") and not _mcp_key_ok(headers):
            return await _send_json(send, 401, b'{"detail":"Missing or invalid API key"}')
        if _INTERNAL_ONLY.match(path):
            return await _send_json(send, 404, b'{"detail":"Not Found"}')

        state = {"kind": None, "start": None, "buf": b""}

        async def wrapped(message):
            if message["type"] == "http.response.start":
                ctype = dict((k.decode().lower(), v.decode()) for k, v in message.get("headers", [])).get("content-type", "")
                status = message.get("status", 200)
                if "text/event-stream" in ctype:
                    state["kind"] = "sse"
                elif "application/json" in ctype and status < 400:
                    state["kind"] = "json"
                    state["start"] = message  # sent once the full body is known (length changes)
                    return
                return await send(message)
            if message["type"] != "http.response.body":
                return await send(message)
            if state["kind"] == "sse":
                body = message.get("body", b"")
                if body:
                    message = {**message, "body": _redact_sse(body.decode("utf-8", "ignore")).encode()}
                return await send(message)
            if state["kind"] == "json":
                state["buf"] += message.get("body", b"")
                if message.get("more_body"):
                    return
                try:
                    body = json.dumps(redact(json.loads(state["buf"]))).encode()
                except ValueError:
                    body = state["buf"]
                start = state["start"]
                hdrs = [(k, v) for k, v in start.get("headers", []) if k.lower() != b"content-length"]
                hdrs.append((b"content-length", str(len(body)).encode()))
                await send({**start, "headers": hdrs})
                return await send({"type": "http.response.body", "body": body})
            return await send(message)

        return await self.app(scope, receive, wrapped)
