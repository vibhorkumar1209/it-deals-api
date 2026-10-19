"""Surface Gemini billing/quota failures to the user.

Pipelines swallow per-call errors, so a scan during an outage ends as a
normal "complete" with empty results. One hook on google-genai records
billing/quota failures for every module, and an ASGI middleware appends an
"API issue" error event to any SSE stream that hit one.
"""

from __future__ import annotations

import json
import threading
import time

API_ISSUE_MESSAGE = "API issue - run again in some time"

_lock = threading.Lock()
_state = {"fail_ts": 0.0, "kind": "", "ok_ts": 0.0}


def _classify(err: Exception) -> str:
    s = str(err)
    if "402" in s or "Payment Required" in s or "billing" in s.lower():
        return "billing"
    if "RESOURCE_EXHAUSTED" in s or "429" in s or "quota" in s.lower():
        return "rate"
    return ""


def _record(kind: str):
    with _lock:
        now = time.time()
        if kind:
            _state["fail_ts"], _state["kind"] = now, kind
        else:
            _state["ok_ts"] = now


def failed_since(start: float) -> bool:
    """Billing failures always count; rate/quota failures only if no Gemini
    call has succeeded since (a retried 429 that then worked is not an issue)."""
    with _lock:
        if _state["fail_ts"] < start:
            return False
        return _state["kind"] == "billing" or _state["ok_ts"] < _state["fail_ts"]


def install_gemini_hook():
    from google.genai import models

    original = models.Models.generate_content
    if getattr(original, "_api_health", False):
        return

    def generate_content(self, *args, **kwargs):
        try:
            resp = original(self, *args, **kwargs)
        except Exception as e:
            _record(_classify(e))
            raise
        _record("")
        return resp

    generate_content._api_health = True
    models.Models.generate_content = generate_content


_ERR_EVENT = f"data: {json.dumps({'type': 'error', 'message': API_ISSUE_MESSAGE})}\n\n".encode()


class ApiIssueMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("method") != "POST":
            return await self.app(scope, receive, send)
        start = time.time()
        state = {"sse": False, "done": False}

        async def _send(msg):
            if msg["type"] == "http.response.start":
                headers = dict(msg.get("headers") or [])
                state["sse"] = b"text/event-stream" in headers.get(b"content-type", b"")
            elif msg["type"] == "http.response.body" and state["sse"] and not state["done"]:
                body, more = msg.get("body", b""), msg.get("more_body", False)
                is_complete = b'"type": "complete"' in body or b'"type":"complete"' in body
                if (is_complete or not more) and failed_since(start):
                    state["done"] = True
                    # After "complete", so it is the last status the UI shows.
                    await send({"type": "http.response.body", "body": body + _ERR_EVENT, "more_body": more})
                    return
            await send(msg)

        await self.app(scope, receive, _send)
