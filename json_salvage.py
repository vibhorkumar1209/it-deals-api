"""Lenient JSON parsing for LLM replies that were cut off at the output cap,
contain a malformed element, or are wrapped in prose."""

from __future__ import annotations

import json
import re

_TRAILING_COMMA = re.compile(r",\s*([\]}])")


def _first_container(text: str) -> int:
    idx = [i for i in (text.find("["), text.find("{")) if i != -1]
    return min(idx) if idx else -1


def repair_truncated(text: str):
    """Parse the first JSON container in text. If it never closes (truncated),
    cut back to the last complete element and close the open brackets."""
    start = _first_container(text)
    if start == -1:
        return None
    stack, in_str, esc = [], False, False
    last_safe, safe_stack = -1, []
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if not stack:
                break
            stack.pop()
            if not stack:
                try:
                    return json.loads(_TRAILING_COMMA.sub(r"\1", text[start:i + 1]))
                except json.JSONDecodeError:
                    return None
            last_safe, safe_stack = i + 1, list(stack)
    if last_safe == -1:
        return None
    body = text[start:last_safe].rstrip().rstrip(",")
    try:
        return json.loads(_TRAILING_COMMA.sub(r"\1", body + "".join(reversed(safe_stack))))
    except json.JSONDecodeError:
        return None


def salvage_objects(text: str) -> list[dict]:
    """Decode complete objects one by one from the first array in text —
    survives truncation and missing commas between objects."""
    dec, out = json.JSONDecoder(), []
    i = text.find("[")
    if i == -1:
        return out
    i += 1
    while True:
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text) or text[i] != "{":
            return out
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            return out
        if isinstance(obj, dict):
            out.append(obj)


def parse_lenient(text: str):
    """Best-effort parse: strict → repaired truncation → per-object salvage.
    Returns a list/dict, or None if nothing usable."""
    text = re.sub(r"```(?:json)?\s*", "", (text or "").strip())
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    repaired = repair_truncated(text)
    if repaired is not None:
        return repaired
    objs = salvage_objects(text)
    return objs or None
