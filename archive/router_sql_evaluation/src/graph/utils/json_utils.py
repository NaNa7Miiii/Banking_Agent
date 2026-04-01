from __future__ import annotations

import json
import re
from typing import Any, Dict


class JsonParseError(ValueError):
    pass


def strip_json_fence(text: str) -> str:
    cleaned = (text or "").strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    return cleaned


def _extract_first_json_object(text: str) -> str:
    s = (text or "").strip()
    start = s.find("{")
    if start < 0:
        raise JsonParseError("No JSON object found (missing '{').")

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    raise JsonParseError("No complete JSON object found (unmatched braces).")


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def safe_json_loads(raw: str) -> Dict[str, Any]:
    cleaned = strip_json_fence(raw)

    try:
        parsed = json.loads(cleaned)
    except Exception:
        cleaned2 = _remove_trailing_commas(cleaned)
        try:
            parsed = json.loads(cleaned2)
        except Exception:
            obj = _extract_first_json_object(cleaned2)
            obj2 = _remove_trailing_commas(obj)
            try:
                parsed = json.loads(obj2)
            except Exception as e:
                raise JsonParseError(f"Invalid JSON from router: {e}") from e

    if not isinstance(parsed, dict):
        raise JsonParseError("Router output JSON must be an object at top-level.")

    return parsed
