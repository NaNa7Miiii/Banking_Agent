from __future__ import annotations

import re


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is unsafe to execute."""

_FORBIDDEN_STMTS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)

_FORBIDDEN_COLUMN = re.compile(r"\bis_fraud\b", re.IGNORECASE)


def clean_sql(text: str) -> str:
    q = (text or "").strip()

    if q.startswith("```"):
        lines = q.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        q = "\n".join(lines).strip()

    if q.lower().startswith("sql:"):
        q = q[4:].strip()

    return q.strip()


def ensure_limit(query: str, default_limit: int = 50) -> str:
    """
    Ensure the SQL query includes a LIMIT clause.
    If missing, append 'LIMIT <default_limit>'.
    """
    q = query.strip()
    if " limit " in q.lower():
        return q
    return f"{q} LIMIT {int(default_limit)}"


def validate_select_only(query: str) -> None:
    """
    Enforce safe SQL:
    - single statement only (no semicolons)
    - SELECT-only
    - forbid dangerous statements
    - forbid referencing is_fraud anywhere
    """
    q = (query or "").strip()

    if not q:
        raise UnsafeSQLError("Empty SQL query.")

    if ";" in q:
        raise UnsafeSQLError("Semicolons are not allowed (single statement only).")

    if _FORBIDDEN_STMTS.search(q):
        raise UnsafeSQLError("Non-SELECT statements are not allowed.")

    if not q.lower().startswith("select"):
        raise UnsafeSQLError("Only SELECT queries are allowed.")

    if _FORBIDDEN_COLUMN.search(q):
        raise UnsafeSQLError("Forbidden column referenced: is_fraud")

