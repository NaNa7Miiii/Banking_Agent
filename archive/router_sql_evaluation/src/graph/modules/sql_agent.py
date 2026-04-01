from __future__ import annotations

import os
import json
import re
from datetime import datetime
from typing import Any, Dict, Optional, List

from sqlalchemy import inspect, text

from src.data.utils import get_sql_db
from src.graph.models.llm import get_llm
from src.graph.prompts.sql_prompts import create_answer_prompt, create_sql_prompt
from src.graph.utils.sql_safety import (
    UnsafeSQLError,
    clean_sql,
    ensure_limit,
    validate_select_only,
)


def _sql_logging_enabled() -> bool:
    return os.getenv("SQL_LOGGING", "NO").upper() == "YES"


def _log(question: str, query: str, mode: str, note: Optional[str] = None) -> None:
    if not _sql_logging_enabled():
        return
    os.makedirs("logs", exist_ok=True)
    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "query": query,
    }
    if note:
        record["note"] = note
    if os.getenv("SQL_LOG_QUESTION", "NO").upper() == "YES":
        record["question"] = question

    with open("logs/sql_agent_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _allow_llm_calls() -> bool:
    return os.getenv("ALLOW_LLM_CALLS", "NO").upper() == "YES"


def _sql_model_from_env(default: str = "gpt-4.1-mini") -> str:
    return (os.getenv("SQL_LLM_MODEL", "") or default).strip()


def _sql_temp_from_env(default: float = 0.0) -> float:
    v = os.getenv("SQL_TEMPERATURE", "").strip()
    if not v:
        return default
    try:
        return float(v)
    except Exception:
        return default


def _sql_default_limit_from_env(default: int = 50) -> int:
    v = os.getenv("SQL_DEFAULT_LIMIT", "").strip()
    if not v:
        return default
    try:
        n = int(v)
        return n if n > 0 else default
    except Exception:
        return default


def _format_schema(columns_info, table_name: str) -> str:
    lines = [f"Table: {table_name}", "-" * (7 + len(table_name))]
    for col in columns_info:
        name = col["name"]
        col_type = str(col["type"])
        nullable = col.get("nullable", True)
        nullable_str = "NULL" if nullable else "NOT NULL"
        default = col.get("default")
        default_str = f", DEFAULT {default}" if default is not None else ""
        lines.append(f"{name} {col_type} {nullable_str}{default_str}")
    return "\n".join(lines)


def _get_schema_text(db, table_name: str) -> str:
    inspector = inspect(db._engine)
    cols = inspector.get_columns(table_name)
    return _format_schema(cols, table_name)


def _fallback_query(table_name: str) -> str:
    return (
        f"SELECT transaction_datetime, merchant_name, merchant_category, "
        f"transaction_amount, transaction_id "
        f"FROM {table_name} "
        f"ORDER BY transaction_datetime DESC "
        f"LIMIT 20"
    )


def _drop_trailing_semicolon(sql: str) -> str:
    s = (sql or "").strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _remove_duplicate_limits(sql: str) -> str:
    s = (sql or "").strip()
    parts = re.split(r"(?i)\blimit\b", s)
    if len(parts) <= 2:
        return s
    base = parts[0].rstrip()
    last = parts[-1].strip()
    s2 = f"{base} LIMIT {last}"
    return s2.strip()


def _clamp_limit(sql: str, max_limit: int) -> str:
    m = re.search(r"(?i)\blimit\s+(\d+)\b", sql)
    if not m:
        return sql
    n = int(m.group(1))
    if n <= max_limit:
        return sql
    return re.sub(r"(?i)\blimit\s+\d+\b", f"LIMIT {max_limit}", sql, count=1)


class SQLAgent:
    def __init__(
        self,
        db=None,
        table_name: str = "transactions",
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        required_columns: Optional[List[str]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[str] = None,
        database: Optional[str] = None,
    ):
        if db is None and all([username, password, host, port, database]):
            db = get_sql_db(
                username=username,
                password=password,
                host=host,
                port=port,
                database=database,
                echo=False,
            )

        if db is None:
            raise RuntimeError("SQLAgent init failed: db is None (provide db or full DB credentials).")

        if not _allow_llm_calls():
            raise RuntimeError("ALLOW_LLM_CALLS is NO, but this SQLAgent requires LLM calls.")

        self.db = db
        self.table_name = table_name

        self.llm_model = llm_model or _sql_model_from_env()
        self.temperature = _sql_temp_from_env() if temperature is None else float(temperature)
        self.default_limit = _sql_default_limit_from_env()

        self.schema_text = _get_schema_text(self.db, self.table_name)

        self.sql_prompt = create_sql_prompt(required_columns=required_columns)
        self.answer_prompt = create_answer_prompt()

        self.llm = get_llm(
            role="sql",
            model_name=self.llm_model,
            temperature=self.temperature,
            use_langchain=True,
        )

    def _postprocess_sql(self, raw_text: str) -> str:
        sql = clean_sql(raw_text)
        sql = _drop_trailing_semicolon(sql)
        sql = _remove_duplicate_limits(sql)

        validate_select_only(sql)

        sql = ensure_limit(sql, default_limit=self.default_limit)
        sql = _drop_trailing_semicolon(sql)
        sql = _remove_duplicate_limits(sql)
        sql = _clamp_limit(sql, max_limit=self.default_limit)

        validate_select_only(sql)
        return sql.strip()

    def _generate_sql_once(self, question: str) -> str:
        raw_sql = (self.sql_prompt | self.llm).invoke({"schema": self.schema_text, "question": question})
        return self._postprocess_sql(str(raw_sql))

    def generate_sql(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return _fallback_query(self.table_name)

        try:
            sql1 = self._generate_sql_once(q)
            _log(q, sql1, "LLM")
            return sql1
        except Exception as e1:
            _log(q, "", "LLM", note=f"gen_fail_1={type(e1).__name__}")

        q2 = q + "\n\n(Important: output exactly ONE PostgreSQL SELECT statement, no semicolon.)"
        try:
            sql2 = self._generate_sql_once(q2)
            _log(q, sql2, "LLM", note="retry_2")
            return sql2
        except Exception as e2:
            _log(q, "", "LLM", note=f"gen_fail_2={type(e2).__name__}")

        fb = _fallback_query(self.table_name)
        _log(q, fb, "FALLBACK", note="fallback_query_used")
        return fb

    def run_query(self, query: str) -> Any:
        q = (query or "").strip()
        if not q:
            q = _fallback_query(self.table_name)

        if hasattr(self.db, "run"):
            return self.db.run(q)

        with self.db._engine.connect() as conn:
            res = conn.execute(text(q))
            return [dict(r._mapping) for r in res]

    def invoke(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "No question provided."

        sql = self.generate_sql(q)

        try:
            rows = self.run_query(sql)
        except Exception as e:
            fb = _fallback_query(self.table_name)
            rows = self.run_query(fb)
            sql = fb
            _log(q, fb, "FALLBACK", note=f"exec_fail={type(e).__name__}")

        try:
            ans = (self.answer_prompt | self.llm).invoke(
                {"schema": self.schema_text, "question": q, "query": sql, "response": rows}
            )
            out_text = ans.content if hasattr(ans, "content") else str(ans)
            _log(q, sql, "LLM", note="answer_ok")
            return out_text
        except Exception as e:
            _log(q, sql, "LLM", note=f"answer_fail={type(e).__name__}")
            return "Generated SQL and executed it, but failed to produce a natural language answer."


def create_sql_agent_node(
    db,
    table_name: str,
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    agent = SQLAgent(
        db=db,
        table_name=table_name,
        llm_model=llm_model,
        temperature=temperature,
    )

    def sql_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get("question", "")
        if not question:
            return {
                **state,
                "schema": agent.schema_text,
                "query": "",
                "response": None,
                "answer": "No question provided.",
            }

        sql = agent.generate_sql(question)
        resp = agent.run_query(sql)
        ans = agent.invoke(question)

        return {
            **state,
            "schema": agent.schema_text,
            "query": sql,
            "response": resp,
            "answer": ans,
        }

    return sql_agent_node


def create_transaction_sql_agent_node(
    username: str,
    password: str,
    host: str,
    port: str,
    database: str,
    table_name: str = "transactions",
    llm_model: Optional[str] = None,
    temperature: Optional[float] = None,
):
    db = get_sql_db(
        username=username,
        password=password,
        host=host,
        port=port,
       database=database,
        echo=False,
    )
    return create_sql_agent_node(
        db=db,
        table_name=table_name,
        llm_model=llm_model,
        temperature=temperature,
    )