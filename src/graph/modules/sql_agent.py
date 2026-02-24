from __future__ import annotations

import os
from typing import Any

from sqlalchemy import inspect

from src.data.utils import get_sql_db
from src.graph.models.llm import get_llm
from src.graph.prompts.sql_prompts_v1 import create_answer_prompt, create_sql_prompt
from src.graph.utils.sql_safety import (
    UnsafeSQLError,
    clean_sql,
    ensure_limit,
    validate_select_only,
)


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


def _use_llm_sql() -> bool:
    """
    Keep it simple:
    - default: RULE (no LLM)
    - if SQL_MODE=LLM and ALLOW_LLM_CALLS=YES -> use LLM
    """
    sql_mode = os.getenv("SQL_MODE", "RULE").upper()
    allow_llm = os.getenv("ALLOW_LLM_CALLS", "NO").upper()
    return sql_mode == "LLM" and allow_llm == "YES"


def _run_rule_sql(db, table_name: str) -> dict:
    sql_query = f"SELECT * FROM {table_name} LIMIT 20;"
    sql_result = db.run(sql_query)
    return {
        "schema": "",
        "query": sql_query,
        "response": sql_result,
        "answer": str(sql_result),
    }


def _run_llm_sql(
    db,
    table_name: str,
    question: str,
    llm_model: str,
    temperature: float,
    default_limit: int,
    required_columns=None,
) -> dict:
    schema_text = _get_schema_text(db, table_name)

    llm = get_llm(
        role="sql",
        model_name=llm_model,
        temperature=temperature,
        use_langchain=True,
    )

    sql_prompt = create_sql_prompt(required_columns=required_columns)
    answer_prompt = create_answer_prompt()

    raw_sql = (sql_prompt | llm).invoke({"schema": schema_text, "question": question})
    if not isinstance(raw_sql, str):
        raw_sql = str(raw_sql)

    sql_query = clean_sql(raw_sql)

    validate_select_only(sql_query)
    sql_query = ensure_limit(sql_query, default_limit=default_limit)

    sql_result = db.run(sql_query)

    answer = (answer_prompt | llm).invoke(
        {
            "schema": schema_text,
            "question": question,
            "query": sql_query,
            "response": sql_result,
        }
    )
    if not isinstance(answer, str):
        answer = str(answer)

    return {
        "schema": schema_text,
        "query": sql_query,
        "response": sql_result,
        "answer": answer,
    }


def create_sql_agent_node(
    db,
    table_name: str,
    llm_model: str = "gpt-4.1",
    temperature: float = 0.0,
    default_limit: int = 50,
    required_columns=None,
):
    if db is None:

        def sql_agent_node(state: dict) -> dict:
            question = state.get("question", "")
            if not question:
                return {
                    **state,
                    "answer": "No question provided.",
                    "schema": "",
                    "query": "",
                    "response": None,
                }

            return {
                **state,
                "schema": "",
                "query": "",
                "response": None,
                "answer": (
                    "Database connection is currently unavailable. "
                    "Please check DB status and network settings."
                ),
            }

        return sql_agent_node

    def sql_agent_node(state: dict) -> dict:
        question = state.get("question", "")
        if not question:
            return {
                **state,
                "answer": "No question provided.",
                "schema": "",
                "query": "",
                "response": None,
            }

        # RULE mode (default)
        if not _use_llm_sql():
            try:
                out = _run_rule_sql(db, table_name)
                return {**state, **out}
            except Exception as e:
                return {
                    **state,
                    "schema": "",
                    "query": f"SELECT * FROM {table_name} LIMIT 20;",
                    "response": None,
                    "answer": f"SQL execution failed: {e}",
                }

        try:
            out = _run_llm_sql(
                db=db,
                table_name=table_name,
                question=question,
                llm_model=llm_model,
                temperature=temperature,
                default_limit=default_limit,
                required_columns=required_columns,
            )
            return {**state, **out}
        except UnsafeSQLError as e:
            return {
                **state,
                "schema": _get_schema_text(db, table_name),
                "query": "",
                "response": None,
                "answer": f"Refused to run unsafe SQL: {e}",
            }
        except Exception as e:
            return {
                **state,
                "schema": _get_schema_text(db, table_name),
                "query": "",
                "response": None,
                "answer": f"SQL agent failed: {e}",
            }

    return sql_agent_node


def create_transaction_sql_agent_node(
    username: str,
    password: str,
    host: str,
    port: str,
    database: str,
    table_name: str = "transactions",
    llm_model: str = "gpt-4.1",
    temperature: float = 0.0,
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
