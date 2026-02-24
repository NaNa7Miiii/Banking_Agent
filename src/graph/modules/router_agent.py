from __future__ import annotations

import os
from typing import Optional

from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory

from src.graph.graphs.router_schema import RouterOutput
from src.graph.utils.memory import create_memory
from src.graph.utils.json_sanitize import JsonParseError


def format_conversation_history(
    memory: Optional[ConversationSummaryBufferMemory],
    current_query: str,
    max_messages: int = 10,
    prefix: str = "Previous conversation",
) -> str:
    if not memory:
        return current_query

    messages = memory.chat_memory.messages
    if not messages:
        return current_query

    history_text = f"\n\n{prefix}:\n"
    for msg in messages[-max_messages:]:
        role = "User" if msg.type == "human" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    return history_text + f"\nCurrent user input: {current_query}"


def _basic_validate_router_output(data: dict) -> RouterOutput:
    if "queries" not in data or not isinstance(data["queries"], list) or not data["queries"]:
        raise ValueError("Router output must contain non-empty 'queries' list.")

    queries = data["queries"]
    for item in queries:
        if not isinstance(item, dict):
            raise ValueError("Each query item must be an object.")
        for key in ("id", "original_text", "primary_intent"):
            if key not in item or not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Missing/invalid field '{key}' in query item.")

    meta = data.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Router output must contain 'meta' object.")
    for key in ("router", "confidence", "margin"):
        if key not in meta:
            raise ValueError(f"Router meta missing '{key}'.")

    return data


def rule_route_output(user_query: str) -> RouterOutput:
    text = (user_query or "").strip()
    lowered = text.lower()

    if any(k in lowered for k in ["fraud", "suspicious", "scam", "stolen", "chargeback"]):
        intent = "FRAUD_DETECTION"
    elif any(k in lowered for k in ["search", "news", "latest", "web", "rate", "rates", "stock", "market"]):
        intent = "WEB_SEARCH"
    elif any(
        k in lowered
        for k in [
            "transaction",
            "transactions",
            "recent",
            "spent",
            "spend",
            "balance",
            "merchant",
            "amount",
            "category",
            "trend",
            "month",
            "week",
        ]
    ):
        intent = "PERSONAL_SPENDING_ANALYSIS"
    elif any(
        k in lowered
        for k in ["loan", "mortgage", "apr", "interest", "credit card", "points", "fee", "fees", "repayment", "invest"]
    ):
        intent = "FINANCIAL_KNOWLEDGE_QA"
    else:
        intent = "CHITCHAT_OR_OTHER"

    return {
        "queries": [
            {
                "id": "q1",
                "original_text": text,
                "primary_intent": intent,  # type: ignore[typeddict-item]
            }
        ],
        "meta": {"router": "rule", "confidence": 1.0, "margin": 1.0},
    }


def _llm_router_enabled() -> bool:
    router_mode = os.getenv("ROUTER_MODE", "RULE").upper()
    allow_llm = os.getenv("ALLOW_LLM_CALLS", "NO").upper()
    return router_mode == "LLM" and allow_llm == "YES"


def route_query(
    user_query: str,
    memory: Optional[ConversationSummaryBufferMemory] = None,
) -> RouterOutput:
    user_prompt = format_conversation_history(memory, user_query, max_messages=10)
    router_mode = os.getenv("ROUTER_MODE", "RULE").upper()

    if router_mode != "LLM":
        return rule_route_output(user_query)

    if not _llm_router_enabled():
        raise RuntimeError(
            "ROUTER_MODE=LLM is set, but ALLOW_LLM_CALLS is not YES. "
            "Refusing to call LLM router to prevent unintended cost."
        )

    raise RuntimeError(
        "LLM router wiring is disabled in this branch. "
        "To enable, implement llm.chat(...) and parse the output with safe_json_loads()."
    )


def create_router_node():
    def router_node(state: dict) -> dict:
        user_input = state.get("user_input", "")
        customer_id_number = state.get("customer_id_number", "")
        session_id = state.get("session_id", "")

        print(f"\n[Router] Starting routing for user input: {user_input}")

        if not user_input:
            print("[Router] WARNING: Empty user input received.")
            return {**state, "queries": [], "current_query_index": 0}

        memory = None
        if customer_id_number and session_id:
            memory = create_memory(customer_id_number, session_id, max_token_limit=2000)
            memory.chat_memory.add_user_message(user_input)

        try:
            router_output = route_query(user_input, memory=memory)
        except JsonParseError as e:
            print(f"[Router] ERROR: LLM router JSON parse failed: {e}. Falling back to RULE.")
            router_output = rule_route_output(user_input)

        router_output = _basic_validate_router_output(router_output)

        print(f"[Router] Routing completed. Found {len(router_output['queries'])} sub-queries:")
        for q in router_output["queries"]:
            print(f"  - {q['id']}: {q['original_text']} (Intent: {q['primary_intent']})")

        return {
            **state,
            "queries": router_output["queries"],
            "current_query_index": 0,
            "results": [],
        }

    return router_node
