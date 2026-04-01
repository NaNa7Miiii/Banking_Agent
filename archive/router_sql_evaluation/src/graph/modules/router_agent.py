from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Literal, cast

try:
    from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory
except ImportError:
    ConversationSummaryBufferMemory = Any

from src.graph.models.llm import get_llm
from src.graph.prompts.router_prompts import ROUTER_SYSTEM_PROMPT
from src.graph.utils.json_utils import safe_json_loads, JsonParseError


Intent = Literal[
    "PERSONAL_SPENDING_ANALYSIS",
    "FINANCIAL_KNOWLEDGE_QA",
    "WEB_SEARCH",
    "FRAUD_DETECTION",
    "CHITCHAT_OR_OTHER",
]


class QueryItem(TypedDict):
    id: str
    original_text: str
    primary_intent: Intent


class RouterOutput(TypedDict):
    queries: List[QueryItem]


class RouterState(TypedDict, total=False):
    user_input: str
    customer_id_number: str
    session_id: str

    queries: List[QueryItem]
    current_query_index: int
    results: List[dict]
    final_answer: str
    summary_enabled: bool


_ALLOWED_INTENTS = {
    "PERSONAL_SPENDING_ANALYSIS",
    "FINANCIAL_KNOWLEDGE_QA",
    "WEB_SEARCH",
    "FRAUD_DETECTION",
    "CHITCHAT_OR_OTHER",
}


def normalize_router_output(raw: Dict[str, Any], fallback_text: str) -> RouterOutput:
    raw_queries = raw.get("queries")
    queries_out: List[QueryItem] = []

    if isinstance(raw_queries, list) and raw_queries:
        for idx, qi in enumerate(raw_queries, start=1):
            if not isinstance(qi, dict):
                continue

            qid = qi.get("id")
            if not isinstance(qid, str) or not qid.strip():
                qid = f"q{idx}"

            original_text = qi.get("original_text")
            if not isinstance(original_text, str) or not original_text.strip():
                original_text = fallback_text

            intent = qi.get("primary_intent")
            if intent not in _ALLOWED_INTENTS:
                intent = "CHITCHAT_OR_OTHER"

            queries_out.append(
                {
                    "id": qid.strip(),
                    "original_text": original_text.strip(),
                    "primary_intent": cast(Intent, intent),
                }
            )

    if not queries_out:
        queries_out = [
            {
                "id": "q1",
                "original_text": fallback_text,
                "primary_intent": cast(Intent, "CHITCHAT_OR_OTHER"),
            }
        ]

    return {"queries": queries_out}


def route_query(
    user_query: str,
    use_history: bool = False,
    memory: Optional[ConversationSummaryBufferMemory] = None,
    system_prompt: Optional[str] = None,
) -> RouterOutput:
    llm = get_llm(role="router")

    raw_output = llm.chat(
        system_prompt=system_prompt or ROUTER_SYSTEM_PROMPT,
        user_prompt=user_query,
        response_format=None,
    )

    if isinstance(raw_output, str):
        try:
            parsed = safe_json_loads(raw_output)
        except JsonParseError:
            parsed = {"queries": []}
    elif isinstance(raw_output, dict):
        parsed = raw_output
    else:
        parsed = {"queries": []}

    return normalize_router_output(parsed, fallback_text=user_query)


def create_router_node():
    def router_node(state: RouterState) -> RouterState:
        user_input = state.get("user_input", "")
        if not user_input.strip():
            return {
                **state,
                "queries": [],
                "current_query_index": 0,
                "results": [],
            }

        router_output = route_query(user_input)

        return {
            **state,
            "queries": router_output["queries"],
            "current_query_index": 0,
            "results": [],
        }

    return router_node
