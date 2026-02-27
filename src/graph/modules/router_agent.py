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


class RouterMeta(TypedDict, total=False):
    router: str
    confidence: float
    margin: float


class RouterOutput(TypedDict):
    queries: List[QueryItem]
    meta: RouterMeta


class RouterState(TypedDict, total=False):
    user_input: str
    customer_id_number: str
    session_id: str

    queries: List[QueryItem]
    current_query_index: int
    results: List[dict]
    final_answer: str
    summary_enabled: bool

    meta: RouterMeta


_ALLOWED_INTENTS = {
    "PERSONAL_SPENDING_ANALYSIS",
    "FINANCIAL_KNOWLEDGE_QA",
    "WEB_SEARCH",
    "FRAUD_DETECTION",
    "CHITCHAT_OR_OTHER",
}


def _safe_float_01(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        v = default
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def normalize_router_output(raw: Dict[str, Any], fallback_text: str) -> RouterOutput:
    meta_out: RouterMeta = {"router": "llm", "confidence": 0.0, "margin": 0.0}
    raw_meta = raw.get("meta")
    if isinstance(raw_meta, dict):
        if isinstance(raw_meta.get("router"), str):
            meta_out["router"] = raw_meta["router"]
        if "confidence" in raw_meta:
            meta_out["confidence"] = _safe_float_01(raw_meta.get("confidence"), 0.0)
        if "margin" in raw_meta:
            meta_out["margin"] = _safe_float_01(raw_meta.get("margin"), 0.0)

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

    return {"queries": queries_out, "meta": meta_out}


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
            parsed = {"queries": [], "meta": {"router": "llm", "confidence": 0.0, "margin": 0.0}}
    elif isinstance(raw_output, dict):
        parsed = raw_output
    else:
        parsed = {"queries": [], "meta": {"router": "llm", "confidence": 0.0, "margin": 0.0}}

    return normalize_router_output(parsed, fallback_text=user_query)


def create_router_node():
    def router_node(state: RouterState) -> RouterState:
        user_input = state.get("user_input", "")
        if not user_input.strip():
            return {**state, "queries": [], "current_query_index": 0, "results": [], "meta": {"router": "llm", "confidence": 0.0, "margin": 0.0}}

        router_output = route_query(user_input)

        return {
            **state,
            "queries": router_output["queries"],
            "meta": router_output["meta"],
            "current_query_index": 0,
            "results": [],
        }

    return router_node
