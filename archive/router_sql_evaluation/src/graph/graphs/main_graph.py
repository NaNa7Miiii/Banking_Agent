from typing import Literal
from langgraph.graph import StateGraph, END

from src.graph.modules.router_agent import (
    RouterState,
    create_router_node,
    create_query_processor_node,
    create_answer_aggregator_node,
    create_summary_node,
)

# determine if we should continue processing queries or aggregate results
def should_continue_processing(state: RouterState) -> Literal["process_query", "aggregate", END]:
    queries = state.get("queries", [])
    current_index = state.get("current_query_index", 0)

    if current_index < len(queries):
        return "process_query"
    elif current_index >= len(queries) and len(queries) > 0:
        return "aggregate"
    else:
        return END

# determine if we should go to summary or end
def should_summarize(state: RouterState) -> Literal["summary", END]:
    summary_enabled = state.get("summary_enabled", False)
    if summary_enabled:
        return "summary"
    else:
        return END

# create the main graph that routes queries to appropriate agents
def create_main_graph(
    sql_agent_node_func,
    knowledge_agent_node_func=None,
    web_search_agent_node_func=None,
    fraud_agent_node_func=None,
    chitchat_agent_node_func=None):
    # Create agent map
    agent_map = {
        "PERSONAL_SPENDING_ANALYSIS": sql_agent_node_func,
        "FINANCIAL_KNOWLEDGE_QA": knowledge_agent_node_func,
        "WEB_SEARCH": web_search_agent_node_func,
        "FRAUD_DETECTION": fraud_agent_node_func,
    }
    if chitchat_agent_node_func:
        agent_map["CHITCHAT_OR_OTHER"] = chitchat_agent_node_func

    # Create nodes
    router_node = create_router_node()
    process_query_node = create_query_processor_node(agent_map)
    aggregate_node = create_answer_aggregator_node()
    summary_node = create_summary_node()

    # Build graph
    builder = StateGraph(RouterState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("process_query", process_query_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("summary", summary_node)

    # Set entry point
    builder.set_entry_point("router")

    # Add edges
    builder.add_conditional_edges(
        "router",
        should_continue_processing,
        {
            "process_query": "process_query",
            "aggregate": "aggregate",
            END: END,
        }
    )

    builder.add_conditional_edges(
        "process_query",
        should_continue_processing,
        {
            "process_query": "process_query",  # Continue processing next query
            "aggregate": "aggregate",
            END: END,
        }
    )

    # After aggregate, check if we should summarize
    builder.add_conditional_edges(
        "aggregate",
        should_summarize,
        {
            "summary": "summary",
            END: END,
        }
    )

    builder.add_edge("summary", END)

    return builder.compile()
