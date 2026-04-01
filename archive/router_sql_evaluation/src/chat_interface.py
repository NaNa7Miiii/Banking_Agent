import uuid
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.graph.graphs.main_graph import create_main_graph
from src.graph.modules.sql_agent import create_transaction_sql_agent_node
from src.graph.modules.rag_agent import create_rag_agent_node
from src.graph.modules.tavily_agent import create_tavily_agent_node
from src.graph.modules.fraud_agent import create_transaction_fraud_agent_node
from src.config import get_db_config
from pinecone import Pinecone
from tavily import TavilyClient
import os
import json

_graph_cache = None


def create_graph():
    global _graph_cache

    # Return cached graph if it exists
    if _graph_cache is not None:
        return _graph_cache

    # Get database configuration from environment variables
    db_config = get_db_config()

    # Create graph components
    sql_agent_node = create_transaction_sql_agent_node(
        username=db_config["username"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        table_name=db_config["table_name"],
    )

    # Initialize Pinecone client for RAG agent
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        rag_agent_node = create_rag_agent_node(pc=pc)
    else:
        rag_agent_node = None

    # Initialize Tavily client for web search agent
    tavily_api_key = os.getenv("TAVILY_SEARCH_KEY")
    if tavily_api_key:
        tavily_client = TavilyClient(api_key=tavily_api_key)
        tavily_agent_node = create_tavily_agent_node(tavily_client=tavily_client, topic=None)
    else:
        tavily_agent_node = None

    # Initialize fraud agent
    fraud_agent_node = create_transaction_fraud_agent_node(
        username=db_config["username"],
        password=db_config["password"],
        host=db_config["host"],
        port=db_config["port"],
        database=db_config["database"],
        table_name=db_config["table_name"],
    )

    # Create and cache graph
    _graph_cache = create_main_graph(
        sql_agent_node_func=sql_agent_node,
        knowledge_agent_node_func=rag_agent_node,
        web_search_agent_node_func=tavily_agent_node,
        fraud_agent_node_func=fraud_agent_node,
    )
    return _graph_cache


def chat(user_input: str, customer_id_number: str, session_id: str = None, summary: bool = False):
    if session_id is None:
        session_id = str(uuid.uuid4())

    # Create graph
    graph = create_graph()

    # Prepare initial state
    initial_state = {
        "user_input": user_input,
        "customer_id_number": str(customer_id_number),
        "session_id": session_id,
        "queries": [],
        "current_query_index": 0,
        "results": [],
        "final_answer": "",
        "summary_enabled": summary,
    }

    # Invoke the graph
    result = graph.invoke(initial_state)

    return {
        "session_id": session_id,
        "final_answer": result.get("final_answer", ""),
        "queries": result.get("queries", []),
        "results": result.get("results", []),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python chat_interface.py <user_input> <customer_id_number> [session_id] [--summary]")
        sys.exit(1)

    user_input = sys.argv[1]
    customer_id_number = sys.argv[2]
    session_id = None
    summary = False

    # Parse optional arguments
    for arg in sys.argv[3:]:
        if arg == "--summary":
            summary = True
        elif not arg.startswith("--"):
            session_id = arg

    result = chat(user_input, customer_id_number, session_id, summary=summary)

    print("=" * 60)
    print("Session ID:", result["session_id"])
    print("=" * 60)

    # Check if this is a fraud detection result
    results = result.get("results", [])
    fraud_result = None
    for res in results:
        if res.get("intent") == "FRAUD_DETECTION":
            fraud_result = res.get("details", {})
            break

    if fraud_result:
        # Format fraud detection output
        # Check if this is a detailed explanation (has LLM analysis) or just the initial report
        llm_analysis_json = fraud_result.get("llm_analysis_json", [])
        answer = fraud_result.get("answer", "")
        
        # If there's LLM analysis, this is a detailed explanation for a specific transaction
        if llm_analysis_json or (answer and "LLM Analysis for Transaction" in answer):
            # This is a detailed explanation - print it directly
            print("\n" + answer)
        else:
            # This is the initial fraud detection report - print it as formatted
            print("\n" + answer)
    else:
        # Regular output format
        print("\nFinal Answer:")
        print(result["final_answer"])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

