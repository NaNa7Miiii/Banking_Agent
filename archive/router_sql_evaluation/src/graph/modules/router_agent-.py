import json
from typing import TypedDict, List, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory.summary_buffer import ConversationSummaryBufferMemory

from src.graph.models.llm import get_llm
from src.graph.prompts.router_prompts import ROUTER_SYSTEM_PROMPT
from src.graph.prompts.summary_prompts import SUMMARY_SYSTEM_PROMPT, create_summary_user_prompt
from src.graph.utils.memory import create_memory


# query item
class QueryItem(TypedDict):
    id: str
    original_text: str
    primary_intent: str

# router output
class RouterOutput(TypedDict):
    queries: List[QueryItem]

# router state
class RouterState(TypedDict):
    user_input: str
    customer_id_number: str
    session_id: str
    queries: List[QueryItem]
    current_query_index: int
    results: List[dict]
    final_answer: str
    summary_enabled: bool


def format_conversation_history(
    memory: Optional[ConversationSummaryBufferMemory],
    current_query: str,
    max_messages: int = 10,
    prefix: str = "Previous conversation"
):
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


def route_query(user_query: str, memory: Optional[ConversationSummaryBufferMemory] = None) -> RouterOutput:
    # Build user prompt with conversation history if available
    user_prompt = format_conversation_history(memory, user_query, max_messages=10)

    # Get LLM instance for router role
    llm = get_llm(role="router")
    raw_output = llm.chat(
        system_prompt=ROUTER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_format=None,
    )

    # JSON parsing
    if isinstance(raw_output, str):
        try:
            # clean up potential markdown code blocks
            cleaned = raw_output.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            return RouterOutput(**parsed)
        except json.JSONDecodeError as e:
            print(f"JSON parsing has failed, raw output: {raw_output}")
            raise
    else:
        return RouterOutput(**raw_output)


# router node
def create_router_node():
    # router node that splits the user input into queries and classifies their intents
    def router_node(state: RouterState) -> RouterState:
        user_input = state.get("user_input", "")
        customer_id_number = state.get("customer_id_number", "")
        session_id = state.get("session_id", "")

        print(f"\n[Router] Starting routing for user input: {user_input}")

        if not user_input:
            print("[Router] WARNING: Empty user input received.")
            return {**state, "queries": [], "current_query_index": 0}

        # Load memory for this session
        memory = None
        if customer_id_number and session_id:
            memory = create_memory(customer_id_number, session_id, max_token_limit=2000)
            # Save user input to memory
            memory.chat_memory.add_user_message(user_input)

        # Route the query with conversation history
        router_output = route_query(user_input, memory=memory)

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


# query processor node
def create_query_processor_node(agent_map: dict):
    def process_query_node(state: RouterState) -> RouterState:
        queries = state.get("queries", [])
        current_index = state.get("current_query_index", 0)
        results = state.get("results", [])
        customer_id_number = state.get("customer_id_number", "")
        session_id = state.get("session_id", "")

        # if all queries have been processed, return the state
        if current_index >= len(queries):
            return state

        # get the current query
        current_query = queries[current_index]
        primary_intent = current_query["primary_intent"]

        # Load memory for this session
        memory = None
        if customer_id_number and session_id:
            memory = create_memory(customer_id_number, session_id, max_token_limit=2000)

        # get the appropriate agent for this intent
        agent_node = agent_map.get(primary_intent)

        if not agent_node:
            # no agent available for this intent, return a default response
            result = {
                "query_id": current_query["id"],
                "query_text": current_query["original_text"],
                "intent": primary_intent,
                "answer": f"No agent available for the intent: {primary_intent}",
            }
        else:
            # Build question with conversation context if memory exists
            question = format_conversation_history(
                memory,
                current_query["original_text"],
                max_messages=5,
                prefix="Previous conversation context"
            )

            # Process with the appropriate agent
            print(f"\n[Router] Processing query with {primary_intent} agent...")
            print(f"[Router] Query: {current_query['original_text']}")
            agent_state = agent_node({"question": question})
            result = {
                "query_id": current_query["id"],
                "query_text": current_query["original_text"],
                "intent": primary_intent,
                "answer": agent_state.get("answer", ""),
                "details": agent_state,
            }
            print(f"[Router] Agent processing completed for {primary_intent}.")

        # Add result and move to next query
        updated_results = results + [result]
        return {
            **state,
            "results": updated_results,
            "current_query_index": current_index + 1,
        }

    return process_query_node


# answer aggregator node
def create_answer_aggregator_node():
    def aggregate_node(state: RouterState) -> RouterState:
        results = state.get("results", [])
        queries = state.get("queries", [])
        customer_id_number = state.get("customer_id_number", "")
        session_id = state.get("session_id", "")

        if not results:
            return {**state, "final_answer": "No results to aggregate."}

        # Combine all answers
        answer_parts = []
        for i, result in enumerate(results):
            query_text = result.get("query_text", "")
            answer = result.get("answer", "")
            answer_parts.append(f"Q{i+1}: {query_text}\nA{i+1}: {answer}\n")

        final_answer = "\n".join(answer_parts)

        # Save conversation to memory only if summary is not enabled
        # (if summary is enabled, we'll save in summary node instead)
        summary_enabled = state.get("summary_enabled", False)
        if customer_id_number and session_id and not summary_enabled:
            memory = create_memory(customer_id_number, session_id, max_token_limit=2000)
            # Save assistant response to memory
            # ConversationSummaryBufferMemory will automatically handle summarization
            # when token limit is reached
            memory.chat_memory.add_ai_message(final_answer)

        return {
            **state,
            "final_answer": final_answer,
        }

    return aggregate_node


# summary node
def create_summary_node():
    """Create a summary node that summarizes multiple answers into a single response"""
    def summary_node(state: RouterState) -> RouterState:
        results = state.get("results", [])

        if not results:
            return {**state, "final_answer": "No results to summarize."}

        # Extract only the answers (A1, A2, ..., An)
        answers = []
        for i, result in enumerate(results):
            answer = result.get("answer", "")
            if answer:
                answers.append(f"A{i+1}: {answer}")

        if not answers:
            return {**state, "final_answer": "No answers found to summarize."}

        # Combine all answers into a single text
        answers_text = "\n\n".join(answers)

        # Generate summary using LLM
        summary_prompt = create_summary_user_prompt(answers_text)

        llm = get_llm(role="summary")
        summary = llm.chat(
            system_prompt=SUMMARY_SYSTEM_PROMPT,
            user_prompt=summary_prompt,
            response_format=None,
        )

        summary_text = summary.strip()

        # Save conversation to memory (summary version)
        customer_id_number = state.get("customer_id_number", "")
        session_id = state.get("session_id", "")
        if customer_id_number and session_id:
            memory = create_memory(customer_id_number, session_id, max_token_limit=2000)
            # Save summarized response to memory
            memory.chat_memory.add_ai_message(summary_text)

        return {
            **state,
            "final_answer": summary_text,
        }

    return summary_node

