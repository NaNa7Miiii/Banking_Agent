"""
Entrypoint: Input+Memory -> Runtime graph (planner -> init -> select -> execute -> merge -> evaluate -> aggregate) -> Memory.
Usage (from project root):
  python -m src.run "your question"
  python -m src.run "How much did I spend?" [customer_id] [session_id]  # with memory
"""
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure project root is on path when run as __main__
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.graph.runtime_graph import create_runtime_graph


def run_task(
    user_input: str,
    customer_id_number: str = "",
    session_id: str = "",
) -> dict:
    """
    Full task flow: memory enrichment -> runtime graph (plan + orchestration + subagent execution) -> save memory.
    """
    prompt_for_planner = user_input
    memory = None

    if customer_id_number and session_id:
        try:
            from src.utils.memory import (
                create_memory,
                format_conversation_history,
                save_context_and_persist,
            )
            memory = create_memory(customer_id_number, session_id)
            prompt_for_planner = format_conversation_history(
                memory, user_input, max_messages=10, prefix="Previous conversation"
            )
        except Exception as e:
            logger.warning(
                "Memory unavailable (e.g. Redis down): %s. Proceeding without conversation history.",
                e,
                exc_info=True,
            )
            memory = None
            prompt_for_planner = user_input

    initial = {
        "user_input": prompt_for_planner,
        "customer_id_number": customer_id_number,
        "session_id": session_id,
        "plan": None,
    }

    graph = create_runtime_graph()
    result = graph.invoke(initial)
    plan = result.get("plan")
    result["execution_state"] = {
        "plan": plan,
        "step_results": result.get("step_results"),
        "final_answer": result.get("final_answer"),
    } if plan else None

    if memory and customer_id_number and session_id:
        if result.get("final_answer"):
            to_save = result["final_answer"]
        elif plan:
            goal = plan.get("goal") or "N/A"
            steps = plan.get("steps") or []
            to_save = f"Plan: {goal} ({len(steps)} steps)."
        else:
            to_save = "No plan or result generated."
        save_context_and_persist(
            memory, customer_id_number, session_id, user_input, to_save
        )

    return result


def print_plan(plan: dict) -> None:
    """Pretty-print plan for CLI."""
    if not plan:
        print("(No plan generated)")
        return
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.run <user_input> [customer_id] [session_id]", file=sys.stderr)
        sys.exit(1)
    user_input = sys.argv[1]
    customer_id_number = sys.argv[2] if len(sys.argv) > 2 else ""
    session_id = sys.argv[3] if len(sys.argv) > 3 else ""

    result = run_task(user_input, customer_id_number, session_id)
    plan = result.get("plan")
    final_answer = result.get("final_answer")

    print("--- Plan ---")
    print_plan(plan)
    if final_answer is not None:
        print("--- Final answer ---")
        print(final_answer)
    print("--- Execution state ---")
    print(json.dumps(result.get("execution_state"), indent=2, ensure_ascii=False, default=str))
    print("--- End ---")


if __name__ == "__main__":
    main()
