"""
Result aggregation: turn step_results + user_goal into a single final answer via LLM.
Uses data.summary from successful steps (contract: executor normalizes all agents to summary + artifacts).
"""
from typing import Any

from src.graph.executor.schema import StepResult
from src.models.llm import get_llm
from src.utils.prompt_loader import load_system_prompt


AGGREGATION_MODULE = "aggregation"


def aggregate_results(
    user_goal: str,
    step_results: dict[str, StepResult],
    plan: dict[str, Any],
) -> str:
    """
    Collect successful step summaries, then ask LLM to synthesize one coherent final answer.
    If no successful steps, returns fallback message without calling LLM.
    """
    parts = []
    steps_order = [s.get("id") for s in (plan.get("steps") or [])]
    for step_id in steps_order:
        sr = step_results.get(step_id)
        if not sr or sr.get("status") != "ok":
            continue
        data = sr.get("data") or {}
        summary = (data.get("summary") or "").strip()
        if summary:
            parts.append(summary)
    if not parts:
        for step_id in steps_order:
            sr = step_results.get(step_id)
            if sr and sr.get("status") == "error":
                err = (sr.get("error_message") or "").strip()
                if err:
                    return err
        return "No results could be produced for your request."

    system_prompt = load_system_prompt(AGGREGATION_MODULE)
    user_prompt = (
        f"User's goal:\n{user_goal or '(not specified)'}\n\n"
        "Step results (summaries from subagents):\n"
        + "\n---\n".join(parts)
    )
    llm = get_llm(role="aggregation")
    final_answer = llm.chat(system_prompt=system_prompt, user_prompt=user_prompt)
    return (final_answer or "").strip() or "\n\n".join(parts)
