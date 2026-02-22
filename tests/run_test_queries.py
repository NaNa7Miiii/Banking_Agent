"""
Test runner for the 30 fraud detection queries.

Runs each query through the full pipeline and saves outputs to
tests/test_results.json for human annotator review.

Usage:
    python tests/run_test_queries.py              # Run all 30 queries
    python tests/run_test_queries.py --ids Q01 Q02  # Run specific queries
    python tests/run_test_queries.py --start Q11    # Resume from a query
"""

import sys
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.chat_interface import chat

# ── Config ─────────────────────────────────────────────────────────────────────
QUERIES_FILE = project_root / "tests" / "test_queries.json"
RESULTS_FILE = project_root / "tests" / "test_results.json"
CUSTOMER_ID  = "test_evaluator_001"   # dummy customer ID for test runs

# Queries expected to be slow (large batch) — print a warning before running
SLOW_QUERIES = {"Q23"}


def load_queries() -> list:
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def load_existing_results() -> dict:
    """Load previously saved results so we can resume without re-running."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)


def run_query(query_obj: dict) -> dict:
    """Run a single query through the pipeline and return the result record."""
    qid      = query_obj["id"]
    question = query_obj["query"]

    print(f"\n{'='*60}")
    print(f"[{qid}] {question}")
    print(f"{'='*60}")

    if qid in SLOW_QUERIES:
        print(f"  [!]  This query may take a while (large batch)...")

    start = time.time()
    error = None
    output = ""

    try:
        result = chat(
            user_input=question,
            customer_id_number=CUSTOMER_ID,
        )
        output = result.get("final_answer", "")

        # For fraud detection results, the answer is inside results[].details
        if not output:
            for res in result.get("results", []):
                details = res.get("details", {})
                if details.get("answer"):
                    output = details["answer"]
                    break

    except Exception as e:
        error = str(e)
        traceback.print_exc()

    elapsed = round(time.time() - start, 2)

    print(f"\n--- Output ---")
    print(output if output else f"[ERROR] {error}")
    print(f"\n[{qid}] Completed in {elapsed}s")

    return {
        "id": qid,
        "category": query_obj["category"],
        "pipeline_flow": query_obj["pipeline_flow"],
        "difficulty": query_obj["difficulty"],
        "query": question,
        "note": query_obj["note"],
        "expected_output": query_obj["expected_output"],
        "actual_output": output,
        "error": error,
        "elapsed_seconds": elapsed,
        "run_timestamp": datetime.now().isoformat(),
        "evaluation": {
            "relevance_criteria":    query_obj["evaluation"]["relevance_criteria"],
            "completeness_criteria": query_obj["evaluation"]["completeness_criteria"],
            "readability_criteria":  query_obj["evaluation"]["readability_criteria"],
            "relevance":   None,   # to be filled by human annotator (1-5)
            "completeness": None,
            "readability": None,
            "notes": "",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Run test queries through the fraud pipeline.")
    parser.add_argument("--ids",   nargs="+", help="Run only these query IDs (e.g. Q01 Q02)")
    parser.add_argument("--start", type=str,  help="Resume from this query ID onwards (e.g. Q11)")
    args = parser.parse_args()

    queries = load_queries()
    existing = load_existing_results()

    # Filter queries to run
    if args.ids:
        target_ids = set(args.ids)
        queries = [q for q in queries if q["id"] in target_ids]
    elif args.start:
        ids = [q["id"] for q in queries]
        if args.start not in ids:
            print(f"ERROR: --start ID '{args.start}' not found in queries.")
            sys.exit(1)
        start_idx = ids.index(args.start)
        queries = queries[start_idx:]

    print(f"\nRunning {len(queries)} queries...")
    print(f"Results will be saved to: {RESULTS_FILE}\n")

    for query_obj in queries:
        qid = query_obj["id"]

        # Skip if already run (resume support)
        if qid in existing and existing[qid].get("actual_output") and not existing[qid].get("error"):
            print(f"[{qid}] Already completed — skipping. (Use --ids {qid} to re-run)")
            continue

        result = run_query(query_obj)
        existing[qid] = result
        save_results(existing)   # save after each query so progress is never lost

    print(f"\n{'='*60}")
    print(f"All done. Results saved to: {RESULTS_FILE}")
    completed = sum(1 for r in existing.values() if r.get("actual_output") and not r.get("error"))
    errored   = sum(1 for r in existing.values() if r.get("error"))
    print(f"Completed: {completed} | Errors: {errored} | Total: {len(existing)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
