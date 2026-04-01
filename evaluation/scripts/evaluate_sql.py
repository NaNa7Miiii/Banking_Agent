from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from src.run import run_task


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("text", "").strip()]


def get_first_owner(result: Dict) -> str:
    plan = result.get("plan") or {}
    steps = plan.get("steps") or []
    if not steps:
        return "none"
    return (steps[0].get("owner") or "").strip().lower() or "none"


def shorten(text: str, max_len: int = 200) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="evaluation/data/sql_eval.csv")
    parser.add_argument("--out-dir", default="evaluation/output")
    parser.add_argument("--customer-id", default="123")
    parser.add_argument("--session-prefix", default="eval-sql")
    parser.add_argument("--limit", type=int, default=0, help="0 = run all rows")
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = out_dir / "sql_predictions.csv"
    summary_path = out_dir / "sql_summary.json"

    detailed_rows: List[Dict[str, str]] = []

    total = len(rows)
    sql_routed = 0
    success_count = 0
    nonempty_answer_count = 0
    error_count = 0

    for i, row in enumerate(rows, start=1):
        text = row["text"].strip()
        print(f"[{i}/{total}] {text}", flush=True)

        owner = "none"
        final_answer = ""
        error_text = ""
        status = "ok"

        try:
            result = run_task(
                user_input=text,
                customer_id_number=args.customer_id,
                session_id=f"{args.session_prefix}-{i}",
            )

            owner = get_first_owner(result)
            final_answer = str(result.get("final_answer") or "")

            if owner == "subagent:sql":
                sql_routed += 1

            if final_answer.strip():
                nonempty_answer_count += 1

            success_count += 1

        except Exception as e:
            status = "error"
            error_text = repr(e)
            error_count += 1

        detailed_rows.append(
            {
                "text": text,
                "first_owner": owner,
                "status": status,
                "has_final_answer": str(bool(final_answer.strip())),
                "final_answer_preview": shorten(final_answer),
                "error": error_text,
            }
        )

    summary = {
        "num_samples": total,
        "sql_routing_rate": round(sql_routed / total, 4) if total else 0.0,
        "execution_success_rate": round(success_count / total, 4) if total else 0.0,
        "nonempty_answer_rate": round(nonempty_answer_count / total, 4) if total else 0.0,
        "error_rate": round(error_count / total, 4) if total else 0.0,
    }

    with open(predictions_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "first_owner",
                "status",
                "has_final_answer",
                "final_answer_preview",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved: {predictions_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
