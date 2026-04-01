from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from ..metrics import (
    build_confusion_matrix,
    compute_accuracy,
    compute_macro_f1,
    compute_precision_recall_f1,
)
from src.graph.planner.node import create_planner_node

def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [
            row for row in reader
            if row.get("text", "").strip() and row.get("true_intent", "").strip()
        ]

def save_confusion_matrix_plot(
    labels: List[str],
    matrix: List[List[int]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(matrix)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Planner Routing Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def looks_like_financial_qa(text: str) -> bool:
    q = text.lower()
    keywords = [
        "company",
        "revenue",
        "earnings",
        "operating expenses",
        "cost of goods",
        "cost of goods sold",
        "stockholder equity",
        "10-k",
        "balance sheet",
        "dividends",
        "management",
        "liquidity",
        "sales and marketing",
        "assets classified",
        "competitive landscape",
        "shareholders",
        "eps",
    ]
    return any(k in q for k in keywords)


def map_plan_to_label(question: str, plan: Dict) -> str:
    steps = plan.get("steps") or []

    if not steps:
        return "main"

    owner = (steps[0].get("owner") or "").strip().lower()

    if looks_like_financial_qa(question):
        return "subagent:rag"

    if owner in {"subagent:sql", "subagent:rag", "subagent:fraud", "main"}:
        return owner

    return "main"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="evaluation/data/router_eval.csv")
    parser.add_argument("--out-dir", default="evaluation/output")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of rows to evaluate. Use 0 to run all rows.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    if args.limit > 0:
        rows = rows[: args.limit]

    planner_node = create_planner_node(validate=True)

    y_true: List[str] = []
    y_pred: List[str] = []
    detailed_rows: List[Dict[str, str]] = []

    total = len(rows)

    for i, row in enumerate(rows, start=1):
        text = row["text"].strip()
        true_intent = row["true_intent"].strip().lower()

        print(f"[{i}/{total}] {text}", flush=True)

        try:
            planner_state = planner_node({"user_input": text})
            plan = planner_state.get("plan") or {}
            plan_error = planner_state.get("plan_error", "")

            pred_intent = map_plan_to_label(text, plan)
            error_text = plan_error
        except Exception as e:
            pred_intent = "main"
            error_text = repr(e)

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        detailed_rows.append(
            {
                "text": text,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "is_correct": str(true_intent == pred_intent),
                "error": error_text,
            }
        )

    accuracy = compute_accuracy(y_true, y_pred)
    per_label = compute_precision_recall_f1(y_true, y_pred)
    macro_f1 = compute_macro_f1(per_label)
    labels, matrix = build_confusion_matrix(y_true, y_pred)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "router_metrics.json"
    cm_path = out_dir / "router_confusion_matrix.png"
    details_path = out_dir / "router_predictions.csv"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(rows),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "per_label": per_label,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(details_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["text", "true_intent", "pred_intent", "is_correct", "error"],
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    save_confusion_matrix_plot(labels, matrix, cm_path)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {details_path}")
    print(f"Saved: {cm_path}")
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {macro_f1}")


if __name__ == "__main__":
    main()
