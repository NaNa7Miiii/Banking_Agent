import argparse
import csv
from collections import Counter
from typing import Any, Dict, List, Tuple

from src.graph.modules.router_agent import rule_route_output as rule_router


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def main(csv_path: str, low_margin_threshold: float = 0.10) -> None:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        print(f"[Error] No rows found in {csv_path}.")
        return

    y_true: List[str] = []
    y_pred: List[str] = []

    mistakes: List[Tuple[str, str, str, float, float, str]] = []
    low_margin_cases: List[Tuple[str, str, str, float, float]] = []

    for r in rows:
        text = (r.get("text") or "").strip()
        true_intent = (r.get("true_intent") or "").strip()

        out = rule_router(text)
        pred_intent = out["queries"][0]["primary_intent"]

        meta = out.get("meta", {})
        confidence = _safe_float(meta.get("confidence", 0.0))
        margin = _safe_float(meta.get("margin", 0.0))
        reasoning = str(meta.get("reasoning", ""))

        y_true.append(true_intent)
        y_pred.append(pred_intent)

        if margin <= low_margin_threshold:
            low_margin_cases.append((text, true_intent, pred_intent, confidence, margin))

        if pred_intent != true_intent:
            mistakes.append((text, true_intent, pred_intent, confidence, margin, reasoning))

    total = len(rows)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / total if total else 0.0

    conf_counts = Counter(zip(y_true, y_pred))

    mistakes_sorted = sorted(mistakes, key=lambda x: (x[4], x[3]))
    low_margin_sorted = sorted(low_margin_cases, key=lambda x: (x[4], x[3]))

    print("==== Router Evaluation (RULE) ====")
    print(f"CSV file: {csv_path}")
    print(f"Total queries: {total}")
    print(f"Accuracy: {acc:.3f} ({correct}/{total})")
    print(f"Low-margin threshold: {low_margin_threshold:.2f}")
    print(f"Low-margin queries: {len(low_margin_cases)} ({len(low_margin_cases)/total:.1%})")

    print("\n-- Confusion Counts (true -> pred) [errors only] --")
    any_confusion = False
    for (t, p), c in conf_counts.most_common():
        if t != p:
            any_confusion = True
            print(f"{t:28s} -> {p:28s} : {c}")
    if not any_confusion:
        print("(none)")

    print("\n-- Top 5 Mistakes (lowest margin first) --")
    if not mistakes_sorted:
        print("(none)")
    else:
        for i, (text, t, p, confv, m, reasoning) in enumerate(mistakes_sorted[:5], start=1):
            reason_short = reasoning if len(reasoning) <= 120 else reasoning[:117] + "..."
            print(
                f"{i}. '{text}'\n"
                f"   true={t} pred={p} confidence={confv:.3f} margin={m:.3f}\n"
                f"   reasoning={reason_short}"
            )

    print("\n-- Top 5 Low-Margin (Ambiguous) Queries --")
    if not low_margin_sorted:
        print("(none)")
    else:
        for i, (text, t, p, confv, m) in enumerate(low_margin_sorted[:5], start=1):
            print(
                f"{i}. '{text}'\n"
                f"   true={t} pred={p} confidence={confv:.3f} margin={m:.3f}"
            )

    print("\nReport tip:")
    print("- Most routing errors occur on low-margin (ambiguous) queries.")
    print("- These are good candidates for future rule refinement.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="router_eval.csv", help="Path to router evaluation CSV")
    parser.add_argument("--low-margin", type=float, default=0.10, help="Low-margin threshold")
    args = parser.parse_args()

    main(csv_path=args.csv, low_margin_threshold=args.low_margin)
