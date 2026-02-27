import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def _compute_metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float, Dict[str, Dict[str, float]]]:
    labels = sorted(set(y_true) | set(y_pred))
    cm = Counter(zip(y_true, y_pred))

    correct = sum(cm.get((l, l), 0) for l in labels)
    accuracy = correct / len(y_true) if y_true else 0.0

    per_label: Dict[str, Dict[str, float]] = {}
    macro_f1_sum = 0.0

    for lab in labels:
        tp = cm.get((lab, lab), 0)
        fp = sum(cm.get((t, lab), 0) for t in labels if t != lab)
        fn = sum(cm.get((lab, p), 0) for p in labels if p != lab)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        macro_f1_sum += f1
        per_label[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(sum(cm.get((lab, p), 0) for p in labels)),
        }

    macro_f1 = macro_f1_sum / len(labels) if labels else 0.0
    return accuracy, macro_f1, per_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-prefix", default="router_eval")
    args = parser.parse_args()

    from src.graph.modules.router_agent import route_query

    rows = _load_rows(args.csv)

    y_true: List[str] = []
    y_pred: List[str] = []
    results: List[Dict[str, Any]] = []

    for i, r in enumerate(rows, 1):
        text_in = (r.get("text") or "").strip()
        true_intent = (r.get("true_intent") or "").strip()

        router_out = route_query(text_in)

        queries = router_out.get("queries", []) or []
        meta = router_out.get("meta", {}) or {}

        num_queries = len(queries)
        q1 = queries[0] if queries else {}

        pred_intent = str(q1.get("primary_intent", "CHITCHAT_OR_OTHER"))
        y_true.append(true_intent)
        y_pred.append(pred_intent)

        meta_conf = _safe_float(meta.get("confidence", 0.0), 0.0)
        meta_margin = _safe_float(meta.get("margin", 0.0), 0.0)
        meta_router = meta.get("router", "llm")

        results.append(
            {
                "row_id": i,
                "text": text_in,
                "true_intent": true_intent,
                "pred_intent": pred_intent,
                "is_correct": int(true_intent == pred_intent),
                "num_queries": num_queries,
                "meta_router": meta_router,
                "meta_confidence": round(meta_conf, 4),
                "meta_margin": round(meta_margin, 4),
                "queries_json": json.dumps(queries, ensure_ascii=False),
            }
        )

        print(f"[{i}/{len(rows)}] {true_intent} -> {pred_intent} | num_queries={num_queries} | conf={meta_conf:.2f} margin={meta_margin:.2f}")

    accuracy, macro_f1, per_label = _compute_metrics(y_true, y_pred)

    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Macro F1   : {macro_f1:.4f}")

    for lab, m in per_label.items():
        print(f"{lab}: P={m['precision']} R={m['recall']} F1={m['f1']} Support={int(m['support'])}")

    ts = _timestamp()
    results_file = f"{args.out_prefix}_{ts}_results.csv"
    metrics_file = f"{args.out_prefix}_{ts}_metrics.json"

    # Save detailed results
    with open(results_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "text",
                "true_intent",
                "pred_intent",
                "is_correct",
                "num_queries",
                "meta_router",
                "meta_confidence",
                "meta_margin",
                "queries_json",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Save metrics summary
    summary = {
        "timestamp": ts,
        "total_samples": len(rows),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_label": per_label,
        "note": "Evaluation uses the FIRST query's primary_intent as prediction (CSV has one true_intent per row).",
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"- {results_file}")
    print(f"- {metrics_file}")


if __name__ == "__main__":
    main()
