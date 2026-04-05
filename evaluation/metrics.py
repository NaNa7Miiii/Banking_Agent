from __future__ import annotations

from typing import Dict, List, Tuple


def compute_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


def compute_precision_recall_f1(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    result: Dict[str, Dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        result[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label),
        }

    return result


def compute_macro_f1(per_label_metrics: Dict[str, Dict[str, float]]) -> float:
    if not per_label_metrics:
        return 0.0
    f1_values = [m["f1"] for m in per_label_metrics.values()]
    return round(sum(f1_values) / len(f1_values), 4)


def build_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
) -> Tuple[List[str], List[List[int]]]:
    labels = sorted(set(y_true) | set(y_pred))
    label_to_idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t]][label_to_idx[p]] += 1

    return labels, matrix
