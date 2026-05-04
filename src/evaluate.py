from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.compute_similarity import threshold_similarity


def evaluate_similarity_as_classifier(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    threshold: float = 0.75,
) -> Dict[str, float]:
    """
    Evaluate whether similarity can classify predictions as correct / incorrect.
    """
    y_true = np.array([record[label_field] for record in records])
    y_score = np.array([record[similarity_field] for record in records])
    y_pred = np.array([threshold_similarity(score, threshold) for score in y_score])

    metrics = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def find_best_threshold(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    thresholds: List[float] | None = None,
) -> Dict[str, float]:
    """
    Search for the best threshold based on F1.
    """
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.0, 1.01, 0.01)]

    best_result = None

    for threshold in thresholds:
        result = evaluate_similarity_as_classifier(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            threshold=threshold,
        )

        if best_result is None or result["f1"] > best_result["f1"]:
            best_result = result

    return best_result


def summarize_similarity_by_correctness(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
) -> Dict[str, float]:
    """
    Summarize similarity scores for correct and incorrect predictions.
    """
    correct_scores = [
        record[similarity_field]
        for record in records
        if record[label_field] == 1
    ]

    incorrect_scores = [
        record[similarity_field]
        for record in records
        if record[label_field] == 0
    ]

    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    def safe_std(values: List[float]) -> float:
        return float(np.std(values)) if values else float("nan")

    summary = {
        "num_correct": len(correct_scores),
        "num_incorrect": len(incorrect_scores),
        "correct_mean": safe_mean(correct_scores),
        "correct_std": safe_std(correct_scores),
        "incorrect_mean": safe_mean(incorrect_scores),
        "incorrect_std": safe_std(incorrect_scores),
        "gap": safe_mean(correct_scores) - safe_mean(incorrect_scores),
    }

    return summary


def get_failure_cases(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    high_threshold: float = 0.8,
    low_threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    Extract two types of failure cases:
        1. high similarity but incorrect
        2. low similarity but correct
    """
    high_similarity_wrong = []
    low_similarity_correct = []

    for record in records:
        score = record[similarity_field]
        label = record[label_field]

        if score >= high_threshold and label == 0:
            high_similarity_wrong.append(record)

        if score <= low_threshold and label == 1:
            low_similarity_correct.append(record)

    return {
        "high_similarity_wrong": high_similarity_wrong,
        "low_similarity_correct": low_similarity_correct,
    }