from math import isfinite
from typing import Dict, Iterable, List

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


DEFAULT_MULTI_VIEW_HYBRID_WEIGHTS = {
    "sentence": 0.35,
    "span": 0.30,
    "overlap": 0.15,
    "conflict_penalty": 0.25,
}


def missing_fields(records: List[Dict], fields: Iterable[str]) -> List[str]:
    """
    Return fields that are absent from at least one record.
    """
    missing = []
    for field in fields:
        if any(field not in record for record in records):
            missing.append(field)
    return missing


def records_have_fields(records: List[Dict], fields: Iterable[str]) -> bool:
    """
    Check whether every record contains every requested field.
    """
    return not missing_fields(records, fields)


def finite_score(value: object, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    return score if isfinite(score) else default


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, score))


def multi_view_hybrid_score(
    *,
    sentence_similarity: object,
    span_max_similarity: object | None = None,
    entity_or_token_overlap: object | None = None,
    factual_conflict_penalty: object | None = None,
    weights: Dict[str, float] | None = None,
) -> float:
    """
    Unit 6 reduced multi-view hybrid score.

    Unit 5's factual embedding view is intentionally omitted. The remaining
    positive weights are renormalized over present finite components, then the
    factual conflict penalty is subtracted and the final score is clamped to
    the classifier score range.
    """
    weights = weights or DEFAULT_MULTI_VIEW_HYBRID_WEIGHTS
    components = [
        ("sentence", finite_score(sentence_similarity)),
        ("span", finite_score(span_max_similarity)),
        ("overlap", finite_score(entity_or_token_overlap)),
    ]
    present_components = [
        (name, score)
        for name, score in components
        if score is not None and weights.get(name, 0.0) > 0.0
    ]
    if not present_components:
        return 0.0

    positive_weight_total = sum(weights[name] for name, _ in present_components)
    combined_score = sum(
        (weights[name] / positive_weight_total) * float(score)
        for name, score in present_components
    )
    penalty = finite_score(factual_conflict_penalty, 0.0) or 0.0
    adjusted_score = combined_score - weights.get("conflict_penalty", 0.0) * penalty
    return clamp_score(adjusted_score)


def require_metric_fields(records: List[Dict], fields: Iterable[str]) -> None:
    missing = missing_fields(records, fields)
    if missing:
        raise ValueError(
            "Cannot evaluate metrics because records are missing fields: "
            + ", ".join(missing)
        )


def evaluate_similarity_as_classifier(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    threshold: float = 0.75,
) -> Dict[str, float]:
    """
    Evaluate whether similarity can classify predictions as correct / incorrect.
    """
    require_metric_fields(records, [label_field, similarity_field])

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
    require_metric_fields(records, [label_field, similarity_field])

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
    require_metric_fields(records, [label_field, similarity_field])

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
