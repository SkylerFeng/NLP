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

DEFAULT_QUESTION_TYPE_MIN_EXAMPLES = 50
DEFAULT_QUESTION_TYPE_MIN_POSITIVE = 10
DEFAULT_QUESTION_TYPE_MIN_NEGATIVE = 10
DEFAULT_QUESTION_TYPE_NUM_FOLDS = 5


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


def classification_metrics_from_predictions(
    y_true: Iterable[int],
    y_score: Iterable[float],
    y_pred: Iterable[int],
    threshold: float,
) -> Dict[str, float]:
    y_true = np.array(list(y_true))
    y_score = np.array(list(y_score))
    y_pred = np.array(list(y_pred))

    metrics = {
        "threshold": float(threshold),
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


def best_threshold_from_arrays(
    y_true: Iterable[int],
    y_score: Iterable[float],
    thresholds: List[float],
) -> Dict[str, float]:
    y_true = np.array(list(y_true))
    y_score = np.array(list(y_score))
    best_threshold = thresholds[0]
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = y_score >= threshold
        true_positive = int(np.sum((y_true == 1) & y_pred))
        false_positive = int(np.sum((y_true == 0) & y_pred))
        false_negative = int(np.sum((y_true == 1) & ~y_pred))
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = (
            true_positive / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1_denominator = precision + recall
        f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    y_pred = np.array(
        [threshold_similarity(score, best_threshold) for score in y_score]
    )
    return classification_metrics_from_predictions(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        threshold=best_threshold,
    )


def score_standard_deviation(records: List[Dict], score_field: str) -> float:
    require_metric_fields(records, [score_field])
    scores = [float(record[score_field]) for record in records]
    return float(np.std(scores)) if scores else float("nan")


def question_type_threshold_support(
    records: List[Dict],
    score_field: str,
    label_field: str = "correct_label",
    min_examples: int = DEFAULT_QUESTION_TYPE_MIN_EXAMPLES,
    min_positive: int = DEFAULT_QUESTION_TYPE_MIN_POSITIVE,
    min_negative: int = DEFAULT_QUESTION_TYPE_MIN_NEGATIVE,
) -> Dict[str, object]:
    """
    Decide whether a question-type bucket is large enough for its own threshold.
    """
    missing = missing_fields(records, [label_field, score_field])
    if missing:
        return {
            "supported": False,
            "reason": "missing_fields:" + ",".join(missing),
            "num_examples": len(records),
            "num_positive": 0,
            "num_negative": 0,
            "score_std": float("nan"),
        }

    labels = [int(record[label_field]) for record in records]
    num_positive = sum(labels)
    num_negative = len(labels) - num_positive
    score_std = score_standard_deviation(records, score_field) if records else float("nan")

    if len(records) < min_examples:
        reason = f"num_examples<{min_examples}"
    elif num_positive < min_positive:
        reason = f"num_positive<{min_positive}"
    elif num_negative < min_negative:
        reason = f"num_negative<{min_negative}"
    elif not isfinite(score_std) or score_std == 0.0:
        reason = "zero_score_std"
    else:
        reason = "supported"

    return {
        "supported": reason == "supported",
        "reason": reason,
        "num_examples": len(records),
        "num_positive": num_positive,
        "num_negative": num_negative,
        "score_std": score_std,
    }


def stratified_fold_indices(
    records: List[Dict],
    label_field: str,
    num_folds: int,
) -> List[List[int]]:
    require_metric_fields(records, [label_field])
    num_folds = max(2, min(num_folds, len(records)))
    folds = [[] for _ in range(num_folds)]
    by_label: Dict[int, List[int]] = {}

    for index, record in enumerate(records):
        label = int(record[label_field])
        by_label.setdefault(label, []).append(index)

    for label in sorted(by_label):
        for offset, index in enumerate(by_label[label]):
            folds[offset % num_folds].append(index)

    return [sorted(fold) for fold in folds if fold]


def cross_validated_best_threshold_metrics(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    thresholds: List[float] | None = None,
    num_folds: int = DEFAULT_QUESTION_TYPE_NUM_FOLDS,
) -> Dict[str, object]:
    """
    Select thresholds on training folds and report held-out predictions.
    """
    require_metric_fields(records, [label_field, similarity_field])
    if thresholds is None:
        thresholds = [round(x, 2) for x in np.arange(0.0, 1.01, 0.01)]

    folds = stratified_fold_indices(records, label_field, num_folds)
    selected_thresholds = []
    y_true = []
    y_score = []
    y_pred = []
    all_indices = set(range(len(records)))

    for validation_indices in folds:
        validation_set = set(validation_indices)
        training_indices = sorted(all_indices - validation_set)
        if not training_indices:
            continue

        validation_records = [records[index] for index in validation_indices]
        best_metrics = best_threshold_from_arrays(
            y_true=[int(records[index][label_field]) for index in training_indices],
            y_score=[
                float(records[index][similarity_field])
                for index in training_indices
            ],
            thresholds=thresholds,
        )
        threshold = float(best_metrics["threshold"])
        selected_thresholds.append(threshold)

        for record in validation_records:
            score = float(record[similarity_field])
            y_true.append(int(record[label_field]))
            y_score.append(score)
            y_pred.append(threshold_similarity(score, threshold))

    if not y_true:
        raise ValueError("Cannot cross-validate threshold without validation records.")

    mean_threshold = float(np.mean(selected_thresholds)) if selected_thresholds else 0.0
    metrics = classification_metrics_from_predictions(
        y_true=y_true,
        y_score=y_score,
        y_pred=y_pred,
        threshold=mean_threshold,
    )
    metrics.update(
        {
            "num_folds": len(folds),
            "selected_thresholds": ";".join(
                f"{threshold:.2f}" for threshold in selected_thresholds
            ),
            "mean_selected_threshold": mean_threshold,
            "threshold_std": float(np.std(selected_thresholds))
            if selected_thresholds
            else float("nan"),
        }
    )
    return metrics


def add_group_zscore_scores(
    records: List[Dict],
    score_field: str,
    group_field: str,
    output_field: str,
) -> List[Dict]:
    """
    Add within-group z-scores for optional calibrated reporting.
    """
    require_metric_fields(records, [score_field, group_field])
    grouped_scores: Dict[str, List[float]] = {}
    for record in records:
        grouped_scores.setdefault(str(record[group_field]), []).append(
            float(record[score_field])
        )

    group_stats = {
        group: (float(np.mean(scores)), float(np.std(scores)))
        for group, scores in grouped_scores.items()
    }

    output_records = []
    for record in records:
        group = str(record[group_field])
        score_mean, score_std = group_stats[group]
        new_record = dict(record)
        if score_std == 0.0:
            new_record[output_field] = 0.0
        else:
            new_record[output_field] = (float(record[score_field]) - score_mean) / score_std
        output_records.append(new_record)

    return output_records


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
