import re
from typing import Dict, List

from src.utils import normalize_text


def remove_articles(text: str) -> str:
    """
    Remove English articles.
    """
    return re.sub(r"\b(a|an|the)\b", " ", text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation.
    """
    return re.sub(r"[^\w\s]", " ", text)


def normalize_answer(text: str) -> str:
    """
    Normalize answer for exact match / token F1.
    """
    text = normalize_text(text)
    text = remove_articles(text)
    text = remove_punctuation(text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Exact match after normalization.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score between prediction and ground truth.
    Useful for short-answer QA.
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0

    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    num_same = sum(min(pred_tokens.count(tok), gt_tokens.count(tok)) for tok in common)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)

    return 2 * precision * recall / (precision + recall)


def contains_ground_truth(prediction: str, ground_truth: str) -> int:
    """
    Check whether normalized ground truth appears in normalized prediction.
    Useful for short answers like SciQ.
    """
    pred_norm = normalize_answer(prediction)
    gt_norm = normalize_answer(ground_truth)

    if not gt_norm:
        return 0

    return int(gt_norm in pred_norm)


def label_correctness_for_record(
    record: Dict,
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
    f1_threshold: float = 0.8,
) -> Dict:
    """
    Add correctness labels to one record.

    Labels:
        exact_match
        token_f1
        contains_ground_truth
        correct_label

    For the first SciQ baseline, correct_label uses:
        exact_match OR contains_ground_truth OR token_f1 >= threshold
    """
    prediction = record.get(prediction_field, "")
    ground_truth = record.get(reference_field, "")

    em = exact_match_score(prediction, ground_truth)
    f1 = token_f1_score(prediction, ground_truth)
    contains = contains_ground_truth(prediction, ground_truth)

    correct_label = int(em == 1 or contains == 1 or f1 >= f1_threshold)

    new_record = dict(record)
    new_record["exact_match"] = em
    new_record["token_f1"] = f1
    new_record["contains_ground_truth"] = contains
    new_record["correct_label"] = correct_label

    return new_record


def label_correctness_for_records(
    records: List[Dict],
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
    f1_threshold: float = 0.8,
) -> List[Dict]:
    """
    Add correctness labels to all records.
    """
    return [
        label_correctness_for_record(
            record=record,
            prediction_field=prediction_field,
            reference_field=reference_field,
            f1_threshold=f1_threshold,
        )
        for record in records
    ]