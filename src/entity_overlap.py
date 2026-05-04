import re
from typing import Dict, List, Set

from src.utils import normalize_text


def extract_simple_entities(text: str) -> Set[str]:
    """
    A lightweight entity-like extractor.

    This baseline extracts:
        - capitalized words / phrases
        - numbers
        - short scientific terms may not be captured well

    For a stronger version, later use spaCy NER.
    """
    if text is None:
        return set()

    entities = set()

    capitalized_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)

    for item in capitalized_phrases + numbers:
        item = normalize_text(item)
        if item:
            entities.add(item)

    return entities


def token_set(text: str) -> Set[str]:
    """
    Fallback token set.
    """
    text = normalize_text(text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = set(text.split())

    stopwords = {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "of",
        "in",
        "on",
        "to",
        "and",
        "or",
        "for",
        "with",
        "by",
        "as",
        "at",
    }

    return {tok for tok in tokens if tok not in stopwords}


def overlap_score(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute overlap score between two sets.

    Uses F1-style overlap:
        2 * precision * recall / (precision + recall)
    """
    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    common = set_a & set_b

    if not common:
        return 0.0

    precision = len(common) / len(set_a)
    recall = len(common) / len(set_b)

    return 2 * precision * recall / (precision + recall)


def entity_overlap_score(prediction: str, reference: str) -> float:
    """
    Compute entity overlap score.

    If no entities are found, fall back to content-token overlap.
    """
    pred_entities = extract_simple_entities(prediction)
    ref_entities = extract_simple_entities(reference)

    if pred_entities or ref_entities:
        return overlap_score(pred_entities, ref_entities)

    return overlap_score(token_set(prediction), token_set(reference))


def add_entity_overlap_scores(
    records: List[Dict],
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
) -> List[Dict]:
    """
    Add entity overlap scores to records.
    """
    output_records = []

    for record in records:
        score = entity_overlap_score(
            prediction=record.get(prediction_field, ""),
            reference=record.get(reference_field, ""),
        )

        new_record = dict(record)
        new_record["entity_overlap"] = score
        output_records.append(new_record)

    return output_records


def hybrid_similarity_score(
    embedding_similarity: float,
    entity_score: float,
    alpha: float = 0.7,
) -> float:
    """
    Combine embedding similarity and entity overlap.

    final_score = alpha * embedding_similarity + (1 - alpha) * entity_score
    """
    return alpha * embedding_similarity + (1 - alpha) * entity_score