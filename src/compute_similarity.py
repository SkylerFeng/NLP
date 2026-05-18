from typing import Dict, List

import numpy as np

from src.compute_embeddings import BaseEmbeddingModel, EmbeddingCache, compute_text_embeddings


DEFAULT_SPAN_BLEND_WEIGHT = 0.5


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_cosine_similarities(
    prediction_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> List[float]:
    """
    Compute cosine similarity for each prediction-reference pair.
    """
    if len(prediction_embeddings) != len(reference_embeddings):
        raise ValueError("prediction_embeddings and reference_embeddings must have same length.")

    similarities = []

    for pred_emb, ref_emb in zip(prediction_embeddings, reference_embeddings):
        similarities.append(cosine_similarity(pred_emb, ref_emb))

    return similarities


def add_similarity_scores(
    records: List[Dict],
    embedding_model: BaseEmbeddingModel,
    embedding_model_name: str,
    batch_size: int = 32,
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
    output_field_prefix: str = "similarity",
    embedding_cache: EmbeddingCache | None = None,
) -> List[Dict]:
    """
    Compute prediction-reference cosine similarity and add it to records.

    Output field examples:
        similarity_all_MiniLM_L6_v2
        similarity_v2_all_MiniLM_L6_v2
    """
    predictions = [record.get(prediction_field, "") for record in records]
    references = [record.get(reference_field, "") for record in records]

    if embedding_cache is None:
        prediction_embeddings = compute_text_embeddings(
            predictions,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )

        reference_embeddings = compute_text_embeddings(
            references,
            embedding_model=embedding_model,
            batch_size=batch_size,
        )
    else:
        prediction_embeddings = embedding_cache.embeddings_for(predictions)
        reference_embeddings = embedding_cache.embeddings_for(references)

    similarities = compute_cosine_similarities(
        prediction_embeddings=prediction_embeddings,
        reference_embeddings=reference_embeddings,
    )

    safe_model_name = embedding_model_name.replace("/", "_").replace("-", "_")
    similarity_field = f"{output_field_prefix}_{safe_model_name}"

    output_records = []

    for record, similarity in zip(records, similarities):
        new_record = dict(record)
        new_record[similarity_field] = similarity
        output_records.append(new_record)

    return output_records


def add_blended_similarity_scores(
    records: List[Dict],
    base_score_field: str,
    span_score_field: str,
    output_field: str,
    span_weight: float = DEFAULT_SPAN_BLEND_WEIGHT,
) -> List[Dict]:
    """
    Add a conservative Unit 2 score that blends full-prediction and span views.

    The raw answer-span cosine is useful for ranking, but it can over-credit
    same-type wrong dates or numbers. The blended field keeps the answer-focus
    signal while retaining sentence-level context as a guardrail.
    """
    if not 0.0 <= span_weight <= 1.0:
        raise ValueError("span_weight must be between 0.0 and 1.0.")

    base_weight = 1.0 - span_weight
    output_records = []
    for record in records:
        missing = [
            field
            for field in (base_score_field, span_score_field)
            if field not in record
        ]
        if missing:
            raise ValueError(
                "Cannot blend similarity scores because record is missing fields: "
                + ", ".join(missing)
            )

        new_record = dict(record)
        base_score = float(record[base_score_field])
        span_score = float(record[span_score_field])
        new_record[output_field] = base_weight * base_score + span_weight * span_score
        output_records.append(new_record)
    return output_records


def threshold_similarity(
    similarity: float,
    threshold: float = 0.75,
) -> int:
    """
    Convert a similarity score into a predicted correctness label.

    Returns:
        1 if predicted correct, 0 otherwise.
    """
    return int(similarity >= threshold)
