import re
from typing import Dict, List

import numpy as np

from src.compute_embeddings import BaseEmbeddingModel, compute_text_embeddings
from src.compute_similarity import cosine_similarity


def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.

    This is enough for a baseline.
    Later you can replace it with nltk or spacy.
    """
    if text is None:
        return []

    text = text.strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    return sentences


def sentence_level_max_similarity(
    prediction: str,
    reference: str,
    embedding_model: BaseEmbeddingModel,
) -> float:
    """
    Compute sentence-level similarity.

    For each prediction sentence, find the maximum similarity
    with any reference sentence. Then average over prediction sentences.

    This is useful for long-form QA.
    """
    pred_sentences = split_into_sentences(prediction)
    ref_sentences = split_into_sentences(reference)

    if not pred_sentences or not ref_sentences:
        return 0.0

    pred_embeddings = compute_text_embeddings(
        pred_sentences,
        embedding_model=embedding_model,
        batch_size=len(pred_sentences),
    )

    ref_embeddings = compute_text_embeddings(
        ref_sentences,
        embedding_model=embedding_model,
        batch_size=len(ref_sentences),
    )

    pred_scores = []

    for pred_emb in pred_embeddings:
        max_score = max(cosine_similarity(pred_emb, ref_emb) for ref_emb in ref_embeddings)
        pred_scores.append(max_score)

    return float(np.mean(pred_scores))


def add_sentence_level_similarity_scores(
    records: List[Dict],
    embedding_model: BaseEmbeddingModel,
    embedding_model_name: str,
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
) -> List[Dict]:
    """
    Add sentence-level similarity scores to records.
    """
    safe_model_name = embedding_model_name.replace("/", "_").replace("-", "_")
    output_field = f"sentence_similarity_{safe_model_name}"

    output_records = []

    for record in records:
        score = sentence_level_max_similarity(
            prediction=record.get(prediction_field, ""),
            reference=record.get(reference_field, ""),
            embedding_model=embedding_model,
        )

        new_record = dict(record)
        new_record[output_field] = score
        output_records.append(new_record)

    return output_records