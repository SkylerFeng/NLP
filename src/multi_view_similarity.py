import re
from typing import Dict, Iterable, List

import numpy as np

from src.compute_embeddings import BaseEmbeddingModel, EmbeddingCache, compute_text_embeddings
from src.compute_similarity import cosine_similarity
from src.reference_answer import clean_extracted_answer


DEFAULT_SPAN_TOP_K = 3
DEFAULT_MAX_SPAN_TOKENS = 5
DEFAULT_MAX_CANDIDATES = 32
DEFAULT_MULTI_VIEW_SPAN_WEIGHT = 0.1

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "were",
    "with",
}

MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|"
    "October|November|December"
)
ERA_PATTERN = r"(?:BCE|BC|CE|AD)"
DATE_PATTERNS = [
    rf"\b(?:{MONTH_PATTERN})\s+\d{{1,2}},\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
    rf"\b\d{{1,2}}\s+(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
    rf"\b(?:{MONTH_PATTERN})\s+\d{{3,4}}(?:\s*{ERA_PATTERN})?\b",
    rf"\b\d{{1,4}}\s*{ERA_PATTERN}\b",
    r"\b\d{3,4}\s*(?:-|–|to)\s*\d{2,4}\b",
]
NUMBER_PATTERNS = [
    r"\b\d+\s*/\s*\d+\b",
    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\s+[A-Za-z%]+)?\b",
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|percentage)\b",
    r"\b\d+(?:\.\d+)?\s+[A-Za-z]+(?:\s+[A-Za-z]+){0,2}\b",
    r"\b\d+(?:\.\d+)?\b",
]
ENTITY_PATTERN = (
    r"\b[A-Z][A-Za-z.'-]+(?:\s+(?:of|the|and|&|de|del|la|le|van|von|"
    r"[A-Z][A-Za-z.'-]+)){0,5}\b"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def normalized_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def clean_candidate(text: str) -> str:
    text = clean_extracted_answer(str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(str(text or "")))


def dedupe_candidates(
    candidates: Iterable[str],
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[str]:
    output: List[str] = []
    seen = set()
    for candidate in candidates:
        cleaned = clean_candidate(candidate)
        key = normalized_key(cleaned)
        if not key or key in seen:
            continue
        if all(token.lower() in STOPWORDS for token in TOKEN_PATTERN.findall(cleaned)):
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= max_candidates:
            break
    return output


def regex_candidates(patterns: Iterable[str], text: str, flags: int = 0) -> List[str]:
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags):
            candidates.append(match.group(1) if match.groups() else match.group(0))
    return candidates


def factual_unit_candidates(text: str) -> List[str]:
    return dedupe_candidates(
        [
            *regex_candidates(DATE_PATTERNS, text, flags=re.I),
            *regex_candidates(NUMBER_PATTERNS, text, flags=re.I),
            *regex_candidates([ENTITY_PATTERN], text),
        ]
    )


def short_ngram_candidates(
    text: str,
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
) -> List[str]:
    tokens = TOKEN_PATTERN.findall(str(text or ""))
    candidates = []
    for size in range(1, max_span_tokens + 1):
        for start in range(0, max(len(tokens) - size + 1, 0)):
            span_tokens = tokens[start:start + size]
            if all(token.lower() in STOPWORDS for token in span_tokens):
                continue
            candidates.append(" ".join(span_tokens))
    return dedupe_candidates(candidates)


def prediction_span_candidates(
    record: Dict,
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[str]:
    prediction = str(record.get("prediction", ""))
    seed_span = str(record.get("prediction_answer_span", ""))
    candidates = []
    if token_count(seed_span) <= max_span_tokens:
        candidates.append(seed_span)
    candidates.extend(factual_unit_candidates(prediction))
    candidates.extend(short_ngram_candidates(prediction, max_span_tokens=max_span_tokens))
    return dedupe_candidates(candidates, max_candidates=max_candidates)


def reference_span_candidates(
    record: Dict,
    reference_field: str = "reference_answer_v2",
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> List[str]:
    reference = str(record.get(reference_field, ""))
    candidates = [
        reference,
        *factual_unit_candidates(reference),
    ]
    if token_count(reference) > max_span_tokens:
        candidates.extend(short_ngram_candidates(reference, max_span_tokens=max_span_tokens))
    return dedupe_candidates(candidates, max_candidates=max_candidates)


def topk_mean_similarity(
    reference_candidates: List[str],
    prediction_candidates: List[str],
    embedding_lookup: Dict[str, np.ndarray],
    top_k: int = DEFAULT_SPAN_TOP_K,
) -> float | None:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not reference_candidates or not prediction_candidates:
        return None

    similarities = []
    for reference in reference_candidates:
        reference_embedding = embedding_lookup.get(reference)
        if reference_embedding is None:
            continue
        for prediction in prediction_candidates:
            prediction_embedding = embedding_lookup.get(prediction)
            if prediction_embedding is None:
                continue
            similarities.append(cosine_similarity(reference_embedding, prediction_embedding))

    if not similarities:
        return None

    similarities.sort(reverse=True)
    return float(np.mean(similarities[:top_k]))


def max_reference_to_prediction_similarity(
    reference: str,
    prediction_candidates: List[str],
    embedding_lookup: Dict[str, np.ndarray],
) -> float | None:
    reference_embedding = embedding_lookup.get(reference)
    if reference_embedding is None or not prediction_candidates:
        return None

    similarities = [
        cosine_similarity(reference_embedding, embedding_lookup[candidate])
        for candidate in prediction_candidates
        if candidate in embedding_lookup
    ]
    if not similarities:
        return None
    return float(max(similarities))


def pair_similarity(
    reference: str,
    prediction: str,
    embedding_lookup: Dict[str, np.ndarray],
) -> float | None:
    reference_embedding = embedding_lookup.get(reference)
    prediction_embedding = embedding_lookup.get(prediction)
    if reference_embedding is None or prediction_embedding is None:
        return None
    return cosine_similarity(reference_embedding, prediction_embedding)


def build_embedding_lookup(
    texts: Iterable[str],
    embedding_model: BaseEmbeddingModel,
    batch_size: int,
    embedding_cache: EmbeddingCache | None = None,
) -> Dict[str, np.ndarray]:
    unique_texts = dedupe_candidates(texts, max_candidates=10**9)
    if not unique_texts:
        return {}

    if embedding_cache is not None:
        return embedding_cache.embedding_lookup(unique_texts)

    embeddings = compute_text_embeddings(
        unique_texts,
        embedding_model=embedding_model,
        batch_size=batch_size,
    )
    return {
        text: embedding
        for text, embedding in zip(unique_texts, embeddings)
    }


def add_span_level_similarity_scores(
    records: List[Dict],
    embedding_model: BaseEmbeddingModel,
    embedding_model_name: str,
    batch_size: int = 32,
    reference_field: str = "reference_answer_v2",
    sentence_similarity_field: str | None = None,
    top_k: int = DEFAULT_SPAN_TOP_K,
    max_span_tokens: int = DEFAULT_MAX_SPAN_TOKENS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    multi_view_span_weight: float = DEFAULT_MULTI_VIEW_SPAN_WEIGHT,
    embedding_cache: EmbeddingCache | None = None,
) -> List[Dict]:
    """
    Add Unit 3 multi-granularity span similarity fields.

    Span scores are evaluated against frozen labels in the evaluator. If a
    record has no usable candidates, the score falls back to the existing
    sentence-level v2 similarity field when available.
    """
    model_key = safe_model_name(embedding_model_name)
    sentence_similarity_field = sentence_similarity_field or f"similarity_v2_{model_key}"
    base_multi_view_field = f"prediction_span_blend_similarity_{model_key}"
    if not 0.0 <= multi_view_span_weight <= 1.0:
        raise ValueError("multi_view_span_weight must be between 0.0 and 1.0.")

    per_record_candidates = []
    texts_to_embed = []
    for record in records:
        reference_text = clean_candidate(record.get(reference_field, ""))
        prediction_span = clean_candidate(record.get("prediction_answer_span", ""))
        prediction_candidates = prediction_span_candidates(
            record,
            max_span_tokens=max_span_tokens,
            max_candidates=max_candidates,
        )
        reference_candidates = reference_span_candidates(
            record,
            reference_field=reference_field,
            max_span_tokens=max_span_tokens,
            max_candidates=max_candidates,
        )
        per_record_candidates.append(
            {
                "reference_text": reference_text,
                "prediction_span": prediction_span,
                "prediction_candidates": prediction_candidates,
                "reference_candidates": reference_candidates,
            }
        )
        texts_to_embed.extend([reference_text, prediction_span])
        texts_to_embed.extend(prediction_candidates)
        texts_to_embed.extend(reference_candidates)

    embedding_lookup = build_embedding_lookup(
        texts_to_embed,
        embedding_model=embedding_model,
        batch_size=batch_size,
        embedding_cache=embedding_cache,
    )

    output_records = []
    for record, candidates in zip(records, per_record_candidates):
        fallback = float(record.get(sentence_similarity_field, 0.0))
        reference_text = candidates["reference_text"]
        prediction_span = candidates["prediction_span"]
        prediction_candidates = candidates["prediction_candidates"]
        reference_candidates = candidates["reference_candidates"]

        span_max = max_reference_to_prediction_similarity(
            reference_text,
            prediction_candidates,
            embedding_lookup,
        )
        span_topk = topk_mean_similarity(
            reference_candidates,
            prediction_candidates,
            embedding_lookup,
            top_k=top_k,
        )
        reference_to_prediction_span = pair_similarity(
            reference_text,
            prediction_span,
            embedding_lookup,
        )

        new_record = dict(record)
        span_max_value = fallback if span_max is None else span_max
        span_topk_value = fallback if span_topk is None else span_topk
        reference_to_prediction_span_value = (
            fallback if reference_to_prediction_span is None else reference_to_prediction_span
        )
        base_multi_view_score = float(record.get(base_multi_view_field, fallback))
        new_record[f"span_max_similarity_{model_key}"] = span_max_value
        new_record[f"span_topk_mean_similarity_{model_key}"] = span_topk_value
        new_record[f"reference_to_prediction_span_similarity_{model_key}"] = (
            reference_to_prediction_span_value
        )
        new_record[f"multi_view_score_{model_key}"] = (
            (1.0 - multi_view_span_weight) * base_multi_view_score
            + multi_view_span_weight * span_max_value
        )
        output_records.append(new_record)

    return output_records
