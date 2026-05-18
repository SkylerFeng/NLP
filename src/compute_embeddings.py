from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm

from src.utils import chunk_list


class BaseEmbeddingModel:
    """
    Base class for embedding models.
    """

    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence-Transformers embedding model.

    Requires:
        pip install sentence-transformers
    """

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )


def create_embedding_model(model_name: str) -> BaseEmbeddingModel:
    """
    Create embedding model by name.
    """
    return SentenceTransformerEmbeddingModel(model_name=model_name)


def compute_text_embeddings(
    texts: List[str],
    embedding_model: BaseEmbeddingModel,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts.
    """
    if not texts:
        return np.array([])

    all_embeddings = []

    for batch in tqdm(list(chunk_list(texts, batch_size)), desc="Computing embeddings"):
        batch_embeddings = embedding_model.encode(batch)
        all_embeddings.append(batch_embeddings)

    return np.vstack(all_embeddings)


def embedding_cache_key(text: object) -> str:
    """
    Normalize text just enough for cache lookup without changing casing.
    """
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


class EmbeddingCache:
    """
    In-memory embedding cache for one embedding model.

    The cache is intentionally per model, so keys only need to represent text.
    It reuses embeddings across multiple similarity views in one pipeline run.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        batch_size: int = 32,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self._embeddings: Dict[str, np.ndarray] = {}

    def ensure_texts(self, texts: Iterable[object]) -> None:
        missing_texts = []
        missing_keys = set()
        for text in texts:
            key = embedding_cache_key(text)
            if key in self._embeddings or key in missing_keys:
                continue
            missing_keys.add(key)
            missing_texts.append(key)

        if not missing_texts:
            return

        embeddings = compute_text_embeddings(
            missing_texts,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size,
        )
        for text, embedding in zip(missing_texts, embeddings):
            self._embeddings[text] = embedding

    def embeddings_for(self, texts: List[object]) -> np.ndarray:
        if not texts:
            return np.array([])

        self.ensure_texts(texts)
        return np.vstack(
            [
                self._embeddings[embedding_cache_key(text)]
                for text in texts
            ]
        )

    def embedding_lookup(self, texts: Iterable[str]) -> Dict[str, np.ndarray]:
        text_list = list(texts)
        self.ensure_texts(text_list)
        return {
            text: self._embeddings[embedding_cache_key(text)]
            for text in text_list
        }


def add_embeddings_to_records(
    records: List[Dict],
    embedding_model: BaseEmbeddingModel,
    batch_size: int = 32,
    prediction_field: str = "prediction",
    reference_field: str = "ground_truth",
) -> List[Dict]:
    """
    Add prediction and reference embeddings to records.

    Note:
        Storing full embeddings in JSONL can make files large.
        This function is useful for small experiments.
    """
    predictions = [record.get(prediction_field, "") for record in records]
    references = [record.get(reference_field, "") for record in records]

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

    output_records = []

    for record, pred_emb, ref_emb in zip(records, prediction_embeddings, reference_embeddings):
        new_record = dict(record)
        new_record["prediction_embedding"] = pred_emb.tolist()
        new_record["reference_embedding"] = ref_emb.tolist()
        output_records.append(new_record)

    return output_records
