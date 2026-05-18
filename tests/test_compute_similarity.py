import unittest

import numpy as np

from src.compute_embeddings import EmbeddingCache
from src.compute_similarity import add_blended_similarity_scores, add_similarity_scores


class CountingEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors[text] for text in texts])


class SimilarityScoreTest(unittest.TestCase):
    def test_add_blended_similarity_scores_uses_configured_weight(self):
        records = [{"sentence_score": 0.2, "span_score": 0.8}]

        result = add_blended_similarity_scores(
            records,
            base_score_field="sentence_score",
            span_score_field="span_score",
            output_field="unit2_score",
            span_weight=0.5,
        )

        self.assertEqual(result[0]["unit2_score"], 0.5)

    def test_add_blended_similarity_scores_rejects_invalid_weight(self):
        with self.assertRaisesRegex(ValueError, "span_weight"):
            add_blended_similarity_scores(
                [],
                base_score_field="sentence_score",
                span_score_field="span_score",
                output_field="unit2_score",
                span_weight=1.5,
            )

    def test_add_blended_similarity_scores_rejects_missing_fields(self):
        with self.assertRaisesRegex(ValueError, "missing fields"):
            add_blended_similarity_scores(
                [{"sentence_score": 0.2}],
                base_score_field="sentence_score",
                span_score_field="span_score",
                output_field="unit2_score",
            )

    def test_add_similarity_scores_reuses_embedding_cache_across_views(self):
        records = [
            {
                "prediction": "Paris",
                "reference_answer": "Paris",
                "reference_answer_v2": "Paris",
            }
        ]
        model = CountingEmbeddingModel({"Paris": np.array([1.0, 0.0])})
        cache = EmbeddingCache(model, batch_size=32)

        records = add_similarity_scores(
            records,
            embedding_model=model,
            embedding_model_name="test/model",
            prediction_field="prediction",
            reference_field="reference_answer",
            embedding_cache=cache,
        )
        records = add_similarity_scores(
            records,
            embedding_model=model,
            embedding_model_name="test/model",
            prediction_field="prediction",
            reference_field="reference_answer_v2",
            output_field_prefix="similarity_v2",
            embedding_cache=cache,
        )

        encoded_texts = [text for call in model.calls for text in call]
        self.assertEqual(encoded_texts, ["Paris"])
        self.assertEqual(records[0]["similarity_test_model"], 1.0)
        self.assertEqual(records[0]["similarity_v2_test_model"], 1.0)


if __name__ == "__main__":
    unittest.main()
