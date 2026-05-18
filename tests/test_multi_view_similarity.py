import unittest

import numpy as np

from src.compute_embeddings import EmbeddingCache
from src.multi_view_similarity import (
    add_span_level_similarity_scores,
    prediction_span_candidates,
    topk_mean_similarity,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KEY = "sentence_transformers_all_MiniLM_L6_v2"


class DummyEmbeddingModel:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors.get(text, np.array([0.0, 1.0])) for text in texts])


class MultiViewSimilarityTest(unittest.TestCase):
    def test_span_max_similarity_uses_exact_short_answer_candidate(self):
        record = {
            "reference_answer_v2": "356",
            "prediction": "Alexander defeated the city in 356 BCE during the campaign.",
            "prediction_answer_span": "356 BCE",
            f"similarity_v2_{MODEL_KEY}": 0.2,
        }
        model = DummyEmbeddingModel(
            {
                "356": np.array([1.0, 0.0]),
                "356 BCE": np.array([1.0, 0.0]),
                "Alexander defeated the city in 356 BCE during the campaign": np.array([0.0, 1.0]),
            }
        )

        result = add_span_level_similarity_scores(
            [record],
            embedding_model=model,
            embedding_model_name=MODEL_NAME,
        )[0]

        self.assertGreaterEqual(
            result[f"span_max_similarity_{MODEL_KEY}"],
            record[f"similarity_v2_{MODEL_KEY}"],
        )
        self.assertAlmostEqual(result[f"span_max_similarity_{MODEL_KEY}"], 1.0)
        self.assertGreater(
            result[f"multi_view_score_{MODEL_KEY}"],
            record[f"similarity_v2_{MODEL_KEY}"],
        )

    def test_empty_span_candidates_fall_back_to_sentence_similarity(self):
        record = {
            "reference_answer_v2": "",
            "prediction": "",
            "prediction_answer_span": "",
            f"similarity_v2_{MODEL_KEY}": 0.37,
        }
        result = add_span_level_similarity_scores(
            [record],
            embedding_model=DummyEmbeddingModel({}),
            embedding_model_name=MODEL_NAME,
        )[0]

        self.assertEqual(result[f"span_max_similarity_{MODEL_KEY}"], 0.37)
        self.assertEqual(result[f"span_topk_mean_similarity_{MODEL_KEY}"], 0.37)
        self.assertEqual(
            result[f"reference_to_prediction_span_similarity_{MODEL_KEY}"],
            0.37,
        )
        self.assertEqual(result[f"multi_view_score_{MODEL_KEY}"], 0.37)

    def test_long_prediction_answer_span_is_not_used_as_a_raw_span_candidate(self):
        record = {
            "prediction": "The Little League World Series features 10 teams in this example.",
            "prediction_answer_span": "The Little League World Series features 10 teams",
        }

        candidates = prediction_span_candidates(record, max_span_tokens=5)

        self.assertNotIn(record["prediction_answer_span"], candidates)
        self.assertIn("10 teams", candidates)

    def test_topk_aggregation_is_deterministic(self):
        lookup = {
            "alpha": np.array([1.0, 0.0]),
            "beta": np.array([0.0, 1.0]),
            "gamma": np.array([0.0, 1.0]),
        }

        first = topk_mean_similarity(
            ["alpha", "beta"],
            ["alpha", "gamma"],
            lookup,
            top_k=2,
        )
        second = topk_mean_similarity(
            ["alpha", "beta"],
            ["alpha", "gamma"],
            lookup,
            top_k=2,
        )

        self.assertEqual(first, second)
        self.assertEqual(first, 1.0)

    def test_same_topic_wrong_answer_is_not_forced_to_exact_match_score(self):
        records = [
            {
                "reference_answer_v2": "Strasbourg",
                "prediction": "The court is in Berlin.",
                "prediction_answer_span": "Berlin",
                f"similarity_v2_{MODEL_KEY}": 0.1,
            },
            {
                "reference_answer_v2": "Strasbourg",
                "prediction": "The court is in Strasbourg.",
                "prediction_answer_span": "Strasbourg",
                f"similarity_v2_{MODEL_KEY}": 0.1,
            },
        ]
        model = DummyEmbeddingModel(
            {
                "Strasbourg": np.array([1.0, 0.0]),
                "Berlin": np.array([0.0, 1.0]),
            }
        )

        result = add_span_level_similarity_scores(
            records,
            embedding_model=model,
            embedding_model_name=MODEL_NAME,
        )

        wrong_score = result[0][f"reference_to_prediction_span_similarity_{MODEL_KEY}"]
        exact_score = result[1][f"reference_to_prediction_span_similarity_{MODEL_KEY}"]
        self.assertLess(wrong_score, exact_score)

    def test_prediction_candidates_include_answer_span_and_ngrams(self):
        record = {
            "prediction": "The answer is 356 BCE.",
            "prediction_answer_span": "356 BCE",
        }

        candidates = prediction_span_candidates(record)

        self.assertIn("356 BCE", candidates)
        self.assertIn("356", candidates)

    def test_span_level_similarity_reuses_embedding_cache(self):
        record = {
            "reference_answer_v2": "Strasbourg",
            "prediction": "The court is in Strasbourg.",
            "prediction_answer_span": "Strasbourg",
            f"similarity_v2_{MODEL_KEY}": 0.1,
        }
        model = DummyEmbeddingModel({"Strasbourg": np.array([1.0, 0.0])})
        cache = EmbeddingCache(model, batch_size=32)

        add_span_level_similarity_scores(
            [record],
            embedding_model=model,
            embedding_model_name=MODEL_NAME,
            embedding_cache=cache,
        )
        calls_after_first_run = len(model.calls)
        add_span_level_similarity_scores(
            [record],
            embedding_model=model,
            embedding_model_name=MODEL_NAME,
            embedding_cache=cache,
        )

        self.assertEqual(len(model.calls), calls_after_first_run)


if __name__ == "__main__":
    unittest.main()
