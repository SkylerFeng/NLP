import unittest

from src.compute_similarity import add_blended_similarity_scores


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


if __name__ == "__main__":
    unittest.main()
