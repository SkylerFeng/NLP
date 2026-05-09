import unittest

from src.answer_span import (
    add_prediction_answer_spans,
    build_prediction_span_report,
    extract_prediction_answer_span,
)


class PredictionAnswerSpanExtractionTest(unittest.TestCase):
    def test_when_extracts_year_with_era(self):
        result = extract_prediction_answer_span(
            "when did alexander defeat the city",
            "Alexander defeated the city in 356 BCE during the campaign.",
        )

        self.assertEqual(result["prediction_answer_span"], "356 BCE")
        self.assertEqual(result["prediction_answer_span_source"], "when_date")

    def test_how_many_extracts_quantity(self):
        result = extract_prediction_answer_span(
            "how many teams are in the playoff",
            "There are 16 teams in the playoff, though older formats had 10 teams.",
        )

        self.assertEqual(result["prediction_answer_span"], "16 teams")
        self.assertEqual(result["prediction_answer_span_source"], "number_quantity")

    def test_who_extracts_named_person_instead_of_subject_pronoun(self):
        result = extract_prediction_answer_span(
            "who plays batman in the lego batman movie",
            "He was voiced by Will Arnett in the film.",
        )

        self.assertEqual(result["prediction_answer_span"], "Will Arnett")
        self.assertEqual(result["prediction_answer_span_source"], "who_by_phrase")

    def test_yes_no_preserves_polarity(self):
        result = extract_prediction_answer_span(
            "is mount everest in africa",
            "No, Mount Everest is in Asia.",
        )

        self.assertTrue(result["prediction_answer_span"].startswith("no"))
        self.assertEqual(result["prediction_answer_span_source"], "yes_no_polarity")

    def test_empty_or_uncertain_extraction_falls_back_safely(self):
        empty_result = extract_prediction_answer_span("who wrote the song", "")
        uncertain_result = extract_prediction_answer_span(
            "who wrote the song",
            "I don't know the answer.",
        )

        self.assertEqual(empty_result["prediction_answer_span"], "")
        self.assertEqual(empty_result["prediction_answer_span_source"], "empty_prediction")
        self.assertEqual(uncertain_result["prediction_answer_span"], "I don't know the answer")
        self.assertEqual(
            uncertain_result["prediction_answer_span_source"],
            "uncertain_prediction_fallback",
        )

    def test_add_prediction_answer_spans_adds_fields(self):
        records = [
            {
                "question": "where is the court located",
                "prediction": "The court is located in Strasbourg, France.",
            }
        ]

        result = add_prediction_answer_spans(records)[0]

        self.assertEqual(result["prediction_answer_span"], "Strasbourg, France")
        self.assertIn("prediction_answer_span_source", result)

    def test_prediction_span_report_tracks_fallback_rate_and_sources(self):
        records = [
            {
                "prediction_answer_span": "356 BCE",
                "prediction_answer_span_source": "when_date",
            },
            {
                "prediction_answer_span": "I don't know",
                "prediction_answer_span_source": "uncertain_prediction_fallback",
            },
        ]

        rows = build_prediction_span_report(records)
        values = {row["metric"]: row["value"] for row in rows if row["source"] == "all"}

        self.assertEqual(values["num_records"], 2)
        self.assertEqual(values["fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
