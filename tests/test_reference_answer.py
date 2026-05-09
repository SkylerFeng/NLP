import unittest

from src.reference_answer import (
    build_reference_quality_report,
    extract_nq_reference_answer,
    extract_nq_reference_answer_v2,
    prepare_reference_answers,
    question_type_v2,
    resolve_reference_field,
    validate_reference_answer,
)


class ReferenceAnswerExtractionTest(unittest.TestCase):
    def test_extracts_actor_from_nq_passage(self):
        result = extract_nq_reference_answer(
            "who plays batman in the lego batman movie",
            (
                "The Lego Batman Movie is a 2017 film produced by Warner Animation Group. "
                "The story features Will Arnett reprising his role as Batman for the film."
            ),
        )
        self.assertEqual(result["reference_answer"], "Will Arnett")

    def test_extracts_location_from_nq_passage(self):
        result = extract_nq_reference_answer(
            "where is the european court of human rights located",
            "European Court of Human Rights The Court is based in Strasbourg, France.",
        )
        self.assertEqual(result["reference_answer"], "Strasbourg, France")

    def test_extracts_number_from_nq_passage(self):
        result = extract_nq_reference_answer(
            "how many cylinders does a v8 engine have",
            "A V8 engine is an eight-cylinder V configuration engine.",
        )
        self.assertEqual(result["reference_answer"], "eight-cylinder V configuration")

    def test_extracts_date_from_nq_passage(self):
        result = extract_nq_reference_answer(
            "when does the fourth season of the flash come out",
            "The fourth season began airing on October 10, 2017, on The CW.",
        )
        self.assertEqual(result["reference_answer"], "October 10, 2017")

    def test_auto_reference_field_uses_nq_reference_answer(self):
        self.assertEqual(
            resolve_reference_field({"data": {"dataset": "nq"}, "evaluation": {}}),
            "reference_answer",
        )

    def test_question_type_v2_wraps_date_like_questions(self):
        self.assertEqual(question_type_v2("what year did the film come out"), "when")

    def test_rejects_pronoun_reference(self):
        result = validate_reference_answer(
            "He",
            "who wrote the song",
            "John Smith wrote the song after he moved to Nashville.",
        )

        self.assertFalse(result["reference_answer_valid"])
        self.assertEqual(result["reference_validation_reason"], "pronoun_or_determiner")

    def test_rejects_month_as_location_reference(self):
        result = extract_nq_reference_answer_v2(
            "where was the festival held",
            "The festival was held in May. The festival was held in Berlin, Germany.",
        )

        self.assertEqual(result["reference_answer_v2"], "Berlin, Germany")
        self.assertTrue(result["reference_answer_valid"])

    def test_rejects_malformed_year_fragment(self):
        result = validate_reference_answer(
            "000",
            "how many people attended the event",
            "The event drew 5,000 people.",
        )

        self.assertFalse(result["reference_answer_valid"])
        self.assertEqual(result["reference_validation_reason"], "malformed_numeric_fragment")

    def test_marks_long_evidence_sentence_reference(self):
        result = extract_nq_reference_answer_v2(
            "what is the passage about",
            (
                "This article gives a long descriptive overview with many details "
                "about background context, publication history, reception, awards, "
                "production notes, and later influence across several related works."
            ),
            max_evidence_fallback_tokens=8,
        )

        self.assertEqual(result["reference_answer_source_v2"], "nq_evidence_sentence")
        self.assertFalse(result["reference_answer_valid"])
        self.assertEqual(result["reference_validation_reason"], "long_evidence_fallback")

    def test_prepare_reference_answers_adds_v2_fields_without_mutating_baseline(self):
        records = [
            {
                "question": "where is the court located",
                "ground_truth": "The court is based in Strasbourg, France.",
            }
        ]

        result = prepare_reference_answers(records, "nq")[0]

        self.assertEqual(result["reference_answer"], "Strasbourg, France")
        self.assertEqual(result["reference_answer_v2"], "Strasbourg, France")
        self.assertIn("reference_answer_valid", result)
        self.assertIn("reference_answer_source_v2", result)

    def test_reference_quality_report_compares_baseline_and_v2(self):
        rows = build_reference_quality_report(
            [
                {
                    "question": "who wrote the song",
                    "ground_truth": "John Smith wrote the song.",
                    "reference_evidence": "He wrote the song.",
                    "reference_answer": "He",
                    "reference_answer_source": "nq_who_heuristic",
                    "reference_answer_v2": "John Smith",
                    "reference_answer_source_v2": "nq_who_heuristic",
                    "reference_answer_valid": True,
                    "reference_validation_reason": "valid",
                }
            ]
        )

        pronoun_rows = {
            row["reference_field"]: row["value"]
            for row in rows
            if row["metric"] == "pronoun_reference_count"
        }
        self.assertEqual(pronoun_rows["reference_answer"], 1)
        self.assertEqual(pronoun_rows["reference_answer_v2"], 0)


if __name__ == "__main__":
    unittest.main()
