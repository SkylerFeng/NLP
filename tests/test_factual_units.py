import unittest

from src.factual_units import (
    add_factual_unit_features,
    compare_factual_units,
    extract_dates,
    extract_entity_like_spans,
    extract_numbers,
    factual_conflict_adjusted_score,
)


class FactualUnitsTest(unittest.TestCase):
    def test_different_population_numbers_produce_number_conflict(self):
        result = compare_factual_units(
            "The 2010 population was 916,542.",
            "The population is approximately 1,083,460 as of 2023.",
        )

        self.assertIn("916,542", result["reference_numbers"])
        self.assertIn("1,083,460", result["prediction_numbers"])
        self.assertEqual(result["number_conflict"], 1)
        self.assertGreater(result["factual_conflict_penalty"], 0.0)

    def test_same_year_with_different_surface_form_matches(self):
        result = compare_factual_units(
            "EBT has been implemented in all states since June 2004.",
            "Food stamps changed to an EBT card in 2004.",
        )

        self.assertEqual(result["date_match"], 1)
        self.assertEqual(result["date_conflict"], 0)

    def test_date_ranges_are_extracted_as_single_units(self):
        self.assertEqual(extract_dates("The show ran from 1990-1995."), ["1990-1995"])

    def test_parenthetical_entity_aliases_match(self):
        result = compare_factual_units(
            "Electronic Benefit Transfer (EBT) replaced stamps.",
            "The answer is EBT.",
        )

        self.assertIn("EBT", extract_entity_like_spans("Electronic Benefit Transfer (EBT)"))
        self.assertEqual(result["entity_match"], 1)
        self.assertEqual(result["entity_conflict"], 0)

    def test_repeated_entity_token_alias_does_not_conflict(self):
        result = compare_factual_units("Emma Stone Stone", "Emma Stone")

        self.assertEqual(result["entity_match"], 1)
        self.assertEqual(result["entity_conflict"], 0)

    def test_entity_substring_alias_does_not_conflict_for_multiword_entities(self):
        result = compare_factual_units(
            "The United States of America joined later.",
            "The United States joined later.",
        )

        self.assertEqual(result["entity_match"], 1)
        self.assertEqual(result["entity_conflict"], 0)

    def test_date_containment_is_partial_overlap_not_conflict(self):
        cases = [
            ("The war lasted from 1775-1783.", "It ended in 1783."),
            ("The invasion began on 19 September 2017.", "It began in 2017."),
            ("The war lasted 1955 to 1975.", "It began in 1955."),
            ("EBT has been implemented since June 2004.", "It started in 2004."),
        ]

        for reference, prediction in cases:
            with self.subTest(reference=reference, prediction=prediction):
                result = compare_factual_units(reference, prediction)

                self.assertEqual(result["date_match"], 1)
                self.assertEqual(result["date_conflict"], 0)
                self.assertEqual(result["partial_factual_overlap"], 1)

    def test_specific_date_mismatch_still_conflicts(self):
        cases = [
            ("It happened in June 2004.", "It happened in July 2004."),
            ("It happened on 19 September 2017.", "It happened on 20 September 2017."),
            ("The war lasted from 1775-1783.", "The war lasted from 1780-1784."),
        ]

        for reference, prediction in cases:
            with self.subTest(reference=reference, prediction=prediction):
                result = compare_factual_units(reference, prediction)

                self.assertEqual(result["date_match"], 0)
                self.assertEqual(result["date_conflict"], 1)

    def test_ordinal_numbers_extract_as_numbers(self):
        numbers = extract_numbers("She finished first and came 2nd in 2017.")

        self.assertIn("first", numbers)
        self.assertIn("2nd", numbers)

    def test_sentence_initial_false_entities_are_filtered(self):
        entities = extract_entity_like_spans(
            "However, Sound and You are not the answer. The answer is Emma Stone."
        )

        self.assertNotIn("However", entities)
        self.assertNotIn("Sound and You", entities)
        self.assertIn("Emma Stone", entities)

    def test_list_item_number_conflict_is_visible(self):
        result = compare_factual_units(
            "The Little League World Series consists of 16 teams, 8 from the United States and 8 international teams.",
            "The series features 10 teams, 8 from the U.S. and 2 international teams.",
        )

        self.assertEqual(result["number_match"], 1)
        self.assertEqual(result["number_conflict"], 1)
        self.assertLess(result["list_item_f1"], 1.0)

    def test_number_words_extract_as_numbers(self):
        numbers = extract_numbers("Starting between two and six hours after death.")

        self.assertIn("two", numbers)
        self.assertIn("six hours", numbers)

    def test_add_factual_unit_features_uses_configured_fields(self):
        records = [
            {
                "reference_answer_v2": "The population was 916,542.",
                "prediction_answer_span": "1,083,460",
            }
        ]

        result = add_factual_unit_features(records)[0]

        self.assertEqual(result["number_conflict"], 1)
        self.assertIn("factual_unit_score", result)

    def test_factual_conflict_adjusted_score_clamps_to_valid_range(self):
        self.assertEqual(factual_conflict_adjusted_score(0.1, 1.0, penalty_weight=0.25), 0.0)
        self.assertEqual(factual_conflict_adjusted_score(1.2, 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
