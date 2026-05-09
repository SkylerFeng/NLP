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
