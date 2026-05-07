import unittest

from src.reference_answer import extract_nq_reference_answer, resolve_reference_field


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


if __name__ == "__main__":
    unittest.main()
