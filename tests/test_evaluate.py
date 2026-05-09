import importlib.util
import unittest
from pathlib import Path

from src.evaluate import evaluate_similarity_as_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_SCRIPT = PROJECT_ROOT / "scripts" / "03_evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", EVALUATE_SCRIPT)
evaluate_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate_script)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_KEY = "sentence_transformers_all_MiniLM_L6_v2"
SIMILARITY_FIELD = f"similarity_{MODEL_KEY}"
HYBRID_FIELD = f"hybrid_{MODEL_KEY}"


def base_config() -> dict:
    return {
        "data": {"dataset": "nq", "sample_size": 2},
        "evaluation": {"label_field": "correct_label", "similarity_threshold": 0.75},
        "embedding": {"models": [MODEL_NAME]},
    }


def base_records() -> list[dict]:
    return [
        {
            "id": "ok",
            "dataset": "nq",
            "question": "where is the court located",
            "ground_truth": "The court is in Strasbourg, France.",
            "reference_answer": "Strasbourg, France",
            "prediction": "Strasbourg, France",
            "correct_label": 1,
            "exact_match": 1,
            "contains_ground_truth": 1,
            "token_f1": 1.0,
            "entity_overlap": 1.0,
            SIMILARITY_FIELD: 0.9,
            HYBRID_FIELD: 0.85,
        },
        {
            "id": "bad",
            "dataset": "nq",
            "question": "where is the court located",
            "ground_truth": "The court is in Strasbourg, France.",
            "reference_answer": "Strasbourg, France",
            "prediction": "Berlin",
            "correct_label": 0,
            "exact_match": 0,
            "contains_ground_truth": 0,
            "token_f1": 0.0,
            "entity_overlap": 0.0,
            SIMILARITY_FIELD: 0.2,
            HYBRID_FIELD: 0.15,
        },
    ]


class EvaluationHarnessTest(unittest.TestCase):
    def test_baseline_rows_record_field_metadata(self):
        rows = evaluate_script.build_baseline_ablation_rows(
            base_records(),
            base_config(),
            "reference_answer",
        )

        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertEqual(row["stage"], "baseline")
            self.assertEqual(row["label_field"], "correct_label")
            self.assertEqual(row["reference_field"], "reference_answer")
            self.assertIn("score_field", row)
            self.assertIn("method", row)

    def test_v2_label_fields_do_not_change_baseline_rows_when_not_selected(self):
        rows_without_v2 = evaluate_script.build_baseline_ablation_rows(
            base_records(),
            base_config(),
            "reference_answer",
        )
        records_with_v2 = [
            {**record, "correct_label_v2": 1 - record["correct_label"]}
            for record in base_records()
        ]
        rows_with_v2 = evaluate_script.build_baseline_ablation_rows(
            records_with_v2,
            base_config(),
            "reference_answer",
        )

        self.assertEqual(rows_without_v2, rows_with_v2)

    def test_missing_configured_v2_fields_are_skipped(self):
        config = base_config()
        config["evaluation"]["ablation_scores"] = [
            {
                "stage": "unit1",
                "method": "Reference validation v2",
                "family": "embedding_v2",
                "score_field": "similarity_v2_sentence_transformers_all_MiniLM_L6_v2",
                "label_field": "correct_label",
                "reference_field": "reference_answer_v2",
            }
        ]
        records = [
            {
                **record,
                "similarity_v2_sentence_transformers_all_MiniLM_L6_v2": 0.8,
            }
            for record in base_records()
        ]

        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            config,
            "reference_answer",
        )

        self.assertNotIn("Reference validation v2", {row["method"] for row in rows})

    def test_configured_ablation_row_uses_exact_fields(self):
        v2_field = f"span_max_similarity_{MODEL_KEY}"
        config = base_config()
        config["evaluation"]["ablation_scores"] = [
            {
                "stage": "unit2",
                "method": "Prediction span max",
                "family": "span_ablation",
                "score_field": v2_field,
                "label_field": "correct_label",
                "reference_field": "reference_answer_v2",
                "threshold": 0.6,
            }
        ]
        records = [
            {**record, "reference_answer_v2": record["reference_answer"], v2_field: score}
            for record, score in zip(base_records(), [0.95, 0.1])
        ]

        rows = evaluate_script.build_baseline_ablation_rows(records, config, "reference_answer")
        row = next(row for row in rows if row["method"] == "Prediction span max")

        self.assertEqual(row["stage"], "unit2")
        self.assertEqual(row["score_field"], v2_field)
        self.assertEqual(row["label_field"], "correct_label")
        self.assertEqual(row["reference_field"], "reference_answer_v2")
        self.assertEqual(row["fixed_threshold"], 0.6)

    def test_reference_validation_v2_row_is_added_when_scores_exist(self):
        records = [
            {
                **record,
                "reference_answer_v2": record["reference_answer"],
                f"similarity_v2_{MODEL_KEY}": record[SIMILARITY_FIELD],
            }
            for record in base_records()
        ]

        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            base_config(),
            "reference_answer",
        )
        row = next(row for row in rows if row["stage"] == "unit1")

        self.assertEqual(row["method"], f"Reference validation v2: {MODEL_NAME}")
        self.assertEqual(row["score_field"], f"similarity_v2_{MODEL_KEY}")
        self.assertEqual(row["label_field"], "correct_label")
        self.assertEqual(row["reference_field"], "reference_answer_v2")

    def test_prediction_span_unit2_row_is_added_when_scores_exist(self):
        records = [
            {
                **record,
                "reference_answer_v2": record["reference_answer"],
                f"prediction_span_blend_similarity_{MODEL_KEY}": score,
            }
            for record, score in zip(base_records(), [0.95, 0.1])
        ]

        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            base_config(),
            "reference_answer",
        )
        row = next(row for row in rows if row["stage"] == "unit2")

        self.assertEqual(
            row["method"],
            f"Reference validation + 0.5 span blend: {MODEL_NAME}",
        )
        self.assertEqual(
            row["score_field"],
            f"prediction_span_blend_similarity_{MODEL_KEY}",
        )
        self.assertEqual(row["label_field"], "correct_label")
        self.assertEqual(row["reference_field"], "reference_answer_v2")

    def test_label_change_audit_rows_are_written_when_v2_label_exists(self):
        records = [
            {**record, "correct_label_v2": record["correct_label"]}
            for record in base_records()
        ]
        records[1]["correct_label_v2"] = 1

        rows = evaluate_script.build_label_change_audit_rows(
            records,
            base_config(),
            "reference_answer",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(row["label_changed"] for row in rows), 1)
        self.assertEqual(rows[0]["baseline_label_field"], "correct_label")
        self.assertEqual(rows[0]["candidate_label_field"], "correct_label_v2")

    def test_missing_metric_fields_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing_score"):
            evaluate_similarity_as_classifier(
                records=[{"correct_label": 1}],
                similarity_field="missing_score",
            )


if __name__ == "__main__":
    unittest.main()
