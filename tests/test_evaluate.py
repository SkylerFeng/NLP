import importlib.util
import unittest
from pathlib import Path

from src.evaluate import (
    add_group_zscore_scores,
    evaluate_similarity_as_classifier,
    multi_view_hybrid_score,
)


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

    def test_span_similarity_unit3_rows_are_added_when_scores_exist(self):
        records = [
            {
                **record,
                "reference_answer_v2": record["reference_answer"],
                f"multi_view_score_{MODEL_KEY}": score,
                f"span_max_similarity_{MODEL_KEY}": score,
                f"span_topk_mean_similarity_{MODEL_KEY}": score,
                f"reference_to_prediction_span_similarity_{MODEL_KEY}": score,
            }
            for record, score in zip(base_records(), [0.95, 0.1])
        ]

        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            base_config(),
            "reference_answer",
        )
        unit3_rows = [row for row in rows if row["stage"] == "unit3"]

        self.assertEqual(len(unit3_rows), 4)
        self.assertEqual(
            {
                row["score_field"]
                for row in unit3_rows
            },
            {
                f"multi_view_score_{MODEL_KEY}",
                f"span_max_similarity_{MODEL_KEY}",
                f"span_topk_mean_similarity_{MODEL_KEY}",
                f"reference_to_prediction_span_similarity_{MODEL_KEY}",
            },
        )
        self.assertEqual(
            {row["reference_field"] for row in unit3_rows},
            {"reference_answer_v2"},
        )

    def test_factual_unit4_rows_are_added_when_scores_exist(self):
        records = [
            {
                **record,
                "reference_answer_v2": record["reference_answer"],
                "factual_unit_score": score,
                f"factual_conflict_adjusted_similarity_{MODEL_KEY}": score,
                f"factual_conflict_adjusted_prediction_span_blend_similarity_{MODEL_KEY}": score,
                f"factual_conflict_adjusted_span_max_similarity_{MODEL_KEY}": score,
                f"factual_conflict_adjusted_multi_view_score_{MODEL_KEY}": score,
            }
            for record, score in zip(base_records(), [0.95, 0.1])
        ]

        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            base_config(),
            "reference_answer",
        )
        unit4_rows = [row for row in rows if row["stage"] == "unit4"]

        self.assertEqual(len(unit4_rows), 5)
        self.assertEqual(
            {
                row["score_field"]
                for row in unit4_rows
            },
            {
                "factual_unit_score",
                f"factual_conflict_adjusted_similarity_{MODEL_KEY}",
                f"factual_conflict_adjusted_prediction_span_blend_similarity_{MODEL_KEY}",
                f"factual_conflict_adjusted_span_max_similarity_{MODEL_KEY}",
                f"factual_conflict_adjusted_multi_view_score_{MODEL_KEY}",
            },
        )
        self.assertEqual(
            {row["reference_field"] for row in unit4_rows},
            {"reference_answer_v2"},
        )
        self.assertIn("high_similarity_wrong", unit4_rows[0])

    def test_multi_view_hybrid_score_penalizes_same_topic_number_conflict(self):
        clean_score = multi_view_hybrid_score(
            sentence_similarity=0.90,
            span_max_similarity=0.88,
            entity_or_token_overlap=0.80,
            factual_conflict_penalty=0.0,
        )
        conflict_score = multi_view_hybrid_score(
            sentence_similarity=0.90,
            span_max_similarity=0.88,
            entity_or_token_overlap=0.80,
            factual_conflict_penalty=1.0,
        )

        self.assertAlmostEqual(clean_score - conflict_score, 0.25)
        self.assertLess(conflict_score, clean_score)

    def test_multi_view_hybrid_score_rewards_short_answer_span_alignment(self):
        sentence_only_score = multi_view_hybrid_score(
            sentence_similarity=0.40,
            span_max_similarity=0.40,
            entity_or_token_overlap=0.0,
            factual_conflict_penalty=0.0,
        )
        span_aligned_score = multi_view_hybrid_score(
            sentence_similarity=0.40,
            span_max_similarity=0.95,
            entity_or_token_overlap=0.0,
            factual_conflict_penalty=0.0,
        )

        self.assertGreater(span_aligned_score, sentence_only_score)

    def test_multi_view_hybrid_score_clamps_and_ignores_missing_optional_values(self):
        self.assertEqual(
            multi_view_hybrid_score(
                sentence_similarity=0.05,
                span_max_similarity=None,
                entity_or_token_overlap=float("nan"),
                factual_conflict_penalty=1.0,
            ),
            0.0,
        )
        self.assertEqual(
            multi_view_hybrid_score(
                sentence_similarity=1.2,
                span_max_similarity=None,
                entity_or_token_overlap=None,
                factual_conflict_penalty=0.0,
            ),
            1.0,
        )

    def test_multi_view_hybrid_unit6_rows_are_added_when_scores_exist(self):
        records = []
        for record, score in zip(base_records(), [0.95, 0.1]):
            enriched = {
                **record,
                "reference_answer_v2": record["reference_answer"],
                f"similarity_v2_{MODEL_KEY}": score,
                f"span_max_similarity_{MODEL_KEY}": score,
                "factual_conflict_penalty": 0.0,
            }
            records.append(enriched)
        records = evaluate_script.add_multi_view_hybrid_scores(records, [MODEL_NAME])

        self.assertIn("unit6_entity_or_token_overlap", records[0])
        rows = evaluate_script.build_baseline_ablation_rows(
            records,
            base_config(),
            "reference_answer",
        )
        unit6_rows = [row for row in rows if row["stage"] == "unit6"]

        self.assertEqual(len(unit6_rows), 5)
        self.assertEqual(
            {
                row["score_field"]
                for row in unit6_rows
            },
            {
                f"unit6_fixed_multi_view_hybrid_score_{MODEL_KEY}",
                f"unit6_span_precision_multi_view_hybrid_score_{MODEL_KEY}",
                f"unit6_span_guarded_multi_view_hybrid_score_{MODEL_KEY}",
                f"unit6_span_ranked_multi_view_hybrid_score_{MODEL_KEY}",
                f"unit6_semantic_recall_multi_view_hybrid_score_{MODEL_KEY}",
            },
        )
        self.assertEqual(
            {row["family"] for row in unit6_rows},
            {"multi_view_hybrid_scoring"},
        )

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

    def test_question_type_metric_rows_report_global_and_inherited_thresholds(self):
        rows = evaluate_script.build_question_type_metric_rows(
            base_records(),
            base_config(),
            "reference_answer",
        )

        similarity_rows = [
            row
            for row in rows
            if row["score_field"] == SIMILARITY_FIELD
        ]

        self.assertIn("all", {row["question_type"] for row in similarity_rows})
        self.assertIn("where", {row["question_type"] for row in similarity_rows})
        self.assertIn("dataset_global", {row["threshold_scope"] for row in similarity_rows})
        self.assertIn("global_fixed", {row["threshold_scope"] for row in similarity_rows})
        inherited = next(
            row
            for row in similarity_rows
            if row["threshold_scope"] == "inherited_global"
            and row["question_type"] == "where"
        )
        self.assertEqual(inherited["calibration_status"], "skipped")
        self.assertEqual(inherited["skip_reason"], "num_examples<50")
        self.assertEqual(inherited["embedding_model"], MODEL_NAME)

    def test_question_type_calibration_uses_cv_when_bucket_is_supported(self):
        records = []
        for index, (label, score) in enumerate(
            [(1, 0.9), (1, 0.8), (0, 0.1), (0, 0.1)]
        ):
            records.append(
                {
                    **base_records()[index % 2],
                    "id": f"record-{index}",
                    "question_type_v2": "where",
                    "correct_label": label,
                    SIMILARITY_FIELD: score,
                    HYBRID_FIELD: score,
                }
            )
        config = base_config()
        config["evaluation"]["question_type_calibration"] = {
            "min_examples": 4,
            "min_positive": 2,
            "min_negative": 2,
            "num_folds": 2,
        }

        rows = evaluate_script.build_question_type_metric_rows(
            records,
            config,
            "reference_answer",
        )
        cv_row = next(
            row
            for row in rows
            if row["score_field"] == SIMILARITY_FIELD
            and row["threshold_scope"] == "question_type_cv"
            and row["question_type"] == "where"
        )

        self.assertEqual(cv_row["calibration_status"], "applied")
        self.assertEqual(cv_row["cv_num_folds"], 2)
        self.assertEqual(cv_row["fixed_f1"], 1.0)
        self.assertNotEqual(cv_row["cv_selected_thresholds"], "")

    def test_group_zscore_scores_are_added_per_question_type(self):
        records = [
            {"question_type_v2": "who", "score": 0.2},
            {"question_type_v2": "who", "score": 0.8},
            {"question_type_v2": "when", "score": 0.5},
            {"question_type_v2": "when", "score": 0.5},
        ]

        output = add_group_zscore_scores(records, "score", "question_type_v2", "score_z")

        self.assertAlmostEqual(output[0]["score_z"], -1.0)
        self.assertAlmostEqual(output[1]["score_z"], 1.0)
        self.assertEqual(output[2]["score_z"], 0.0)
        self.assertEqual(output[3]["score_z"], 0.0)

    def test_missing_metric_fields_raise_clear_error(self):
        with self.assertRaisesRegex(ValueError, "missing_score"):
            evaluate_similarity_as_classifier(
                records=[{"correct_label": 1}],
                similarity_field="missing_score",
            )


if __name__ == "__main__":
    unittest.main()
