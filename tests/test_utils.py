import os
import tempfile
import unittest
from pathlib import Path

from src.utils import resolve_config


def base_config() -> dict:
    return {
        "project": {
            "auto_paths": True,
            "preserve_runs": True,
            "run_id": "unit1-check",
        },
        "data": {
            "dataset": "nq",
            "sample_size": 500,
            "data_root": "data/raw",
            "data_file": "merged_fb.json",
            "prediction_dir": "data/interim/predictions",
            "similarity_dir": "data/interim/similarity",
        },
        "llm": {"run_name": "qwen25_7b_instruct"},
        "prediction": {},
        "similarity": {},
        "evaluation": {},
        "output": {},
    }


class ConfigPathResolutionTest(unittest.TestCase):
    def test_preserved_run_paths_do_not_use_legacy_similarity_output(self):
        config = resolve_config(base_config(), stage="similarity")

        self.assertEqual(config["project"]["resolved_run_id"], "unit1_check")
        self.assertEqual(
            config["prediction"]["output_file"],
            "data/interim/predictions/nq_qwen25_7b_instruct_predictions_500.jsonl",
        )
        self.assertEqual(
            config["similarity"]["output_file"],
            "outputs/experiments/results_nq_500/runs/unit1_check/similarity/"
            "nq_qwen25_7b_instruct_similarity_500.jsonl",
        )
        self.assertEqual(
            config["output"]["table_dir"],
            "outputs/experiments/results_nq_500/runs/unit1_check/tables",
        )

    def test_evaluation_uses_latest_run_marker_when_run_id_is_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                marker_dir = Path("outputs/experiments/results_nq_500")
                marker_dir.mkdir(parents=True)
                (marker_dir / "latest_run_id.txt").write_text(
                    "latest_unit1\n",
                    encoding="utf-8",
                )
                config = base_config()
                config["project"]["run_id"] = "auto"

                resolved = resolve_config(config, stage="evaluation")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(resolved["project"]["resolved_run_id"], "latest_unit1")
        self.assertIn(
            "outputs/experiments/results_nq_500/runs/latest_unit1/similarity/",
            resolved["evaluation"]["input_file"],
        )

    def test_evaluation_can_read_legacy_similarity_when_no_latest_run_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                legacy_similarity = Path(
                    "data/interim/similarity/nq_qwen25_7b_instruct_similarity_500.jsonl"
                )
                legacy_similarity.parent.mkdir(parents=True)
                legacy_similarity.write_text("", encoding="utf-8")
                config = base_config()
                config["project"]["run_id"] = "auto"

                resolved = resolve_config(config, stage="evaluation")
            finally:
                os.chdir(original_cwd)

        self.assertEqual(
            resolved["evaluation"]["input_file"],
            "data/interim/similarity/nq_qwen25_7b_instruct_similarity_500.jsonl",
        )
        self.assertIn("outputs/experiments/results_nq_500/runs/", resolved["output"]["table_dir"])

    def test_preserve_runs_can_be_disabled_for_legacy_paths(self):
        config = base_config()
        config["project"]["preserve_runs"] = False

        resolved = resolve_config(config, stage="similarity")

        self.assertNotIn("resolved_run_id", resolved["project"])
        self.assertEqual(
            resolved["similarity"]["output_file"],
            "data/interim/similarity/nq_qwen25_7b_instruct_similarity_500.jsonl",
        )
        self.assertEqual(resolved["output"]["table_dir"], "outputs/experiments/results_nq_500/tables")

    def test_prediction_stage_keeps_reusable_legacy_paths(self):
        resolved = resolve_config(base_config(), stage="prediction")

        self.assertNotIn("resolved_run_id", resolved["project"])
        self.assertEqual(
            resolved["prediction"]["output_file"],
            "data/interim/predictions/nq_qwen25_7b_instruct_predictions_500.jsonl",
        )
        self.assertEqual(
            resolved["similarity"]["output_file"],
            "data/interim/similarity/nq_qwen25_7b_instruct_similarity_500.jsonl",
        )


if __name__ == "__main__":
    unittest.main()
