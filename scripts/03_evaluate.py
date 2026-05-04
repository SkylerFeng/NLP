import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.evaluate import (
    evaluate_similarity_as_classifier,
    find_best_threshold,
    get_failure_cases,
    summarize_similarity_by_correctness,
)
from src.utils import ensure_dir, load_config, load_jsonl, save_jsonl


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def main() -> None:
    config = load_config("config.yaml")

    input_file = config["evaluation"]["input_file"]
    threshold = config["evaluation"].get("similarity_threshold", 0.75)
    label_field = config["evaluation"].get("label_field", "correct_label")

    embedding_models = config["embedding"]["models"]

    table_dir = Path(config["output"]["table_dir"])
    failure_case_dir = Path(config["output"]["failure_case_dir"])

    ensure_dir(table_dir)
    ensure_dir(failure_case_dir)

    print(f"Loading similarity results from: {input_file}")
    records = load_jsonl(input_file)
    print(f"Loaded {len(records)} records.")

    all_results = []

    for model_name in embedding_models:
        model_key = safe_model_name(model_name)
        similarity_field = f"similarity_{model_key}"

        print("=" * 80)
        print(f"Evaluating similarity field: {similarity_field}")

        summary = summarize_similarity_by_correctness(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
        )

        fixed_threshold_metrics = evaluate_similarity_as_classifier(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            threshold=threshold,
        )

        best_threshold_metrics = find_best_threshold(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
        )

        result = {
            "embedding_model": model_name,
            "similarity_field": similarity_field,

            "num_correct": summary["num_correct"],
            "num_incorrect": summary["num_incorrect"],
            "correct_mean": summary["correct_mean"],
            "correct_std": summary["correct_std"],
            "incorrect_mean": summary["incorrect_mean"],
            "incorrect_std": summary["incorrect_std"],
            "gap": summary["gap"],

            "fixed_threshold": fixed_threshold_metrics["threshold"],
            "fixed_accuracy": fixed_threshold_metrics["accuracy"],
            "fixed_precision": fixed_threshold_metrics["precision"],
            "fixed_recall": fixed_threshold_metrics["recall"],
            "fixed_f1": fixed_threshold_metrics["f1"],
            "fixed_roc_auc": fixed_threshold_metrics["roc_auc"],
            "fixed_pr_auc": fixed_threshold_metrics["pr_auc"],

            "best_threshold": best_threshold_metrics["threshold"],
            "best_accuracy": best_threshold_metrics["accuracy"],
            "best_precision": best_threshold_metrics["precision"],
            "best_recall": best_threshold_metrics["recall"],
            "best_f1": best_threshold_metrics["f1"],
            "best_roc_auc": best_threshold_metrics["roc_auc"],
            "best_pr_auc": best_threshold_metrics["pr_auc"],
        }

        all_results.append(result)

        print("Summary:")
        for key, value in result.items():
            print(f"{key}: {value}")

        failure_cases = get_failure_cases(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            high_threshold=0.8,
            low_threshold=0.5,
        )

        high_wrong_path = failure_case_dir / f"high_similarity_wrong_{model_key}.jsonl"
        low_correct_path = failure_case_dir / f"low_similarity_correct_{model_key}.jsonl"

        save_jsonl(failure_cases["high_similarity_wrong"], high_wrong_path)
        save_jsonl(failure_cases["low_similarity_correct"], low_correct_path)

        print(f"Saved high-similarity wrong cases to: {high_wrong_path}")
        print(f"Saved low-similarity correct cases to: {low_correct_path}")

    output_csv = table_dir / "evaluation_results.csv"

    print(f"Saving evaluation table to: {output_csv}")

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)

    print("Evaluation finished.")


if __name__ == "__main__":
    main()