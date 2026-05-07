import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.entity_overlap import add_entity_overlap_scores, hybrid_similarity_score
from src.evaluate import (
    evaluate_similarity_as_classifier,
    find_best_threshold,
    get_failure_cases,
    summarize_similarity_by_correctness,
)
from src.reference_answer import resolve_reference_field
from src.utils import (
    dataset_task_type,
    ensure_dir,
    load_config,
    load_jsonl,
    print_config_summary,
    save_jsonl,
    validate_records_dataset,
)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    ensure_dir(path.parent)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def average_token_count(records: list[dict], field: str) -> float:
    if not records:
        return 0.0
    return mean(len(str(record.get(field, "")).split()) for record in records)


def dataset_statistics(records: list[dict], config: dict, reference_field: str) -> list[dict]:
    total = len(records)
    correct = sum(int(record.get("correct_label", 0)) for record in records)
    incorrect = total - correct

    return [
        {
            "dataset": config["data"]["dataset"],
            "task_type": dataset_task_type(config["data"]["dataset"]),
            "sample_size": config["data"].get("sample_size"),
            "num_records": total,
            "num_correct_label": correct,
            "num_incorrect_label": incorrect,
            "correct_rate": correct / total if total else 0.0,
            "avg_prediction_tokens": average_token_count(records, "prediction"),
            "avg_ground_truth_tokens": average_token_count(records, "ground_truth"),
            "avg_evaluation_reference_tokens": average_token_count(records, reference_field),
            "evaluation_reference_field": reference_field,
            "empty_predictions": sum(
                1 for record in records if not str(record.get("prediction", "")).strip()
            ),
            "exact_match_count": sum(int(record.get("exact_match", 0)) for record in records),
            "contains_ground_truth_count": sum(
                int(record.get("contains_ground_truth", 0)) for record in records
            ),
            "avg_token_f1": mean(float(record.get("token_f1", 0.0)) for record in records)
            if records
            else 0.0,
        }
    ]


def add_hybrid_scores(
    records: list[dict],
    embedding_models: list[str],
    reference_field: str,
) -> list[dict]:
    records = add_entity_overlap_scores(records, reference_field=reference_field)
    output_records = []

    for record in records:
        new_record = dict(record)
        for model_name in embedding_models:
            model_key = safe_model_name(model_name)
            similarity_field = f"similarity_{model_key}"
            hybrid_field = f"hybrid_{model_key}"
            new_record[hybrid_field] = hybrid_similarity_score(
                embedding_similarity=float(record.get(similarity_field, 0.0)),
                entity_score=float(record.get("entity_overlap", 0.0)),
                alpha=0.7,
            )
        output_records.append(new_record)

    return output_records


def metric_row(
    records: list[dict],
    method: str,
    family: str,
    score_field: str,
    label_field: str,
    threshold: float,
) -> dict:
    summary = summarize_similarity_by_correctness(
        records=records,
        similarity_field=score_field,
        label_field=label_field,
    )
    fixed_metrics = evaluate_similarity_as_classifier(
        records=records,
        similarity_field=score_field,
        label_field=label_field,
        threshold=threshold,
    )
    best_metrics = find_best_threshold(
        records=records,
        similarity_field=score_field,
        label_field=label_field,
    )

    return {
        "method": method,
        "family": family,
        "score_field": score_field,
        "num_correct": summary["num_correct"],
        "num_incorrect": summary["num_incorrect"],
        "correct_mean": summary["correct_mean"],
        "incorrect_mean": summary["incorrect_mean"],
        "gap": summary["gap"],
        "fixed_threshold": fixed_metrics["threshold"],
        "fixed_accuracy": fixed_metrics["accuracy"],
        "fixed_precision": fixed_metrics["precision"],
        "fixed_recall": fixed_metrics["recall"],
        "fixed_f1": fixed_metrics["f1"],
        "roc_auc": fixed_metrics["roc_auc"],
        "pr_auc": fixed_metrics["pr_auc"],
        "best_threshold": best_metrics["threshold"],
        "best_accuracy": best_metrics["accuracy"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f1": best_metrics["f1"],
    }


def build_baseline_ablation_rows(records: list[dict], config: dict) -> list[dict]:
    label_field = config["evaluation"].get("label_field", "correct_label")
    similarity_threshold = config["evaluation"].get("similarity_threshold", 0.75)

    rows = [
        metric_row(records, "Exact match", "lexical_baseline", "exact_match", label_field, 0.5),
        metric_row(
            records,
            "Ground-truth containment",
            "lexical_baseline",
            "contains_ground_truth",
            label_field,
            0.5,
        ),
        metric_row(records, "Token F1", "lexical_baseline", "token_f1", label_field, 0.8),
        metric_row(
            records,
            "Entity/token overlap",
            "structured_baseline",
            "entity_overlap",
            label_field,
            0.75,
        ),
    ]

    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        rows.append(
            metric_row(
                records,
                f"Embedding cosine: {model_name}",
                "embedding",
                f"similarity_{model_key}",
                label_field,
                similarity_threshold,
            )
        )
        rows.append(
            metric_row(
                records,
                f"Hybrid 0.7*embedding+0.3*overlap: {model_name}",
                "hybrid_ablation",
                f"hybrid_{model_key}",
                label_field,
                similarity_threshold,
            )
        )

    return rows


def case_hint(record: dict, failure_kind: str) -> str:
    token_f1 = float(record.get("token_f1", 0.0))
    contains = int(record.get("contains_ground_truth", 0) or 0)
    entity_overlap = float(record.get("entity_overlap", 0.0))

    if failure_kind == "low_similarity_correct" and contains:
        return "Correct short answer is contained, but whole-answer embedding is low."
    if failure_kind == "high_similarity_wrong" and token_f1 >= 0.5:
        return "High semantic score with partial lexical overlap; inspect specificity."
    if failure_kind == "high_similarity_wrong" and entity_overlap < 0.2:
        return "Semantic relatedness may be high while key factual units differ."
    return "Manual review needed for factual equivalence."


def build_case_studies(
    records: list[dict],
    config: dict,
    reference_field: str,
    case_limit: int = 10,
) -> list[dict]:
    label_field = config["evaluation"].get("label_field", "correct_label")
    rows = []

    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        similarity_field = f"similarity_{model_key}"
        failure_cases = get_failure_cases(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            high_threshold=0.8,
            low_threshold=0.5,
        )

        grouped_cases = [
            (
                "high_similarity_wrong",
                sorted(
                    failure_cases["high_similarity_wrong"],
                    key=lambda item: item[similarity_field],
                    reverse=True,
                ),
            ),
            (
                "low_similarity_correct",
                sorted(
                    failure_cases["low_similarity_correct"],
                    key=lambda item: item[similarity_field],
                ),
            ),
        ]

        for failure_kind, cases in grouped_cases:
            for record in cases[:case_limit]:
                score = float(record.get(similarity_field, 0.0))
                rows.append(
                    {
                        "dataset": config["data"]["dataset"],
                        "task_type": dataset_task_type(config["data"]["dataset"]),
                        "model": model_name,
                        "failure_kind": failure_kind,
                        "id": record.get("id", ""),
                        "question": record.get("question", ""),
                        "ground_truth": record.get("ground_truth", ""),
                        "evaluation_reference": record.get(reference_field, ""),
                        "reference_answer_source": record.get("reference_answer_source", ""),
                        "reference_evidence": record.get("reference_evidence", ""),
                        "prediction": record.get("prediction", ""),
                        "correct_label": record.get(label_field, ""),
                        "token_f1": record.get("token_f1", ""),
                        "contains_ground_truth": record.get("contains_ground_truth", ""),
                        "entity_overlap": record.get("entity_overlap", ""),
                        "similarity": score,
                        "distance": 1.0 - score,
                        "explanation_hint": case_hint(record, failure_kind),
                    }
                )

    return rows


def main() -> None:
    config = load_config("config.yaml")

    input_file = config["evaluation"]["input_file"]
    threshold = config["evaluation"].get("similarity_threshold", 0.75)
    label_field = config["evaluation"].get("label_field", "correct_label")
    reference_field = resolve_reference_field(config)

    embedding_models = config["embedding"]["models"]

    table_dir = Path(config["output"]["table_dir"])
    failure_case_dir = Path(config["output"]["failure_case_dir"])

    ensure_dir(table_dir)
    ensure_dir(failure_case_dir)

    print_config_summary(config)
    print(f"Loading similarity results from: {input_file}")
    records = load_jsonl(input_file)
    validate_records_dataset(records, config["data"]["dataset"])
    print(f"Loaded {len(records)} records.")
    print(f"Using evaluation reference field: {reference_field}")
    records = add_hybrid_scores(records, embedding_models, reference_field)

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

    write_csv(table_dir / "dataset_statistics.csv", dataset_statistics(records, config, reference_field))
    write_csv(table_dir / "baseline_ablation_results.csv", build_baseline_ablation_rows(records, config))
    write_csv(table_dir / "case_studies.csv", build_case_studies(records, config, reference_field))

    metadata = {
        "dataset": config["data"]["dataset"],
        "task_type": dataset_task_type(config["data"]["dataset"]),
        "sample_size": config["data"].get("sample_size"),
        "prediction_input_file": config["prediction"]["input_file"],
        "prediction_output_file": config["prediction"]["output_file"],
        "similarity_input_file": config["similarity"]["input_file"],
        "similarity_output_file": config["similarity"]["output_file"],
        "evaluation_input_file": config["evaluation"]["input_file"],
        "results_dir": config["output"]["results_dir"],
        "llm_provider": config["llm"].get("provider"),
        "llm_model": config["llm"].get("model"),
        "embedding_models": config["embedding"]["models"],
        "evaluation_reference_field": reference_field,
    }
    metadata_path = table_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
