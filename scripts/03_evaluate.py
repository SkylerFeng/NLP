import csv
import json
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.answer_span import build_prediction_span_report
from src.compute_similarity import DEFAULT_SPAN_BLEND_WEIGHT
from src.entity_overlap import add_entity_overlap_scores, hybrid_similarity_score
from src.evaluate import (
    evaluate_similarity_as_classifier,
    find_best_threshold,
    get_failure_cases,
    records_have_fields,
    summarize_similarity_by_correctness,
)
from src.reference_answer import build_reference_quality_report, resolve_reference_field
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


def default_reference_field(config: dict) -> str:
    if config.get("data", {}).get("dataset") == "nq":
        return "reference_answer"
    return "ground_truth"


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
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
    stage: str,
    method: str,
    family: str,
    score_field: str,
    label_field: str,
    reference_field: str,
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
        "stage": stage,
        "method": method,
        "family": family,
        "label_field": label_field,
        "score_field": score_field,
        "reference_field": reference_field,
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


def build_metric_row_if_available(
    records: list[dict],
    *,
    stage: str,
    method: str,
    family: str,
    score_field: str,
    label_field: str,
    reference_field: str,
    threshold: float,
) -> dict | None:
    if not records_have_fields(records, [label_field, score_field, reference_field]):
        return None

    return metric_row(
        records=records,
        stage=stage,
        method=method,
        family=family,
        score_field=score_field,
        label_field=label_field,
        reference_field=reference_field,
        threshold=threshold,
    )


def configured_ablation_rows(
    records: list[dict],
    config: dict,
    default_label_field: str,
    default_reference_field: str,
) -> list[dict]:
    rows = []
    for spec in config.get("evaluation", {}).get("ablation_scores", []):
        score_field = spec["score_field"]
        row = build_metric_row_if_available(
            records,
            stage=spec.get("stage", "configured"),
            method=spec.get("method", score_field),
            family=spec.get("family", "configured_ablation"),
            score_field=score_field,
            label_field=spec.get("label_field", default_label_field),
            reference_field=spec.get("reference_field", default_reference_field),
            threshold=float(
                spec.get(
                    "threshold",
                    config["evaluation"].get("similarity_threshold", 0.75),
                )
            ),
        )
        if row is not None:
            rows.append(row)
    return rows


def reference_validation_ablation_rows(
    records: list[dict],
    config: dict,
    label_field: str,
) -> list[dict]:
    rows = []
    similarity_threshold = config["evaluation"].get("similarity_threshold", 0.75)
    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        score_field = f"similarity_v2_{model_key}"
        row = build_metric_row_if_available(
            records,
            stage="unit1",
            method=f"Reference validation v2: {model_name}",
            family="embedding_v2",
            score_field=score_field,
            label_field=label_field,
            reference_field="reference_answer_v2",
            threshold=similarity_threshold,
        )
        if row is not None:
            rows.append(row)
    return rows


def prediction_span_ablation_rows(
    records: list[dict],
    config: dict,
    label_field: str,
) -> list[dict]:
    rows = []
    similarity_threshold = config["evaluation"].get("similarity_threshold", 0.75)
    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        score_field = f"prediction_span_blend_similarity_{model_key}"
        row = build_metric_row_if_available(
            records,
            stage="unit2",
            method=(
                f"Reference validation + {DEFAULT_SPAN_BLEND_WEIGHT:.1f} span blend: "
                f"{model_name}"
            ),
            family="prediction_span_ablation",
            score_field=score_field,
            label_field=label_field,
            reference_field="reference_answer_v2",
            threshold=similarity_threshold,
        )
        if row is not None:
            rows.append(row)
    return rows


def span_similarity_ablation_rows(
    records: list[dict],
    config: dict,
    label_field: str,
) -> list[dict]:
    rows = []
    similarity_threshold = config["evaluation"].get("similarity_threshold", 0.75)
    score_specs = [
        ("multi_view_score", "Conservative multi-view score"),
        ("span_max_similarity", "Span max similarity"),
        ("span_topk_mean_similarity", "Span top-k mean similarity"),
        ("reference_to_prediction_span_similarity", "Reference-to-prediction span similarity"),
    ]
    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        for field_prefix, method_prefix in score_specs:
            score_field = f"{field_prefix}_{model_key}"
            row = build_metric_row_if_available(
                records,
                stage="unit3",
                method=f"{method_prefix}: {model_name}",
                family="span_similarity_ablation",
                score_field=score_field,
                label_field=label_field,
                reference_field="reference_answer_v2",
                threshold=similarity_threshold,
            )
            if row is not None:
                rows.append(row)
    return rows


def build_baseline_ablation_rows(
    records: list[dict],
    config: dict,
    reference_field: str,
) -> list[dict]:
    label_field = config["evaluation"].get("label_field", "correct_label")
    similarity_threshold = config["evaluation"].get("similarity_threshold", 0.75)

    rows = [
        metric_row(
            records,
            "baseline",
            "Exact match",
            "lexical_baseline",
            "exact_match",
            label_field,
            reference_field,
            0.5,
        ),
        metric_row(
            records,
            "baseline",
            "Ground-truth containment",
            "lexical_baseline",
            "contains_ground_truth",
            label_field,
            reference_field,
            0.5,
        ),
        metric_row(
            records,
            "baseline",
            "Token F1",
            "lexical_baseline",
            "token_f1",
            label_field,
            reference_field,
            0.8,
        ),
        metric_row(
            records,
            "baseline",
            "Entity/token overlap",
            "structured_baseline",
            "entity_overlap",
            label_field,
            reference_field,
            0.75,
        ),
    ]

    for model_name in config["embedding"]["models"]:
        model_key = safe_model_name(model_name)
        rows.append(
            metric_row(
                records,
                "baseline",
                f"Embedding cosine: {model_name}",
                "embedding",
                f"similarity_{model_key}",
                label_field,
                reference_field,
                similarity_threshold,
            )
        )
        rows.append(
            metric_row(
                records,
                "baseline",
                f"Hybrid 0.7*embedding+0.3*overlap: {model_name}",
                "hybrid_ablation",
                f"hybrid_{model_key}",
                label_field,
                reference_field,
                similarity_threshold,
            )
        )

    rows.extend(reference_validation_ablation_rows(records, config, label_field))
    rows.extend(prediction_span_ablation_rows(records, config, label_field))
    rows.extend(span_similarity_ablation_rows(records, config, label_field))
    rows.extend(configured_ablation_rows(records, config, label_field, reference_field))
    return rows


def build_label_change_audit_rows(
    records: list[dict],
    config: dict,
    reference_field: str,
) -> list[dict]:
    evaluation_config = config.get("evaluation", {})
    baseline_label_field = evaluation_config.get("baseline_label_field", "correct_label")
    candidate_label_field = evaluation_config.get("candidate_label_field", "correct_label_v2")
    baseline_reference_field = evaluation_config.get(
        "baseline_reference_field",
        default_reference_field(config),
    )
    candidate_reference_field = evaluation_config.get(
        "candidate_reference_field",
        "reference_answer_v2",
    )

    if not records_have_fields(records, [baseline_label_field, candidate_label_field]):
        return []

    rows = []
    for record in records:
        baseline_label = int(record.get(baseline_label_field, 0))
        candidate_label = int(record.get(candidate_label_field, 0))
        rows.append(
            {
                "dataset": config["data"]["dataset"],
                "id": record.get("id", ""),
                "baseline_label_field": baseline_label_field,
                "candidate_label_field": candidate_label_field,
                "baseline_label": baseline_label,
                "candidate_label": candidate_label,
                "label_changed": int(baseline_label != candidate_label),
                "evaluation_reference_field": reference_field,
                "baseline_reference_field": baseline_reference_field,
                "candidate_reference_field": candidate_reference_field,
                "baseline_reference": record.get(baseline_reference_field, ""),
                "candidate_reference": record.get(candidate_reference_field, ""),
                "question": record.get("question", ""),
                "prediction": record.get("prediction", ""),
                "ground_truth": record.get("ground_truth", ""),
            }
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
                        "label_field": label_field,
                        "score_field": similarity_field,
                        "reference_field": reference_field,
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


def config_path_from_args() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "config.yaml"


def main() -> None:
    config = load_config(config_path_from_args(), stage="evaluation")

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
            "method": "Embedding cosine",
            "embedding_model": model_name,
            "label_field": label_field,
            "score_field": similarity_field,
            "similarity_field": similarity_field,
            "reference_field": reference_field,

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
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_results[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_results)

    write_csv(
        table_dir / "dataset_statistics.csv",
        dataset_statistics(records, config, reference_field),
    )
    write_csv(
        table_dir / "baseline_ablation_results.csv",
        build_baseline_ablation_rows(records, config, reference_field),
    )
    write_csv(
        table_dir / "case_studies.csv",
        build_case_studies(records, config, reference_field),
    )
    reference_quality_report_path = table_dir / "reference_quality_report.csv"
    reference_quality_rows = build_reference_quality_report(records)
    if reference_quality_rows:
        write_csv(reference_quality_report_path, reference_quality_rows)
    elif reference_quality_report_path.exists():
        reference_quality_report_path.unlink()

    prediction_span_report_path = table_dir / "prediction_span_report.csv"
    prediction_span_rows = build_prediction_span_report(records)
    if prediction_span_rows:
        write_csv(prediction_span_report_path, prediction_span_rows)
    elif prediction_span_report_path.exists():
        prediction_span_report_path.unlink()

    label_change_audit_path = table_dir / "label_change_audit.csv"
    label_change_audit_rows = build_label_change_audit_rows(records, config, reference_field)
    if label_change_audit_rows:
        write_csv(label_change_audit_path, label_change_audit_rows)
    elif label_change_audit_path.exists():
        label_change_audit_path.unlink()

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
        "base_results_dir": config["output"].get("base_results_dir", ""),
        "run_id": config["project"].get("resolved_run_id", ""),
        "llm_provider": config["llm"].get("provider"),
        "llm_model": config["llm"].get("model"),
        "embedding_models": config["embedding"]["models"],
        "label_field": label_field,
        "evaluation_reference_field": reference_field,
    }
    metadata_path = table_dir / "run_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Evaluation finished.")


if __name__ == "__main__":
    main()
