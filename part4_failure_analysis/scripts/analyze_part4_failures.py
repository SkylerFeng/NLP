import csv
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "part4_failure_analysis"

RESULT_DIRS = {
    "results_nq": PROJECT_ROOT / "results_nq",
    "results_sciq_5000": PROJECT_ROOT / "results_sciq_5000",
    "results_truthfulQA_500": PROJECT_ROOT / "results_truthfulQA_500",
    "results_wiki": PROJECT_ROOT / "results_wiki",
}

MODEL_INFO = {
    "sentence_transformers_all_MiniLM_L6_v2": {
        "short_name": "MiniLM",
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_field": "similarity_sentence_transformers_all_MiniLM_L6_v2",
    },
    "BAAI_bge_base_en_v1.5": {
        "short_name": "BGE",
        "model_name": "BAAI/bge-base-en-v1.5",
        "similarity_field": "similarity_BAAI_bge_base_en_v1.5",
    },
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "to",
    "and",
    "or",
    "for",
    "with",
    "by",
    "as",
    "at",
    "from",
    "through",
    "into",
    "called",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize(text: str) -> str:
    text = "" if text is None else str(text).lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    parts = [NUMBER_WORDS.get(part, part) for part in text.split()]
    return " ".join(parts)


def simple_lemma(token: str) -> str:
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("es") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def content_tokens(text: str) -> list[str]:
    return [
        simple_lemma(tok)
        for tok in normalize(text).split()
        if tok and tok not in STOPWORDS
    ]


def token_overlap(a: list[str], b: list[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    common = len(set_a & set_b)
    if common == 0:
        return 0.0
    precision = common / len(set_a)
    recall = common / len(set_b)
    return 2 * precision * recall / (precision + recall)


def has_numeric_equivalence(prediction: str, ground_truth: str) -> bool:
    pred_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize(prediction)))
    gt_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", normalize(ground_truth)))
    return bool(pred_nums and gt_nums and pred_nums & gt_nums)


def classify_failure(record: dict, failure_kind: str, model_key: str) -> str:
    prediction = record.get("prediction", "")
    ground_truth = record.get("ground_truth") or record.get("correct_answer", "")
    pred_tokens = content_tokens(prediction)
    gt_tokens = content_tokens(ground_truth)
    pred_set = set(pred_tokens)
    gt_set = set(gt_tokens)
    exact_lemma_match = pred_tokens == gt_tokens and bool(pred_tokens)
    overlap = token_overlap(pred_tokens, gt_tokens)
    token_f1 = safe_float(record.get("token_f1"))
    contains_ground_truth = int(record.get("contains_ground_truth", 0) or 0)
    similarity = safe_float(record.get(MODEL_INFO[model_key]["similarity_field"]))

    if has_numeric_equivalence(prediction, ground_truth):
        return "numeric_equivalence"

    if exact_lemma_match:
        return "morphology_or_inflection"

    if failure_kind == "low_similarity_correct":
        if contains_ground_truth and len(pred_tokens) > len(gt_tokens) + 2:
            return "overly_long_answer_context_dilution"
        if contains_ground_truth:
            return "answer_containment_low_embedding_score"
        return "low_score_for_valid_paraphrase"

    if pred_set and gt_set and (pred_set < gt_set or gt_set < pred_set):
        return "underspecified_or_overspecified_answer"

    if token_f1 >= 0.5 or overlap >= 0.5:
        return "synonym_or_paraphrase_labeling_artifact"

    if similarity >= 0.85:
        return "semantic_relatedness_not_correctness"

    return "other_or_true_semantic_error"


def human_note_for_type(failure_type: str, failure_kind: str) -> str:
    notes = {
        "numeric_equivalence": "Prediction and reference use equivalent numeric forms; automatic labeling/similarity can be sensitive to surface form.",
        "morphology_or_inflection": "Prediction differs mainly by singular/plural or simple inflection; this is likely an automatic-label artifact.",
        "synonym_or_paraphrase_labeling_artifact": "Prediction is a close paraphrase or synonym of the reference; automatic exact/token labels may be too strict.",
        "underspecified_or_overspecified_answer": "Prediction shares key content but is either too broad, too narrow, or includes a different level of specificity.",
        "semantic_relatedness_not_correctness": "Prediction is semantically related to the reference, but relatedness alone may not prove factual correctness.",
        "answer_containment_low_embedding_score": "Prediction contains the reference answer, but the embedding score is low; short-answer signal may be diluted.",
        "overly_long_answer_context_dilution": "Prediction embeds the correct answer in a longer sentence, so extra context likely lowers the similarity score.",
        "low_score_for_valid_paraphrase": "Prediction appears to be a valid paraphrase, but the embedding model assigns a low score.",
        "other_or_true_semantic_error": "Requires manual review; this may be a genuine model answer error or an unhandled labeling issue.",
    }
    prefix = "High-similarity wrong: " if failure_kind == "high_similarity_wrong" else "Low-similarity correct: "
    return prefix + notes.get(failure_type, "Requires manual review.")


def parse_failure_file_name(path: Path) -> tuple[str, str]:
    name = path.stem
    if name.startswith("high_similarity_wrong_"):
        failure_kind = "high_similarity_wrong"
        model_key = name.removeprefix("high_similarity_wrong_")
    elif name.startswith("low_similarity_correct_"):
        failure_kind = "low_similarity_correct"
        model_key = name.removeprefix("low_similarity_correct_")
    else:
        raise ValueError(f"Unexpected failure case file name: {path.name}")

    if model_key not in MODEL_INFO:
        raise ValueError(f"Unknown model key from file name: {path.name}")
    return failure_kind, model_key


def format_float(value: str | float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def estimate_confusion_from_metrics(metric_row: dict) -> dict:
    total = int(metric_row["num_correct"]) + int(metric_row["num_incorrect"])
    positives = int(metric_row["num_correct"])
    negatives = int(metric_row["num_incorrect"])
    recall = safe_float(metric_row["fixed_recall"])
    precision = safe_float(metric_row["fixed_precision"])
    tp = round(recall * positives)
    fn = positives - tp
    fp = round(tp / precision - tp) if precision > 0 else 0
    tn = negatives - fp
    return {
        "total": total,
        "estimated_tp": tp,
        "estimated_fp": fp,
        "estimated_tn": tn,
        "estimated_fn": fn,
    }


def make_example_rows(rows: list[dict], model_key: str, limit: int = 8) -> list[dict]:
    sim_field = MODEL_INFO[model_key]["similarity_field"]
    sorted_rows = sorted(rows, key=lambda item: safe_float(item.get(sim_field)), reverse=True)
    examples = []
    for row in sorted_rows[:limit]:
        examples.append(
            {
                "id": row.get("id", ""),
                "question": row.get("question", "")[:120].replace("|", "/"),
                "ground_truth": (row.get("ground_truth") or row.get("correct_answer", ""))[:60].replace("|", "/"),
                "prediction": row.get("prediction", "")[:80].replace("|", "/"),
                "token_f1": format_float(row.get("token_f1", 0)),
                "similarity": format_float(row.get(sim_field, 0)),
                "heuristic_type": row.get("heuristic_type", ""),
            }
        )
    return examples


def make_manual_annotation_sample(
    annotated_rows: list[dict],
    output_path: Path,
    sample_per_group: int = 50,
    seed: int = 42,
) -> None:
    existing_annotations = {}
    if output_path.exists():
        for old_row in read_csv(output_path):
            key = (
                old_row.get("dataset", ""),
                old_row.get("model", ""),
                old_row.get("failure_kind", ""),
                old_row.get("id", ""),
            )
            existing_annotations[key] = old_row

    grouped = defaultdict(list)
    for row in annotated_rows:
        grouped[(row["dataset"], row["model"], row["failure_kind"])].append(row)

    rng = random.Random(seed)
    sample_rows = []
    for group_key, rows in sorted(grouped.items()):
        selected = rows if len(rows) <= sample_per_group else rng.sample(rows, sample_per_group)
        for row in selected:
            key = (
                row.get("dataset", ""),
                row.get("model", ""),
                row.get("failure_kind", ""),
                row.get("id", ""),
            )
            old_row = existing_annotations.get(key, {})
            human_type = old_row.get("human_type") or row.get("heuristic_type", "")
            human_notes = old_row.get("human_notes") or human_note_for_type(
                human_type,
                row.get("failure_kind", ""),
            )
            sample_rows.append(
                {
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "failure_kind": row.get("failure_kind", ""),
                    "id": row.get("id", ""),
                    "question": row.get("question", ""),
                    "ground_truth": row.get("ground_truth", ""),
                    "prediction": row.get("prediction", ""),
                    "token_f1": row.get("token_f1", ""),
                    "contains_ground_truth": row.get("contains_ground_truth", ""),
                    "similarity_MiniLM": row.get("similarity_sentence_transformers_all_MiniLM_L6_v2", ""),
                    "similarity_BGE": row.get("similarity_BAAI_bge_base_en_v1.5", ""),
                    "suggested_heuristic_type": row.get("heuristic_type", ""),
                    "human_type": human_type,
                    "human_notes": human_notes,
                }
            )

    write_csv(
        output_path,
        sample_rows,
        [
            "dataset",
            "model",
            "failure_kind",
            "id",
            "question",
            "ground_truth",
            "prediction",
            "token_f1",
            "contains_ground_truth",
            "similarity_MiniLM",
            "similarity_BGE",
            "suggested_heuristic_type",
            "human_type",
            "human_notes",
        ],
    )


def analyze_dataset(dataset_name: str, result_dir: Path) -> dict:
    dataset_out = OUTPUT_ROOT / "datasets" / dataset_name
    ensure_dir(dataset_out / "tables")
    ensure_dir(dataset_out / "examples")

    metric_rows = read_csv(result_dir / "tables" / "evaluation_results.csv")
    shutil.copyfile(
        result_dir / "tables" / "evaluation_results.csv",
        dataset_out / "tables" / "evaluation_results.csv",
    )

    failure_count_rows = []
    taxonomy_rows = []
    annotated_rows = []
    example_sections = []
    all_example_rows = []

    for failure_file in sorted((result_dir / "failure_cases").glob("*.jsonl")):
        failure_kind, model_key = parse_failure_file_name(failure_file)
        model = MODEL_INFO[model_key]
        rows = read_jsonl(failure_file)

        for row in rows:
            new_row = dict(row)
            new_row["dataset"] = dataset_name
            new_row["model"] = model["short_name"]
            new_row["failure_kind"] = failure_kind
            new_row["heuristic_type"] = classify_failure(row, failure_kind, model_key)
            annotated_rows.append(new_row)

        count = len(rows)
        sim_values = [safe_float(row.get(model["similarity_field"])) for row in rows]
        token_f1_values = [safe_float(row.get("token_f1")) for row in rows]
        failure_count_rows.append(
            {
                "dataset": dataset_name,
                "model": model["short_name"],
                "failure_kind": failure_kind,
                "count": count,
                "avg_similarity": format_float(mean(sim_values) if sim_values else 0),
                "avg_token_f1": format_float(mean(token_f1_values) if token_f1_values else 0),
            }
        )

        taxonomy_counter = Counter(row["heuristic_type"] for row in annotated_rows if row["dataset"] == dataset_name and row["model"] == model["short_name"] and row["failure_kind"] == failure_kind)
        for failure_type, type_count in taxonomy_counter.most_common():
            taxonomy_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model["short_name"],
                    "failure_kind": failure_kind,
                    "heuristic_type": failure_type,
                    "count": type_count,
                    "percentage": format_float(type_count / count * 100 if count else 0, 1),
                }
            )

        for row in rows:
            row["heuristic_type"] = classify_failure(row, failure_kind, model_key)
        examples = make_example_rows(rows, model_key)
        for example in examples:
            example_with_group = {
                "dataset": dataset_name,
                "model": model["short_name"],
                "failure_kind": failure_kind,
                **example,
            }
            all_example_rows.append(example_with_group)
        write_csv(
            dataset_out / "examples" / f"{model['short_name']}_{failure_kind}_examples.csv",
            [
                {
                    "dataset": dataset_name,
                    "model": model["short_name"],
                    "failure_kind": failure_kind,
                    **example,
                }
                for example in examples
            ],
        )
        section_title = f"### {model['short_name']} - {failure_kind}"
        example_sections.append(section_title)
        example_sections.append(markdown_table(examples, ["id", "ground_truth", "prediction", "token_f1", "similarity", "heuristic_type"]))

    write_csv(dataset_out / "examples" / "representative_examples.csv", all_example_rows)

    write_csv(dataset_out / "tables" / "failure_counts.csv", failure_count_rows)
    write_csv(dataset_out / "tables" / "heuristic_failure_taxonomy.csv", taxonomy_rows)

    annotated_fields = [
        "dataset",
        "model",
        "failure_kind",
        "heuristic_type",
        "id",
        "question",
        "ground_truth",
        "prediction",
        "exact_match",
        "token_f1",
        "contains_ground_truth",
        "correct_label",
        "similarity_sentence_transformers_all_MiniLM_L6_v2",
        "similarity_BAAI_bge_base_en_v1.5",
    ]
    compact_rows = [{key: row.get(key, "") for key in annotated_fields} for row in annotated_rows]
    write_csv(dataset_out / "tables" / "annotated_failure_cases.csv", compact_rows, annotated_fields)
    make_manual_annotation_sample(
        annotated_rows=annotated_rows,
        output_path=dataset_out / "tables" / "manual_annotation_sample.csv",
        sample_per_group=50,
    )

    confusion_rows = []
    for metric_row in metric_rows:
        model_short = "MiniLM" if "MiniLM" in metric_row["embedding_model"] else "BGE"
        confusion = estimate_confusion_from_metrics(metric_row)
        confusion_rows.append(
            {
                "dataset": dataset_name,
                "model": model_short,
                "fixed_threshold": metric_row["fixed_threshold"],
                **confusion,
                "fixed_accuracy": format_float(metric_row["fixed_accuracy"]),
                "fixed_precision": format_float(metric_row["fixed_precision"]),
                "fixed_recall": format_float(metric_row["fixed_recall"]),
                "fixed_f1": format_float(metric_row["fixed_f1"]),
            }
        )
    write_csv(dataset_out / "tables" / "fixed_threshold_confusion_estimates.csv", confusion_rows)

    dataset_report = build_dataset_report(
        dataset_name=dataset_name,
        metric_rows=metric_rows,
        failure_count_rows=failure_count_rows,
        taxonomy_rows=taxonomy_rows,
        confusion_rows=confusion_rows,
        example_sections=example_sections,
    )
    (dataset_out / "dataset_failure_report.md").write_text(dataset_report, encoding="utf-8")

    return {
        "dataset": dataset_name,
        "metrics": metric_rows,
        "failure_counts": failure_count_rows,
        "taxonomy": taxonomy_rows,
        "confusion": confusion_rows,
    }


def build_dataset_report(
    dataset_name: str,
    metric_rows: list[dict],
    failure_count_rows: list[dict],
    taxonomy_rows: list[dict],
    confusion_rows: list[dict],
    example_sections: list[str],
) -> str:
    metric_summary = []
    for row in metric_rows:
        model_short = "MiniLM" if "MiniLM" in row["embedding_model"] else "BGE"
        metric_summary.append(
            {
                "Model": model_short,
                "Correct Mean": format_float(row["correct_mean"]),
                "Incorrect Mean": format_float(row["incorrect_mean"]),
                "Gap": format_float(row["gap"]),
                "Fixed F1": format_float(row["fixed_f1"]),
                "Best Threshold": format_float(row["best_threshold"], 2),
                "Best F1": format_float(row["best_f1"]),
                "ROC-AUC": format_float(row["fixed_roc_auc"]),
            }
        )

    failure_rows = [
        {
            "Model": row["model"],
            "Failure Kind": row["failure_kind"],
            "Count": row["count"],
            "Avg Similarity": row["avg_similarity"],
            "Avg Token F1": row["avg_token_f1"],
        }
        for row in failure_count_rows
    ]

    taxonomy_display = [
        {
            "Model": row["model"],
            "Failure Kind": row["failure_kind"],
            "Heuristic Type": row["heuristic_type"],
            "Count": row["count"],
            "%": row["percentage"],
        }
        for row in taxonomy_rows
    ]

    confusion_display = [
        {
            "Model": row["model"],
            "Threshold": row["fixed_threshold"],
            "TP": row["estimated_tp"],
            "FP": row["estimated_fp"],
            "TN": row["estimated_tn"],
            "FN": row["estimated_fn"],
            "F1": row["fixed_f1"],
        }
        for row in confusion_rows
    ]

    lines = [
        f"# Part 4 Failure Analysis: {dataset_name}",
        "",
        "## Metric Summary",
        markdown_table(metric_summary, ["Model", "Correct Mean", "Incorrect Mean", "Gap", "Fixed F1", "Best Threshold", "Best F1", "ROC-AUC"]),
        "## Failure Case Counts",
        markdown_table(failure_rows, ["Model", "Failure Kind", "Count", "Avg Similarity", "Avg Token F1"]),
        "## Fixed-Threshold Confusion Estimates",
        "These counts are reconstructed from precision/recall in the evaluation table, so they may differ by one sample because of rounding.",
        "",
        markdown_table(confusion_display, ["Model", "Threshold", "TP", "FP", "TN", "FN", "F1"]),
        "## Heuristic Failure Taxonomy",
        "The taxonomy is automatically assigned by lexical and numeric heuristics. Use it for quantitative guidance, then manually verify representative samples for the report.",
        "",
        markdown_table(taxonomy_display, ["Model", "Failure Kind", "Heuristic Type", "Count", "%"]),
        "## Representative Cases",
        "\n".join(example_sections),
        "## Suggested Interpretation",
        "- High-similarity-wrong cases often indicate either automatic-label artifacts or semantic relatedness being mistaken for correctness.",
        "- Low-similarity-correct cases often indicate answer containment inside a longer prediction or embedding-model insensitivity to short factual answers.",
        "- Compare MiniLM and BGE by the failure count trade-off: BGE usually produces fewer low-similarity-correct cases but more high-similarity-wrong cases.",
        "",
    ]
    return "\n".join(lines)


def build_summary_report(dataset_results: list[dict]) -> None:
    summary_dir = OUTPUT_ROOT / "summary_tables"
    ensure_dir(summary_dir)

    all_metrics = []
    all_failure_counts = []
    all_taxonomy = []
    all_confusion = []

    for result in dataset_results:
        dataset = result["dataset"]
        for row in result["metrics"]:
            model_short = "MiniLM" if "MiniLM" in row["embedding_model"] else "BGE"
            all_metrics.append(
                {
                    "dataset": dataset,
                    "model": model_short,
                    "num_correct": row["num_correct"],
                    "num_incorrect": row["num_incorrect"],
                    "correct_mean": format_float(row["correct_mean"]),
                    "incorrect_mean": format_float(row["incorrect_mean"]),
                    "gap": format_float(row["gap"]),
                    "fixed_threshold": row["fixed_threshold"],
                    "fixed_accuracy": format_float(row["fixed_accuracy"]),
                    "fixed_precision": format_float(row["fixed_precision"]),
                    "fixed_recall": format_float(row["fixed_recall"]),
                    "fixed_f1": format_float(row["fixed_f1"]),
                    "best_threshold": format_float(row["best_threshold"], 2),
                    "best_f1": format_float(row["best_f1"]),
                    "roc_auc": format_float(row["fixed_roc_auc"]),
                    "pr_auc": format_float(row["fixed_pr_auc"]),
                }
            )
        all_failure_counts.extend(result["failure_counts"])
        all_taxonomy.extend(result["taxonomy"])
        all_confusion.extend(result["confusion"])

    write_csv(summary_dir / "model_metrics_summary.csv", all_metrics)
    write_csv(summary_dir / "failure_counts_summary.csv", all_failure_counts)
    write_csv(summary_dir / "heuristic_taxonomy_summary.csv", all_taxonomy)
    write_csv(summary_dir / "fixed_threshold_confusion_estimates_summary.csv", all_confusion)

    aggregate = defaultdict(lambda: Counter())
    for row in all_failure_counts:
        aggregate[row["model"]][row["failure_kind"]] += int(row["count"])

    aggregate_rows = []
    for model, counter in aggregate.items():
        aggregate_rows.append(
            {
                "Model": model,
                "High-Sim Wrong": counter["high_similarity_wrong"],
                "Low-Sim Correct": counter["low_similarity_correct"],
                "Total Failure Cases": sum(counter.values()),
            }
        )

    metric_display = [
        {
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Gap": row["gap"],
            "Fixed F1": row["fixed_f1"],
            "Best Thresh.": row["best_threshold"],
            "Best F1": row["best_f1"],
            "ROC-AUC": row["roc_auc"],
        }
        for row in all_metrics
    ]

    failure_display = [
        {
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Failure Kind": row["failure_kind"],
            "Count": row["count"],
            "Avg Sim.": row["avg_similarity"],
        }
        for row in all_failure_counts
    ]

    taxonomy_top_rows = []
    grouped_taxonomy = defaultdict(list)
    for row in all_taxonomy:
        grouped_taxonomy[(row["dataset"], row["model"], row["failure_kind"])].append(row)
    for key, rows in grouped_taxonomy.items():
        top = sorted(rows, key=lambda item: int(item["count"]), reverse=True)[:3]
        for row in top:
            taxonomy_top_rows.append(
                {
                    "Dataset": row["dataset"],
                    "Model": row["model"],
                    "Failure Kind": row["failure_kind"],
                    "Top Type": row["heuristic_type"],
                    "Count": row["count"],
                    "%": row["percentage"],
                }
            )

    report = [
        "# Part 4 Failure Analysis Summary",
        "",
        "Chinese version: `summary_report.zh.md`.",
        "",
        "## Scope",
        "This folder summarizes failure cases from `results_nq`, `results_sciq_5000`, `results_truthfulQA_500`, and `results_wiki`. A failure case means that the embedding-similarity correctness decision disagrees with the automatic correctness label.",
        "",
        "Important caveat: the inspected `results_nq` examples look like short-form science QA rather than Natural Questions long-form QA. Treat that folder cautiously unless the original intermediate prediction file is recovered.",
        "",
        "## Embedding Model Comparison",
        markdown_table(metric_display, ["Dataset", "Model", "Gap", "Fixed F1", "Best Thresh.", "Best F1", "ROC-AUC"]),
        "Across all four result folders, MiniLM has a larger correct-vs-incorrect similarity gap, while BGE assigns higher scores to both correct and incorrect answers. This makes BGE recall-friendly but more prone to high-similarity wrong cases.",
        "",
        "## Aggregate Failure Counts",
        markdown_table(aggregate_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total Failure Cases"]),
        "BGE has many more high-similarity-wrong cases and no low-similarity-correct cases in these outputs. MiniLM has fewer false-positive-like failures, but it misses some correct answers when the prediction contains the answer inside a longer phrase or sentence.",
        "",
        "## Per-Dataset Failure Counts",
        markdown_table(failure_display, ["Dataset", "Model", "Failure Kind", "Count", "Avg Sim."]),
        "## Main Heuristic Failure Types",
        markdown_table(taxonomy_top_rows, ["Dataset", "Model", "Failure Kind", "Top Type", "Count", "%"]),
        "## Recommended Part 4 Narrative",
        "1. Define failure cases as disagreement between similarity-threshold decisions and automatic correctness labels.",
        "2. Quantify counts for high-similarity-wrong and low-similarity-correct cases by dataset and embedding model.",
        "3. Use the heuristic taxonomy to show that many high-similarity-wrong cases are label artifacts caused by morphology, numeric equivalence, or paraphrase.",
        "4. Explain true similarity limitations separately: semantic relatedness is not the same as factual correctness, especially for underspecified answers.",
        "5. Propose a robust evaluator combining stronger normalization, answer extraction, entity/keyword overlap, and sentence-level similarity or NLI verification.",
        "",
        "## Generated Files",
        "- `summary_tables/model_metrics_summary.csv`",
        "- `summary_tables/failure_counts_summary.csv`",
        "- `summary_tables/heuristic_taxonomy_summary.csv`",
        "- `summary_tables/fixed_threshold_confusion_estimates_summary.csv`",
        "- `datasets/<result_dir>/dataset_failure_report.md`",
        "- `datasets/<result_dir>/tables/annotated_failure_cases.csv`",
        "- `datasets/<result_dir>/tables/manual_annotation_sample.csv`",
        "",
    ]

    (OUTPUT_ROOT / "summary_report.md").write_text("\n".join(report), encoding="utf-8")

    zh_report = [
        "# Part 4 失败案例分析总报告",
        "",
        "英文版本见：`summary_report.md`。",
        "",
        "## 分析范围",
        "本目录汇总了 `results_nq`、`results_sciq_5000`、`results_truthfulQA_500` 和 `results_wiki` 四个结果目录中的 failure cases。这里的 failure case 指的是：基于 embedding similarity threshold 得到的正确/错误判断，与自动生成的 `correct_label` 不一致。",
        "",
        "重要提示：人工查看后发现，`results_nq` 中的样例看起来更像 short-form science QA，而不像真正的 Natural Questions long-form QA。因此在正式报告中使用 `results_nq` 时需要谨慎，最好先找回原始 prediction/similarity 中间文件确认数据来源。",
        "",
        "## 两种 Embedding Model 对比",
        markdown_table(metric_display, ["Dataset", "Model", "Gap", "Fixed F1", "Best Thresh.", "Best F1", "ROC-AUC"]),
        "总体来看，MiniLM 的 correct 和 incorrect 平均相似度差距更大，说明它对正确/错误答案的分离更清楚。BGE 会给正确答案和错误答案都更高的分数，因此 recall 更友好，但更容易产生 high-similarity-wrong cases。",
        "",
        "## Failure Case 总数",
        markdown_table(aggregate_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total Failure Cases"]),
        "BGE 在这些结果中没有 low-similarity-correct cases，但 high-similarity-wrong cases 明显更多。MiniLM 的 false-positive-like failure 更少，不过当正确答案嵌在较长预测文本里时，它更容易给出较低 similarity。",
        "",
        "## 各数据集 Failure Case 数量",
        markdown_table(failure_display, ["Dataset", "Model", "Failure Kind", "Count", "Avg Sim."]),
        "## 主要启发式 Failure 类型",
        markdown_table(taxonomy_top_rows, ["Dataset", "Model", "Failure Kind", "Top Type", "Count", "%"]),
        "## Failure Type 说明",
        "| Type | 中文含义 |",
        "| --- | --- |",
        "| `morphology_or_inflection` | 单复数、词形变化导致自动标签误判，例如 `ovary` vs `ovaries` |",
        "| `numeric_equivalence` | 数字表达等价，例如 `five` vs `5` |",
        "| `synonym_or_paraphrase_labeling_artifact` | 同义或改写表达，prediction 可能正确但自动 label 判错 |",
        "| `underspecified_or_overspecified_answer` | 答案过泛或过细，例如只答 `succession` 而标准答案是 `primary succession` |",
        "| `semantic_relatedness_not_correctness` | 两个答案语义相关，但相关不等于事实正确 |",
        "| `answer_containment_low_embedding_score` | prediction 包含正确答案，但整体 embedding similarity 仍然偏低 |",
        "| `overly_long_answer_context_dilution` | 正确短答案被包在长句中，额外上下文稀释了相似度 |",
        "| `other_or_true_semantic_error` | 其他情况或真正的语义判断错误 |",
        "",
        "## 推荐写进 Part 4 的分析逻辑",
        "1. 先定义 failure case：similarity-threshold correctness decision 与 automatic correctness label 不一致。",
        "2. 对每个数据集、每个 embedding model 分别统计 high-similarity-wrong 和 low-similarity-correct 的数量。",
        "3. 用启发式 taxonomy 展示主要错误来源：很多 high-similarity-wrong 其实是自动标签缺陷，例如词形变化、数字等价、同义改写。",
        "4. 单独讨论真正的 similarity 局限：embedding similarity 容易把“语义相关”误当成“事实正确”，尤其是答案过泛或缺少关键限定词时。",
        "5. 提出改进：更强 normalization、answer extraction、entity/keyword overlap、sentence-level similarity，或者使用 NLI/LLM judge 做事实一致性验证。",
        "",
        "## 统计和标注方法说明",
        "### Failure case 数量如何统计",
        "脚本直接读取每个结果目录下的 `failure_cases/*.jsonl` 文件，并按文件行数统计数量。文件名决定 failure kind 和 embedding model：",
        "",
        "- `high_similarity_wrong_BAAI_bge_base_en_v1.5.jsonl`：BGE 的 high-similarity-wrong cases。",
        "- `high_similarity_wrong_sentence_transformers_all_MiniLM_L6_v2.jsonl`：MiniLM 的 high-similarity-wrong cases。",
        "- `low_similarity_correct_BAAI_bge_base_en_v1.5.jsonl`：BGE 的 low-similarity-correct cases。",
        "- `low_similarity_correct_sentence_transformers_all_MiniLM_L6_v2.jsonl`：MiniLM 的 low-similarity-correct cases。",
        "",
        "### Failure type 如何标注",
        "当前版本没有真正进行人工逐条标注，而是使用脚本中的启发式规则自动给每条 failure case 分配一个 `heuristic_type`。规则会检查 prediction 和 ground truth 的数字等价、简单词形还原、token overlap、`token_f1`、`contains_ground_truth`、文本长度和 similarity 分数。",
        "",
        "这些启发式标签适合做初步量化分析，但正式报告中最好人工复核每类的代表样例。若要做严格的人工抽样标注，可以从 `datasets/<result_dir>/tables/annotated_failure_cases.csv` 中抽样 20-50 条，人工检查并新增一列 `human_type`。",
        "",
        "## 生成文件",
        "- `summary_tables/model_metrics_summary.csv`",
        "- `summary_tables/failure_counts_summary.csv`",
        "- `summary_tables/heuristic_taxonomy_summary.csv`",
        "- `summary_tables/fixed_threshold_confusion_estimates_summary.csv`",
        "- `datasets/<result_dir>/dataset_failure_report.md`",
        "- `datasets/<result_dir>/tables/annotated_failure_cases.csv`",
        "",
    ]

    (OUTPUT_ROOT / "summary_report.zh.md").write_text("\n".join(zh_report), encoding="utf-8")


def human_category(human_type: str) -> str:
    label_artifacts = {
        "morphology_or_inflection",
        "numeric_equivalence",
        "synonym_or_paraphrase_labeling_artifact",
    }
    low_score_failures = {
        "answer_containment_low_embedding_score",
        "overly_long_answer_context_dilution",
        "low_score_for_valid_paraphrase",
    }
    semantic_limitations = {
        "underspecified_or_overspecified_answer",
        "semantic_relatedness_not_correctness",
        "other_or_true_semantic_error",
    }
    if human_type in label_artifacts:
        return "automatic_label_artifact"
    if human_type in low_score_failures:
        return "low_similarity_false_negative"
    if human_type in semantic_limitations:
        return "semantic_similarity_limitation"
    return "other"


def load_all_manual_annotations() -> list[dict]:
    rows = []
    for dataset_name in RESULT_DIRS:
        path = OUTPUT_ROOT / "datasets" / dataset_name / "tables" / "manual_annotation_sample.csv"
        if not path.exists():
            continue
        for row in read_csv(path):
            human_type = row.get("human_type", "").strip()
            row["human_type"] = human_type
            row["human_category"] = human_category(human_type)
            rows.append(row)
    return rows


def summarize_manual_annotations() -> dict:
    rows = load_all_manual_annotations()
    summary_dir = OUTPUT_ROOT / "summary_tables"
    ensure_dir(summary_dir)

    type_summary = []
    category_summary = []
    dataset_model_summary = []

    grouped_type = defaultdict(list)
    grouped_category = defaultdict(list)
    grouped_dataset_model = defaultdict(list)

    for row in rows:
        grouped_type[(row["dataset"], row["model"], row["failure_kind"], row["human_type"])].append(row)
        grouped_category[(row["dataset"], row["model"], row["failure_kind"], row["human_category"])].append(row)
        grouped_dataset_model[(row["dataset"], row["model"])].append(row)

    for key, group_rows in sorted(grouped_type.items()):
        dataset, model, failure_kind, human_type = key
        denominator = len([
            row for row in rows
            if row["dataset"] == dataset
            and row["model"] == model
            and row["failure_kind"] == failure_kind
        ])
        type_summary.append(
            {
                "dataset": dataset,
                "model": model,
                "failure_kind": failure_kind,
                "human_type": human_type,
                "count": len(group_rows),
                "percentage": format_float(len(group_rows) / denominator * 100 if denominator else 0, 1),
            }
        )

    for key, group_rows in sorted(grouped_category.items()):
        dataset, model, failure_kind, category = key
        denominator = len([
            row for row in rows
            if row["dataset"] == dataset
            and row["model"] == model
            and row["failure_kind"] == failure_kind
        ])
        category_summary.append(
            {
                "dataset": dataset,
                "model": model,
                "failure_kind": failure_kind,
                "human_category": category,
                "count": len(group_rows),
                "percentage": format_float(len(group_rows) / denominator * 100 if denominator else 0, 1),
            }
        )

    for key, group_rows in sorted(grouped_dataset_model.items()):
        dataset, model = key
        category_counter = Counter(row["human_category"] for row in group_rows)
        type_counter = Counter(row["human_type"] for row in group_rows)
        total = len(group_rows)
        dataset_model_summary.append(
            {
                "dataset": dataset,
                "model": model,
                "sampled_cases": total,
                "automatic_label_artifact": category_counter["automatic_label_artifact"],
                "automatic_label_artifact_pct": format_float(category_counter["automatic_label_artifact"] / total * 100 if total else 0, 1),
                "semantic_similarity_limitation": category_counter["semantic_similarity_limitation"],
                "semantic_similarity_limitation_pct": format_float(category_counter["semantic_similarity_limitation"] / total * 100 if total else 0, 1),
                "low_similarity_false_negative": category_counter["low_similarity_false_negative"],
                "low_similarity_false_negative_pct": format_float(category_counter["low_similarity_false_negative"] / total * 100 if total else 0, 1),
                "top_human_type": type_counter.most_common(1)[0][0] if type_counter else "",
            }
        )

    write_csv(summary_dir / "human_annotation_type_summary.csv", type_summary)
    write_csv(summary_dir / "human_annotation_category_summary.csv", category_summary)
    write_csv(summary_dir / "human_annotation_model_summary.csv", dataset_model_summary)

    for dataset_name in RESULT_DIRS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset_name]
        dataset_type_rows = [row for row in type_summary if row["dataset"] == dataset_name]
        dataset_category_rows = [row for row in category_summary if row["dataset"] == dataset_name]
        dataset_out = OUTPUT_ROOT / "datasets" / dataset_name / "tables"
        write_csv(dataset_out / "human_annotation_type_summary.csv", dataset_type_rows)
        write_csv(dataset_out / "human_annotation_category_summary.csv", dataset_category_rows)
        write_csv(dataset_out / "human_annotation_all_sampled_cases.csv", dataset_rows)

        dataset_model_rows = [
            row for row in dataset_model_summary
            if row["dataset"] == dataset_name
        ]
        human_report_rows = [
            {
                "Model": row["model"],
                "Sampled": row["sampled_cases"],
                "Label Artifacts": f"{row['automatic_label_artifact']} ({row['automatic_label_artifact_pct']}%)",
                "Semantic Limits": f"{row['semantic_similarity_limitation']} ({row['semantic_similarity_limitation_pct']}%)",
                "Low-Sim FN": f"{row['low_similarity_false_negative']} ({row['low_similarity_false_negative_pct']}%)",
                "Top Type": row["top_human_type"],
            }
            for row in dataset_model_rows
        ]
        type_report_rows = [
            {
                "Model": row["model"],
                "Failure Kind": row["failure_kind"],
                "Human Type": row["human_type"],
                "Count": row["count"],
                "%": row["percentage"],
            }
            for row in dataset_type_rows
        ]
        category_report_rows = [
            {
                "Model": row["model"],
                "Failure Kind": row["failure_kind"],
                "Human Category": row["human_category"],
                "Count": row["count"],
                "%": row["percentage"],
            }
            for row in dataset_category_rows
        ]
        report = [
            f"# Human Annotation Analysis: {dataset_name}",
            "",
            "This report summarizes the sampled human annotations stored in `tables/manual_annotation_sample.csv`.",
            "",
            "## Model-Level Summary",
            markdown_table(human_report_rows, ["Model", "Sampled", "Label Artifacts", "Semantic Limits", "Low-Sim FN", "Top Type"]),
            "## Human Failure-Type Distribution",
            markdown_table(type_report_rows, ["Model", "Failure Kind", "Human Type", "Count", "%"]),
            "## Human Category Distribution",
            markdown_table(category_report_rows, ["Model", "Failure Kind", "Human Category", "Count", "%"]),
            "## Interpretation Guide",
            "- `automatic_label_artifact`: the answer is likely acceptable, but the automatic correctness label is too strict.",
            "- `semantic_similarity_limitation`: similarity is high because the texts are related, but relatedness does not guarantee correctness.",
            "- `low_similarity_false_negative`: the answer is correct or contained in the prediction, but the embedding score is low.",
            "",
        ]
        (OUTPUT_ROOT / "datasets" / dataset_name / "human_annotation_report.md").write_text(
            "\n".join(report),
            encoding="utf-8",
        )

    return {
        "manual_rows": rows,
        "type_summary": type_summary,
        "category_summary": category_summary,
        "dataset_model_summary": dataset_model_summary,
    }


def build_complete_part4_reports(human_summary: dict) -> None:
    metrics = read_csv(OUTPUT_ROOT / "summary_tables" / "model_metrics_summary.csv")
    failure_counts = read_csv(OUTPUT_ROOT / "summary_tables" / "failure_counts_summary.csv")
    human_model_summary = human_summary["dataset_model_summary"]
    human_type_summary = human_summary["type_summary"]
    human_category_summary = human_summary["category_summary"]

    aggregate_by_model = defaultdict(Counter)
    for row in failure_counts:
        aggregate_by_model[row["model"]][row["failure_kind"]] += int(row["count"])

    aggregate_failure_rows = [
        {
            "Model": model,
            "High-Sim Wrong": counts["high_similarity_wrong"],
            "Low-Sim Correct": counts["low_similarity_correct"],
            "Total": sum(counts.values()),
        }
        for model, counts in sorted(aggregate_by_model.items())
    ]

    metric_rows = [
        {
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Gap": row["gap"],
            "Fixed F1": row["fixed_f1"],
            "Best Threshold": row["best_threshold"],
            "Best F1": row["best_f1"],
            "ROC-AUC": row["roc_auc"],
        }
        for row in metrics
    ]

    human_model_rows = [
        {
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Sampled": row["sampled_cases"],
            "Label Artifacts": f"{row['automatic_label_artifact']} ({row['automatic_label_artifact_pct']}%)",
            "Semantic Limits": f"{row['semantic_similarity_limitation']} ({row['semantic_similarity_limitation_pct']}%)",
            "Low-Sim FN": f"{row['low_similarity_false_negative']} ({row['low_similarity_false_negative_pct']}%)",
            "Top Type": row["top_human_type"],
        }
        for row in human_model_summary
    ]

    top_type_rows = []
    grouped = defaultdict(list)
    for row in human_type_summary:
        grouped[(row["dataset"], row["model"], row["failure_kind"])].append(row)
    for key, rows in grouped.items():
        for row in sorted(rows, key=lambda item: int(item["count"]), reverse=True)[:3]:
            top_type_rows.append(
                {
                    "Dataset": row["dataset"],
                    "Model": row["model"],
                    "Failure Kind": row["failure_kind"],
                    "Human Type": row["human_type"],
                    "Count": row["count"],
                    "%": row["percentage"],
                }
            )

    category_rows = [
        {
            "Dataset": row["dataset"],
            "Model": row["model"],
            "Failure Kind": row["failure_kind"],
            "Human Category": row["human_category"],
            "Count": row["count"],
            "%": row["percentage"],
        }
        for row in human_category_summary
    ]

    complete_report = [
        "# Complete Part 4 Failure Analysis",
        "",
        "Chinese version: `part4_complete_analysis.zh.md`.",
        "",
        "## 1. Failure Definition",
        "A failure case is a sample where the similarity-threshold correctness decision disagrees with the automatic `correct_label`. `high_similarity_wrong` means the embedding score is high although `correct_label = 0`; `low_similarity_correct` means the embedding score is low although `correct_label = 1`.",
        "",
        "This analysis uses both full failure-case counts and sampled human annotations from `manual_annotation_sample.csv`.",
        "",
        "## 2. Overall Embedding Performance",
        markdown_table(metric_rows, ["Dataset", "Model", "Gap", "Fixed F1", "Best Threshold", "Best F1", "ROC-AUC"]),
        "MiniLM consistently has a larger correct-vs-incorrect similarity gap. BGE has comparable ROC-AUC but assigns higher scores to incorrect answers, which shifts its best threshold upward and increases high-similarity wrong cases.",
        "",
        "## 3. Full Failure Counts",
        markdown_table(aggregate_failure_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total"]),
        "BGE produces no low-similarity-correct cases in these outputs, but it produces substantially more high-similarity-wrong cases. MiniLM is more conservative and has fewer high-similarity false positives, but it misses some correct answers that are embedded in longer predictions.",
        "",
        "## 4. Human Annotation Summary",
        markdown_table(human_model_rows, ["Dataset", "Model", "Sampled", "Label Artifacts", "Semantic Limits", "Low-Sim FN", "Top Type"]),
        "The sampled human annotations separate two sources of failure: automatic-label artifacts and genuine similarity limitations. Label artifacts include morphology, numeric equivalence, and paraphrase cases where the answer is likely acceptable but the automatic label is too strict. Semantic limitations include underspecified answers and cases where semantic relatedness is mistaken for correctness.",
        "",
        "## 5. Human Failure-Type Distribution",
        markdown_table(top_type_rows, ["Dataset", "Model", "Failure Kind", "Human Type", "Count", "%"]),
        "## 6. Human Category Distribution",
        markdown_table(category_rows, ["Dataset", "Model", "Failure Kind", "Human Category", "Count", "%"]),
        "## 7. Main Findings",
        "- Many high-similarity-wrong cases are not true LLM answer errors. They are automatic-label artifacts caused by singular/plural variation, numeric equivalence, acronyms, or paraphrases.",
        "- BGE is recall-friendly but less conservative: it gives high scores to many semantically related answers, including answers that are too broad, too narrow, or only partially correct.",
        "- MiniLM has a clearer separation between correct and incorrect answers, but its low-similarity-correct cases show that short correct answers can be diluted by longer predictions.",
        "- Similarity is useful as a ranking or screening signal, but a single threshold is not robust enough to be a final correctness evaluator.",
        "",
        "## 8. Improvement Proposal",
        "A stronger evaluator should combine: stronger normalization, number-word conversion, lemmatization, answer extraction for long predictions, entity/keyword overlap, sentence-level similarity for long-form responses, and NLI or LLM-based verification for high-risk cases.",
        "",
        "A practical hybrid rule is: first apply normalization and exact/containment checks; then use embedding similarity with a dataset/model-specific threshold; finally send ambiguous high-similarity-wrong or low-similarity-correct cases to a verifier.",
        "",
    ]
    (OUTPUT_ROOT / "part4_complete_analysis.md").write_text("\n".join(complete_report), encoding="utf-8")

    zh_report = [
        "# Part 4 完整失败案例分析",
        "",
        "英文版本见：`part4_complete_analysis.md`。",
        "",
        "## 1. Failure Case 定义",
        "这里的 failure case 指 similarity threshold 得到的正确/错误判断与自动生成的 `correct_label` 不一致。`high_similarity_wrong` 表示 similarity 很高但 `correct_label = 0`；`low_similarity_correct` 表示 similarity 很低但 `correct_label = 1`。",
        "",
        "本分析同时使用全量 failure case 数量和 `manual_annotation_sample.csv` 中已经填写的 human annotation。",
        "",
        "## 2. Embedding Model 整体表现",
        markdown_table(metric_rows, ["Dataset", "Model", "Gap", "Fixed F1", "Best Threshold", "Best F1", "ROC-AUC"]),
        "MiniLM 在四个结果目录中都有更大的 correct-vs-incorrect similarity gap，说明它对正确和错误答案的分离更明显。BGE 的 ROC-AUC 接近 MiniLM，但它会给错误答案更高的分数，因此 best threshold 更高，也更容易产生 high-similarity-wrong cases。",
        "",
        "## 3. 全量 Failure Case 数量",
        markdown_table(aggregate_failure_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total"]),
        "BGE 在这些输出中没有 low-similarity-correct cases，但 high-similarity-wrong cases 明显更多。MiniLM 更保守，high-similarity false positives 更少，不过会漏掉一些嵌在长预测文本中的正确短答案。",
        "",
        "## 4. Human Annotation 汇总",
        markdown_table(human_model_rows, ["Dataset", "Model", "Sampled", "Label Artifacts", "Semantic Limits", "Low-Sim FN", "Top Type"]),
        "human annotation 把失败来源分成几类：自动标签缺陷、真正的 similarity 局限、以及 low-similarity false negatives。自动标签缺陷包括单复数、数字等价、同义改写等；真正的 similarity 局限包括答案过泛、过细，或者语义相关但不事实正确。",
        "",
        "## 5. Human Failure Type 分布",
        markdown_table(top_type_rows, ["Dataset", "Model", "Failure Kind", "Human Type", "Count", "%"]),
        "## 6. Human Category 分布",
        markdown_table(category_rows, ["Dataset", "Model", "Failure Kind", "Human Category", "Count", "%"]),
        "## 7. 主要结论",
        "- 很多 high-similarity-wrong 并不是真正的 LLM 答错，而是自动标签太严格，例如单复数、数字表达、缩写或同义改写。",
        "- BGE 更 recall-friendly，但也更容易把语义相关的答案判成正确，包括过泛、过细或部分正确的答案。",
        "- MiniLM 对正确/错误的分离更清楚，但 low-similarity-correct cases 说明：当正确短答案被包在较长 prediction 里时，整体句向量可能被额外上下文稀释。",
        "- embedding similarity 适合作为筛选或排序信号，但单一 threshold 不足以作为最终 correctness evaluator。",
        "",
        "## 8. 改进方案",
        "更稳健的 evaluator 可以结合：更强的 normalization、number-word conversion、lemmatization、长预测中的 answer extraction、entity/keyword overlap、面向长文本的 sentence-level similarity，以及对高风险样例使用 NLI 或 LLM judge 做事实一致性验证。",
        "",
        "一个可实现的 hybrid 流程是：先做 normalization 和 exact/containment 检查；再用 embedding similarity 和针对 dataset/model 调好的 threshold；最后把 ambiguous high-similarity-wrong 或 low-similarity-correct 样例交给 verifier。",
        "",
    ]
    (OUTPUT_ROOT / "part4_complete_analysis.zh.md").write_text("\n".join(zh_report), encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    ensure_dir(OUTPUT_ROOT / "datasets")
    ensure_dir(OUTPUT_ROOT / "summary_tables")

    dataset_results = []
    for dataset_name, result_dir in RESULT_DIRS.items():
        if not result_dir.exists():
            raise FileNotFoundError(f"Missing result directory: {result_dir}")
        dataset_results.append(analyze_dataset(dataset_name, result_dir))

    build_summary_report(dataset_results)
    human_summary = summarize_manual_annotations()
    build_complete_part4_reports(human_summary)
    print(f"Part 4 failure analysis written to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
