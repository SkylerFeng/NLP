import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Allow running this script directly from the repository root without packaging src.
sys.path.append(str(PROJECT_ROOT))

from src.answer_span import add_prediction_answer_spans
from src.compute_embeddings import EmbeddingCache, create_embedding_model
from src.compute_similarity import (
    DEFAULT_SPAN_BLEND_WEIGHT,
    add_blended_similarity_scores,
    add_similarity_scores,
)
from src.correctness_labeling import label_correctness_for_records
from src.entity_overlap import add_factual_conflict_adjusted_scores
from src.factual_units import add_factual_unit_features
from src.multi_view_similarity import add_span_level_similarity_scores
from src.reference_answer import prepare_reference_answers, resolve_reference_field
from src.utils import (
    ensure_dir,
    load_config,
    load_jsonl,
    print_config_summary,
    save_jsonl,
    validate_records_dataset,
    write_latest_run_marker,
)


CORRECTNESS_FIELDS = (
    "exact_match",
    "token_f1",
    "contains_ground_truth",
    "contains_prediction_in_reference",
    "correct_label",
)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def default_reference_field(config: dict) -> str:
    if config.get("data", {}).get("dataset") == "nq":
        return "reference_answer"
    return "ground_truth"


def resolve_label_reference_field(config: dict, similarity_reference_field: str) -> str:
    # Labels and similarity can intentionally use different references for ablations.
    configured_field = config.get("evaluation", {}).get("label_reference_field")
    if configured_field:
        if configured_field == "auto":
            return default_reference_field(config)
        return configured_field

    if similarity_reference_field.endswith("_v2"):
        return default_reference_field(config)
    return similarity_reference_field


def ensure_records_have_field(records: list[dict], field: str) -> None:
    if any(field not in record for record in records):
        raise ValueError(f"Cannot label correctness because records are missing '{field}'.")


def add_correctness_fields_with_suffix(
    records: list[dict],
    reference_field: str,
    suffix: str,
) -> list[dict]:
    ensure_records_have_field(records, reference_field)
    labeled_records = label_correctness_for_records(
        records,
        prediction_field="prediction",
        reference_field=reference_field,
        f1_threshold=0.8,
    )

    output_records = []
    for record, labeled_record in zip(records, labeled_records):
        new_record = dict(record)
        for field in CORRECTNESS_FIELDS:
            new_record[f"{field}{suffix}"] = labeled_record[field]
        output_records.append(new_record)
    return output_records


def config_path_from_args() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "config.yaml"


def main() -> None:
    config = load_config(config_path_from_args(), stage="similarity")

    input_file = config["similarity"]["input_file"]
    output_file = config["similarity"]["output_file"]

    embedding_models = config["embedding"]["models"]
    batch_size = config["embedding"].get("batch_size", 32)

    ensure_dir(Path(output_file).parent)

    print_config_summary(config)
    print(f"Loading predictions from: {input_file}")
    records = load_jsonl(input_file)
    validate_records_dataset(records, config["data"]["dataset"])
    print(f"Loaded {len(records)} records.")

    # The reference used for similarity may be dataset-specific, especially for NQ.
    reference_field = resolve_reference_field(config)
    label_reference_field = resolve_label_reference_field(config, reference_field)
    print(f"Preparing evaluation references with field: {reference_field}")
    records = prepare_reference_answers(records, config["data"]["dataset"])
    print("Extracting prediction answer spans.")
    records = add_prediction_answer_spans(records)
    # Prefer the cleaned NQ reference for factual checks when it is available.
    factual_reference_field = (
        "reference_answer_v2"
        if config["data"]["dataset"] == "nq"
        and all("reference_answer_v2" in record for record in records)
        else reference_field
    )
    print(
        "Extracting factual units with "
        f"reference field: {factual_reference_field}"
    )
    records = add_factual_unit_features(
        records,
        reference_field=factual_reference_field,
        prediction_field="prediction_answer_span",
    )

    print(f"Labeling correctness with reference field: {label_reference_field}")
    records = label_correctness_for_records(
        records,
        prediction_field="prediction",
        reference_field=label_reference_field,
        f1_threshold=0.8,
    )

    # Optional v2 labels keep label-changing experiments auditable instead of
    # silently replacing the baseline correctness target.
    if config.get("evaluation", {}).get("enable_correct_label_v2", False):
        label_reference_field_v2 = config["evaluation"].get(
            "label_reference_field_v2",
            "reference_answer_v2",
        )
        print(f"Labeling v2 correctness with reference field: {label_reference_field_v2}")
        records = add_correctness_fields_with_suffix(
            records=records,
            reference_field=label_reference_field_v2,
            suffix="_v2",
        )

    for model_name in embedding_models:
        print(f"Computing similarity with embedding model: {model_name}")

        embedding_model = create_embedding_model(model_name)
        embedding_cache = EmbeddingCache(
            embedding_model=embedding_model,
            batch_size=batch_size,
        )

        records = add_similarity_scores(
            records=records,
            embedding_model=embedding_model,
            embedding_model_name=model_name,
            batch_size=batch_size,
            prediction_field="prediction",
            reference_field=reference_field,
            embedding_cache=embedding_cache,
        )
        if config["data"]["dataset"] == "nq" and all(
            "reference_answer_v2" in record for record in records
        ):
            # NQ gets extra answer-focused views because whole-passage similarity
            # can measure topic relatedness instead of answer equivalence.
            print(f"Computing v2 reference similarity with embedding model: {model_name}")
            records = add_similarity_scores(
                records=records,
                embedding_model=embedding_model,
                embedding_model_name=model_name,
                batch_size=batch_size,
                prediction_field="prediction",
                reference_field="reference_answer_v2",
                output_field_prefix="similarity_v2",
                embedding_cache=embedding_cache,
            )
            print(f"Computing prediction span similarity with embedding model: {model_name}")
            records = add_similarity_scores(
                records=records,
                embedding_model=embedding_model,
                embedding_model_name=model_name,
                batch_size=batch_size,
                prediction_field="prediction_answer_span",
                reference_field="reference_answer_v2",
                output_field_prefix="prediction_span_similarity",
                embedding_cache=embedding_cache,
            )
            model_key = safe_model_name(model_name)
            records = add_blended_similarity_scores(
                records=records,
                base_score_field=f"similarity_v2_{model_key}",
                span_score_field=f"prediction_span_similarity_{model_key}",
                output_field=f"prediction_span_blend_similarity_{model_key}",
                span_weight=DEFAULT_SPAN_BLEND_WEIGHT,
            )
            print(f"Computing span-level similarity with embedding model: {model_name}")
            records = add_span_level_similarity_scores(
                records=records,
                embedding_model=embedding_model,
                embedding_model_name=model_name,
                batch_size=batch_size,
                reference_field="reference_answer_v2",
                sentence_similarity_field=f"similarity_v2_{model_key}",
                embedding_cache=embedding_cache,
            )
            records = add_factual_conflict_adjusted_scores(
                records,
                score_field_pairs=[
                    (
                        f"similarity_v2_{model_key}",
                        f"factual_conflict_adjusted_similarity_{model_key}",
                    ),
                    (
                        f"prediction_span_blend_similarity_{model_key}",
                        f"factual_conflict_adjusted_prediction_span_blend_similarity_{model_key}",
                    ),
                    (
                        f"span_max_similarity_{model_key}",
                        f"factual_conflict_adjusted_span_max_similarity_{model_key}",
                    ),
                    (
                        f"multi_view_score_{model_key}",
                        f"factual_conflict_adjusted_multi_view_score_{model_key}",
                    ),
                ],
            )

    print(f"Saving similarity results to: {output_file}")
    save_jsonl(records, output_file)
    # Evaluation and visualization use this marker when project.run_id is auto.
    write_latest_run_marker(config)

    print("Similarity computation finished.")


if __name__ == "__main__":
    main()
