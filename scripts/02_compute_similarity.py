import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.compute_embeddings import create_embedding_model
from src.compute_similarity import add_similarity_scores
from src.correctness_labeling import label_correctness_for_records
from src.reference_answer import prepare_reference_answers, resolve_reference_field
from src.utils import (
    ensure_dir,
    load_config,
    load_jsonl,
    print_config_summary,
    save_jsonl,
    validate_records_dataset,
)


def main() -> None:
    config = load_config("config.yaml")

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

    reference_field = resolve_reference_field(config)
    print(f"Preparing evaluation references with field: {reference_field}")
    records = prepare_reference_answers(records, config["data"]["dataset"])

    print("Labeling correctness...")
    records = label_correctness_for_records(
        records,
        prediction_field="prediction",
        reference_field=reference_field,
        f1_threshold=0.8,
    )

    for model_name in embedding_models:
        print(f"Computing similarity with embedding model: {model_name}")

        embedding_model = create_embedding_model(model_name)

        records = add_similarity_scores(
            records=records,
            embedding_model=embedding_model,
            embedding_model_name=model_name,
            batch_size=batch_size,
            prediction_field="prediction",
            reference_field=reference_field,
        )

    print(f"Saving similarity results to: {output_file}")
    save_jsonl(records, output_file)

    print("Similarity computation finished.")


if __name__ == "__main__":
    main()
