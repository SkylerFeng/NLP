import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.utils import ensure_dir, load_config, load_jsonl, print_config_summary, validate_records_dataset
from src.visualize import (
    plot_pr_curve,
    plot_roc_curve,
    plot_similarity_correlation,
    plot_similarity_distribution,
)


def safe_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def config_path_from_args() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "config.yaml"


def main() -> None:
    config = load_config(config_path_from_args(), stage="visualization")

    input_file = config["evaluation"]["input_file"]
    label_field = config["evaluation"].get("label_field", "correct_label")

    embedding_models = config["embedding"]["models"]
    figure_dir = Path(config["output"]["figure_dir"])

    ensure_dir(figure_dir)

    print_config_summary(config)
    print(f"Loading results from: {input_file}")
    records = load_jsonl(input_file)
    validate_records_dataset(records, config["data"]["dataset"])
    print(f"Loaded {len(records)} records.")

    for model_name in embedding_models:
        model_key = safe_model_name(model_name)
        similarity_field = f"similarity_{model_key}"

        print(f"Generating plots for: {similarity_field}")

        distribution_path = figure_dir / f"similarity_distribution_{model_key}.png"
        roc_path = figure_dir / f"roc_curve_{model_key}.png"
        pr_path = figure_dir / f"pr_curve_{model_key}.png"

        plot_similarity_distribution(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            output_path=distribution_path,
        )

        plot_roc_curve(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            output_path=roc_path,
        )

        plot_pr_curve(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            output_path=pr_path,
        )

        corr_path = figure_dir / f"similarity_correlation_{model_key}.png"
        plot_similarity_correlation(
            records=records,
            similarity_field=similarity_field,
            label_field=label_field,
            lexical_field="token_f1",
            output_path=corr_path,
        )

        print(f"Saved: {distribution_path}")
        print(f"Saved: {roc_path}")
        print(f"Saved: {pr_path}")
        print(f"Saved: {corr_path}")

    print("Visualization finished.")


if __name__ == "__main__":
    main()
