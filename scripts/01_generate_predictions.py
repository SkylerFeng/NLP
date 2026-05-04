import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.generate_predictions import (
    create_llm_client,
    generate_predictions_from_file,
)
from src.utils import ensure_dir, load_config


def main() -> None:
    config = load_config("config.yaml")

    input_file = config["prediction"]["input_file"]
    output_file = config["prediction"]["output_file"]
    use_support = config["prediction"].get("use_support", False)

    ensure_dir(Path(output_file).parent)

    llm_client = create_llm_client(config)

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Use support: {use_support}")
    print(f"LLM provider: {config['llm']['provider']}")
    print(f"LLM model: {config['llm']['model']}")

    generate_predictions_from_file(
        input_path=input_file,
        output_path=output_file,
        llm_client=llm_client,
        use_support=use_support,
        sample_size=config["data"].get("sample_size"),
    )

    print("Prediction generation finished.")


if __name__ == "__main__":
    main()