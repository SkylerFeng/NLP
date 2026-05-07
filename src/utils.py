import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


SUPPORTED_DATASETS = {
    "nq",
    "sciq",
    "simple_questions_wiki",
    "truthfulQA",
}

DATASET_TASK_TYPES = {
    "sciq": "short_form",
    "simple_questions_wiki": "short_form",
    "nq": "long_form",
    "truthfulQA": "long_form",
}


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return resolve_config(config)


def slugify(value: str) -> str:
    """
    Convert dataset/model names into stable path fragments.
    """
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def dataset_task_type(dataset_name: str) -> str:
    return DATASET_TASK_TYPES.get(dataset_name, "unknown")


def sample_tag(sample_size: Any) -> str:
    return "full" if sample_size is None else str(sample_size)


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve dataset-specific paths from config.

    When project.auto_paths is true, users only need to change data.dataset and
    data.sample_size. All input, intermediate, and result paths are derived from
    those fields to avoid mixing SciQ, NQ, Wiki, and TruthfulQA outputs.
    """
    config = dict(config)
    config.setdefault("project", {})
    config.setdefault("data", {})
    config.setdefault("llm", {})
    config.setdefault("prediction", {})
    config.setdefault("similarity", {})
    config.setdefault("evaluation", {})
    config.setdefault("output", {})

    if config["project"].get("auto_paths", False):
        dataset = config["data"].get("dataset")
        tag = sample_tag(config["data"].get("sample_size"))
        llm_name = slugify(config["llm"].get("run_name") or config["llm"].get("model", "llm"))

        data_root = Path(config["data"].get("data_root", "processed_data"))
        data_file = config["data"].get("data_file", "merged_fb.json")
        prediction_dir = Path(config["data"].get("prediction_dir", "data/predictions"))
        similarity_dir = Path(config["data"].get("similarity_dir", "data/similarity"))
        results_dir = Path(f"results_{dataset}_{tag}")

        prediction_file = prediction_dir / f"{dataset}_{llm_name}_predictions_{tag}.jsonl"
        similarity_file = similarity_dir / f"{dataset}_{llm_name}_similarity_{tag}.jsonl"

        config["prediction"]["input_file"] = str(data_root / dataset / data_file)
        config["prediction"]["output_file"] = str(prediction_file)
        config["similarity"]["input_file"] = str(prediction_file)
        config["similarity"]["output_file"] = str(similarity_file)
        config["evaluation"]["input_file"] = str(similarity_file)
        config["output"]["results_dir"] = str(results_dir)
        config["output"]["figure_dir"] = str(results_dir / "figures")
        config["output"]["table_dir"] = str(results_dir / "tables")
        config["output"]["failure_case_dir"] = str(results_dir / "failure_cases")
        config["output"]["report_asset_dir"] = str(results_dir / "report_assets")

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    dataset = config.get("data", {}).get("dataset")

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    expected_input = (
        Path(config["data"].get("data_root", "processed_data"))
        / dataset
        / config["data"].get("data_file", "merged_fb.json")
    )

    if Path(config["prediction"]["input_file"]) != expected_input:
        raise ValueError(
            "Config mismatch: prediction.input_file does not match data.dataset. "
            f"Expected '{expected_input}', got '{config['prediction']['input_file']}'. "
            "Set project.auto_paths: true or update all paths consistently."
        )

    if config["similarity"]["input_file"] != config["prediction"]["output_file"]:
        raise ValueError("Config mismatch: similarity.input_file must equal prediction.output_file.")

    if config["evaluation"]["input_file"] != config["similarity"]["output_file"]:
        raise ValueError("Config mismatch: evaluation.input_file must equal similarity.output_file.")

    results_dir = Path(config["output"]["results_dir"])
    for key in ("figure_dir", "table_dir", "failure_case_dir"):
        output_path = Path(config["output"][key])
        if output_path != results_dir and results_dir not in output_path.parents:
            raise ValueError(f"Config mismatch: output.{key} should live under output.results_dir.")


def print_config_summary(config: Dict[str, Any]) -> None:
    print("Resolved experiment:")
    print(f"  dataset: {config['data']['dataset']}")
    print(f"  task_type: {dataset_task_type(config['data']['dataset'])}")
    print(f"  sample_size: {config['data'].get('sample_size')}")
    print(f"  prediction_input: {config['prediction']['input_file']}")
    print(f"  prediction_output: {config['prediction']['output_file']}")
    print(f"  similarity_output: {config['similarity']['output_file']}")
    print(f"  results_dir: {config['output']['results_dir']}")


def validate_records_dataset(records: List[Dict[str, Any]], expected_dataset: str) -> None:
    """
    Guard against using a JSONL file produced for another dataset.
    """
    observed = {
        record.get("dataset")
        for record in records
        if record.get("dataset") is not None
    }

    if not observed:
        raise ValueError(
            "Loaded records do not contain a dataset field. Regenerate predictions/similarity with the current code."
        )

    if observed != {expected_dataset}:
        raise ValueError(
            "Dataset mismatch in loaded records. "
            f"Expected '{expected_dataset}', observed {sorted(observed)}. "
            "Regenerate predictions/similarity with the current config."
        )


def ensure_dir(path: str | Path) -> None:
    """
    Create directory if it does not exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_jsonl(records: Iterable[Dict[str, Any]], output_path: str | Path) -> None:
    """
    Save records to a JSONL file.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load records from a JSONL file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {input_path}")

    records = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def save_json(data: Any, output_path: str | Path) -> None:
    """
    Save data to a JSON file.
    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str | Path) -> Any:
    """
    Load data from a JSON file.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"JSON file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text: str) -> str:
    """
    Basic text normalization.
    """
    if text is None:
        return ""

    return " ".join(str(text).strip().lower().split())


def chunk_list(items: List[Any], batch_size: int) -> Iterable[List[Any]]:
    """
    Yield batches from a list.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]
