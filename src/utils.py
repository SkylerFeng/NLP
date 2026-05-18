import json
import os
import random
import re
from copy import deepcopy
from datetime import datetime
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

LATEST_RUN_FILENAME = "latest_run_id.txt"
# Default project layout used when config.yaml asks for automatic path resolution.
DEFAULT_DATA_ROOT = "data/raw"
DEFAULT_PREDICTION_DIR = "data/interim/predictions"
DEFAULT_SIMILARITY_DIR = "data/interim/similarity"
DEFAULT_EXPERIMENTS_DIR = "outputs/experiments"


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def load_config(config_path: str = "config.yaml", stage: str | None = None) -> Dict[str, Any]:
    """
    Load YAML config file.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return resolve_config(config, stage=stage)


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


def generated_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def latest_run_id_path(base_results_dir: Path) -> Path:
    return base_results_dir / LATEST_RUN_FILENAME


def read_latest_run_id(base_results_dir: Path) -> str:
    path = latest_run_id_path(base_results_dir)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def resolve_run_id(project_config: Dict[str, Any], base_results_dir: Path, stage: str | None) -> str:
    configured_run_id = project_config.get("run_id")
    if configured_run_id and configured_run_id != "auto":
        return slugify(configured_run_id)

    # Environment override is useful for rerunning all stages against one fixed run.
    environment_run_id = os.environ.get("EXPERIMENT_RUN_ID")
    if environment_run_id:
        return slugify(environment_run_id)

    if stage in {"evaluation", "visualization"}:
        # Downstream stages default to the latest similarity run created earlier.
        latest_run_id = read_latest_run_id(base_results_dir)
        if latest_run_id:
            return latest_run_id

    return generated_run_id()


def write_latest_run_marker(config: Dict[str, Any]) -> None:
    run_id = config.get("project", {}).get("resolved_run_id")
    base_results_dir = config.get("output", {}).get("base_results_dir")
    if not run_id or not base_results_dir:
        return

    path = latest_run_id_path(Path(base_results_dir))
    ensure_dir(path.parent)
    path.write_text(str(run_id) + "\n", encoding="utf-8")


def resolve_config(config: Dict[str, Any], stage: str | None = None) -> Dict[str, Any]:
    """
    Resolve dataset-specific paths from config.

    When project.auto_paths is true, users only need to change data.dataset and
    data.sample_size. All input, intermediate, and result paths are derived from
    those fields to avoid mixing SciQ, NQ, Wiki, and TruthfulQA outputs.
    """
    config = deepcopy(config)
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

        data_root = Path(config["data"].get("data_root", DEFAULT_DATA_ROOT))
        data_file = config["data"].get("data_file", "merged_fb.json")
        prediction_dir = Path(config["data"].get("prediction_dir", DEFAULT_PREDICTION_DIR))
        similarity_dir = Path(config["data"].get("similarity_dir", DEFAULT_SIMILARITY_DIR))
        experiments_dir = Path(config["output"].get("experiments_dir", DEFAULT_EXPERIMENTS_DIR))
        results_base_dir = experiments_dir / f"results_{dataset}_{tag}"

        prediction_file = prediction_dir / f"{dataset}_{llm_name}_predictions_{tag}.jsonl"
        legacy_similarity_file = similarity_dir / f"{dataset}_{llm_name}_similarity_{tag}.jsonl"
        preserve_runs = (
            config["project"].get("preserve_runs", True)
            and stage != "prediction"
        )

        if preserve_runs:
            run_id = resolve_run_id(config["project"], results_base_dir, stage)
            results_dir = results_base_dir / "runs" / run_id
            similarity_file = (
                results_dir
                / "similarity"
                / f"{dataset}_{llm_name}_similarity_{tag}.jsonl"
            )
            configured_run_id = config["project"].get("run_id")
            if (
                stage in {"evaluation", "visualization"}
                and not read_latest_run_id(results_base_dir)
                and not os.environ.get("EXPERIMENT_RUN_ID")
                and (not configured_run_id or configured_run_id == "auto")
                and legacy_similarity_file.exists()
            ):
                # Keep old fixed-path artifacts readable when no run marker exists yet.
                similarity_file = legacy_similarity_file
            config["project"]["resolved_run_id"] = run_id
            config["output"]["experiments_dir"] = str(experiments_dir)
            config["output"]["base_results_dir"] = str(results_base_dir)
        else:
            results_dir = results_base_dir
            similarity_file = legacy_similarity_file
            config["project"].pop("resolved_run_id", None)
            config["output"]["experiments_dir"] = str(experiments_dir)
            config["output"].pop("base_results_dir", None)

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
        Path(config["data"].get("data_root", DEFAULT_DATA_ROOT))
        / dataset
        / config["data"].get("data_file", "merged_fb.json")
    )

    if Path(config["prediction"]["input_file"]) != expected_input:
        # This guard prevents evaluating one dataset while accidentally reading another.
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
    if config.get("project", {}).get("resolved_run_id"):
        print(f"  run_id: {config['project']['resolved_run_id']}")


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
