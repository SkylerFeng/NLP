import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


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

    return config


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