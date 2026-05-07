from pathlib import Path
from typing import Dict, List, Optional
import json
import random


SUPPORTED_DATASETS = {
    "nq",
    "sciq",
    "simple_questions_wiki",
    "truthfulQA",
}


def load_teacher_processed_dataset(
    dataset_name: str,
    data_root: str = "processed_data",
    filename: str = "merged_fb.json",
) -> List[Dict]:
    """
    Load teacher-provided processed QA dataset.

    Expected path:
        processed_data/{dataset_name}/merged_fb.json

    Expected JSONL format:
        {"question": "...", "correct_answer": "..."}

    Output project format:
        id
        dataset
        split
        question
        ground_truth
        raw
    """
    if dataset_name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}. "
            f"Supported datasets: {sorted(SUPPORTED_DATASETS)}"
        )

    path = Path(data_root) / dataset_name / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}, line {line_id}: {e}")

            if "question" not in item:
                raise KeyError(f"Missing 'question' in {path}, line {line_id}")

            if "correct_answer" not in item:
                raise KeyError(f"Missing 'correct_answer' in {path}, line {line_id}")

            record = {
                "id": f"{dataset_name}_{line_id}",
                "dataset": dataset_name,
                "split": "processed",
                "question": item["question"],
                "ground_truth": item["correct_answer"],
                "raw": item,
            }

            records.append(record)

    print(f"Loaded {len(records)} records from {path}")
    return records


def sample_records(
    records: List[Dict],
    sample_size: Optional[int] = None,
    seed: int = 42,
) -> List[Dict]:
    """
    Randomly sample records.

    If sample_size is None or larger than dataset size, return all records.
    """
    if sample_size is None:
        return records

    if sample_size >= len(records):
        return records

    rng = random.Random(seed)
    return rng.sample(records, sample_size)


def build_prompt_for_qa(
    question: str,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Build a general short-answer QA prompt.

    This works for:
        nq
        sciq
        simple_questions_wiki
        truthfulQA
    """
    if dataset_name == "truthfulQA":
        return (
            "Answer the following question truthfully in one or two concise sentences.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    if dataset_name == "sciq":
        return (
            "Answer the following science question with a short phrase.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    if dataset_name == "simple_questions_wiki":
        return (
            "Answer the following factual question with a short phrase.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    if dataset_name == "nq":
        return (
            "Answer the following question in one or two concise sentences.\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    return (
        "Answer the following question concisely.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def build_prompt_for_short_answer(question: str) -> str:
    """
    Backward-compatible wrapper for old SciQ code.
    """
    return build_prompt_for_qa(question, dataset_name="sciq")


def load_sciq_split(
    dataset_name: str = "sciq",
    split: str = "processed",
    local_path: Optional[str] = None,
) -> List[Dict]:
    """
    Backward-compatible wrapper.

    The old project used load_sciq_split().
    Now it loads the teacher-provided SciQ processed data by default.
    """
    if local_path is not None:
        path = Path(local_path)

        if not path.exists():
            raise FileNotFoundError(f"Local file not found: {path}")

        records = []

        with path.open("r", encoding="utf-8") as f:
            for line_id, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                records.append({
                    "id": f"sciq_{line_id}",
                    "dataset": "sciq",
                    "split": split,
                    "question": item["question"],
                    "ground_truth": item["correct_answer"],
                    "raw": item,
                })

        return records

    return load_teacher_processed_dataset("sciq")


def convert_sciq_to_qa_format(
    examples: List[Dict],
    dataset_split: str = "processed",
) -> List[Dict]:
    """
    Backward-compatible wrapper.

    If examples are already in project format, return them.
    Otherwise convert raw SciQ-style examples.
    """
    processed = []

    for idx, item in enumerate(examples):
        if "ground_truth" in item:
            processed.append(item)
            continue

        record = {
            "id": f"sciq_{dataset_split}_{idx}",
            "dataset": "sciq",
            "split": dataset_split,
            "question": item["question"],
            "ground_truth": item["correct_answer"],
            "support": item.get("support", ""),
            "distractor1": item.get("distractor1", ""),
            "distractor2": item.get("distractor2", ""),
            "distractor3": item.get("distractor3", ""),
            "raw": item,
        }

        processed.append(record)

    return processed


def build_prompt_with_support(question: str, support: str) -> str:
    """
    Optional context-based prompt.
    This is mainly useful for SciQ if support is available.
    Teacher-processed data may not include support.
    """
    return (
        "Answer the following science question using the provided context. "
        "Use a short phrase.\n\n"
        f"Context: {support}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
