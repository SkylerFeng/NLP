from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.utils import ensure_dir


def plot_similarity_distribution(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    output_path: str | Path = "results/figures/similarity_distribution.png",
) -> None:
    """
    Plot similarity distributions for correct and incorrect predictions.
    """
    correct_scores = [
        record[similarity_field]
        for record in records
        if record[label_field] == 1
    ]

    incorrect_scores = [
        record[similarity_field]
        for record in records
        if record[label_field] == 0
    ]

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(8, 5))
    plt.hist(correct_scores, bins=30, alpha=0.6, label="Correct")
    plt.hist(incorrect_scores, bins=30, alpha=0.6, label="Incorrect")
    plt.xlabel("Similarity")
    plt.ylabel("Count")
    plt.title("Similarity Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_roc_curve(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    output_path: str | Path = "results/figures/roc_curve.png",
) -> None:
    """
    Plot ROC curve.
    """
    y_true = np.array([record[label_field] for record in records])
    y_score = np.array([record[similarity_field] for record in records])

    if len(set(y_true)) <= 1:
        raise ValueError("ROC curve requires both positive and negative labels.")

    fpr, tpr, _ = roc_curve(y_true, y_score)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="Similarity")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pr_curve(
    records: List[Dict],
    similarity_field: str,
    label_field: str = "correct_label",
    output_path: str | Path = "results/figures/pr_curve.png",
) -> None:
    """
    Plot Precision-Recall curve.
    """
    y_true = np.array([record[label_field] for record in records])
    y_score = np.array([record[similarity_field] for record in records])

    if len(set(y_true)) <= 1:
        raise ValueError("PR curve requires both positive and negative labels.")

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="Similarity")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()