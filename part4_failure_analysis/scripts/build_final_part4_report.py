import csv
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "summary_tables"
FIGURES = ROOT / "figures"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])


def short_dataset(name: str) -> str:
    mapping = {
        "results_nq_5000": "Natural Questions",
        "results_sciq_5000": "SciQ",
        "results_truthfulQA_500": "TruthfulQA",
        "results_wiki": "Wiki",
    }
    return mapping.get(name, name)


def make_metric_rows(metrics: list[dict]) -> list[dict]:
    return [
        {
            "Dataset": short_dataset(row["dataset"]),
            "Model": row["model"],
            "Gap": row["gap"],
            "Fixed F1": row["fixed_f1"],
            "Best Threshold": row["best_threshold"],
            "Best F1": row["best_f1"],
            "ROC-AUC": row["roc_auc"],
        }
        for row in metrics
    ]


def make_human_rows(human_model: list[dict]) -> list[dict]:
    return [
        {
            "Dataset": short_dataset(row["dataset"]),
            "Model": row["model"],
            "Sampled": row["sampled_cases"],
            "Label Artifacts": f"{row['automatic_label_artifact']} ({row['automatic_label_artifact_pct']}%)",
            "Semantic Limits": f"{row['semantic_similarity_limitation']} ({row['semantic_similarity_limitation_pct']}%)",
            "Low-Sim FN": f"{row['low_similarity_false_negative']} ({row['low_similarity_false_negative_pct']}%)",
            "Top Type": row["top_human_type"],
        }
        for row in human_model
    ]


def aggregate_failure_counts(failure_counts: list[dict]) -> list[dict]:
    aggregate: dict[str, dict[str, int]] = {}
    for row in failure_counts:
        model = row["model"]
        kind = row["failure_kind"]
        aggregate.setdefault(model, {"High-Sim Wrong": 0, "Low-Sim Correct": 0})
        if kind == "high_similarity_wrong":
            aggregate[model]["High-Sim Wrong"] += int(row["count"])
        elif kind == "low_similarity_correct":
            aggregate[model]["Low-Sim Correct"] += int(row["count"])

    output = []
    for model in sorted(aggregate):
        high = aggregate[model]["High-Sim Wrong"]
        low = aggregate[model]["Low-Sim Correct"]
        output.append(
            {
                "Model": model,
                "High-Sim Wrong": high,
                "Low-Sim Correct": low,
                "Total": high + low,
            }
        )
    return output


def top_human_type_rows(human_type: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in human_type:
        key = (row["dataset"], row["model"], row["failure_kind"])
        grouped.setdefault(key, []).append(row)

    output = []
    for rows in grouped.values():
        rows = sorted(rows, key=lambda item: int(item["count"]), reverse=True)
        for row in rows[:3]:
            output.append(
                {
                    "Dataset": short_dataset(row["dataset"]),
                    "Model": row["model"],
                    "Failure Kind": row["failure_kind"],
                    "Human Type": row["human_type"],
                    "Count": row["count"],
                    "%": row["percentage"],
                }
            )
    return output


def representative_failure_examples() -> list[dict]:
    return [
        {
            "Category": "automatic_label_artifact",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "ovaries",
            "Prediction": "Ovary",
            "Similarity": "0.895",
            "Distance": "0.105",
            "Interpretation": "Singular/plural variation; the automatic label is too strict.",
        },
        {
            "Category": "automatic_label_artifact",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "four",
            "Prediction": "4",
            "Similarity": "0.863",
            "Distance": "0.137",
            "Interpretation": "Numeric equivalence should be normalized before labeling.",
        },
        {
            "Category": "automatic_label_artifact",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "wider pelvis",
            "Prediction": "wider hips",
            "Similarity": "0.887",
            "Distance": "0.113",
            "Interpretation": "Close paraphrase/anatomical wording difference.",
        },
        {
            "Category": "semantic_similarity_limitation",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "bone fractures",
            "Prediction": "fractures",
            "Similarity": "0.891",
            "Distance": "0.109",
            "Interpretation": "Prediction is related but underspecified; it misses the bone modifier.",
        },
        {
            "Category": "semantic_similarity_limitation",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "proto-oncogenes",
            "Prediction": "Oncogenes",
            "Similarity": "0.835",
            "Distance": "0.165",
            "Interpretation": "Related biological term, but not the same answer.",
        },
        {
            "Category": "semantic_similarity_limitation",
            "Dataset": "SciQ",
            "Model": "BGE",
            "Failure Kind": "high_similarity_wrong",
            "Reference": "solar energy",
            "Prediction": "Solar panels",
            "Similarity": "0.837",
            "Distance": "0.163",
            "Interpretation": "Related concept, but source vs. device distinction matters.",
        },
        {
            "Category": "low_similarity_false_negative",
            "Dataset": "SciQ",
            "Model": "MiniLM",
            "Failure Kind": "low_similarity_correct",
            "Reference": "three",
            "Prediction": "Three main types: elliptical, spiral, and irregular.",
            "Similarity": "0.190",
            "Distance": "0.810",
            "Interpretation": "Correct short answer is embedded in a much longer sentence.",
        },
        {
            "Category": "low_similarity_false_negative",
            "Dataset": "SciQ",
            "Model": "MiniLM",
            "Failure Kind": "low_similarity_correct",
            "Reference": "negative",
            "Prediction": "Partial negative charge",
            "Similarity": "0.377",
            "Distance": "0.623",
            "Interpretation": "Reference is contained, but added context changes the sentence vector.",
        },
        {
            "Category": "low_similarity_false_negative",
            "Dataset": "SciQ",
            "Model": "MiniLM",
            "Failure Kind": "low_similarity_correct",
            "Reference": "bacteria",
            "Prediction": "Yogurt is made from milk fermented with bacteria.",
            "Similarity": "0.399",
            "Distance": "0.601",
            "Interpretation": "Answer containment is clear, but whole-sentence embedding is diluted.",
        },
    ]


def representative_failure_examples_zh() -> list[dict]:
    rows = []
    category_map = {
        "automatic_label_artifact": "自动标签缺陷",
        "semantic_similarity_limitation": "语义相似度局限",
        "low_similarity_false_negative": "低相似度假阴性",
    }
    kind_map = {
        "high_similarity_wrong": "高相似但标为错",
        "low_similarity_correct": "低相似但标为对",
    }
    interpretation_map = {
        "Singular/plural variation; the automatic label is too strict.": "单复数差异，自动标签过于严格。",
        "Numeric equivalence should be normalized before labeling.": "数字表达等价，标注前应做数字规范化。",
        "Close paraphrase/anatomical wording difference.": "近义表达或术语说法差异。",
        "Prediction is related but underspecified; it misses the bone modifier.": "预测相关但过泛，缺少 bone 这个关键限定。",
        "Related biological term, but not the same answer.": "相关生物术语，但并不是同一个答案。",
        "Related concept, but source vs. device distinction matters.": "概念相关，但 energy source 与 device 的区别会影响正确性。",
        "Correct short answer is embedded in a much longer sentence.": "正确短答案嵌在长句中。",
        "Reference is contained, but added context changes the sentence vector.": "包含参考答案，但额外上下文改变了句向量。",
        "Answer containment is clear, but whole-sentence embedding is diluted.": "答案包含关系很清楚，但整句 embedding 被稀释。",
    }
    for row in representative_failure_examples():
        rows.append(
            {
                "类别": category_map[row["Category"]],
                "数据集": row["Dataset"],
                "模型": row["Model"],
                "Failure Kind": kind_map[row["Failure Kind"]],
                "参考答案": row["Reference"],
                "预测答案": row["Prediction"],
                "相似度": row["Similarity"],
                "距离": row["Distance"],
                "解释": interpretation_map[row["Interpretation"]],
            }
        )
    return rows


def plot_metric_comparison(metrics: list[dict]) -> None:
    ensure_dir(FIGURES)
    datasets = []
    for row in metrics:
        label = short_dataset(row["dataset"])
        if label not in datasets:
            datasets.append(label)

    width = 1100
    height = 420
    margin = 60
    panel_gap = 50
    panel_w = (width - 2 * margin - panel_gap) / 2
    panel_h = 260
    colors = {"MiniLM": "#2f6fbb", "BGE": "#d07a2d"}

    def metric_value(dataset: str, model: str, metric: str) -> float:
        row = next(
            item
            for item in metrics
            if short_dataset(item["dataset"]) == dataset and item["model"] == model
        )
        return float(row[metric])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="28" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">Embedding Model Metric Comparison</text>',
    ]
    for panel_idx, (metric, title) in enumerate([("gap", "Similarity Gap"), ("fixed_f1", "Fixed-Threshold F1")]):
        x0 = margin + panel_idx * (panel_w + panel_gap)
        y0 = 65
        parts.append(f'<text x="{x0 + panel_w / 2:.1f}" y="{y0 - 20}" text-anchor="middle" font-size="15" font-family="Arial" font-weight="700">{title}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + panel_h}" x2="{x0 + panel_w}" y2="{y0 + panel_h}" stroke="#444"/>')
        parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0 + panel_h}" stroke="#444"/>')
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            y = y0 + panel_h - tick * panel_h
            parts.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + panel_w}" y2="{y:.1f}" stroke="#ddd"/>')
            parts.append(f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{tick:.2f}</text>')
        group_w = panel_w / len(datasets)
        bar_w = group_w * 0.28
        for i, dataset in enumerate(datasets):
            cx = x0 + i * group_w + group_w / 2
            for j, model in enumerate(["MiniLM", "BGE"]):
                value = metric_value(dataset, model, metric)
                h = value * panel_h
                x = cx + (j - 0.5) * bar_w
                y = y0 + panel_h - h
                parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors[model]}"/>')
            parts.append(f'<text x="{cx:.1f}" y="{y0 + panel_h + 20}" text-anchor="middle" font-size="11" font-family="Arial">{escape(dataset)}</text>')
    parts.append('<rect x="435" y="375" width="14" height="14" fill="#2f6fbb"/><text x="455" y="386" font-size="12" font-family="Arial">MiniLM</text>')
    parts.append('<rect x="530" y="375" width="14" height="14" fill="#d07a2d"/><text x="550" y="386" font-size="12" font-family="Arial">BGE</text>')
    parts.append("</svg>")
    (FIGURES / "model_metric_comparison.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_failure_counts(failure_counts: list[dict]) -> None:
    ensure_dir(FIGURES)
    datasets = []
    for row in failure_counts:
        label = short_dataset(row["dataset"])
        if label not in datasets:
            datasets.append(label)

    width = 960
    height = 430
    margin = 70
    plot_w = width - 2 * margin
    plot_h = 280
    y0 = 60
    max_count = max(int(row["count"]) for row in failure_counts) + 20
    colors = {
        ("MiniLM", "high_similarity_wrong"): "#2f6fbb",
        ("MiniLM", "low_similarity_correct"): "#78aee8",
        ("BGE", "high_similarity_wrong"): "#d07a2d",
        ("BGE", "low_similarity_correct"): "#f0b36f",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="28" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">Failure Cases by Dataset and Model</text>',
        f'<line x1="{margin}" y1="{y0 + plot_h}" x2="{margin + plot_w}" y2="{y0 + plot_h}" stroke="#444"/>',
        f'<line x1="{margin}" y1="{y0}" x2="{margin}" y2="{y0 + plot_h}" stroke="#444"/>',
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        count = frac * max_count
        y = y0 + plot_h - frac * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{margin + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{int(count)}</text>')
    group_w = plot_w / len(datasets)
    bar_w = group_w * 0.25
    for i, dataset in enumerate(datasets):
        cx = margin + i * group_w + group_w / 2
        for j, model in enumerate(["MiniLM", "BGE"]):
            high = sum(int(row["count"]) for row in failure_counts if short_dataset(row["dataset"]) == dataset and row["model"] == model and row["failure_kind"] == "high_similarity_wrong")
            low = sum(int(row["count"]) for row in failure_counts if short_dataset(row["dataset"]) == dataset and row["model"] == model and row["failure_kind"] == "low_similarity_correct")
            x = cx + (j - 0.5) * bar_w
            high_h = high / max_count * plot_h
            low_h = low / max_count * plot_h
            y_high = y0 + plot_h - high_h
            y_low = y_high - low_h
            parts.append(f'<rect x="{x:.1f}" y="{y_high:.1f}" width="{bar_w:.1f}" height="{high_h:.1f}" fill="{colors[(model, "high_similarity_wrong")]}"/>')
            if low > 0:
                parts.append(f'<rect x="{x:.1f}" y="{y_low:.1f}" width="{bar_w:.1f}" height="{low_h:.1f}" fill="{colors[(model, "low_similarity_correct")]}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{y0 + plot_h + 22}" text-anchor="middle" font-size="11" font-family="Arial">{escape(dataset)}</text>')
    legend = [
        ("MiniLM high-sim wrong", "#2f6fbb"),
        ("MiniLM low-sim correct", "#78aee8"),
        ("BGE high-sim wrong", "#d07a2d"),
        ("BGE low-sim correct", "#f0b36f"),
    ]
    for i, (label, color) in enumerate(legend):
        x = 170 + i * 175
        parts.append(f'<rect x="{x}" y="385" width="12" height="12" fill="{color}"/><text x="{x + 18}" y="395" font-size="11" font-family="Arial">{escape(label)}</text>')
    parts.append("</svg>")
    (FIGURES / "failure_case_counts.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_human_categories(human_model: list[dict]) -> None:
    ensure_dir(FIGURES)
    labels = [f"{short_dataset(row['dataset'])}\n{row['model']}" for row in human_model]
    label_art = [float(row["automatic_label_artifact_pct"]) for row in human_model]
    semantic = [float(row["semantic_similarity_limitation_pct"]) for row in human_model]
    low_fn = [float(row["low_similarity_false_negative_pct"]) for row in human_model]
    width = 1160
    height = 470
    margin = 70
    y0 = 60
    plot_w = width - 2 * margin
    plot_h = 300
    colors = {
        "label": "#70a37f",
        "semantic": "#d28c45",
        "low": "#6a8dcc",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="580" y="28" text-anchor="middle" font-size="20" font-family="Arial" font-weight="700">Human Annotation Categories</text>',
        f'<line x1="{margin}" y1="{y0 + plot_h}" x2="{margin + plot_w}" y2="{y0 + plot_h}" stroke="#444"/>',
        f'<line x1="{margin}" y1="{y0}" x2="{margin}" y2="{y0 + plot_h}" stroke="#444"/>',
    ]
    for pct in [0, 25, 50, 75, 100]:
        y = y0 + plot_h - pct / 100 * plot_h
        parts.append(f'<line x1="{margin}" y1="{y:.1f}" x2="{margin + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{margin - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="10" font-family="Arial">{pct}%</text>')
    group_w = plot_w / len(labels)
    bar_w = group_w * 0.55
    for i, label in enumerate(labels):
        x = margin + i * group_w + (group_w - bar_w) / 2
        y_base = y0 + plot_h
        for value, color in [(label_art[i], colors["label"]), (semantic[i], colors["semantic"]), (low_fn[i], colors["low"])]:
            h = value / 100 * plot_h
            y_base -= h
            parts.append(f'<rect x="{x:.1f}" y="{y_base:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        line1, line2 = label.split("\n")
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + plot_h + 18}" text-anchor="middle" font-size="10" font-family="Arial">{escape(line1)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y0 + plot_h + 32}" text-anchor="middle" font-size="10" font-family="Arial">{escape(line2)}</text>')
    legend = [
        ("Automatic label artifact", colors["label"]),
        ("Semantic similarity limitation", colors["semantic"]),
        ("Low-sim false negative", colors["low"]),
    ]
    for i, (label, color) in enumerate(legend):
        x = 275 + i * 215
        parts.append(f'<rect x="{x}" y="430" width="12" height="12" fill="{color}"/><text x="{x + 18}" y="440" font-size="11" font-family="Arial">{escape(label)}</text>')
    parts.append("</svg>")
    (FIGURES / "human_annotation_categories.svg").write_text("\n".join(parts), encoding="utf-8")


def build_reports() -> None:
    metrics = read_csv(SUMMARY / "model_metrics_summary.csv")
    failure_counts = read_csv(SUMMARY / "failure_counts_summary.csv")
    human_model = read_csv(SUMMARY / "human_annotation_model_summary.csv")
    human_type = read_csv(SUMMARY / "human_annotation_type_summary.csv")

    plot_metric_comparison(metrics)
    plot_failure_counts(failure_counts)
    plot_human_categories(human_model)

    metric_rows = make_metric_rows(metrics)
    aggregate_rows = aggregate_failure_counts(failure_counts)
    human_rows = make_human_rows(human_model)
    top_types = top_human_type_rows(human_type)

    english = [
        "# Part 4 Failure Analysis",
        "",
        "Chinese version: [`part4_report.zh.md`](part4_report.zh.md).",
        "",
        "## 1. Purpose and Definition",
        "Part 4 asks us to identify where embedding similarity fails and to propose improvements. We define a failure case as a sample where the similarity-threshold correctness decision disagrees with the automatic `correct_label`.",
        "",
        "- `high_similarity_wrong`: `correct_label = 0`, but similarity is high.",
        "- `low_similarity_correct`: `correct_label = 1`, but similarity is low.",
        "",
        "This distinction is important: these are failures of the similarity-as-classifier pipeline, not always failures of the LLM answer. Some cases reveal defects in the automatic label itself.",
        "",
        "## 2. Data and Outputs Used",
        "We analyze four result folders: `results_nq_5000`, `results_sciq_5000`, `results_truthfulQA_500`, and `results_wiki`. For each folder, we use the evaluation table, failure-case JSONL files, and the sampled human annotations in `manual_annotation_sample.csv`.",
        "",
        "Caveat: older `results_nq` outputs were path-mismatched and looked like short-form science QA. Natural Questions claims should use regenerated `results_nq_5000` outputs.",
        "",
        "## 3. Embedding Model Comparison",
        "![Model metric comparison](figures/model_metric_comparison.svg)",
        "",
        markdown_table(metric_rows, ["Dataset", "Model", "Gap", "Fixed F1", "Best Threshold", "Best F1", "ROC-AUC"]),
        "",
        "MiniLM consistently has a larger correct-vs-incorrect similarity gap, around 0.45-0.46. BGE has comparable ROC-AUC, but its incorrect-answer mean similarity is much higher, so its best threshold is also higher. In practice, BGE is more recall-friendly but less conservative.",
        "",
        "## 4. Full Failure Counts",
        "![Failure case counts](figures/failure_case_counts.svg)",
        "",
        markdown_table(aggregate_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total"]),
        "",
        "BGE produces no low-similarity-correct cases in these outputs, but it produces many more high-similarity-wrong cases. MiniLM has fewer high-similarity false positives, but it misses some correct answers when the correct short answer is embedded in a longer prediction.",
        "",
        "## 5. Human Annotation Analysis",
        "![Human annotation categories](figures/human_annotation_categories.svg)",
        "",
        markdown_table(human_rows, ["Dataset", "Model", "Sampled", "Label Artifacts", "Semantic Limits", "Low-Sim FN", "Top Type"]),
        "",
        "The sampled human annotations split failures into three broad categories:",
        "",
        "- `automatic_label_artifact`: the model answer is likely acceptable, but the automatic correctness label is too strict.",
        "- `semantic_similarity_limitation`: the prediction is semantically related to the reference but is not necessarily correct.",
        "- `low_similarity_false_negative`: the prediction contains or expresses the correct answer, but the embedding score is low.",
        "",
        "## 6. Main Failure Types",
        markdown_table(top_types, ["Dataset", "Model", "Failure Kind", "Human Type", "Count", "%"]),
        "",
        "The common high-similarity-wrong cases include singular/plural variants, numeric equivalence, paraphrases, and underspecified answers. The most important true limitation is that embedding similarity often measures relatedness rather than factual correctness. For example, a broad answer can be semantically close to the reference but still miss a key modifier.",
        "",
        "Low-similarity-correct cases mostly appear for MiniLM. These cases often contain the correct short answer inside a longer phrase or sentence, so the sentence embedding is diluted by surrounding context.",
        "",
        "## 7. Conclusions",
        "- MiniLM is more conservative and separates correct and incorrect answers more clearly.",
        "- BGE gives stronger semantic matches but is more likely to over-score related yet incomplete answers.",
        "- Many apparent errors are actually automatic-label artifacts, not true LLM answer errors.",
        "- A single cosine-similarity threshold is useful as a screening signal, but it is not robust enough as the final evaluator.",
        "",
        "## 8. Improvement Proposal",
        "A stronger evaluator should combine multiple checks:",
        "",
        "1. Stronger normalization: lowercasing, punctuation/hyphen handling, lemmatization, and number-word conversion.",
        "2. Answer containment and answer extraction, especially when predictions are longer than references.",
        "3. Entity or keyword overlap to catch important factual units.",
        "4. Sentence-level similarity for longer answers.",
        "5. NLI or LLM-based verification for ambiguous high-similarity cases.",
        "",
        "A practical hybrid pipeline is: first normalize and check exact/containment matches; then apply embedding similarity with a dataset/model-specific threshold; finally send ambiguous cases to a verifier.",
        "",
        "## 9. Reproducibility",
        "Run the scripts below from the project root:",
        "",
        "```bash",
        "python part4_failure_analysis/scripts/analyze_part4_failures.py",
        "python part4_failure_analysis/scripts/build_final_part4_report.py",
        "```",
        "",
        "Key tables are stored under `part4_failure_analysis/summary_tables/`; dataset-level reports and annotation files are under `part4_failure_analysis/datasets/`.",
        "",
    ]

    chinese = [
        "# Part 4 失败案例分析",
        "",
        "英文版见：[`part4_report.md`](part4_report.md)。",
        "",
        "## 1. 目标与定义",
        "Part 4 要求我们分析 embedding similarity 在哪些情况下失败，并提出改进方案。这里我们把 failure case 定义为：基于 similarity threshold 得到的正确/错误判断，与自动生成的 `correct_label` 不一致。",
        "",
        "- `high_similarity_wrong`：`correct_label = 0`，但 similarity 很高。",
        "- `low_similarity_correct`：`correct_label = 1`，但 similarity 很低。",
        "",
        "需要注意的是，这些 failure case 是 similarity-as-classifier pipeline 的失败，不一定都是 LLM 答错。有些样例反而暴露了自动标签本身的问题。",
        "",
        "## 2. 使用的数据与结果",
        "我们分析了四个结果目录：`results_nq_5000`、`results_sciq_5000`、`results_truthfulQA_500` 和 `results_wiki`。每个目录使用 evaluation table、failure-case JSONL 文件，以及 `manual_annotation_sample.csv` 中的人工标注样本。",
        "",
        "注意：旧的 `results_nq` 输出存在路径混用风险，样例看起来更像 short-form science QA。Natural Questions 结论应基于重新生成的 `results_nq_5000`。",
        "",
        "## 3. Embedding Model 对比",
        "![Model metric comparison](figures/model_metric_comparison.svg)",
        "",
        markdown_table(metric_rows, ["Dataset", "Model", "Gap", "Fixed F1", "Best Threshold", "Best F1", "ROC-AUC"]),
        "",
        "MiniLM 在四个结果目录中都有更大的 correct-vs-incorrect similarity gap，大约为 0.45-0.46。BGE 的 ROC-AUC 与 MiniLM 接近，但它给错误答案的平均 similarity 也更高，因此最佳阈值更高。实际使用时，BGE 更 recall-friendly，但也更不保守。",
        "",
        "## 4. 全量 Failure Case 数量",
        "![Failure case counts](figures/failure_case_counts.svg)",
        "",
        markdown_table(aggregate_rows, ["Model", "High-Sim Wrong", "Low-Sim Correct", "Total"]),
        "",
        "BGE 在这些输出中没有 low-similarity-correct cases，但 high-similarity-wrong 数量明显更多。MiniLM 的 high-similarity false positives 更少，但当正确短答案嵌在更长 prediction 中时，它更容易给出较低 similarity。",
        "",
        "## 5. Human Annotation 分析",
        "![Human annotation categories](figures/human_annotation_categories.svg)",
        "",
        markdown_table(human_rows, ["Dataset", "Model", "Sampled", "Label Artifacts", "Semantic Limits", "Low-Sim FN", "Top Type"]),
        "",
        "人工标注样本把 failure 分成三大类：",
        "",
        "- `automatic_label_artifact`：模型答案可能是可接受的，但自动正确性标签太严格。",
        "- `semantic_similarity_limitation`：prediction 和 reference 语义相关，但不一定事实正确。",
        "- `low_similarity_false_negative`：prediction 包含或表达了正确答案，但 embedding similarity 偏低。",
        "",
        "## 6. 主要 Failure Type",
        markdown_table(top_types, ["Dataset", "Model", "Failure Kind", "Human Type", "Count", "%"]),
        "",
        "常见的 high-similarity-wrong 包括单复数变化、数字等价、同义改写和答案过泛/过细。真正的 similarity 局限在于：embedding similarity 经常衡量的是“语义相关”，而不是“事实正确”。例如，一个更泛的答案可能和标准答案很接近，但缺少关键限定词。",
        "",
        "low-similarity-correct 主要出现在 MiniLM 中。这类样例通常是在较长 prediction 中包含了正确短答案，额外上下文稀释了整体句向量。",
        "",
        "## 7. 结论",
        "- MiniLM 更保守，对正确和错误答案的分离更明显。",
        "- BGE 的语义匹配能力更强，但更容易把相关但不完整的答案打高分。",
        "- 很多表面上的错误其实是自动标签缺陷，而不是 LLM 真正答错。",
        "- 单一 cosine similarity threshold 可以作为筛选信号，但不足以作为最终 correctness evaluator。",
        "",
        "## 8. 改进方案",
        "更稳健的 evaluator 应该结合多种检查：",
        "",
        "1. 更强 normalization：大小写、标点/连字符、词形还原、数字词转换。",
        "2. Answer containment 和 answer extraction，尤其是 prediction 比 reference 更长时。",
        "3. Entity 或 keyword overlap，用来检查关键事实单元。",
        "4. 对长答案使用 sentence-level similarity。",
        "5. 对模糊的 high-similarity cases 使用 NLI 或 LLM judge 做事实一致性验证。",
        "",
        "一个可实现的 hybrid pipeline 是：先做 normalization 和 exact/containment 检查；再使用 dataset/model-specific threshold 的 embedding similarity；最后把 ambiguous cases 交给 verifier。",
        "",
        "## 9. 可复现方式",
        "在项目根目录运行：",
        "",
        "```bash",
        "python part4_failure_analysis/scripts/analyze_part4_failures.py",
        "python part4_failure_analysis/scripts/build_final_part4_report.py",
        "```",
        "",
        "关键统计表保存在 `part4_failure_analysis/summary_tables/`；每个数据集的报告和标注文件保存在 `part4_failure_analysis/datasets/`。",
        "",
    ]

    (ROOT / "part4_report.md").write_text("\n".join(english), encoding="utf-8")
    (ROOT / "part4_report.zh.md").write_text("\n".join(chinese), encoding="utf-8")
    enhance_final_reports()

    for old_name in [
        "part4_complete_analysis.md",
        "part4_complete_analysis.zh.md",
        "summary_report.md",
        "summary_report.zh.md",
        "README.md",
        "README.zh.md",
    ]:
        old_path = ROOT / old_name
        if old_path.exists():
            old_path.unlink()


def enhance_final_reports() -> None:
    english_path = ROOT / "part4_report.md"
    zh_path = ROOT / "part4_report.zh.md"

    english = english_path.read_text(encoding="utf-8")
    if "## 6. Similarity Distance and Concrete Examples" not in english:
        english_examples = markdown_table(
            representative_failure_examples(),
            ["Category", "Dataset", "Model", "Failure Kind", "Reference", "Prediction", "Similarity", "Distance", "Interpretation"],
        )
        english_insert = "\n".join([
            "## 6. Similarity Distance and Concrete Examples",
            "The similarity score is cosine similarity between the prediction embedding and the reference embedding. For interpretation, we also use:",
            "",
            "```text",
            "distance = 1 - cosine_similarity",
            "```",
            "",
            "A small distance means the two answers are close in embedding space. In `high_similarity_wrong` cases, the distance is small even though the automatic label says the prediction is wrong. In `low_similarity_correct` cases, the distance is large even though the automatic label says the prediction is correct.",
            "",
            english_examples,
            "",
            "These examples show that the same distance pattern can mean different things. A small distance can indicate a correct paraphrase that was mislabeled, but it can also indicate that the embedding model confuses semantic relatedness with factual correctness. A large distance can indicate that a short correct answer is surrounded by extra context.",
            "",
        ])
        english = english.replace("## 6. Main Failure Types", english_insert + "## 7. Main Failure Types")
        english = english.replace("## 7. Conclusions", "## 8. Conclusions")
        english = english.replace("## 8. Improvement Proposal", "## 9. Detailed Improvement Proposal")
        english = english.replace("## 9. Reproducibility", "## 10. Reproducibility")

    old_improvement = "\n".join([
        "A stronger evaluator should combine multiple checks:",
        "",
        "1. Stronger normalization: lowercasing, punctuation/hyphen handling, lemmatization, and number-word conversion.",
        "2. Answer containment and answer extraction, especially when predictions are longer than references.",
        "3. Entity or keyword overlap to catch important factual units.",
        "4. Sentence-level similarity for longer answers.",
        "5. NLI or LLM-based verification for ambiguous high-similarity cases.",
        "",
        "A practical hybrid pipeline is: first normalize and check exact/containment matches; then apply embedding similarity with a dataset/model-specific threshold; finally send ambiguous cases to a verifier.",
    ])
    detailed_improvement = "\n".join([
        "The failure analysis suggests that a better evaluator should not replace embedding similarity entirely. Instead, similarity should become one component in a more structured correctness pipeline.",
        "",
        "### 9.1 Normalization and Canonicalization",
        "Before computing labels or similarity thresholds, normalize both prediction and reference. This should include lowercasing, punctuation removal, hyphen normalization, singular/plural lemmatization, number-word conversion, and common abbreviation expansion such as `CO2` to `carbon dioxide`. This directly targets label artifacts such as `ovary` vs. `ovaries`, `four` vs. `4`, and `intra-plate` vs. `intraplate`.",
        "",
        "### 9.2 Answer Extraction for Long Predictions",
        "For short-answer QA, many false negatives happen because the prediction is a full sentence while the reference is a short phrase. Before embedding comparison, extract the likely answer span from the prediction. A simple version can use containment rules and noun-phrase heuristics; a stronger version can use an LLM prompt that rewrites the prediction into the shortest answer phrase. This addresses cases like `Three main types: elliptical, spiral, and irregular.` vs. `three`.",
        "",
        "### 9.3 Hybrid Scoring",
        "Use a score that combines semantic similarity with lexical and factual overlap:",
        "",
        "```text",
        "hybrid_score = 0.55 * embedding_similarity",
        "             + 0.20 * token_f1",
        "             + 0.15 * entity_or_keyword_overlap",
        "             + 0.10 * normalization_bonus",
        "```",
        "",
        "The weights can be tuned on a small validation subset. The goal is to keep the paraphrase sensitivity of embeddings while preventing related but incomplete answers from receiving too much credit.",
        "",
        "### 9.4 Dataset- and Model-Specific Thresholds",
        "The best thresholds are not the same for MiniLM and BGE. MiniLM works best around 0.70-0.76 in these results, while BGE often needs around 0.78-0.81. Therefore, a single global threshold such as 0.75 is not ideal. Thresholds should be selected separately for each dataset and embedding model using validation F1 or a precision-recall trade-off.",
        "",
        "### 9.5 Ambiguity-Aware Verification",
        "Some cases should not be decided by similarity alone. Send uncertain cases to a verifier when the score is near the threshold, when entity overlap is low despite high similarity, or when the prediction is much shorter or more general than the reference. The verifier can be an NLI model or an LLM judge asked whether the prediction entails the reference answer under the question context.",
        "",
        "### 9.6 Expected Effect",
        "Normalization should reduce label artifacts; answer extraction should reduce MiniLM's low-similarity false negatives; entity overlap and verifier checks should reduce BGE's high-similarity wrong cases caused by semantic relatedness without correctness. These modules directly target the three failure categories found in the human annotation analysis.",
    ])
    english = english.replace(old_improvement, detailed_improvement)
    english_path.write_text(english, encoding="utf-8")

    zh = zh_path.read_text(encoding="utf-8")
    if "## 6. 相似度距离说明与具体例子" not in zh:
        zh_examples = markdown_table(
            representative_failure_examples_zh(),
            ["类别", "数据集", "模型", "Failure Kind", "参考答案", "预测答案", "相似度", "距离", "解释"],
        )
        zh_insert = "\n".join([
            "## 6. 相似度距离说明与具体例子",
            "这里的 similarity 是 prediction embedding 和 reference embedding 之间的 cosine similarity。为了更直观地解释 failure case，我们也使用一个简单的距离：",
            "",
            "```text",
            "distance = 1 - cosine_similarity",
            "```",
            "",
            "距离越小，表示两个答案在 embedding space 中越接近。`high_similarity_wrong` 的特点是：距离很小，但自动标签认为 prediction 错；`low_similarity_correct` 的特点是：距离很大，但自动标签认为 prediction 对。",
            "",
            zh_examples,
            "",
            "这些例子说明，同样是“小距离”，可能代表正确改写被自动标签误判，也可能代表 embedding 把“语义相关”误当成“事实正确”。而“大距离”也不一定代表答案错，它可能只是因为正确短答案被放进了更长的句子里。",
            "",
        ])
        zh = zh.replace("## 6. 主要 Failure Type", zh_insert + "## 7. 主要 Failure Type")
        zh = zh.replace("## 7. 结论", "## 8. 结论")
        zh = zh.replace("## 8. 改进方案", "## 9. 详细改进方案")
        zh = zh.replace("## 9. 可复现方式", "## 10. 可复现方式")

    old_zh_improvement = "\n".join([
        "更稳健的 evaluator 应该结合多种检查：",
        "",
        "1. 更强 normalization：大小写、标点/连字符、词形还原、数字词转换。",
        "2. Answer containment 和 answer extraction，尤其是 prediction 比 reference 更长时。",
        "3. Entity 或 keyword overlap，用来检查关键事实单元。",
        "4. 对长答案使用 sentence-level similarity。",
        "5. 对模糊的 high-similarity cases 使用 NLI 或 LLM judge 做事实一致性验证。",
        "",
        "一个可实现的 hybrid pipeline 是：先做 normalization 和 exact/containment 检查；再使用 dataset/model-specific threshold 的 embedding similarity；最后把 ambiguous cases 交给 verifier。",
    ])
    detailed_zh_improvement = "\n".join([
        "failure analysis 表明，改进方案不应该简单地抛弃 embedding similarity，而应该把它作为一个更完整 evaluator 的组成部分。",
        "",
        "### 9.1 Normalization and Canonicalization",
        "在生成 correctness label 或使用 similarity threshold 之前，先对 prediction 和 reference 做更强的规范化，包括：小写化、标点清理、连字符统一、单复数/词形还原、数字词转换，以及常见缩写展开，例如把 `CO2` 映射到 `carbon dioxide`。这可以直接修复 `ovary` vs. `ovaries`、`four` vs. `4`、`intra-plate` vs. `intraplate` 这类自动标签缺陷。",
        "",
        "### 9.2 面向长预测的 Answer Extraction",
        "对于 short-answer QA，很多 low-similarity-correct 是因为 prediction 是完整句子，而 reference 是短语。计算 embedding similarity 之前，应先从 prediction 中抽取最可能的答案片段。简单版本可以用 containment rule 和 noun phrase heuristic；更强版本可以让 LLM 把预测改写成最短答案。这能处理 `Three main types: elliptical, spiral, and irregular.` vs. `three` 这类样例。",
        "",
        "### 9.3 Hybrid Scoring",
        "可以把 embedding similarity 和词面/事实重叠结合起来：",
        "",
        "```text",
        "hybrid_score = 0.55 * embedding_similarity",
        "             + 0.20 * token_f1",
        "             + 0.15 * entity_or_keyword_overlap",
        "             + 0.10 * normalization_bonus",
        "```",
        "",
        "权重可以在小验证集上调参。这样做的目标是：既保留 embedding 对 paraphrase 的识别能力，又避免它给语义相关但不完整的答案过高分。",
        "",
        "### 9.4 Dataset- and Model-Specific Thresholds",
        "MiniLM 和 BGE 的最佳阈值并不相同。当前结果里，MiniLM 的 best threshold 大约在 0.70-0.76，而 BGE 往往需要 0.78-0.81。因此，不建议使用统一的 0.75 threshold。应针对每个 dataset 和 embedding model，用 validation F1 或 precision-recall trade-off 单独选择阈值。",
        "",
        "### 9.5 Ambiguity-Aware Verification",
        "一些样例不适合只靠 similarity 决定。例如：分数接近阈值、similarity 高但 entity overlap 低、prediction 明显比 reference 更短或更泛。这些样例可以交给 verifier，例如 NLI model 或 LLM judge，让它在 question context 下判断 prediction 是否 entail reference answer。",
        "",
        "### 9.6 预期效果",
        "Normalization 主要减少自动标签缺陷；answer extraction 主要减少 MiniLM 的 low-similarity false negatives；entity overlap 和 verifier 主要减少 BGE 因“语义相关但不正确”产生的 high-similarity wrong cases。这三个方向正好对应 human annotation 中发现的三大 failure 类别。",
    ])
    zh = zh.replace(old_zh_improvement, detailed_zh_improvement)
    zh_path.write_text(zh, encoding="utf-8")


def main() -> None:
    build_reports()
    print(f"Final Part 4 reports written to: {ROOT}")


if __name__ == "__main__":
    main()
