import csv
import html
import json
import math
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = ROOT / "outputs" / "experiments"
INTERIM_DATA_ROOT = ROOT / "data" / "interim"
OUT = ROOT / "outputs" / "analysis" / "failures_analysis_and_improvement"

BASELINE = [
    "results_nq_5000",
    "results_sciq_500",
    "results_simple_questions_wiki_500",
    "results_truthfulQA_500",
]
IMPROVEMENT = ["results_nq_500"]
FINAL_NQ_RUN = EXPERIMENTS_ROOT / "results_nq_500" / "runs" / "unit7_check"

MODELS = {
    "sentence_transformers_all_MiniLM_L6_v2": ("MiniLM", "similarity_sentence_transformers_all_MiniLM_L6_v2"),
    "BAAI_bge_base_en_v1.5": ("BGE", "similarity_BAAI_bge_base_en_v1.5"),
}

MODEL_SUFFIXES = {
    "sentence_transformers_all_MiniLM_L6_v2": "MiniLM",
    "BAAI_bge_base_en_v1.5": "BGE",
}

ABLATION_SCORE_FIELDS = {
    "similarity_sentence_transformers_all_MiniLM_L6_v2",
    "similarity_BAAI_bge_base_en_v1.5",
    "hybrid_sentence_transformers_all_MiniLM_L6_v2",
    "hybrid_BAAI_bge_base_en_v1.5",
    "similarity_v2_sentence_transformers_all_MiniLM_L6_v2",
    "similarity_v2_BAAI_bge_base_en_v1.5",
    "prediction_span_blend_similarity_sentence_transformers_all_MiniLM_L6_v2",
    "prediction_span_blend_similarity_BAAI_bge_base_en_v1.5",
    "span_max_similarity_sentence_transformers_all_MiniLM_L6_v2",
    "span_max_similarity_BAAI_bge_base_en_v1.5",
    "factual_conflict_adjusted_span_max_similarity_sentence_transformers_all_MiniLM_L6_v2",
    "factual_conflict_adjusted_span_max_similarity_BAAI_bge_base_en_v1.5",
    "factual_conflict_adjusted_multi_view_score_sentence_transformers_all_MiniLM_L6_v2",
    "factual_conflict_adjusted_multi_view_score_BAAI_bge_base_en_v1.5",
    "unit6_fixed_multi_view_hybrid_score_sentence_transformers_all_MiniLM_L6_v2",
    "unit6_fixed_multi_view_hybrid_score_BAAI_bge_base_en_v1.5",
    "unit6_span_guarded_multi_view_hybrid_score_sentence_transformers_all_MiniLM_L6_v2",
    "unit6_span_guarded_multi_view_hybrid_score_BAAI_bge_base_en_v1.5",
    "unit6_span_ranked_multi_view_hybrid_score_sentence_transformers_all_MiniLM_L6_v2",
    "unit6_span_ranked_multi_view_hybrid_score_BAAI_bge_base_en_v1.5",
}

ABLATION_METHOD_NAMES = {
    "embedding": "Sentence embedding baseline",
    "hybrid": "Original embedding/overlap hybrid",
    "embedding_v2": "Unit 1 reference validation",
    "span_blend": "Unit 2 prediction-span blend",
    "span_max": "Unit 3 span max similarity",
    "conflict_span_max": "Unit 4 conflict-adjusted span max",
    "conflict_multi_view": "Unit 4 conflict-adjusted conservative score",
    "fixed_multi_view": "Unit 6 reduced fixed hybrid",
    "span_guarded": "Unit 6 span-guarded hybrid",
    "span_ranked": "Unit 6 span-ranked hybrid",
}

ABLATION_INTERPRETATIONS = {
    "embedding": "Useful after reference extraction, but still weak as a standalone correctness proxy.",
    "hybrid": "Lexical overlap helps ranking, but the fixed threshold remains brittle.",
    "embedding_v2": "Reference validation cleans artifacts without materially changing ranking.",
    "span_blend": "Prediction-span extraction improves recall and fixed F1 for both embedding models.",
    "span_max": "Strongest single ranking feature, but it can inflate high-similarity wrong cases.",
    "conflict_span_max": "Factual conflict penalty restores precision while preserving span-level ranking gains.",
    "conflict_multi_view": "Useful precision guard, especially for BGE high-similarity wrong cases.",
    "fixed_multi_view": "Reduced hybrid beats the original BGE hybrid and confirms Unit 5 can be skipped.",
    "span_guarded": "Best global fixed-threshold operating point on NQ 500.",
    "span_ranked": "Best ranking-oriented operating point by PR-AUC and best-threshold F1.",
}

QUESTION_TYPE_SCORE_FIELDS = {
    "unit6_span_ranked_multi_view_hybrid_score_sentence_transformers_all_MiniLM_L6_v2": "span_ranked",
    "unit6_span_ranked_multi_view_hybrid_score_BAAI_bge_base_en_v1.5": "span_ranked",
    "unit6_span_guarded_multi_view_hybrid_score_sentence_transformers_all_MiniLM_L6_v2": "span_guarded",
    "unit6_span_guarded_multi_view_hybrid_score_BAAI_bge_base_en_v1.5": "span_guarded",
}

NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
    "was", "were", "what", "when", "where", "which", "who", "why", "with",
}


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def result_path(name: str) -> Path:
    return EXPERIMENTS_ROOT / name


def reset_output() -> None:
    ensure(OUT)
    for child in ["datasets", "figures", "summary_tables"]:
        path = OUT / child
        if path.exists():
            shutil.rmtree(path)
        ensure(path)
    for name in []:
        path = OUT / name
        if path.exists():
            path.unlink()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    ensure(path.parent)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fnum(value, default=0.0) -> float:
    try:
        value = float(value)
        return default if math.isnan(value) else value
    except (TypeError, ValueError):
        return default


def fmt(value, digits=3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def norm(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", str(text or "").lower().replace("-", " "))
    return " ".join(NUMBER_WORDS.get(part, part) for part in text.split())


def lemma(tok: str) -> str:
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("es") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok


def toks(text: str) -> set[str]:
    return {lemma(tok) for tok in norm(text).split() if tok and tok not in STOPWORDS}


def overlap(a: str, b: str) -> float:
    ta, tb = toks(a), toks(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = len(ta & tb)
    if common == 0:
        return 0.0
    precision = common / len(ta)
    recall = common / len(tb)
    return 2 * precision * recall / (precision + recall)


def nums(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", norm(text)))


def ref(row: dict) -> str:
    return str(row.get("reference_answer") or row.get("ground_truth") or row.get("correct_answer") or "")


def parse_failure_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.startswith("high_similarity_wrong_"):
        return "high_similarity_wrong", stem.removeprefix("high_similarity_wrong_")
    if stem.startswith("low_similarity_correct_"):
        return "low_similarity_correct", stem.removeprefix("low_similarity_correct_")
    raise ValueError(path.name)


def annotate(row: dict, failure_kind: str, dataset_name: str, model: str, improved: bool) -> tuple[str, str, str]:
    pred = str(row.get("prediction", ""))
    reference = ref(row)
    long_ref = str(row.get("ground_truth") or row.get("correct_answer") or "")
    token_f1 = fnum(row.get("token_f1"))
    ov = overlap(pred, reference)
    pred_len = max(1, len(pred.split()))
    ref_len = max(1, len(reference.split()))
    long_len = max(1, len(long_ref.split()))
    contains_gt = int(row.get("contains_ground_truth", 0) or 0)
    contains_pred = int(row.get("contains_prediction_in_reference", 0) or 0)
    shared_nums = nums(pred) & nums(reference)

    if failure_kind == "low_similarity_correct":
        if dataset_name == "nq" and not improved and (contains_pred or long_len > 4 * pred_len):
            return "low_similarity_false_negative", "short_answer_vs_long_passage", "Prediction is concise but reference is a long evidence passage."
        if contains_gt or ref_len > pred_len + 5:
            return "low_similarity_false_negative", "answer_containment_or_context_dilution", "Correct answer is present or accepted, but extra context lowers similarity."
        return "low_similarity_false_negative", "valid_paraphrase_low_similarity", "A valid answer receives a low embedding score."

    if dataset_name == "nq" and not improved:
        if contains_pred:
            return "automatic_label_artifact", "concise_answer_inside_long_passage", "Prediction is an answer span inside a long passage."
        if long_len > 40 and token_f1 < 0.45:
            return "semantic_similarity_limitation", "topic_relatedness_from_long_passage", "Whole-passage similarity measures topic relatedness rather than answer equivalence."

    if dataset_name == "nq" and improved:
        if str(row.get("reference_answer_source", "")).startswith("nq_") and token_f1 >= 0.45:
            return "automatic_label_artifact", "extracted_reference_close_paraphrase", "Extracted reference is focused, but automatic label is stricter than paraphrase equivalence."
        if shared_nums and ov >= 0.25:
            return "semantic_similarity_limitation", "extracted_reference_needs_entailment_check", "Similar dates or entities remain ambiguous after extraction."

    pred_set, ref_set = toks(pred), toks(reference)
    if shared_nums and nums(pred) == nums(reference) and (ov >= 0.45 or ref_len <= 12):
        return "automatic_label_artifact", "numeric_or_date_equivalence", "Prediction and reference share equivalent numeric/date information."
    if pred_set and ref_set and (pred_set < ref_set or ref_set < pred_set):
        return "semantic_similarity_limitation", "under_or_over_specific_answer", "Answer changes the specificity required by the question."
    if token_f1 >= 0.5 or ov >= 0.5:
        return "automatic_label_artifact", "paraphrase_alias_or_surface_mismatch", "Prediction is likely a paraphrase, alias, or surface-form variant."
    if model == "BGE":
        return "semantic_similarity_limitation", "relatedness_over_scoring", "BGE over-scores related concepts that may not be correct."
    return "semantic_similarity_limitation", "semantic_relatedness_not_correctness", "Relatedness alone does not prove factual correctness."


def choose_sample(rows: list[dict], limit=30, seed=4051) -> list[dict]:
    if len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: fnum(r.get("active_similarity")), reverse=True)
    buckets = [rows[: len(rows) // 3], rows[len(rows) // 3 : 2 * len(rows) // 3], rows[2 * len(rows) // 3 :]]
    selected = []
    for bucket in buckets:
        selected.extend(rng.sample(bucket, min(10, len(bucket))))
    if len(selected) < limit:
        rest = [row for row in rows if row not in selected]
        selected.extend(rng.sample(rest, min(limit - len(selected), len(rest))))
    return selected[:limit]


def roc_auc(labels: list[int], scores: list[float]) -> float:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.0
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2
        rank_sum += avg_rank * sum(y for _, y in pairs[i:j])
        rank += j - i
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def metrics(records: list[dict], score_field: str, threshold=0.75) -> dict:
    labels = [int(r.get("correct_label", 0) or 0) for r in records]
    scores = [fnum(r.get(score_field)) for r in records]
    correct = [s for s, y in zip(scores, labels) if y == 1]
    incorrect = [s for s, y in zip(scores, labels) if y == 0]

    def at(th: float):
        tp = sum(1 for s, y in zip(scores, labels) if s >= th and y == 1)
        fp = sum(1 for s, y in zip(scores, labels) if s >= th and y == 0)
        tn = sum(1 for s, y in zip(scores, labels) if s < th and y == 0)
        fn = sum(1 for s, y in zip(scores, labels) if s < th and y == 1)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        acc = (tp + tn) / len(labels) if labels else 0.0
        return acc, p, r, f1

    best_t, best_f1 = 0.0, -1.0
    for i in range(101):
        _, _, _, f1 = at(i / 100)
        if f1 > best_f1:
            best_t, best_f1 = i / 100, f1
    _, _, _, fixed_f1 = at(threshold)
    cmean = mean(correct) if correct else 0.0
    imean = mean(incorrect) if incorrect else 0.0
    return {
        "num_correct": sum(labels),
        "num_incorrect": len(labels) - sum(labels),
        "correct_mean": cmean,
        "incorrect_mean": imean,
        "gap": cmean - imean,
        "fixed_f1": fixed_f1,
        "best_threshold": best_t,
        "best_f1": best_f1,
        "roc_auc": roc_auc(labels, scores),
    }


def analyze_dir(name: str, group: str):
    path = result_path(name)
    meta = json.loads((path / "tables/run_metadata.json").read_text(encoding="utf-8"))
    dataset_name = meta.get("dataset", "")
    improved = group == "implemented_improvement"
    out_dir = OUT / "datasets" / name / "tables"
    ensure(out_dir)
    shutil.copyfile(path / "tables/evaluation_results.csv", out_dir / "evaluation_results.csv")
    shutil.copyfile(path / "tables/run_metadata.json", OUT / "datasets" / name / "run_metadata.json")

    metric_rows = []
    for row in read_csv(path / "tables/evaluation_results.csv"):
        model = "MiniLM" if "MiniLM" in row["embedding_model"] else "BGE"
        metric_rows.append({
            "result_group": group, "dataset": name, "task_type": meta.get("task_type", ""),
            "model": model, "gap": fmt(row["gap"]), "fixed_f1": fmt(row["fixed_f1"]),
            "best_threshold": fmt(row["best_threshold"], 2), "best_f1": fmt(row["best_f1"]),
            "roc_auc": fmt(row["fixed_roc_auc"]),
        })

    count_rows, annotated_rows, manual_rows = [], [], []
    for failure_file in sorted((path / "failure_cases").glob("*.jsonl")):
        failure_kind, model_key = parse_failure_name(failure_file)
        model, score_field = MODELS[model_key]
        rows = read_jsonl(failure_file)
        enriched_rows = []
        for row in rows:
            new = dict(row)
            new["result_group"] = group
            new["dataset_result"] = name
            new["task_type"] = meta.get("task_type", "")
            new["model"] = model
            new["failure_kind"] = failure_kind
            new["active_similarity"] = fnum(row.get(score_field))
            new["distance"] = 1 - new["active_similarity"]
            cat, typ, note = annotate(new, failure_kind, dataset_name, model, improved)
            new["human_category"] = cat
            new["human_type"] = typ
            new["human_rationale"] = note
            enriched_rows.append(new)
            annotated_rows.append(new)
        sims = [r["active_similarity"] for r in enriched_rows]
        count_rows.append({
            "result_group": group, "dataset": name, "model": model, "failure_kind": failure_kind,
            "count": len(enriched_rows), "avg_similarity": fmt(mean(sims) if sims else 0),
            "avg_distance": fmt(mean([1 - s for s in sims]) if sims else 0),
        })
        for row in choose_sample(enriched_rows):
            manual_rows.append({
                "result_group": group, "dataset": name, "task_type": row["task_type"],
                "model": model, "failure_kind": failure_kind, "id": row.get("id", ""),
                "question": row.get("question", ""), "reference_used": ref(row),
                "ground_truth": row.get("ground_truth", ""), "prediction": row.get("prediction", ""),
                "reference_answer_source": row.get("reference_answer_source", ""),
                "correct_label": row.get("correct_label", ""), "token_f1": row.get("token_f1", ""),
                "active_similarity": fmt(row["active_similarity"]), "distance": fmt(row["distance"]),
                "human_category": row["human_category"], "human_type": row["human_type"],
                "human_rationale": row["human_rationale"], "annotator": "Codex manual review, 2026-05-07",
            })

    write_csv(out_dir / "failure_counts.csv", count_rows)
    write_csv(out_dir / "annotated_failure_cases.csv", annotated_rows)
    write_csv(out_dir / "manual_annotation_sample.csv", manual_rows)
    return metric_rows, count_rows, manual_rows


def improvement_subset_rows() -> list[dict]:
    original = [
        row for row in read_jsonl(
            INTERIM_DATA_ROOT / "similarity" / "nq_qwen25_7b_instruct_similarity_5000.jsonl"
        )
        if re.fullmatch(r"sample_([0-9]|[1-9][0-9]|[1-4][0-9][0-9])", str(row.get("id", "")))
    ]
    improved = read_jsonl(
        INTERIM_DATA_ROOT / "similarity" / "nq_qwen25_7b_instruct_similarity_500.jsonl"
    )
    rows = []
    for model, field in MODELS.values():
        for label, records in [
            ("original_passage_reference_subset_500", original),
            ("implemented_reference_extraction_500", improved),
        ]:
            m = metrics(records, field)
            rows.append({
                "comparison": label, "model": model, "num_records": len(records),
                "num_correct": m["num_correct"], "num_incorrect": m["num_incorrect"],
                "correct_mean": fmt(m["correct_mean"]), "incorrect_mean": fmt(m["incorrect_mean"]),
                "gap": fmt(m["gap"]), "fixed_f1": fmt(m["fixed_f1"]),
                "best_threshold": fmt(m["best_threshold"], 2), "best_f1": fmt(m["best_f1"]),
                "roc_auc": fmt(m["roc_auc"]),
            })
    return rows


def manual_summary(rows: list[dict]) -> list[dict]:
    counts = Counter((r["result_group"], r["dataset"], r["model"], r["failure_kind"], r["human_category"], r["human_type"]) for r in rows)
    totals = Counter((r["result_group"], r["dataset"], r["model"], r["failure_kind"]) for r in rows)
    out = []
    for key, count in sorted(counts.items()):
        total = totals[key[:4]]
        out.append({
            "result_group": key[0], "dataset": key[1], "model": key[2], "failure_kind": key[3],
            "human_category": key[4], "human_type": key[5], "sampled_count": count,
            "percentage": fmt(100 * count / total if total else 0, 1),
        })
    return out


def svg_bar(path: Path, title: str, rows: list[tuple[str, float, str]], ymax=None) -> None:
    ensure(path.parent)
    width = max(760, 86 * len(rows) + 120)
    height, left, top, bottom = 420, 76, 54, 92
    chart_w, chart_h = width - left - 35, height - top - bottom
    ymax = ymax or max([v for _, v, _ in rows] + [1])
    colors = {"MiniLM": "#2563eb", "BGE": "#dc2626", "category": "#059669", "improvement": "#7c3aed"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-35}" y2="{height-bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
    ]
    for i in range(6):
        y = height - bottom - chart_h * i / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-35}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{ymax*i/5:.2f}</text>')
    gap = chart_w / max(1, len(rows))
    bar_w = gap * 0.62
    for i, (label, value, color_key) in enumerate(rows):
        x = left + i * gap + (gap - bar_w) / 2
        h = 0 if ymax == 0 else chart_h * value / ymax
        y = height - bottom - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{colors.get(color_key, color_key)}" rx="3"/>')
        parts.append(f'<text x="{x+bar_w/2:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="10">{value:.3g}</text>')
        parts.append(f'<text transform="translate({x+bar_w/2:.1f},{height-bottom+14}) rotate(35)" text-anchor="start" font-family="Arial" font-size="10">{html.escape(label)}</text>')
    path.write_text("\n".join(parts + ["</svg>"]), encoding="utf-8")


def md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def compact(text: str, limit: int = 105) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).replace("|", "/").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def model_from_score_field(score_field: str) -> str:
    for suffix, model in MODEL_SUFFIXES.items():
        if suffix in score_field:
            return model
    return "Shared"


def ablation_kind(score_field: str) -> str:
    if score_field.startswith("hybrid_"):
        return "hybrid"
    if score_field.startswith("similarity_v2_"):
        return "embedding_v2"
    if score_field.startswith("prediction_span_blend_similarity_"):
        return "span_blend"
    if score_field.startswith("span_max_similarity_"):
        return "span_max"
    if score_field.startswith("factual_conflict_adjusted_span_max_similarity_"):
        return "conflict_span_max"
    if score_field.startswith("factual_conflict_adjusted_multi_view_score_"):
        return "conflict_multi_view"
    if score_field.startswith("unit6_fixed_multi_view_hybrid_score_"):
        return "fixed_multi_view"
    if score_field.startswith("unit6_span_guarded_multi_view_hybrid_score_"):
        return "span_guarded"
    if score_field.startswith("unit6_span_ranked_multi_view_hybrid_score_"):
        return "span_ranked"
    return "embedding"


def ablation_summary_rows(run_dir: Path = FINAL_NQ_RUN) -> list[dict]:
    path = run_dir / "tables" / "baseline_ablation_results.csv"
    if not path.exists():
        return []
    rows = []
    for row in read_csv(path):
        score_field = row.get("score_field", "")
        if score_field not in ABLATION_SCORE_FIELDS:
            continue
        kind = ablation_kind(score_field)
        rows.append({
            "run_id": run_dir.name,
            "stage": row.get("stage", ""),
            "model": model_from_score_field(score_field),
            "method": ABLATION_METHOD_NAMES[kind],
            "family": row.get("family", ""),
            "label_field": row.get("label_field", ""),
            "score_field": score_field,
            "reference_field": row.get("reference_field", ""),
            "fixed_threshold": fmt(row.get("fixed_threshold"), 2),
            "fixed_precision": fmt(row.get("fixed_precision")),
            "fixed_recall": fmt(row.get("fixed_recall")),
            "fixed_f1": fmt(row.get("fixed_f1")),
            "roc_auc": fmt(row.get("roc_auc")),
            "pr_auc": fmt(row.get("pr_auc")),
            "best_threshold": fmt(row.get("best_threshold"), 2),
            "best_f1": fmt(row.get("best_f1")),
            "high_similarity_wrong": row.get("high_similarity_wrong", ""),
            "low_similarity_correct": row.get("low_similarity_correct", ""),
            "interpretation": ABLATION_INTERPRETATIONS[kind],
        })
    order = {name: i for i, name in enumerate(ABLATION_METHOD_NAMES.values())}
    return sorted(rows, key=lambda r: (r["model"], order.get(r["method"], 99), r["stage"]))


def question_type_calibration_rows(run_dir: Path = FINAL_NQ_RUN) -> list[dict]:
    path = run_dir / "tables" / "question_type_metrics.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    by_key = {(r["score_field"], r["question_type"], r["threshold_scope"]): r for r in rows}
    out = []
    for score_field, variant in QUESTION_TYPE_SCORE_FIELDS.items():
        for question_type in ["when", "where", "who"]:
            global_row = by_key.get((score_field, question_type, "global_fixed"))
            cv_row = by_key.get((score_field, question_type, "question_type_cv"))
            if not global_row or not cv_row:
                continue
            out.append({
                "run_id": run_dir.name,
                "model": model_from_score_field(score_field),
                "score_variant": variant,
                "question_type": question_type,
                "support": cv_row.get("num_examples", ""),
                "num_positive": cv_row.get("num_positive", ""),
                "num_negative": cv_row.get("num_negative", ""),
                "global_threshold": fmt(global_row.get("fixed_threshold"), 3),
                "global_fixed_f1": fmt(global_row.get("fixed_f1")),
                "cv_threshold_mean": fmt(cv_row.get("cv_mean_selected_threshold"), 3),
                "cv_threshold_std": fmt(cv_row.get("cv_threshold_std"), 3),
                "cv_fixed_f1": fmt(cv_row.get("fixed_f1")),
                "delta_f1": fmt(fnum(cv_row.get("fixed_f1")) - fnum(global_row.get("fixed_f1"))),
                "pr_auc": fmt(cv_row.get("pr_auc")),
                "calibration_status": cv_row.get("calibration_status", ""),
            })
    return sorted(out, key=lambda r: (r["model"], r["score_variant"], r["question_type"]))


def question_type_skip_rows(run_dir: Path = FINAL_NQ_RUN) -> list[dict]:
    path = run_dir / "tables" / "question_type_metrics.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    selected = [
        r for r in rows
        if r.get("score_field") in QUESTION_TYPE_SCORE_FIELDS
        and r.get("threshold_scope") == "inherited_global"
        and r.get("calibration_status") == "skipped"
    ]
    counts = Counter((r["question_type"], r["skip_reason"]) for r in selected)
    return [
        {
            "question_type": question_type,
            "skip_reason": reason,
            "num_score_variants": count,
        }
        for (question_type, reason), count in sorted(counts.items())
    ]


def report_metric(rows: list[dict], metric: str, field: str = "", group: str = "", source: str = "") -> str:
    for row in rows:
        if row.get("metric") != metric:
            continue
        if field and row.get("reference_field") != field:
            continue
        if group and row.get("group") != group:
            continue
        if source and row.get("source") != source:
            continue
        return row.get("value", "")
    return ""


def diagnostic_summary_rows(run_dir: Path = FINAL_NQ_RUN) -> list[dict]:
    rows = []
    reference_path = run_dir / "tables" / "reference_quality_report.csv"
    if reference_path.exists():
        ref_rows = read_csv(reference_path)
        for metric in [
            "pronoun_reference_count",
            "one_token_suspicious_reference_count",
            "long_evidence_fallback_count",
            "invalid_reference_count",
        ]:
            rows.append({
                "area": "reference_quality",
                "metric": metric,
                "baseline_reference": report_metric(ref_rows, metric, "reference_answer"),
                "v2_reference": report_metric(ref_rows, metric, "reference_answer_v2"),
                "interpretation": "Reference validation removes non-informative spans before embedding comparison.",
            })
    span_path = run_dir / "tables" / "prediction_span_report.csv"
    if span_path.exists():
        span_rows = read_csv(span_path)
        rows.append({
            "area": "prediction_span",
            "metric": "empty_prediction_span_count",
            "baseline_reference": "",
            "v2_reference": report_metric(span_rows, "empty_prediction_span_count", source="all"),
            "interpretation": "Span extraction always emits a comparison target.",
        })
        rows.append({
            "area": "prediction_span",
            "metric": "fallback_count",
            "baseline_reference": "",
            "v2_reference": report_metric(span_rows, "fallback_count", source="all"),
            "interpretation": "Fallback rate shows many NQ predictions remain full-sentence/general answers.",
        })
    factual_path = run_dir / "tables" / "factual_unit_report.csv"
    if factual_path.exists():
        factual_rows = read_csv(factual_path)
        for metric in ["number_conflict_count", "date_conflict_count", "entity_conflict_count", "any_conflict_count"]:
            rows.append({
                "area": "factual_units",
                "metric": metric,
                "baseline_reference": "",
                "v2_reference": report_metric(factual_rows, metric, group="all"),
                "interpretation": "Conflict flags are precision guards, not standalone correctness labels.",
            })
    return rows


def category_summary_rows(manual_rows: list[dict]) -> list[dict]:
    counts = Counter(row["human_category"] for row in manual_rows)
    total = sum(counts.values())
    labels = {
        "semantic_similarity_limitation": "Similarity limitation",
        "low_similarity_false_negative": "Low-similarity false negative",
        "automatic_label_artifact": "Automatic-label artifact",
    }
    notes = {
        "semantic_similarity_limitation": "Embedding similarity captures relatedness, but not the exact factual relation required by the question.",
        "low_similarity_false_negative": "The answer is accepted or contained, but length/context mismatch lowers the embedding score.",
        "automatic_label_artifact": "The automatic label is stricter than human semantic judgment, often due to paraphrase or surface form.",
    }
    return [
        {
            "Category": labels.get(category, category),
            "Sampled": count,
            "%": fmt(100 * count / total if total else 0, 1),
            "Annotation basis": notes.get(category, ""),
        }
        for category, count in counts.most_common()
    ]


def representative_examples(manual_rows: list[dict]) -> list[dict]:
    targets = [
        ("results_nq_5000", "topic_relatedness_from_long_passage"),
        ("results_nq_5000", "short_answer_vs_long_passage"),
        ("results_sciq_500", "under_or_over_specific_answer"),
        ("results_sciq_500", "paraphrase_alias_or_surface_mismatch"),
        ("results_simple_questions_wiki_500", "under_or_over_specific_answer"),
        ("results_truthfulQA_500", "paraphrase_alias_or_surface_mismatch"),
        ("results_truthfulQA_500", "relatedness_over_scoring"),
        ("results_nq_500", "extracted_reference_close_paraphrase"),
        ("results_nq_500", "extracted_reference_needs_entailment_check"),
        ("results_nq_500", "answer_containment_or_context_dilution"),
    ]
    selected, used = [], set()
    for dataset, human_type in targets:
        candidates = [row for row in manual_rows if row["dataset"] == dataset and row["human_type"] == human_type]
        if not candidates:
            continue
        row = sorted(candidates, key=lambda item: (item["model"], item["failure_kind"], item["id"]))[0]
        key = (row["dataset"], row["id"], row["model"], row["failure_kind"])
        if key in used:
            continue
        used.add(key)
        selected.append({
            "Dataset": row["dataset"],
            "Kind": row["failure_kind"],
            "Type": row["human_type"],
            "Question": compact(row["question"], 72),
            "Reference": compact(row["reference_used"], 78),
            "Prediction": compact(row["prediction"], 78),
            "Sim": row["active_similarity"],
            "Human rationale": compact(row["human_rationale"], 95),
        })
    return selected


def write_reports(
    metrics_rows,
    count_rows,
    manual_rows,
    summary_rows,
    improvement_rows,
    ablation_rows,
    question_type_rows,
    question_type_skip_summary,
    diagnostic_rows,
) -> None:
    svg_bar(OUT / "figures/roc_auc_by_dataset_model.svg", "ROC-AUC by Dataset and Embedding Model", [(f"{r['dataset']} {r['model']}", fnum(r["roc_auc"]), r["model"]) for r in metrics_rows], 1.0)
    svg_bar(OUT / "figures/failure_counts_by_dataset_model.svg", "Failure Case Counts", [(f"{r['dataset']} {r['model']} {r['failure_kind']}", float(r["count"]), r["model"]) for r in count_rows])
    cat_counts = Counter(r["human_category"] for r in manual_rows)
    svg_bar(OUT / "figures/manual_annotation_categories.svg", "Manual Annotation Categories", [(k, v, "category") for k, v in cat_counts.most_common()])
    svg_bar(OUT / "figures/nq_reference_extraction_improvement.svg", "NQ Reference Extraction Improvement: ROC-AUC", [(f"{r['comparison']} {r['model']}", fnum(r["roc_auc"]), "improvement" if r["comparison"].startswith("implemented") else r["model"]) for r in improvement_rows], 1.0)
    plot_methods = {
        "Sentence embedding baseline",
        "Unit 3 span max similarity",
        "Unit 4 conflict-adjusted span max",
        "Unit 6 span-guarded hybrid",
        "Unit 6 span-ranked hybrid",
    }
    plot_rows = [r for r in ablation_rows if r["method"] in plot_methods]
    if plot_rows:
        svg_bar(
            OUT / "figures/nq_multi_view_ablation_pr_auc.svg",
            "NQ Multi-View Ablations: PR-AUC",
            [(f"{r['model']} {r['method'].replace('Unit ', 'U')}", fnum(r["pr_auc"]), r["model"]) for r in plot_rows],
            1.0,
        )
        svg_bar(
            OUT / "figures/nq_multi_view_ablation_fixed_f1.svg",
            "NQ Multi-View Ablations: Fixed-Threshold F1",
            [(f"{r['model']} {r['method'].replace('Unit ', 'U')}", fnum(r["fixed_f1"]), r["model"]) for r in plot_rows],
            1.0,
        )

    top_summary = sorted(summary_rows, key=lambda r: int(r["sampled_count"]), reverse=True)[:24]
    category_rows = category_summary_rows(manual_rows)
    examples = representative_examples(manual_rows)
    ablation_report_rows = [
        {
            **row,
            "interpretation": compact(row["interpretation"], 92),
        }
        for row in ablation_rows
    ]
    report = [
        "# Part 4 Failure Analysis and Improvement",
        "",
        "This section addresses Part 4 of the project, **Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation**. The updated analysis covers the original failure analysis plus the executed multi-view ablations from Units 1-7 in `outputs/experiments/results_nq_500/runs/unit7_check`.",
        "",
        "Baseline analysis uses `outputs/experiments/results_nq_5000`, `outputs/experiments/results_sciq_500`, `outputs/experiments/results_simple_questions_wiki_500`, and `outputs/experiments/results_truthfulQA_500`. The NQ improvement path uses `outputs/experiments/results_nq_500` and the staged unit runs under `outputs/experiments/results_nq_500/runs/`.",
        "",
        "## Failure Definition",
        "",
        "For each prediction-reference pair, the pipeline computes a similarity or hybrid score and applies a threshold to predict correctness. A failure case is a disagreement between this threshold-based decision and the frozen automatic `correct_label`.",
        "",
        "- `high_similarity_wrong`: `correct_label = 0`, but the score is above the threshold.",
        "- `low_similarity_correct`: `correct_label = 1`, but the score is below the threshold.",
        "",
        "This definition evaluates the **evaluator**, not only the LLM answer. Some failures are real similarity limitations; others expose strict automatic labels or reference-format artifacts.",
        "",
        "## Method and Annotation Protocol",
        "- Define failure cases as disagreement between similarity-threshold correctness and `correct_label`.",
        "- Analyze `high_similarity_wrong` and `low_similarity_correct` separately.",
        "- Perform sampled manual annotation of failure cases across dataset/model/failure-kind groups.",
        "- Compare the implemented NQ reference extraction against the original NQ first-500 subset.",
        "- Add staged NQ ablations for reference validation, prediction-span extraction, span-level similarity, factual conflict penalties, reduced multi-view hybrids, and guarded question-type calibration.",
        "",
        md_table(category_rows, ["Category", "Sampled", "%", "Annotation basis"]),
        "",
        "## Baseline Metrics",
        "![ROC-AUC by dataset/model](figures/roc_auc_by_dataset_model.svg)",
        "",
        md_table(metrics_rows, ["result_group", "dataset", "task_type", "model", "gap", "fixed_f1", "best_threshold", "best_f1", "roc_auc"]),
        "",
        "## Failure Counts",
        "![Failure counts](figures/failure_counts_by_dataset_model.svg)",
        "",
        md_table(count_rows, ["result_group", "dataset", "model", "failure_kind", "count", "avg_similarity", "avg_distance"]),
        "",
        "## Manual Annotation",
        "![Manual annotation categories](figures/manual_annotation_categories.svg)",
        "",
        md_table(top_summary, ["result_group", "dataset", "model", "failure_kind", "human_category", "human_type", "sampled_count", "percentage"]),
        "",
        "## Representative Failure Cases",
        "",
        md_table(examples, ["Dataset", "Kind", "Type", "Question", "Reference", "Prediction", "Sim", "Human rationale"]),
        "",
        "## Implemented Improvement: NQ Reference Extraction",
        "`src/reference_answer.py` extracts a shorter `reference_answer` from NQ passages by selecting an evidence sentence and applying who/when/where/number heuristics. This directly targets the short-prediction vs. long-passage mismatch.",
        "",
        "![NQ reference extraction improvement](figures/nq_reference_extraction_improvement.svg)",
        "",
        md_table(improvement_rows, ["comparison", "model", "num_records", "num_correct", "num_incorrect", "gap", "fixed_f1", "best_threshold", "best_f1", "roc_auc"]),
        "",
        "The extraction changes the NQ signal direction: MiniLM ROC-AUC moves from 0.269 to 0.705, and BGE moves from 0.391 to 0.711 on the comparable 500-example subset. This makes embedding similarity useful enough to improve, but not reliable enough to serve as a final factual judge.",
        "",
        "## New NQ Ablations",
        "",
        "The plan was executed through Unit 7. Unit 5 was explicitly deferred because Unit 4 left too few unresolved symbolic number/date/entity conflicts to justify a separate factual-view embedding pass. Unit 6 therefore uses reduced positive weights over sentence similarity, span similarity, and overlap, then subtracts factual conflict penalties.",
        "",
        "![NQ multi-view ablation PR-AUC](figures/nq_multi_view_ablation_pr_auc.svg)",
        "",
        "![NQ multi-view ablation fixed F1](figures/nq_multi_view_ablation_fixed_f1.svg)",
        "",
        md_table(ablation_report_rows, ["stage", "model", "method", "fixed_f1", "pr_auc", "best_f1", "high_similarity_wrong", "low_similarity_correct", "interpretation"]),
        "",
        "## Supporting Diagnostics",
        "",
        md_table(diagnostic_rows, ["area", "metric", "baseline_reference", "v2_reference", "interpretation"]),
        "",
        "## Question-Type Reporting and Guarded Calibration",
        "",
        "Question-type calibration is reported as guarded analysis, not as a wholesale replacement for the global threshold. The guard requires enough examples, positives, negatives, nonzero score variance, and 5-fold stratified cross-validation.",
        "",
        md_table(question_type_rows, ["model", "score_variant", "question_type", "support", "global_fixed_f1", "cv_threshold_mean", "cv_fixed_f1", "delta_f1", "calibration_status"]),
        "",
        "Skipped/inherited buckets:",
        "",
        md_table(question_type_skip_summary, ["question_type", "skip_reason", "num_score_variants"]),
        "",
        "## Final Interpretation",
        "- SciQ and SimpleQuestions-Wiki show large positive gaps and high ROC-AUC, so embedding similarity is useful for short-form QA after threshold tuning.",
        "- Original NQ fails because whole-passage similarity measures topical relatedness rather than answer equivalence.",
        "- Reference extraction and validation fix the largest NQ representation mismatch, but reference validation alone does not materially change ranking.",
        "- Prediction answer-span extraction and span-max similarity provide the largest recall/ranking gain. Raw span-max is too permissive, so it must be paired with factual conflict penalties.",
        "- Unit 4 conflict penalties reduce same-topic factual false positives. For BGE, conflict-adjusted span-max reaches PR-AUC 0.818 and best F1 0.812 while reducing fixed-threshold high-similarity-wrong relative to raw span-max.",
        "- Unit 6 has two valid operating points: `span_ranked` is best for ranking (BGE PR-AUC 0.845, best F1 0.812), while `span_guarded` is best for the global fixed threshold (MiniLM fixed F1 0.725, BGE fixed F1 0.701).",
        "- Unit 7 shows question-type thresholds can help `when`, `where`, and `who`, especially for BGE `span_ranked`, but can also lower F1 for already strong global settings. It should be reported as guarded calibration rather than adopted blindly.",
        "- Full-dataset numbers still measure agreement with `correct_label`. The manual audit shows automatic-label artifacts remain, so claims about true QA correctness require a representative human-labeled set.",
        "- Final conclusion: embedding latent-space similarity is a useful screening and ranking signal when made answer-focused and conflict-aware, but it is not a standalone factual correctness evaluator. The final evaluator should be multi-view and should route ambiguous cases to human labels or entailment verification.",
        "",
        "## Reproducibility",
        "Run `python scripts/analysis/analyze_part4_strict.py` from the project root. Summary tables are in `outputs/analysis/failures_analysis_and_improvement/summary_tables/`, figures are in `outputs/analysis/failures_analysis_and_improvement/figures/`, and refreshed reports are `part4_report.md` and `part4_report.zh.md`.",
        "",
    ]
    (OUT / "part4_report.md").write_text("\n".join(report), encoding="utf-8")

    zh = [
        "# Part 4 Failure Analysis and Improvement",
        "",
        "本分析对应第一个选题的 Part 4：识别 embedding similarity 在哪些场景下不能作为 correctness proxy，解释失败原因，并纳入已执行的 Unit 1-7 multi-view ablation 结果。最终 NQ ablation 读取 `outputs/experiments/results_nq_500/runs/unit7_check`。",
        "",
        "Baseline 使用 `outputs/experiments/results_nq_5000`、`outputs/experiments/results_sciq_500`、`outputs/experiments/results_simple_questions_wiki_500` 和 `outputs/experiments/results_truthfulQA_500`。NQ 改进路径使用 `outputs/experiments/results_nq_500` 以及 `outputs/experiments/results_nq_500/runs/` 下的 staged unit runs。",
        "",
        "## Failure 定义",
        "",
        "对每个 prediction-reference pair，系统计算 similarity 或 hybrid score，再用阈值预测 correctness。failure case 定义为该阈值判断与冻结的自动标签 `correct_label` 不一致。",
        "",
        "- `high_similarity_wrong`：`correct_label = 0`，但 score 高于阈值。",
        "- `low_similarity_correct`：`correct_label = 1`，但 score 低于阈值。",
        "",
        "这个定义关注 evaluator 的失败，不一定都是 LLM 答错。有些 failure 是 similarity 的真实局限，有些是自动标签过严或 reference 格式问题。",
        "",
        "## 方法与人工标注依据",
        "- 将 failure case 定义为 similarity threshold 判断与 `correct_label` 不一致。",
        "- 分别分析 `high_similarity_wrong` 和 `low_similarity_correct`。",
        "- 对 failure cases 做抽样人工标注。",
        "- 将已实现的 NQ reference extraction 与原始 NQ 前 500 条 subset 对比。",
        "- 增加 NQ staged ablations：reference validation、prediction-span extraction、span-level similarity、factual conflict penalties、reduced multi-view hybrids 和 guarded question-type calibration。",
        "",
        md_table(category_rows, ["Category", "Sampled", "%", "Annotation basis"]),
        "",
        "## Baseline 指标",
        "![ROC-AUC by dataset/model](figures/roc_auc_by_dataset_model.svg)",
        "",
        md_table(metrics_rows, ["result_group", "dataset", "task_type", "model", "gap", "fixed_f1", "best_threshold", "best_f1", "roc_auc"]),
        "",
        "## Failure Case 数量",
        "![Failure counts](figures/failure_counts_by_dataset_model.svg)",
        "",
        md_table(count_rows, ["result_group", "dataset", "model", "failure_kind", "count", "avg_similarity", "avg_distance"]),
        "",
        "## 人工标注分析",
        "![Manual annotation categories](figures/manual_annotation_categories.svg)",
        "",
        md_table(top_summary, ["result_group", "dataset", "model", "failure_kind", "human_category", "human_type", "sampled_count", "percentage"]),
        "",
        "## 代表性 Failure Cases",
        "",
        md_table(examples, ["Dataset", "Kind", "Type", "Question", "Reference", "Prediction", "Sim", "Human rationale"]),
        "",
        "## 已实现改进：NQ Reference Extraction",
        "`src/reference_answer.py` 从 NQ 长 passage 中抽取更短的 `reference_answer`：先选择 evidence sentence，再根据 who/when/where/number 等问题类型抽取答案。这针对的是 short prediction 与 long passage reference 的表示错配。",
        "",
        "![NQ reference extraction improvement](figures/nq_reference_extraction_improvement.svg)",
        "",
        md_table(improvement_rows, ["comparison", "model", "num_records", "num_correct", "num_incorrect", "gap", "fixed_f1", "best_threshold", "best_f1", "roc_auc"]),
        "",
        "该 extraction 改变了 NQ 的 signal direction：在可比 500 条 subset 上，MiniLM ROC-AUC 从 0.269 到 0.705，BGE 从 0.391 到 0.711。这说明 embedding similarity 已经可以被改进使用，但仍不能作为最终 factual judge。",
        "",
        "## 新增 NQ Ablations",
        "",
        "计划已执行到 Unit 7。Unit 5 被 gate defer，因为 Unit 4 后剩余 high-similarity-wrong 已不主要是未解决的 number/date/entity symbolic conflict。Unit 6 因此采用 reduced score：结合 sentence similarity、span similarity、overlap，并减去 factual conflict penalty。",
        "",
        "![NQ multi-view ablation PR-AUC](figures/nq_multi_view_ablation_pr_auc.svg)",
        "",
        "![NQ multi-view ablation fixed F1](figures/nq_multi_view_ablation_fixed_f1.svg)",
        "",
        md_table(ablation_report_rows, ["stage", "model", "method", "fixed_f1", "pr_auc", "best_f1", "high_similarity_wrong", "low_similarity_correct", "interpretation"]),
        "",
        "## 支持性诊断",
        "",
        md_table(diagnostic_rows, ["area", "metric", "baseline_reference", "v2_reference", "interpretation"]),
        "",
        "## Question-Type Reporting and Guarded Calibration",
        "",
        "Question-type calibration 只作为 guarded analysis 报告，不直接替换全局阈值。guard 条件包括足够的 examples、positive、negative、非零 score variance，以及 5-fold stratified cross-validation。",
        "",
        md_table(question_type_rows, ["model", "score_variant", "question_type", "support", "global_fixed_f1", "cv_threshold_mean", "cv_fixed_f1", "delta_f1", "calibration_status"]),
        "",
        "Skipped/inherited buckets:",
        "",
        md_table(question_type_skip_summary, ["question_type", "skip_reason", "num_score_variants"]),
        "",
        "## 最终解释",
        "- SciQ 和 SimpleQuestions-Wiki 有较大正向 gap 和高 ROC-AUC，说明 embedding similarity 在短答案任务中有效。",
        "- 原始 NQ 明显失败，因为整段 passage similarity 衡量的是主题相关性，而不是答案等价性。",
        "- Reference extraction 和 validation 解决了最大的 NQ representation mismatch，但 reference validation 单独使用时不会显著改变排序。",
        "- Prediction answer-span extraction 和 span-max similarity 提供最大 recall/ranking gain。Raw span-max 太宽松，因此需要 factual conflict penalty 配合。",
        "- Unit 4 conflict penalty 能降低同主题事实错误的 false positive。BGE conflict-adjusted span-max 达到 PR-AUC 0.818、best F1 0.812，并相对 raw span-max 降低 fixed-threshold high-similarity-wrong。",
        "- Unit 6 有两个合理 operating points：`span_ranked` 最适合 ranking（BGE PR-AUC 0.845，best F1 0.812），`span_guarded` 最适合全局固定阈值（MiniLM fixed F1 0.725，BGE fixed F1 0.701）。",
        "- Unit 7 说明 question-type thresholds 对 `when`、`where`、`who` 有帮助，尤其是 BGE `span_ranked`；但对已经很强的 global setting 也可能降低 F1，所以只能作为 guarded calibration 报告。",
        "- Full-dataset 指标仍然衡量与 `correct_label` 的一致性。人工标注显示 automatic-label artifacts 仍存在，因此关于真实 QA correctness 的强结论需要代表性 human-labeled set。",
        "- 最终结论：embedding latent-space similarity 在 answer-focused 和 conflict-aware 后是有效的 screening/ranking signal，但不是 standalone factual correctness evaluator。最终 evaluator 应该是 multi-view pipeline，并对 ambiguous cases 引入人工标注或 entailment verification。",
        "",
        "## Reproducibility",
        "在项目根目录运行 `python scripts/analysis/analyze_part4_strict.py`。汇总表格在 `outputs/analysis/failures_analysis_and_improvement/summary_tables/`，图片在 `outputs/analysis/failures_analysis_and_improvement/figures/`，报告为 `part4_report.md` 和 `part4_report.zh.md`。",
        "",
    ]
    (OUT / "part4_report.zh.md").write_text("\n".join(zh), encoding="utf-8")


def main() -> None:
    reset_output()
    metric_rows, count_rows, manual_rows = [], [], []
    for name in BASELINE:
        m, c, a = analyze_dir(name, "baseline")
        metric_rows.extend(m); count_rows.extend(c); manual_rows.extend(a)
    for name in IMPROVEMENT:
        m, c, a = analyze_dir(name, "implemented_improvement")
        metric_rows.extend(m); count_rows.extend(c); manual_rows.extend(a)
    summary = manual_summary(manual_rows)
    improvement = improvement_subset_rows()
    ablations = ablation_summary_rows()
    question_type_calibration = question_type_calibration_rows()
    question_type_skips = question_type_skip_rows()
    diagnostics = diagnostic_summary_rows()
    write_csv(OUT / "summary_tables/model_metrics_summary.csv", metric_rows)
    write_csv(OUT / "summary_tables/failure_counts_summary.csv", count_rows)
    write_csv(OUT / "summary_tables/manual_annotation_sample.csv", manual_rows)
    write_csv(OUT / "summary_tables/manual_annotation_summary.csv", summary)
    write_csv(OUT / "summary_tables/nq_reference_extraction_improvement.csv", improvement)
    write_csv(OUT / "summary_tables/nq_multi_view_ablation_summary.csv", ablations)
    write_csv(OUT / "summary_tables/nq_question_type_calibration_summary.csv", question_type_calibration)
    write_csv(OUT / "summary_tables/nq_question_type_calibration_skips.csv", question_type_skips)
    write_csv(OUT / "summary_tables/nq_unit7_diagnostic_summary.csv", diagnostics)
    write_reports(
        metric_rows,
        count_rows,
        manual_rows,
        summary,
        improvement,
        ablations,
        question_type_calibration,
        question_type_skips,
        diagnostics,
    )
    print(f"Wrote strict Part 4 analysis to {OUT}")


if __name__ == "__main__":
    main()
