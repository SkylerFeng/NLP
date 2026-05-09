# semantic-similarity-llm-eval Onboarding Guide

## What Is This?

This project studies whether embedding-space semantic similarity can evaluate
LLM question-answering predictions. It compares a model prediction with a
reference answer, computes cosine similarity between their embeddings, and
tests whether that score can separate correct answers from incorrect ones.

The course project is the first topic in
`docs/AIAA 4051 Final Research Projects (2).pdf`: `Semantic Similarity
Measurement in Latent Space for LLM Prediction Evaluation`. The repo covers
short-form QA (`sciq`, `simple_questions_wiki`) and long-form QA (`nq`,
`truthfulQA`), then analyzes where similarity works, where it fails, and how
to improve the evaluator.

The current failure-analysis work lives in
`failures_analysis_and_improvement/`. Start there when you are working on
Part 4 of the report.

---

## Developer Experience

You use this project as a research pipeline, not as an application server.
The main interface is `config.yaml` plus four scripts under `scripts/`.

```bash
conda activate nlp-semsim
python scripts/01_generate_predictions.py
python scripts/02_compute_similarity.py
python scripts/03_evaluate.py
python scripts/04_visualize_results.py
```

For existing result analysis, you usually do not need to regenerate LLM
predictions. The repo already contains `data/predictions/`,
`data/similarity/`, `results_*`, and the Part 4 analysis artifacts.

---

## How Is It Organized?

The project is a single Python experiment repo. It has no web server,
database, or background worker.

```text
semantic-similarity-llm-eval/
  config.yaml                         # Dataset, model, and path config
  processed_data/                     # Teacher-provided QA JSONL data
  data/
    predictions/                      # LLM prediction JSONL files
    similarity/                       # Prediction records with scores
  results_*/                          # Tables, figures, failure cases
  failures_analysis_and_improvement/  # Part 4 analysis outputs
  scripts/                            # Pipeline entry points
  src/                                # Reusable pipeline modules
  tests/                              # Unit tests
```

The core experiment flow is:

```text
processed_data/{dataset}/merged_fb.json
  |
  |  JSONL records with question + correct_answer
  v
scripts/01_generate_predictions.py
  |
  |  adds prompt + prediction
  v
data/predictions/*_predictions_*.jsonl
  |
  |  adds reference_answer, labels, embedding similarities
  v
data/similarity/*_similarity_*.jsonl
  |
  |  evaluates similarity as a correctness classifier
  v
results_*/tables + results_*/failure_cases + results_*/figures
  |
  |  aggregates failure categories and improvement evidence
  v
failures_analysis_and_improvement/
```

| Module | Responsibility |
|--------|----------------|
| `src/data_loader.py` | Loads processed QA data and builds prompts. |
| `src/generate_predictions.py` | Calls Qwen/OpenAI-compatible LLM APIs. |
| `src/reference_answer.py` | Extracts short NQ references from long passages. |
| `src/correctness_labeling.py` | Builds automatic `correct_label` values. |
| `src/compute_embeddings.py` | Wraps `sentence-transformers` embedding models. |
| `src/compute_similarity.py` | Computes cosine similarity fields. |
| `src/evaluate.py` | Computes classifier metrics and failure cases. |
| `src/entity_overlap.py` | Computes token/entity overlap and hybrid scores. |
| `src/visualize.py` | Writes ROC, PR, distribution, and correlation plots. |
| `scripts/analyze_part4_strict.py` | Builds Part 4 tables, figures, and reports. |

External dependencies:

| Dependency | What it is used for | Configured via |
|------------|---------------------|----------------|
| Qwen/DashScope API | LLM prediction generation | `DASHSCOPE_API_KEY` |
| OpenAI-compatible SDK | API client implementation | `llm.*` in `config.yaml` |
| Hugging Face models | Embedding model downloads | `embedding.models` |
| Local processed data | QA source records | `processed_data/{dataset}/` |

---

## Key Concepts and Abstractions

| Concept | What it means in this codebase |
|---------|--------------------------------|
| `correct_answer` | Raw teacher-provided reference field. |
| `ground_truth` | Normalized project reference copied from `correct_answer`. |
| `reference_answer` | Evaluation reference; extracted for NQ, raw for others. |
| `correct_label` | Automatic correctness label used as classifier truth. |
| `similarity_*` | Cosine score for one embedding model. |
| `threshold` | Cutoff that turns similarity into a correctness decision. |
| `high_similarity_wrong` | Similarity says correct, but `correct_label` is wrong. |
| `low_similarity_correct` | Similarity says wrong, but `correct_label` is correct. |
| `auto_paths` | Config mode that derives all paths from dataset and size. |
| `task_type` | `short_form` for SciQ/Wiki, `long_form` for NQ/TruthfulQA. |
| `hybrid_*` | Weighted score combining embedding and token/entity overlap. |
| `run_metadata.json` | Record of dataset, model, paths, and reference field. |

Two details matter for most development work:

1. `correct_label` is generated by code, not supplied by the dataset.
2. NQ uses `reference_answer` by default because raw NQ references are long
   Wikipedia evidence passages.

---

## Primary Flows

### Experiment Pipeline

```text
config.yaml
  selects dataset, sample size, LLM, embeddings, and output paths
  |
  v
scripts/01_generate_predictions.py
  loads processed data and calls `src/generate_predictions.py`
  |
  v
scripts/02_compute_similarity.py
  calls reference extraction, labeling, and embedding similarity
  |
  v
scripts/03_evaluate.py
  writes metrics, ablations, case studies, and failure JSONL files
  |
  v
scripts/04_visualize_results.py
  writes distribution, ROC, PR, and correlation plots
```

### Part 4 Failure Analysis

`scripts/analyze_part4_strict.py` reads selected `results_*` directories,
copies their metric tables, annotates sampled failure cases, writes summary
CSV files, and creates SVG figures under
`failures_analysis_and_improvement/`.

The baseline result groups are `results_nq_5000`, `results_sciq_500`,
`results_simple_questions_wiki_500`, and `results_truthfulQA_500`. The
implemented NQ improvement is analyzed separately with `results_nq_500`.

---

## Developer Guide

### Setup

Use the local conda environment created for this repo:

```bash
conda activate nlp-semsim
```

If you need to recreate it on another machine:

```bash
conda create -n nlp-semsim python=3.11 pip
conda activate nlp-semsim
python -m pip install -r requirements.txt
```

Prediction generation requires an API key:

```bash
export DASHSCOPE_API_KEY="your_key_here"
```

Do not put API keys in `config.yaml` or committed files.

### Running

Run unit tests:

```bash
python -m unittest discover -s tests
```

Run the full pipeline for the dataset selected in `config.yaml`:

```bash
python scripts/01_generate_predictions.py
python scripts/02_compute_similarity.py
python scripts/03_evaluate.py
python scripts/04_visualize_results.py
```

Regenerate the Part 4 analysis:

```bash
python scripts/analyze_part4_strict.py
```

### Common Change Patterns

To run a different dataset, edit only `data.dataset` and `data.sample_size`
in `config.yaml` while `project.auto_paths` stays `true`. The path resolver in
`src/utils.py` derives the prediction, similarity, and result paths.

To change correctness labeling, start in `src/correctness_labeling.py`. The
main hook is `label_correctness_for_record`, which writes `exact_match`,
`token_f1`, containment flags, and `correct_label`.

To improve NQ long-reference handling, start in `src/reference_answer.py`.
The main hook is `extract_nq_reference_answer`, and the tests in
`tests/test_reference_answer.py` cover the current who/where/number/date
heuristics.

To add another embedding model, add its Hugging Face model name to
`embedding.models` in `config.yaml`. The scripts will create a new
`similarity_{model_name}` field after replacing `/` and `-` with `_`.

To change Part 4 tables or figures, start in
`scripts/analyze_part4_strict.py`. Its output directories are
`failures_analysis_and_improvement/summary_tables/` and
`failures_analysis_and_improvement/figures/`.

### Key Files

| Area | File | Why |
|------|------|-----|
| Project overview | `README.md` | Existing human summary and run guide. |
| Experiment config | `config.yaml` | Dataset, paths, thresholds, model names. |
| Path safety | `src/utils.py` | Prevents dataset/result path mixing. |
| Labels | `src/correctness_labeling.py` | Defines the classifier truth labels. |
| NQ improvement | `src/reference_answer.py` | Implements reference extraction. |
| Metrics | `src/evaluate.py` | Similarity classifier and failure cases. |
| Main evaluation | `scripts/03_evaluate.py` | Writes tables and failure JSONL files. |
| Part 4 | `scripts/analyze_part4_strict.py` | Rebuilds failure analysis artifacts. |

### Practical Tips

Keep `project.auto_paths: true` unless you are intentionally reproducing an
old run. It prevents SciQ inputs from being written into NQ result folders.

Treat `results_nq_5000` and `results_nq_500` differently. The former is the
original long-passage NQ baseline; the latter uses the implemented
`reference_answer` extraction.

The markdown reports in `failures_analysis_and_improvement/` are written only
when missing. If you regenerate CSV/SVG outputs and need the reports refreshed,
delete the old `part4_report.md` or `part4_report.zh.md` first.
