# Part 4 Failure Analysis

Chinese version: [`part4_report.zh.md`](part4_report.zh.md).

## 1. Purpose and Definition
Part 4 asks us to identify where embedding similarity fails and to propose improvements. We define a failure case as a sample where the similarity-threshold correctness decision disagrees with the automatic `correct_label`.

- `high_similarity_wrong`: `correct_label = 0`, but similarity is high.
- `low_similarity_correct`: `correct_label = 1`, but similarity is low.

This distinction is important: these are failures of the similarity-as-classifier pipeline, not always failures of the LLM answer. Some cases reveal defects in the automatic label itself.

## 2. Data and Outputs Used
We analyze four result folders: `results_nq`, `results_sciq_5000`, `results_truthfulQA_500`, and `results_wiki`. For each folder, we use the evaluation table, failure-case JSONL files, and the sampled human annotations in `manual_annotation_sample.csv`.

Caveat: inspected examples in `results_nq` look like short-form science QA rather than true Natural Questions long-form QA. We therefore treat `results_nq` as an additional result folder, but avoid making strong claims about long-form NQ behavior from it alone.

## 3. Embedding Model Comparison
![Model metric comparison](figures/model_metric_comparison.svg)

| Dataset | Model | Gap | Fixed F1 | Best Threshold | Best F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| NQ-like | MiniLM | 0.462 | 0.893 | 0.70 | 0.895 | 0.959 |
| NQ-like | BGE | 0.260 | 0.877 | 0.80 | 0.888 | 0.955 |
| SciQ | MiniLM | 0.462 | 0.891 | 0.70 | 0.895 | 0.958 |
| SciQ | BGE | 0.260 | 0.879 | 0.80 | 0.889 | 0.954 |
| TruthfulQA | MiniLM | 0.453 | 0.883 | 0.76 | 0.893 | 0.958 |
| TruthfulQA | BGE | 0.257 | 0.882 | 0.78 | 0.894 | 0.957 |
| Wiki | MiniLM | 0.463 | 0.894 | 0.74 | 0.896 | 0.959 |
| Wiki | BGE | 0.261 | 0.877 | 0.81 | 0.890 | 0.955 |

MiniLM consistently has a larger correct-vs-incorrect similarity gap, around 0.45-0.46. BGE has comparable ROC-AUC, but its incorrect-answer mean similarity is much higher, so its best threshold is also higher. In practice, BGE is more recall-friendly but less conservative.

## 4. Full Failure Counts
![Failure case counts](figures/failure_case_counts.svg)

| Model | High-Sim Wrong | Low-Sim Correct | Total |
| --- | --- | --- | --- |
| BGE | 1505 | 0 | 1505 |
| MiniLM | 771 | 114 | 885 |

BGE produces no low-similarity-correct cases in these outputs, but it produces many more high-similarity-wrong cases. MiniLM has fewer high-similarity false positives, but it misses some correct answers when the correct short answer is embedded in a longer prediction.

## 5. Human Annotation Analysis
![Human annotation categories](figures/human_annotation_categories.svg)

| Dataset | Model | Sampled | Label Artifacts | Semantic Limits | Low-Sim FN | Top Type |
| --- | --- | --- | --- | --- | --- | --- |
| NQ-like | BGE | 50 | 17 (34.0%) | 33 (66.0%) | 0 (0.0%) | synonym_or_paraphrase_labeling_artifact |
| NQ-like | MiniLM | 90 | 20 (22.2%) | 33 (36.7%) | 37 (41.1%) | answer_containment_low_embedding_score |
| SciQ | BGE | 50 | 16 (32.0%) | 34 (68.0%) | 0 (0.0%) | underspecified_or_overspecified_answer |
| SciQ | MiniLM | 87 | 19 (21.8%) | 34 (39.1%) | 34 (39.1%) | answer_containment_low_embedding_score |
| TruthfulQA | BGE | 50 | 26 (52.0%) | 24 (48.0%) | 0 (0.0%) | synonym_or_paraphrase_labeling_artifact |
| TruthfulQA | MiniLM | 30 | 12 (40.0%) | 16 (53.3%) | 2 (6.7%) | synonym_or_paraphrase_labeling_artifact |
| Wiki | BGE | 50 | 21 (42.0%) | 29 (58.0%) | 0 (0.0%) | synonym_or_paraphrase_labeling_artifact |
| Wiki | MiniLM | 85 | 24 (28.2%) | 29 (34.1%) | 32 (37.6%) | answer_containment_low_embedding_score |

The sampled human annotations split failures into three broad categories:

- `automatic_label_artifact`: the model answer is likely acceptable, but the automatic correctness label is too strict.
- `semantic_similarity_limitation`: the prediction is semantically related to the reference but is not necessarily correct.
- `low_similarity_false_negative`: the prediction contains or expresses the correct answer, but the embedding score is low.

## 6. Main Failure Types
| Dataset | Model | Failure Kind | Human Type | Count | % |
| --- | --- | --- | --- | --- | --- |
| NQ-like | BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 13 | 26.0 |
| NQ-like | BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 13 | 26.0 |
| NQ-like | BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 11 | 22.0 |
| NQ-like | MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 18 | 36.0 |
| NQ-like | MiniLM | high_similarity_wrong | other_or_true_semantic_error | 10 | 20.0 |
| NQ-like | MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 8 | 16.0 |
| NQ-like | MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 25 | 62.5 |
| NQ-like | MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 12 | 30.0 |
| NQ-like | MiniLM | low_similarity_correct | numeric_equivalence | 3 | 7.5 |
| SciQ | BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 16 | 32.0 |
| SciQ | BGE | high_similarity_wrong | other_or_true_semantic_error | 11 | 22.0 |
| SciQ | BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 10 | 20.0 |
| SciQ | MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 18 | 36.0 |
| SciQ | MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 9 | 18.0 |
| SciQ | MiniLM | high_similarity_wrong | other_or_true_semantic_error | 7 | 14.0 |
| SciQ | MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 22 | 59.5 |
| SciQ | MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 12 | 32.4 |
| SciQ | MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.1 |
| TruthfulQA | BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 21 | 42.0 |
| TruthfulQA | BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 12 | 24.0 |
| TruthfulQA | BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 10 | 20.0 |
| TruthfulQA | MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 6 | 21.4 |
| TruthfulQA | MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 6 | 21.4 |
| TruthfulQA | MiniLM | high_similarity_wrong | other_or_true_semantic_error | 5 | 17.9 |
| TruthfulQA | MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 2 | 100.0 |
| Wiki | BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 16 | 32.0 |
| Wiki | BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 13 | 26.0 |
| Wiki | BGE | high_similarity_wrong | other_or_true_semantic_error | 9 | 18.0 |
| Wiki | MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 16 | 32.0 |
| Wiki | MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 12 | 24.0 |
| Wiki | MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 7 | 14.0 |
| Wiki | MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 24 | 68.6 |
| Wiki | MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 8 | 22.9 |
| Wiki | MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.6 |

The common high-similarity-wrong cases include singular/plural variants, numeric equivalence, paraphrases, and underspecified answers. The most important true limitation is that embedding similarity often measures relatedness rather than factual correctness. For example, a broad answer can be semantically close to the reference but still miss a key modifier.

Low-similarity-correct cases mostly appear for MiniLM. These cases often contain the correct short answer inside a longer phrase or sentence, so the sentence embedding is diluted by surrounding context.

## 7. Conclusions
- MiniLM is more conservative and separates correct and incorrect answers more clearly.
- BGE gives stronger semantic matches but is more likely to over-score related yet incomplete answers.
- Many apparent errors are actually automatic-label artifacts, not true LLM answer errors.
- A single cosine-similarity threshold is useful as a screening signal, but it is not robust enough as the final evaluator.

## 8. Improvement Proposal
A stronger evaluator should combine multiple checks:

1. Stronger normalization: lowercasing, punctuation/hyphen handling, lemmatization, and number-word conversion.
2. Answer containment and answer extraction, especially when predictions are longer than references.
3. Entity or keyword overlap to catch important factual units.
4. Sentence-level similarity for longer answers.
5. NLI or LLM-based verification for ambiguous high-similarity cases.

A practical hybrid pipeline is: first normalize and check exact/containment matches; then apply embedding similarity with a dataset/model-specific threshold; finally send ambiguous cases to a verifier.

## 9. Reproducibility
Run the scripts below from the project root:

```bash
python part4_failure_analysis/scripts/analyze_part4_failures.py
python part4_failure_analysis/scripts/build_final_part4_report.py
```

Key tables are stored under `part4_failure_analysis/summary_tables/`; dataset-level reports and annotation files are under `part4_failure_analysis/datasets/`.
