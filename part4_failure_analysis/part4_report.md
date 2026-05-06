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

## 6. Similarity Distance and Concrete Examples
The similarity score is cosine similarity between the prediction embedding and the reference embedding. For interpretation, we also use:

```text
distance = 1 - cosine_similarity
```

A small distance means the two answers are close in embedding space. In `high_similarity_wrong` cases, the distance is small even though the automatic label says the prediction is wrong. In `low_similarity_correct` cases, the distance is large even though the automatic label says the prediction is correct.

| Category | Dataset | Model | Failure Kind | Reference | Prediction | Similarity | Distance | Interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| automatic_label_artifact | SciQ | BGE | high_similarity_wrong | ovaries | Ovary | 0.895 | 0.105 | Singular/plural variation; the automatic label is too strict. |
| automatic_label_artifact | SciQ | BGE | high_similarity_wrong | four | 4 | 0.863 | 0.137 | Numeric equivalence should be normalized before labeling. |
| automatic_label_artifact | SciQ | BGE | high_similarity_wrong | wider pelvis | wider hips | 0.887 | 0.113 | Close paraphrase/anatomical wording difference. |
| semantic_similarity_limitation | SciQ | BGE | high_similarity_wrong | bone fractures | fractures | 0.891 | 0.109 | Prediction is related but underspecified; it misses the bone modifier. |
| semantic_similarity_limitation | SciQ | BGE | high_similarity_wrong | proto-oncogenes | Oncogenes | 0.835 | 0.165 | Related biological term, but not the same answer. |
| semantic_similarity_limitation | SciQ | BGE | high_similarity_wrong | solar energy | Solar panels | 0.837 | 0.163 | Related concept, but source vs. device distinction matters. |
| low_similarity_false_negative | SciQ | MiniLM | low_similarity_correct | three | Three main types: elliptical, spiral, and irregular. | 0.190 | 0.810 | Correct short answer is embedded in a much longer sentence. |
| low_similarity_false_negative | SciQ | MiniLM | low_similarity_correct | negative | Partial negative charge | 0.377 | 0.623 | Reference is contained, but added context changes the sentence vector. |
| low_similarity_false_negative | SciQ | MiniLM | low_similarity_correct | bacteria | Yogurt is made from milk fermented with bacteria. | 0.399 | 0.601 | Answer containment is clear, but whole-sentence embedding is diluted. |

These examples show that the same distance pattern can mean different things. A small distance can indicate a correct paraphrase that was mislabeled, but it can also indicate that the embedding model confuses semantic relatedness with factual correctness. A large distance can indicate that a short correct answer is surrounded by extra context.
## 7. Main Failure Types
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

## 8. Conclusions
- MiniLM is more conservative and separates correct and incorrect answers more clearly.
- BGE gives stronger semantic matches but is more likely to over-score related yet incomplete answers.
- Many apparent errors are actually automatic-label artifacts, not true LLM answer errors.
- A single cosine-similarity threshold is useful as a screening signal, but it is not robust enough as the final evaluator.

## 9. Detailed Improvement Proposal
The failure analysis suggests that a better evaluator should not replace embedding similarity entirely. Instead, similarity should become one component in a more structured correctness pipeline.

### 9.1 Normalization and Canonicalization
Before computing labels or similarity thresholds, normalize both prediction and reference. This should include lowercasing, punctuation removal, hyphen normalization, singular/plural lemmatization, number-word conversion, and common abbreviation expansion such as `CO2` to `carbon dioxide`. This directly targets label artifacts such as `ovary` vs. `ovaries`, `four` vs. `4`, and `intra-plate` vs. `intraplate`.

### 9.2 Answer Extraction for Long Predictions
For short-answer QA, many false negatives happen because the prediction is a full sentence while the reference is a short phrase. Before embedding comparison, extract the likely answer span from the prediction. A simple version can use containment rules and noun-phrase heuristics; a stronger version can use an LLM prompt that rewrites the prediction into the shortest answer phrase. This addresses cases like `Three main types: elliptical, spiral, and irregular.` vs. `three`.

### 9.3 Hybrid Scoring
Use a score that combines semantic similarity with lexical and factual overlap:

```text
hybrid_score = 0.55 * embedding_similarity
             + 0.20 * token_f1
             + 0.15 * entity_or_keyword_overlap
             + 0.10 * normalization_bonus
```

The weights can be tuned on a small validation subset. The goal is to keep the paraphrase sensitivity of embeddings while preventing related but incomplete answers from receiving too much credit.

### 9.4 Dataset- and Model-Specific Thresholds
The best thresholds are not the same for MiniLM and BGE. MiniLM works best around 0.70-0.76 in these results, while BGE often needs around 0.78-0.81. Therefore, a single global threshold such as 0.75 is not ideal. Thresholds should be selected separately for each dataset and embedding model using validation F1 or a precision-recall trade-off.

### 9.5 Ambiguity-Aware Verification
Some cases should not be decided by similarity alone. Send uncertain cases to a verifier when the score is near the threshold, when entity overlap is low despite high similarity, or when the prediction is much shorter or more general than the reference. The verifier can be an NLI model or an LLM judge asked whether the prediction entails the reference answer under the question context.

### 9.6 Expected Effect
Normalization should reduce label artifacts; answer extraction should reduce MiniLM's low-similarity false negatives; entity overlap and verifier checks should reduce BGE's high-similarity wrong cases caused by semantic relatedness without correctness. These modules directly target the three failure categories found in the human annotation analysis.

## 10. Reproducibility
Run the scripts below from the project root:

```bash
python part4_failure_analysis/scripts/analyze_part4_failures.py
python part4_failure_analysis/scripts/build_final_part4_report.py
```

Key tables are stored under `part4_failure_analysis/summary_tables/`; dataset-level reports and annotation files are under `part4_failure_analysis/datasets/`.
