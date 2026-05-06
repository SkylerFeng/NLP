# Human Annotation Analysis: results_truthfulQA_500

This report summarizes the sampled human annotations stored in `tables/manual_annotation_sample.csv`.

## Model-Level Summary
| Model | Sampled | Label Artifacts | Semantic Limits | Low-Sim FN | Top Type |
| --- | --- | --- | --- | --- | --- |
| BGE | 50 | 26 (52.0%) | 24 (48.0%) | 0 (0.0%) | synonym_or_paraphrase_labeling_artifact |
| MiniLM | 30 | 12 (40.0%) | 16 (53.3%) | 2 (6.7%) | synonym_or_paraphrase_labeling_artifact |

## Human Failure-Type Distribution
| Model | Failure Kind | Human Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | morphology_or_inflection | 3 | 6.0 |
| BGE | high_similarity_wrong | numeric_equivalence | 2 | 4.0 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 2 | 4.0 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 10 | 20.0 |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 21 | 42.0 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 12 | 24.0 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 3 | 10.7 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 3 | 10.7 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 5 | 17.9 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 6 | 21.4 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 6 | 21.4 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 5 | 17.9 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 2 | 100.0 |

## Human Category Distribution
| Model | Failure Kind | Human Category | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | automatic_label_artifact | 26 | 52.0 |
| BGE | high_similarity_wrong | semantic_similarity_limitation | 24 | 48.0 |
| MiniLM | high_similarity_wrong | automatic_label_artifact | 12 | 42.9 |
| MiniLM | high_similarity_wrong | semantic_similarity_limitation | 16 | 57.1 |
| MiniLM | low_similarity_correct | low_similarity_false_negative | 2 | 100.0 |

## Interpretation Guide
- `automatic_label_artifact`: the answer is likely acceptable, but the automatic correctness label is too strict.
- `semantic_similarity_limitation`: similarity is high because the texts are related, but relatedness does not guarantee correctness.
- `low_similarity_false_negative`: the answer is correct or contained in the prediction, but the embedding score is low.
