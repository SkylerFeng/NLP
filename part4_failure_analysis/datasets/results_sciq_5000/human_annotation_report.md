# Human Annotation Analysis: results_sciq_5000

This report summarizes the sampled human annotations stored in `tables/manual_annotation_sample.csv`.

## Model-Level Summary
| Model | Sampled | Label Artifacts | Semantic Limits | Low-Sim FN | Top Type |
| --- | --- | --- | --- | --- | --- |
| BGE | 50 | 16 (32.0%) | 34 (68.0%) | 0 (0.0%) | underspecified_or_overspecified_answer |
| MiniLM | 87 | 19 (21.8%) | 34 (39.1%) | 34 (39.1%) | answer_containment_low_embedding_score |

## Human Failure-Type Distribution
| Model | Failure Kind | Human Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | morphology_or_inflection | 5 | 10.0 |
| BGE | high_similarity_wrong | numeric_equivalence | 1 | 2.0 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 11 | 22.0 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 7 | 14.0 |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 10 | 20.0 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 16 | 32.0 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 5 | 10.0 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 4 | 8.0 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 7 | 14.0 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 9 | 18.0 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 7 | 14.0 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 18 | 36.0 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 22 | 59.5 |
| MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.1 |
| MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 12 | 32.4 |

## Human Category Distribution
| Model | Failure Kind | Human Category | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | automatic_label_artifact | 16 | 32.0 |
| BGE | high_similarity_wrong | semantic_similarity_limitation | 34 | 68.0 |
| MiniLM | high_similarity_wrong | automatic_label_artifact | 16 | 32.0 |
| MiniLM | high_similarity_wrong | semantic_similarity_limitation | 34 | 68.0 |
| MiniLM | low_similarity_correct | automatic_label_artifact | 3 | 8.1 |
| MiniLM | low_similarity_correct | low_similarity_false_negative | 34 | 91.9 |

## Interpretation Guide
- `automatic_label_artifact`: the answer is likely acceptable, but the automatic correctness label is too strict.
- `semantic_similarity_limitation`: similarity is high because the texts are related, but relatedness does not guarantee correctness.
- `low_similarity_false_negative`: the answer is correct or contained in the prediction, but the embedding score is low.
