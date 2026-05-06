# Human Annotation Analysis: results_wiki

This report summarizes the sampled human annotations stored in `tables/manual_annotation_sample.csv`.

## Model-Level Summary
| Model | Sampled | Label Artifacts | Semantic Limits | Low-Sim FN | Top Type |
| --- | --- | --- | --- | --- | --- |
| BGE | 50 | 21 (42.0%) | 29 (58.0%) | 0 (0.0%) | synonym_or_paraphrase_labeling_artifact |
| MiniLM | 85 | 24 (28.2%) | 29 (34.1%) | 32 (37.6%) | answer_containment_low_embedding_score |

## Human Failure-Type Distribution
| Model | Failure Kind | Human Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | morphology_or_inflection | 4 | 8.0 |
| BGE | high_similarity_wrong | numeric_equivalence | 1 | 2.0 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 9 | 18.0 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 7 | 14.0 |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 16 | 32.0 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 13 | 26.0 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 4 | 8.0 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 5 | 10.0 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 6 | 12.0 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 7 | 14.0 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 12 | 24.0 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 16 | 32.0 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 24 | 68.6 |
| MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.6 |
| MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 8 | 22.9 |

## Human Category Distribution
| Model | Failure Kind | Human Category | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | automatic_label_artifact | 21 | 42.0 |
| BGE | high_similarity_wrong | semantic_similarity_limitation | 29 | 58.0 |
| MiniLM | high_similarity_wrong | automatic_label_artifact | 21 | 42.0 |
| MiniLM | high_similarity_wrong | semantic_similarity_limitation | 29 | 58.0 |
| MiniLM | low_similarity_correct | automatic_label_artifact | 3 | 8.6 |
| MiniLM | low_similarity_correct | low_similarity_false_negative | 32 | 91.4 |

## Interpretation Guide
- `automatic_label_artifact`: the answer is likely acceptable, but the automatic correctness label is too strict.
- `semantic_similarity_limitation`: similarity is high because the texts are related, but relatedness does not guarantee correctness.
- `low_similarity_false_negative`: the answer is correct or contained in the prediction, but the embedding score is low.
