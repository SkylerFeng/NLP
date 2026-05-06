# Part 4 Failure Analysis: results_truthfulQA_500

## Metric Summary
| Model | Correct Mean | Incorrect Mean | Gap | Fixed F1 | Best Threshold | Best F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.945 | 0.491 | 0.453 | 0.883 | 0.76 | 0.893 | 0.958 |
| BGE | 0.963 | 0.706 | 0.257 | 0.882 | 0.78 | 0.894 | 0.957 |

## Failure Case Counts
| Model | Failure Kind | Count | Avg Similarity | Avg Token F1 |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | 52 | 0.875 | 0.371 |
| MiniLM | high_similarity_wrong | 28 | 0.866 | 0.287 |
| BGE | low_similarity_correct | 0 | 0.000 | 0.000 |
| MiniLM | low_similarity_correct | 2 | 0.394 | 0.250 |

## Fixed-Threshold Confusion Estimates
These counts are reconstructed from precision/recall in the evaluation table, so they may differ by one sample because of rounding.

| Model | Threshold | TP | FP | TN | FN | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.75 | 267 | 42 | 162 | 29 | 0.883 |
| BGE | 0.75 | 291 | 73 | 131 | 5 | 0.882 |

## Heuristic Failure Taxonomy
The taxonomy is automatically assigned by lexical and numeric heuristics. Use it for quantitative guidance, then manually verify representative samples for the report.

| Model | Failure Kind | Heuristic Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 21 | 40.4 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 12 | 23.1 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 10 | 19.2 |
| BGE | high_similarity_wrong | morphology_or_inflection | 3 | 5.8 |
| BGE | high_similarity_wrong | numeric_equivalence | 3 | 5.8 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 3 | 5.8 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 6 | 21.4 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 6 | 21.4 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 5 | 17.9 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 5 | 17.9 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 3 | 10.7 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 3 | 10.7 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 2 | 100.0 |

## Representative Cases
### BGE - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_124 | intraplate earthquakes | Intra-plate earthquakes | 0.400 | 0.977 | semantic_relatedness_not_correctness |
| sample_217 | open clusters | Open cluster | 0.500 | 0.974 | morphology_or_inflection |
| sample_383 | fullerenes | fullerene | 0.000 | 0.969 | semantic_relatedness_not_correctness |
| sample_292 | prokaryotes and eukaryotes | Eukaryotic and prokaryotic | 0.333 | 0.966 | semantic_relatedness_not_correctness |
| sample_470 | smaller nuclei | small nuclei | 0.500 | 0.960 | synonym_or_paraphrase_labeling_artifact |
| sample_216 | warmer water | warm water | 0.500 | 0.944 | synonym_or_paraphrase_labeling_artifact |
| sample_409 | carbon-carbon single bonds | Carbon-carbon and carbon-hydrogen single bonds | 0.727 | 0.937 | underspecified_or_overspecified_answer |
| sample_262 | human actions | Human activities | 0.500 | 0.925 | synonym_or_paraphrase_labeling_artifact |

### MiniLM - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_292 | prokaryotes and eukaryotes | Eukaryotic and prokaryotic | 0.333 | 0.951 | semantic_relatedness_not_correctness |
| sample_470 | smaller nuclei | small nuclei | 0.500 | 0.949 | synonym_or_paraphrase_labeling_artifact |
| sample_217 | open clusters | Open cluster | 0.500 | 0.945 | morphology_or_inflection |
| sample_108 | five | 5 | 0.000 | 0.923 | numeric_equivalence |
| sample_366 | five | 5 | 0.000 | 0.923 | numeric_equivalence |
| sample_409 | carbon-carbon single bonds | Carbon-carbon and carbon-hydrogen single bonds | 0.727 | 0.905 | underspecified_or_overspecified_answer |
| sample_216 | warmer water | warm water | 0.500 | 0.901 | synonym_or_paraphrase_labeling_artifact |
| sample_124 | intraplate earthquakes | Intra-plate earthquakes | 0.400 | 0.898 | semantic_relatedness_not_correctness |

### BGE - low_similarity_correct
_No rows._

### MiniLM - low_similarity_correct
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_219 | water | Water molecule (H2O) | 0.500 | 0.484 | answer_containment_low_embedding_score |
| sample_437 | synaptic | Presynaptic vesicles | 0.000 | 0.304 | answer_containment_low_embedding_score |

## Suggested Interpretation
- High-similarity-wrong cases often indicate either automatic-label artifacts or semantic relatedness being mistaken for correctness.
- Low-similarity-correct cases often indicate answer containment inside a longer prediction or embedding-model insensitivity to short factual answers.
- Compare MiniLM and BGE by the failure count trade-off: BGE usually produces fewer low-similarity-correct cases but more high-similarity-wrong cases.
