# Part 4 Failure Analysis: results_sciq_5000

## Metric Summary
| Model | Correct Mean | Incorrect Mean | Gap | Fixed F1 | Best Threshold | Best F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.940 | 0.477 | 0.462 | 0.891 | 0.70 | 0.895 | 0.958 |
| BGE | 0.958 | 0.699 | 0.260 | 0.879 | 0.80 | 0.889 | 0.954 |

## Failure Case Counts
| Model | Failure Kind | Count | Avg Similarity | Avg Token F1 |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | 479 | 0.870 | 0.337 |
| MiniLM | high_similarity_wrong | 241 | 0.875 | 0.336 |
| BGE | low_similarity_correct | 0 | 0.000 | 0.000 |
| MiniLM | low_similarity_correct | 37 | 0.416 | 0.362 |

## Fixed-Threshold Confusion Estimates
These counts are reconstructed from precision/recall in the evaluation table, so they may differ by one sample because of rounding.

| Model | Threshold | TP | FP | TN | FN | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.75 | 2626 | 328 | 1733 | 313 | 0.891 |
| BGE | 0.75 | 2859 | 710 | 1351 | 80 | 0.879 |

## Heuristic Failure Taxonomy
The taxonomy is automatically assigned by lexical and numeric heuristics. Use it for quantitative guidance, then manually verify representative samples for the report.

| Model | Failure Kind | Heuristic Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 130 | 27.1 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 121 | 25.3 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 93 | 19.4 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 78 | 16.3 |
| BGE | high_similarity_wrong | morphology_or_inflection | 42 | 8.8 |
| BGE | high_similarity_wrong | numeric_equivalence | 15 | 3.1 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 72 | 29.9 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 50 | 20.7 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 40 | 16.6 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 33 | 13.7 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 31 | 12.9 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 15 | 6.2 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 22 | 59.5 |
| MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 12 | 32.4 |
| MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.1 |

## Representative Cases
### BGE - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1016 | pāhoehoe | Pahoehoe | 0.000 | 1.000 | semantic_relatedness_not_correctness |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.982 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.979 | morphology_or_inflection |
| sample_124 | intraplate earthquakes | Intra-plate earthquakes | 0.400 | 0.977 | semantic_relatedness_not_correctness |
| sample_1042 | electron configurations | Electron configuration | 0.500 | 0.976 | morphology_or_inflection |
| sample_1559 | monarch butterflies | Monarch Butterfly | 0.500 | 0.976 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.975 | morphology_or_inflection |
| sample_217 | open clusters | Open cluster | 0.500 | 0.974 | morphology_or_inflection |

### MiniLM - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1016 | pāhoehoe | Pahoehoe | 0.000 | 1.000 | semantic_relatedness_not_correctness |
| sample_2502 | alkaline earth metals | Alkali earth metals | 0.667 | 0.984 | synonym_or_paraphrase_labeling_artifact |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.979 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.976 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.976 | morphology_or_inflection |
| sample_1204 | infectious diseases | Infectious disease | 0.500 | 0.973 | synonym_or_paraphrase_labeling_artifact |
| sample_1042 | electron configurations | Electron configuration | 0.500 | 0.969 | morphology_or_inflection |
| sample_3625 | ammeters | Ammeter | 0.000 | 0.968 | morphology_or_inflection |

### BGE - low_similarity_correct
_No rows._

### MiniLM - low_similarity_correct
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1659 | a fault | Strike-slip fault | 0.500 | 0.499 | answer_containment_low_embedding_score |
| sample_4360 | fuel | biofuel | 0.000 | 0.494 | answer_containment_low_embedding_score |
| sample_1481 | solid | Solid elements except for lithium. | 0.333 | 0.493 | overly_long_answer_context_dilution |
| sample_2643 | state | states of matter | 0.000 | 0.491 | answer_containment_low_embedding_score |
| sample_1394 | size | Cell size | 0.667 | 0.491 | answer_containment_low_embedding_score |
| sample_2146 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |
| sample_3366 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |
| sample_3166 | behavior | mating behavior | 0.667 | 0.484 | answer_containment_low_embedding_score |

## Suggested Interpretation
- High-similarity-wrong cases often indicate either automatic-label artifacts or semantic relatedness being mistaken for correctness.
- Low-similarity-correct cases often indicate answer containment inside a longer prediction or embedding-model insensitivity to short factual answers.
- Compare MiniLM and BGE by the failure count trade-off: BGE usually produces fewer low-similarity-correct cases but more high-similarity-wrong cases.
