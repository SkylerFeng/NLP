# Part 4 Failure Analysis: results_nq

## Metric Summary
| Model | Correct Mean | Incorrect Mean | Gap | Fixed F1 | Best Threshold | Best F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.941 | 0.478 | 0.462 | 0.893 | 0.70 | 0.895 | 0.959 |
| BGE | 0.959 | 0.699 | 0.260 | 0.877 | 0.80 | 0.888 | 0.955 |

## Failure Case Counts
| Model | Failure Kind | Count | Avg Similarity | Avg Token F1 |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | 493 | 0.870 | 0.337 |
| MiniLM | high_similarity_wrong | 252 | 0.876 | 0.322 |
| BGE | low_similarity_correct | 0 | 0.000 | 0.000 |
| MiniLM | low_similarity_correct | 40 | 0.421 | 0.372 |

## Fixed-Threshold Confusion Estimates
These counts are reconstructed from precision/recall in the evaluation table, so they may differ by one sample because of rounding.

| Model | Threshold | TP | FP | TN | FN | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.75 | 2635 | 338 | 1731 | 296 | 0.893 |
| BGE | 0.75 | 2852 | 718 | 1351 | 79 | 0.877 |

## Heuristic Failure Taxonomy
The taxonomy is automatically assigned by lexical and numeric heuristics. Use it for quantitative guidance, then manually verify representative samples for the report.

| Model | Failure Kind | Heuristic Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 140 | 28.4 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 119 | 24.1 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 94 | 19.1 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 78 | 15.8 |
| BGE | high_similarity_wrong | morphology_or_inflection | 43 | 8.7 |
| BGE | high_similarity_wrong | numeric_equivalence | 19 | 3.9 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 67 | 26.6 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 54 | 21.4 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 41 | 16.3 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 38 | 15.1 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 34 | 13.5 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 18 | 7.1 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 25 | 62.5 |
| MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 12 | 30.0 |
| MiniLM | low_similarity_correct | numeric_equivalence | 3 | 7.5 |

## Representative Cases
### BGE - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.982 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.979 | morphology_or_inflection |
| sample_124 | intraplate earthquakes | Intra-plate earthquakes | 0.400 | 0.977 | semantic_relatedness_not_correctness |
| sample_1042 | electron configurations | electron configuration | 0.500 | 0.976 | morphology_or_inflection |
| sample_1559 | monarch butterflies | Monarch Butterfly | 0.500 | 0.976 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.975 | morphology_or_inflection |
| sample_217 | open clusters | Open cluster | 0.500 | 0.974 | morphology_or_inflection |
| sample_2843 | ionic bonds | Ionic bond | 0.500 | 0.974 | morphology_or_inflection |

### MiniLM - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_2502 | alkaline earth metals | Alkali earth metals | 0.667 | 0.984 | synonym_or_paraphrase_labeling_artifact |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.979 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.976 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.976 | morphology_or_inflection |
| sample_301 | stratovolcanoes | Stratovolcanos | 0.000 | 0.974 | morphology_or_inflection |
| sample_1204 | infectious diseases | Infectious disease | 0.500 | 0.973 | synonym_or_paraphrase_labeling_artifact |
| sample_1042 | electron configurations | electron configuration | 0.500 | 0.969 | morphology_or_inflection |
| sample_3625 | ammeters | Ammeter | 0.000 | 0.968 | morphology_or_inflection |

### BGE - low_similarity_correct
_No rows._

### MiniLM - low_similarity_correct
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1738 | fermentation | Aerobic respiration not applicable here; ATP is produced in glycolysis during an | 0.095 | 0.499 | overly_long_answer_context_dilution |
| sample_1659 | a fault | Strike-slip fault | 0.500 | 0.499 | answer_containment_low_embedding_score |
| sample_1067 | simple | simple machine | 0.667 | 0.497 | answer_containment_low_embedding_score |
| sample_4360 | fuel | biofuel | 0.000 | 0.494 | answer_containment_low_embedding_score |
| sample_1394 | size | Cell size | 0.667 | 0.491 | answer_containment_low_embedding_score |
| sample_3366 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |
| sample_2146 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |
| sample_3533 | organisms | Animals obtain their energy from consuming other organisms. | 0.222 | 0.484 | overly_long_answer_context_dilution |

## Suggested Interpretation
- High-similarity-wrong cases often indicate either automatic-label artifacts or semantic relatedness being mistaken for correctness.
- Low-similarity-correct cases often indicate answer containment inside a longer prediction or embedding-model insensitivity to short factual answers.
- Compare MiniLM and BGE by the failure count trade-off: BGE usually produces fewer low-similarity-correct cases but more high-similarity-wrong cases.
