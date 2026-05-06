# Part 4 Failure Analysis: results_wiki

## Metric Summary
| Model | Correct Mean | Incorrect Mean | Gap | Fixed F1 | Best Threshold | Best F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.942 | 0.479 | 0.463 | 0.894 | 0.74 | 0.896 | 0.959 |
| BGE | 0.960 | 0.699 | 0.261 | 0.877 | 0.81 | 0.890 | 0.955 |

## Failure Case Counts
| Model | Failure Kind | Count | Avg Similarity | Avg Token F1 |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | 481 | 0.873 | 0.339 |
| MiniLM | high_similarity_wrong | 250 | 0.876 | 0.324 |
| BGE | low_similarity_correct | 0 | 0.000 | 0.000 |
| MiniLM | low_similarity_correct | 35 | 0.419 | 0.354 |

## Fixed-Threshold Confusion Estimates
These counts are reconstructed from precision/recall in the evaluation table, so they may differ by one sample because of rounding.

| Model | Threshold | TP | FP | TN | FN | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| MiniLM | 0.75 | 2632 | 330 | 1741 | 297 | 0.894 |
| BGE | 0.75 | 2847 | 720 | 1351 | 82 | 0.877 |

## Heuristic Failure Taxonomy
The taxonomy is automatically assigned by lexical and numeric heuristics. Use it for quantitative guidance, then manually verify representative samples for the report.

| Model | Failure Kind | Heuristic Type | Count | % |
| --- | --- | --- | --- | --- |
| BGE | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 133 | 27.7 |
| BGE | high_similarity_wrong | underspecified_or_overspecified_answer | 122 | 25.4 |
| BGE | high_similarity_wrong | semantic_relatedness_not_correctness | 88 | 18.3 |
| BGE | high_similarity_wrong | other_or_true_semantic_error | 77 | 16.0 |
| BGE | high_similarity_wrong | morphology_or_inflection | 44 | 9.1 |
| BGE | high_similarity_wrong | numeric_equivalence | 17 | 3.5 |
| MiniLM | high_similarity_wrong | underspecified_or_overspecified_answer | 68 | 27.2 |
| MiniLM | high_similarity_wrong | synonym_or_paraphrase_labeling_artifact | 53 | 21.2 |
| MiniLM | high_similarity_wrong | morphology_or_inflection | 43 | 17.2 |
| MiniLM | high_similarity_wrong | semantic_relatedness_not_correctness | 36 | 14.4 |
| MiniLM | high_similarity_wrong | other_or_true_semantic_error | 33 | 13.2 |
| MiniLM | high_similarity_wrong | numeric_equivalence | 17 | 6.8 |
| MiniLM | low_similarity_correct | answer_containment_low_embedding_score | 24 | 68.6 |
| MiniLM | low_similarity_correct | overly_long_answer_context_dilution | 8 | 22.9 |
| MiniLM | low_similarity_correct | numeric_equivalence | 3 | 8.6 |

## Representative Cases
### BGE - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1016 | pāhoehoe | Pahoehoe | 0.000 | 1.000 | semantic_relatedness_not_correctness |
| sample_2182 | interphase and mitotic | Interphase and Mitosis | 0.667 | 0.988 | synonym_or_paraphrase_labeling_artifact |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.982 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.979 | morphology_or_inflection |
| sample_124 | intraplate earthquakes | Intra-plate earthquakes | 0.400 | 0.977 | semantic_relatedness_not_correctness |
| sample_1042 | electron configurations | electron configuration | 0.500 | 0.976 | morphology_or_inflection |
| sample_1559 | monarch butterflies | Monarch Butterfly | 0.500 | 0.976 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.975 | morphology_or_inflection |

### MiniLM - high_similarity_wrong
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1016 | pāhoehoe | Pahoehoe | 0.000 | 1.000 | semantic_relatedness_not_correctness |
| sample_917 | silvery grey | Silvery gray | 0.500 | 0.982 | synonym_or_paraphrase_labeling_artifact |
| sample_805 | polar covalent bonds | Polar covalent bond | 0.667 | 0.979 | morphology_or_inflection |
| sample_3536 | atomic numbers | atomic number | 0.500 | 0.976 | morphology_or_inflection |
| sample_2072 | carbon taxes | Carbon tax | 0.500 | 0.976 | morphology_or_inflection |
| sample_301 | stratovolcanoes | Stratovolcanos | 0.000 | 0.974 | morphology_or_inflection |
| sample_1204 | infectious diseases | Infectious disease | 0.500 | 0.973 | synonym_or_paraphrase_labeling_artifact |
| sample_1042 | electron configurations | electron configuration | 0.500 | 0.969 | morphology_or_inflection |

### BGE - low_similarity_correct
_No rows._

### MiniLM - low_similarity_correct
| id | ground_truth | prediction | token_f1 | similarity | heuristic_type |
| --- | --- | --- | --- | --- | --- |
| sample_1659 | a fault | Strike-slip fault | 0.500 | 0.499 | answer_containment_low_embedding_score |
| sample_1067 | simple | simple machine | 0.667 | 0.497 | answer_containment_low_embedding_score |
| sample_4360 | fuel | biofuel | 0.000 | 0.494 | answer_containment_low_embedding_score |
| sample_2643 | state | states of matter | 0.000 | 0.491 | answer_containment_low_embedding_score |
| sample_1394 | size | Cell size | 0.667 | 0.491 | answer_containment_low_embedding_score |
| sample_4879 | acid | acidulent environment | 0.000 | 0.490 | answer_containment_low_embedding_score |
| sample_3366 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |
| sample_2146 | negative | Negative feedback | 0.667 | 0.488 | answer_containment_low_embedding_score |

## Suggested Interpretation
- High-similarity-wrong cases often indicate either automatic-label artifacts or semantic relatedness being mistaken for correctness.
- Low-similarity-correct cases often indicate answer containment inside a longer prediction or embedding-model insensitivity to short factual answers.
- Compare MiniLM and BGE by the failure count trade-off: BGE usually produces fewer low-similarity-correct cases but more high-similarity-wrong cases.
