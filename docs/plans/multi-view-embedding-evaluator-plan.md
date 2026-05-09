---
title: Multi-View Embedding Evaluator Plan
status: active
created: 2026-05-09
scope: NQ-focused QA correctness evaluation improvement
---

# Multi-View Embedding Evaluator Plan

## Problem Frame

The project studies whether sentence embedding similarity can act as a proxy for LLM QA prediction correctness. Current NQ results show that reference extraction is necessary but not sufficient:

- Raw NQ passage reference performs poorly: MiniLM ROC-AUC 0.269, BGE ROC-AUC 0.391.
- Extracted `reference_answer` improves signal: MiniLM ROC-AUC 0.705, BGE ROC-AUC 0.711.
- Fixed F1 remains low: MiniLM 0.225, BGE 0.235.
- Best F1 remains low: roughly 0.28 for embedding-only, roughly 0.36 for BGE hybrid.
- NQ 500 is highly imbalanced: 48 positive labels and 452 negative labels.

The remaining failure mode is not just threshold selection. A single sentence embedding cosine score mixes topic relatedness, factual equivalence, reference extraction quality, and automatic-label artifacts. The next improvement should keep the project inside the existing sentence embedding framework, but extend it into a multi-view and multi-granularity evaluator.

The goal is not to change the ground-truth answer itself. The goal is to make the prediction-reference comparison more answer-focused, then test whether improved embedding-based similarity is a more reliable proxy for QA prediction correctness.

## Scope

Primary scope:

- Dataset: `results_nq_500`.
- Models: `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-base-en-v1.5`.
- Inputs: existing prediction and similarity files under `data/` and `results_nq_500/`.
- Audit data: existing `failures_analysis_and_improvement/summary_tables/manual_annotation_sample.csv`, used for qualitative failure analysis unless a binary human-label contract is added.

Non-goals:

- Do not train a new large embedding model.
- Do not require a large new human-labeled dataset.
- Do not treat a single global threshold as the final method.
- Do not assume `correct_label` is a fully reliable human correctness label.
- Do not remove existing baseline fields; add v2 fields so previous results remain reproducible.
- Do not use a verifier or LLM judge as part of this plan. The project should remain an embedding-similarity evaluation study.

## Research Thesis

Extend the evaluator from single-vector sentence similarity into:

```text
multi-view similarity =
  sentence-level semantic relatedness
+ span-level answer alignment
+ entity/number/date factual-unit agreement
+ question-type-aware reporting and guarded calibration
```

This preserves the academic focus on embedding latent space while addressing known QA correctness failures without large-scale retraining.

## Current Failure Mapping

| Failure type | Current evidence | Root cause | Needed fix |
| --- | --- | --- | --- |
| `low_similarity_correct` | MiniLM 24, BGE 5 in `results_nq_500` | Short reference answer is diluted by full prediction sentence embedding | Prediction answer-span extraction and span-level max similarity |
| `high_similarity_wrong` with numeric/date/entity mismatch | BGE 31, MiniLM 11 | Embedding captures topic but not exact factual units | Entity/number/date-aware features and conflict penalties |
| Reference extraction artifact | Pronoun refs such as `He`, `It`, `This`; one-token suspicious refs | Heuristic extraction accepts non-informative spans | Reference validation and fallback extraction |
| Automatic-label artifact | Correct paraphrases labeled 0 by exact/containment/token-F1 | `correct_label` is too strict for paraphrase and ordering | Human-audited subset reporting and label-change audit |
| Threshold instability | MiniLM best threshold 0.37, BGE best threshold 0.79 | Different embedding score distributions | Dataset/model/question-type calibration |

## Architecture Decisions

1. Keep sentence embedding as one feature, not the sole judge.

   Rationale: NQ reference extraction proves embedding similarity contains useful signal, but high-similarity wrong cases show sentence-level cosine cannot validate factual equivalence.

2. Add new v2 fields instead of mutating existing baseline fields.

   Rationale: Current tables and Part 4 report depend on existing fields. Adding v2 fields preserves reproducibility.

3. Separate automatic-label agreement from human correctness.

   Rationale: `correct_label` is produced by `src/correctness_labeling.py`, so full-dataset metrics are agreement with the automatic evaluator. Human-audited subset metrics should be reported separately.

4. Treat every improvement as an ablation, not an automatic replacement.

   Rationale: The course project asks whether embedding similarity can indicate correctness. Each new feature must prove its contribution against the previous stage before becoming part of the final method.

5. Keep the baseline and v2 artifacts separate.

   Rationale: `scripts/02_compute_similarity.py` currently uses one reference field for both `correct_label` and embedding similarity. V2 reference changes can otherwise silently change both the target label and the score, making ablations incomparable.

## Artifact Contract

Do not overwrite existing fields:

- `reference_answer`
- `correct_label`
- `similarity_{model}`
- `hybrid_{model}`

Add v2 fields beside them:

- `question_type_v2`
- `reference_answer_v2`
- `reference_answer_valid`
- `reference_validation_reason`
- `reference_answer_source_v2`
- `prediction_answer_span`
- `prediction_answer_span_source`
- `correct_label_v2`, only if explicitly evaluating a changed automatic label
- `similarity_v2_{model}` for sentence similarity against `reference_answer_v2`
- `span_max_similarity_{model}`
- `span_topk_mean_similarity_{model}`
- `multi_view_score_{model}`

Required comparison rule:

- Baseline rows use original `correct_label` and original `similarity_{model}`.
- V2 score rows should first be evaluated against frozen original `correct_label` for comparability.
- If `correct_label_v2` is produced, report it in a separate label-change audit and do not mix it with baseline metrics.
- Every output table must state which label field and score field it uses.

## Implementation Units

### Unit 0: Experiment Contract and Ablation Harness

Files:

- `scripts/02_compute_similarity.py`
- `scripts/03_evaluate.py`
- `src/evaluate.py`
- `tests/test_evaluate.py`

Add:

- A stable ablation table that can append one row per improvement stage.
- Field-level metadata for `label_field`, `score_field`, `reference_field`, and `method`.
- A label-change audit if `correct_label_v2` is enabled.

Targets:

- Reproducible baseline.
- Comparable one-improvement-at-a-time experiments.

Tests:

- Existing baseline rows are unchanged when v2 features are disabled.
- Missing v2 fields do not break baseline evaluation.
- Each ablation row records the exact label and score fields used.

Experiment gate:

- Re-run the current NQ 500 baseline and confirm `evaluation_results.csv` and `baseline_ablation_results.csv` are reproducible before starting Unit 1.

### Unit 1: Reference Answer Validation

Files:

- `src/reference_answer.py`
- `tests/test_reference_answer.py`

Add:

- shared `question_type_v2(question)` or a documented wrapper around the existing `question_type(question)`.
- `validate_reference_answer(answer, question, evidence)`.
- `reference_answer_v2`.
- `reference_answer_valid`.
- `reference_validation_reason`.
- `reference_answer_source_v2`.

Rules:

- Reject pronouns and determiners: `he`, `she`, `it`, `they`, `this`, `that`, `his`, `her`, `its`, `their`.
- Reject one-token spans that are months or generic title fragments unless question type is explicitly date-like.
- Reject malformed numeric spans such as `000` when evidence indicates a larger number like `5,000`.
- Flag long evidence fallback spans over a configurable token limit.
- If a candidate is invalid, continue to the next candidate before falling back to evidence sentence.

Targets:

- Reference extraction artifact.
- Automatic-label artifact caused by bad reference.

Why this should work:

Bad references corrupt both embedding comparison and automatic labels. Removing `He/It/May/000` style references prevents downstream features from optimizing against invalid targets.

Tests:

- `test_rejects_pronoun_reference`.
- `test_rejects_month_as_location_reference`.
- `test_rejects_malformed_year_fragment`.
- `test_marks_long_evidence_sentence_reference`.
- Existing reference extraction tests must continue passing.

Experiment gate:

- Write `reference_quality_report.csv`.
- Compare pronoun reference count, one-token suspicious reference count, long evidence fallback count, invalid reference count, and reference source distribution.
- Keep the improvement if reference quality improves and PR-AUC / best F1 do not materially regress against the frozen original `correct_label`.

### Unit 2: Prediction Answer-Span Extraction

Files:

- New: `src/answer_span.py`
- shared question-type helper from Unit 1
- `tests/test_answer_span.py`

Add:

- `extract_prediction_answer_span(question, prediction)`.
- `prediction_answer_span`.
- `prediction_answer_span_source`.

Initial question types:

- `who`
- `when`
- `where`
- `number`
- `yes_no`
- `list`
- `comparison`
- `definition`
- `general`

Extraction behavior:

- `when`: extract full dates, years, date ranges, relative date phrases.
- `number`: extract counts, fractions, percentages, number words.
- `who`: extract person/org-like noun phrases.
- `where`: extract locations and prepositional location phrases.
- `yes_no`: extract polarity plus supporting answer phrase.
- `list`: extract coordinated entities/items.
- `comparison`: extract ordered relation or compared entities.
- `definition/general`: retain concise predicate phrase or fallback to prediction.

Targets:

- `low_similarity_correct`.
- Short answer inside full-sentence prediction.

Why this should work:

MiniLM low-similarity correct cases often have the correct reference span inside a longer prediction. Comparing `356` with a full sentence is weaker than comparing `356` with `356 BCE`.

Tests:

- `reference=356`, prediction sentence containing `356 BCE` extracts `356 BCE`.
- `how many` extracts `16 teams` or `10 teams`.
- `who` extracts named person rather than sentence subject pronoun.
- `yes/no` preserves polarity.
- Empty or uncertain extraction falls back safely.

Experiment gate:

- Add an ablation row for `reference_validation + prediction_span`.
- Track extraction source distribution and fallback rate.
- Keep the improvement if `low_similarity_correct` decreases or PR-AUC improves without a clear increase in `high_similarity_wrong`.

### Unit 3: Span-Level Embedding Similarity

Files:

- New: `src/multi_view_similarity.py`
- `src/compute_similarity.py`
- `scripts/02_compute_similarity.py`
- `tests/test_multi_view_similarity.py`

Add features per embedding model:

- `span_max_similarity_{model}`.
- `span_topk_mean_similarity_{model}`.
- `reference_to_prediction_span_similarity_{model}`.

Computation:

```text
sim_span_max = max_i cosine(emb(reference_answer_v2), emb(prediction_span_i))
sim_topk = mean(top_k cosine(emb(reference_spans), emb(prediction_spans)))
```

Prediction span candidates:

- `prediction_answer_span`.
- Numbers/dates/entities.
- Noun chunks approximated by regex.
- Short n-grams up to a small configurable limit.

Reference span candidates:

- `reference_answer_v2`.
- Numbers/dates/entities from `reference_answer_v2`.
- Short n-grams from `reference_answer_v2` only when the answer is longer than the configured span length.

Targets:

- `low_similarity_correct`.
- Partial paraphrase where local span aligns even if whole sentence does not.

Why this should work:

This converts sentence embedding into multi-granularity representation. It reduces the penalty from full-sentence context when the relevant answer span is short.

Tests:

- `span_max_similarity` is at least sentence similarity when exact short answer appears in prediction.
- Empty span list falls back to sentence-level similarity.
- Top-k aggregation is deterministic.

Negative-control checks:

- Shuffled references should not improve.
- Same-topic wrong-answer pairs should not receive systematic score inflation.
- Increasing candidate count should not sharply increase `high_similarity_wrong`.

Experiment gate:

- Add an ablation row for `span_similarity`.
- Keep the improvement if `low_similarity_correct` decreases or PR-AUC improves without reversing the Unit 4 conflict gains.

### Unit 4: Factual Unit Extraction and Conflict Detection

Files:

- New: `src/factual_units.py`
- `src/entity_overlap.py`
- `tests/test_factual_units.py`

Add:

- `extract_numbers(text)`.
- `extract_dates(text)`.
- `extract_entity_like_spans(text)`.
- `compare_factual_units(reference, prediction)`.

Fields:

- `number_match`.
- `number_conflict`.
- `date_match`.
- `date_conflict`.
- `entity_match`.
- `entity_conflict`.
- `list_item_f1`.
- `specificity_flag`.
- `factual_conflict_penalty`.

Targets:

- High-similarity wrong caused by numeric/date/entity mismatch.
- BGE relatedness over-scoring.

Why this should work:

Embedding models often rate `916,542` and `1,083,460` population answers as semantically close because surrounding topic words match. Explicit conflict flags make this mismatch visible to the scorer.

Tests:

- Different population numbers produce `number_conflict=1`.
- Same year with different surface form produces match.
- Date ranges preserve granularity.
- Entity aliases from parenthetical definitions can match when obvious.

Experiment gate:

- Add ablation rows for `factual_unit_features` and `factual_conflict_penalty`.
- Keep the improvement if BGE `high_similarity_wrong` decreases without collapsing recall or best F1.

Implementation note, 2026-05-09:

- Unit 4 is implemented in run `results_nq_500/runs/unit4_check`.
- BGE `factual_conflict_adjusted_multi_view_score` improved PR-AUC from `0.4928` to `0.5657`, best F1 from `0.5000` to `0.5682`, and reduced high-similarity-wrong from `29` to `14` compared with the Unit 3 conservative multi-view score.
- BGE `factual_conflict_adjusted_span_max_similarity` improved fixed-threshold F1 from `0.4503` to `0.5419` and reduced high-similarity-wrong from `56` to `25` compared with Unit 3 span-max, while keeping PR-AUC roughly stable/improved from `0.8117` to `0.8181`.

Unit 4 rule repairs to consider before or during Unit 6:

- Normalize duplicated entity aliases such as `Emma Stone Stone` before entity conflict comparison.
- Treat obvious alias/substring matches as non-conflicts when one entity is a clean contraction of the other, for example `Emma Stone Stone` versus `Emma Stone`.
- Improve date granularity logic so acceptable containment cases such as `1775-1783` versus `1783`, `19 September 2017` versus `2017`, and `1955 to 1975` versus `1955` are flagged as partial date overlap rather than blunt conflict when the question asks for a broad period.
- Add ordinal and word-number coverage for `first`, `second`, `third`, etc. where they act as factual units.
- Filter sentence-initial discourse words and false entity spans such as `However`, `Sound`, `You`, `The`, and malformed fragments from entity extraction.
- Add a separate `partial_factual_overlap` or reason field so Unit 6 can distinguish strict contradiction from acceptable under/over-specificity.

### Unit 5: Entity/Number-Aware Embedding View

Files:

- `src/multi_view_similarity.py`
- `src/factual_units.py`
- `scripts/02_compute_similarity.py`

Add:

- Entity span embedding similarity.
- Number/date context embedding similarity.
- Combined factual-view score, enabled only after the Unit 4 and Unit 3 gates show remaining high-similarity wrong cases that symbolic conflict flags do not handle.

Formula:

```text
factual_view_similarity =
  weighted_mean(entity_span_similarity, date_span_similarity, number_span_similarity)
```

Empty-path rule:

- Add `entity_view_present`, `date_view_present`, and `number_view_present`.
- Compute the weighted mean over present view types only.
- If no factual views are present, set `factual_view_similarity` to a documented neutral value and set factual-view presence flags to false.

Targets:

- High-similarity wrong.
- Under/over-specific answer cases.

Why this should work:

It keeps factual units in embedding space instead of only symbolic space. This supports the academic claim of knowledge-enhanced or entity-aware embedding without training a new model.

Validation:

- Compare against `entity_overlap` and conflict flags.
- Report whether factual-view similarity improves PR-AUC over sentence similarity.

Experiment gate:

- Build Unit 5 only if Unit 3 + Unit 4 still leave enough high-similarity wrong cases to justify the extra embedding pass.
- Keep the improvement only if it improves PR-AUC or reduces `high_similarity_wrong` beyond conflict flags alone.

Gate decision, 2026-05-09:

- Gate status: failed / deferred. Do not implement Unit 5 before Unit 6 unless new evidence appears.
- After Unit 4, BGE adjusted multi-view leaves only `14` high-similarity-wrong cases at the fixed high threshold. Among those, only `2` retain entity conflicts and none retain number/date conflicts, so the remaining failures are not primarily unresolved symbolic factual-unit mismatches.
- The residual cases are dominated by automatic-label strictness, reference extraction artifacts, or entailment/specificity ambiguity: examples include `Nigeria` / `United Nations`, `JAR` definition paraphrases, `Emma Stone Stone` alias cleanup, and date-surface variants such as `22 July 1947` versus `July 22, 1947`.
- A new factual-unit embedding pass is likely to raise scores for close paraphrases and alias cases that are already high, while adding compute and ablation complexity. The better next step is Unit 6 with the reduced score path that omits `factual_view_similarity`, plus the Unit 4 rule repairs listed above.

### Unit 6: Multi-View Hybrid Scoring

Files:

- `src/evaluate.py`
- `scripts/03_evaluate.py`
- `tests/test_evaluate.py`

Add:

```text
multi_view_score =
  w1 * sentence_similarity
+ w2 * span_max_similarity
+ w3 * factual_view_similarity_or_neutral
+ w4 * entity_or_token_overlap
- w5 * factual_conflict_penalty
```

Start with fixed weights as one ablation, not as the final method:

- `w1 = 0.35`
- `w2 = 0.30`
- `w3 = 0.15`
- `w4 = 0.15`
- `w5 = 0.25`

If Unit 5 is skipped, use a reduced score that omits `w3` and renormalizes the remaining positive weights.

Do not calibrate weights on `manual_annotation_sample.csv` unless a representative binary `human_correct_label` contract is added. Use NQ 500 cross-validation against the frozen automatic label for tuning, then report manual audit category changes separately.

Targets:

- BGE high-similarity wrong.
- MiniLM low-similarity correct.
- Fixed-F1 weakness.

Why this should work:

Sentence similarity supplies broad semantic recall; span and factual views restore answer-level precision; conflict penalty prevents high topic similarity from masking contradiction.

Tests:

- Conflict penalty lowers score for same-topic different-number examples.
- Span score raises score for short-answer containment examples.
- Score remains in a documented range.
- Missing optional feature values do not produce NaN.

Experiment gate:

- Add `multi_view_ablation_results.csv`.
- Compare fixed-weight, reduced-score, and sensitivity-analysis rows.
- Keep the combined score only if it improves over the best single-feature ablation and over the existing BGE hybrid baseline.

Implementation note, 2026-05-09:

- Unit 6 is implemented in run `results_nq_500/runs/unit6_check`.
- Unit 5 remains skipped, so Unit 6 uses reduced positive weights over sentence similarity, span max similarity, and entity/token overlap, then subtracts the factual conflict penalty.
- The fixed reduced score improves over the existing BGE hybrid baseline: PR-AUC `0.3468 -> 0.6156`, best F1 `0.3585 -> 0.5618`, and fixed F1 `0.2121 -> 0.5570`.
- The sensitivity row `span_ranked` (`0.05 sentence + 0.95 span - 0.15 conflict_penalty`) is the strongest Unit 6 ranking setting. For BGE it improves PR-AUC over the best Unit 4 single-feature row from `0.8181` to `0.8454`, with best F1 tied at `0.8125`.
- The sensitivity row `span_guarded` (`0.10 sentence + 0.80 span + 0.10 overlap - 0.10 conflict_penalty`) is the strongest Unit 6 fixed-threshold setting. For BGE it improves fixed-threshold F1 over the best Unit 4 single-feature row from `0.5419` to `0.7009`.
- For MiniLM, `span_guarded` improves fixed-threshold F1 over the best Unit 4 span row from `0.6341` to `0.7255`, and `span_ranked` improves PR-AUC from `0.7950` to `0.8119`.

### Unit 7: Question-Type Reporting and Guarded Calibration

Files:

- `scripts/03_evaluate.py`
- `src/evaluate.py`

Add:

- Per-dataset/model/question-type metrics.
- Per-dataset/model/question-type thresholds only for buckets with enough support.
- Optional z-score calibrated similarity:

```text
z = (score - mean_qtype_model) / std_qtype_model
```

Report:

- Global threshold metrics.
- Dataset/model-specific threshold metrics.
- Question-type-specific threshold metrics.
- Skipped or inherited threshold buckets.

Targets:

- Threshold instability.
- Class imbalance.

Why this should work:

`when`, `who`, `number`, and `definition` questions have different similarity score distributions. MiniLM and BGE also use different score scales.

Validation:

- Report PR-AUC and F1 by question type.
- Avoid selecting thresholds only on the final evaluation split; use cross-validation where possible.
- Minimum bucket rule before per-type thresholding: `num_examples >= 50`, `num_positive >= 10`, `num_negative >= 10`, and nonzero score standard deviation.

Experiment gate:

- Always keep question-type reporting.
- Keep per-question-type thresholds only if they improve held-out or cross-validated metrics and do not rely on undersized buckets.

Implementation note, 2026-05-09:

- Unit 7 is implemented in run `results_nq_500/runs/unit7_check`.
- `question_type_metrics.csv` now reports dataset-global rows, question-type rows under the global fixed threshold, and guarded threshold rows with explicit `question_type_cv` or `inherited_global` status.
- Default calibration guards are `num_examples >= 50`, `num_positive >= 10`, `num_negative >= 10`, nonzero score standard deviation, and 5-fold stratified cross-validation.
- NQ 500 supports calibrated question-type thresholds for `when`, `where`, and `who`. `definition`, `list`, `number`, and `yes_no` are skipped because `num_examples < 50`; `general` is skipped because `num_positive < 10`.
- For BGE `span_ranked`, question-type CV improves fixed-threshold F1 for `when` from `0.5128` to `0.7619`, `where` from `0.6316` to `0.8148`, and `who` from `0.6667` to `0.7805`. Dataset-global BGE `span_ranked` remains strongest for ranking with PR-AUC `0.8454`.
- For BGE `span_guarded`, question-type CV improves `when` fixed-threshold F1 from `0.6897` to `0.7619`, but lowers `where` from `0.7586` to `0.6923` and `who` from `0.8500` to `0.7805`, so per-type calibration should be reported as guarded analysis rather than replacing the global threshold wholesale.

## Experiment Matrix

Run these ablations in order:

1. Current sentence embedding baseline.
2. Reference validation.
3. Reference validation + prediction answer-span extraction.
4. Factual unit features.
5. Factual conflict penalty.
6. Span-level max/top-k similarity.
7. Entity/number-aware embedding view, only if gated in.
8. Multi-view hybrid score.
9. Question-type reporting.
10. Guarded question-type calibration, only for supported buckets.

For each ablation, report metrics for:

- MiniLM.
- BGE.
- NQ 500 automatic labels.
- Human-audited subset where a binary label contract exists.
- Manual audit category changes where only qualitative annotations exist.

## Metrics

Full dataset metrics:

- ROC-AUC.
- PR-AUC.
- fixed precision/recall/F1.
- best-threshold precision/recall/F1.
- `high_similarity_wrong` count.
- `low_similarity_correct` count.
- score gap between `correct_label=1` and `correct_label=0`.

Reference quality metrics:

- Pronoun reference count.
- One-token suspicious reference count.
- Long evidence fallback count.
- Invalid reference count.
- Reference source distribution.

Human-audited subset metrics:

- Accuracy, precision, recall, F1 against human label only if a `human_correct_label` field or deterministic mapping exists.
- Agreement with automatic label.
- Failure category count changes.

Important interpretation rule:

Full-dataset results measure agreement with the automatic evaluator. Human-audited subset results are required for claims about true QA correctness.

## Required Output Tables

Extend `scripts/03_evaluate.py` to write:

- `multi_view_ablation_results.csv`.
- `question_type_metrics.csv`.
- `reference_quality_report.csv`.
- `enriched_failure_cases.csv`.
- `label_change_audit.csv`, if `correct_label_v2` is produced.

Existing tables should remain:

- `evaluation_results.csv`.
- `baseline_ablation_results.csv`.
- `dataset_statistics.csv`.
- `case_studies.csv`.

## Testing Strategy

Unit tests:

- `tests/test_reference_answer.py` for validation and fallback extraction.
- `tests/test_answer_span.py` for prediction span extraction.
- `tests/test_factual_units.py` for number/date/entity extraction and conflict flags.
- `tests/test_multi_view_similarity.py` for span aggregation behavior.
- `tests/test_evaluate.py` for hybrid scoring and threshold calibration helpers.

Characterization checks:

- Current baseline tables remain reproducible before enabling v2 scoring.
- Existing NQ reference extraction tests continue passing.
- Each ablation row records the exact label field, score field, and reference field.
- Optional or skipped features use documented fallback values and never produce NaN.

Regression scenarios:

- `He` should not be accepted as a valid person answer.
- `It` should not be accepted as a valid singer answer.
- `356 BCE` should align with `356`.
- `916,542` vs `1,083,460` should trigger number conflict.
- `16 teams` vs `10 teams` should trigger number/list conflict.
- Speed-of-sound ordering should not be dismissed only because token order differs.

## Sequencing

1. Implement Unit 0 experiment contract and verify the current baseline is reproducible.
2. Implement Unit 1 reference validation and quality report, then run tests and the NQ 500 ablation.
3. Implement Unit 2 prediction answer-span extraction, then run tests and the cumulative ablation.
4. Implement Unit 4 factual unit extraction and conflict flags, then run tests and the cumulative ablation.
5. Implement Unit 3 span-level embedding aggregation, then run tests, negative controls, and the cumulative ablation.
6. Decide whether Unit 5 is justified by remaining failures. If yes, implement it and run its gated ablation.
7. Implement Unit 6 multi-view score and sensitivity analysis.
8. Implement Unit 7 question-type reporting, then add guarded calibration only for supported buckets.
9. Update Part 4 analysis script/report to include new ablations and final interpretation.

## Risks

- Automatic label circularity: lexical/entity features may look strong because `correct_label` itself uses lexical containment. Mitigation: always report human-audited subset separately.
- V2 reference changes can redefine both labels and scores. Mitigation: freeze baseline fields, add v2 fields beside them, and report label-change audit separately.
- Heuristic span extraction may introduce new errors. Mitigation: keep original fields, add validation reasons, and report invalid/uncertain bucket.
- Factual conflict rules may over-penalize acceptable granularity differences. Mitigation: preserve uncertainty state and report affected examples in `enriched_failure_cases.csv`.
- Calibration may overfit NQ 500. Mitigation: start with reporting, require minimum support, and use NQ 5000 or cross-validation when compute allows.
- Max-span similarity may inflate false positives. Mitigation: run shuffled-reference, same-topic wrong-answer, and candidate-count negative controls.

## Success Criteria

Minimum success:

- Baseline rows remain reproducible with v2 disabled.
- Reference quality report shows pronoun reference count reduced or isolated.
- At least one cumulative ablation improves PR-AUC or best F1 over sentence embedding baseline.
- `high_similarity_wrong` decreases for BGE or `low_similarity_correct` decreases for MiniLM without a compensating regression in the other failure type.
- Any claim about true QA correctness is backed by human-audited labels or clearly framed as qualitative audit evidence.

Strong success:

- Multi-view hybrid score improves over existing BGE hybrid best F1.
- Human-audited subset shows fewer semantic-relatedness false positives.
- Report clearly distinguishes automatic-label agreement from true correctness.
- Question-type reporting identifies where similarity succeeds and fails without relying on undersized threshold buckets.

## Research Contribution Statement

The final method can be described as:

> We extend sentence-embedding-based QA correctness estimation from single-vector semantic similarity to a multi-view, multi-granularity evaluator that combines sentence-level semantic relatedness, answer-span embedding alignment, entity/number/date-aware factual units, and question-type-aware reporting with guarded calibration, without retraining embedding models.
