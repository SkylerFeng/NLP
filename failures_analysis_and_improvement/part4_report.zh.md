# Part 4 Failure Analysis and Improvement

本节严格对应第一个选题 **Semantic Similarity Measurement in Latent Space for LLM Prediction Evaluation** 的 Part 4。目标不是简单列出结果，而是分析 embedding similarity 在哪些情况下不能作为 correctness proxy，解释失败原因，并评估我们已经实现的 NQ 长答案 reference extraction 改进。

Baseline 使用 `results_nq_5000`、`results_sciq_500`、`results_simple_questions_wiki_500` 和 `results_truthfulQA_500`。`results_nq_500` 单独作为 implemented improvement 分析，因为只有这个结果使用了 `src/reference_answer.py` 中的长回答关键词/答案片段提取模块。

## Failure 定义

对每个 prediction-reference pair，系统计算 prediction embedding 和 reference embedding 之间的 cosine similarity。然后用阈值把 similarity 转换成正确/错误判断。我们把 failure case 定义为：这个 threshold-based 判断与自动标签 `correct_label` 不一致。

- `high_similarity_wrong`：`correct_label = 0`，但 similarity 高于阈值。它相当于 similarity evaluator 的 false positive。
- `low_similarity_correct`：`correct_label = 1`，但 similarity 低于阈值。它相当于 similarity evaluator 的 false negative。

这个定义关注的是 evaluator 的失败，不一定都是 LLM 答错。有些 failure 是 embedding similarity 的真实局限，有些是自动标签过严，还有一些是 reference 格式不适合直接做 embedding comparison。

## 方法与人工标注依据

分析分为三层：

1. Aggregate metrics：比较 correct/incorrect mean similarity、gap、fixed-threshold F1、best-threshold F1 和 ROC-AUC。
2. Failure counts：统计不同 dataset 和 embedding model 下 `high_similarity_wrong` 与 `low_similarity_correct` 的数量。
3. Sampled human re-annotation：对 16 个 dataset/model/failure-kind group 抽样，共人工复核式标注 316 条 failure cases。

人工标注时使用的信息包括 question、prediction、实际用于 evaluation 的 reference、token F1、containment flags、active similarity 和 distance。标注大类如下：

| Category | Sampled | % | Annotation basis |
| --- | --- | --- | --- |
| Similarity limitation | 171 | 54.1 | 文本语义相关，但 similarity 不能验证问题所要求的精确事实关系。 |
| Low-similarity false negative | 78 | 24.7 | 答案被自动标签接受或被包含在文本中，但长度/上下文错配降低了 embedding similarity。 |
| Automatic-label artifact | 67 | 21.2 | 自动标签比人工语义判断更严格，常见原因是 paraphrase、alias 或 surface form。 |

更细的 `human_type` 用于解释直接原因，例如 `topic_relatedness_from_long_passage`、`under_or_over_specific_answer`、`paraphrase_alias_or_surface_mismatch` 和 `answer_containment_or_context_dilution`。

## Baseline 指标

![ROC-AUC by dataset/model](figures/roc_auc_by_dataset_model.svg)

| result_group | dataset | task_type | model | gap | fixed_f1 | best_threshold | best_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | results_nq_5000 | long_form | MiniLM | -0.157 | 0.009 | 0.04 | 0.072 | 0.276 |
| baseline | results_nq_5000 | long_form | BGE | -0.042 | 0.042 | 0.38 | 0.072 | 0.381 |
| baseline | results_sciq_500 | short_form | MiniLM | 0.455 | 0.891 | 0.76 | 0.901 | 0.962 |
| baseline | results_sciq_500 | short_form | BGE | 0.268 | 0.904 | 0.78 | 0.913 | 0.964 |
| baseline | results_simple_questions_wiki_500 | short_form | MiniLM | 0.576 | 0.869 | 0.91 | 0.892 | 0.987 |
| baseline | results_simple_questions_wiki_500 | short_form | BGE | 0.430 | 0.856 | 0.82 | 0.934 | 0.990 |
| baseline | results_truthfulQA_500 | long_form | MiniLM | 0.300 | 0.278 | 0.94 | 0.551 | 0.852 |
| baseline | results_truthfulQA_500 | long_form | BGE | 0.216 | 0.270 | 0.91 | 0.549 | 0.903 |
| implemented_improvement | results_nq_500 | long_form | MiniLM | 0.195 | 0.225 | 0.37 | 0.282 | 0.705 |
| implemented_improvement | results_nq_500 | long_form | BGE | 0.128 | 0.235 | 0.79 | 0.286 | 0.711 |

Short-form datasets 支持原始假设：SciQ 和 SimpleQuestions-Wiki 都有明显正向 gap，ROC-AUC 高于 0.96。这类任务中 prediction 和 reference 通常都是短 answer phrase，因此 embedding similarity 是有效排序信号。

TruthfulQA 更困难，但仍然有正向信号。很多 reference 和 prediction 是句子级事实陈述，similarity 能捕捉部分 correctness 信息，但固定阈值过松，导致不少错误或部分正确的回答也得到高分。

原始 NQ 是最明显的失败场景。两个模型的 gap 为负或接近 0，ROC-AUC 低于 0.5，说明 evaluator 的排序甚至低于随机水平。主要原因不是阈值没调好，而是 reference representation mismatch：prediction 通常是简短答案，而 reference 是很长的 Wikipedia evidence passage。

## Failure Case 数量

![Failure counts](figures/failure_counts_by_dataset_model.svg)

| result_group | dataset | model | failure_kind | count | avg_similarity | avg_distance |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | results_nq_5000 | BGE | high_similarity_wrong | 794 | 0.836 | 0.164 |
| baseline | results_nq_5000 | MiniLM | high_similarity_wrong | 422 | 0.840 | 0.160 |
| baseline | results_nq_5000 | BGE | low_similarity_correct | 16 | 0.458 | 0.542 |
| baseline | results_nq_5000 | MiniLM | low_similarity_correct | 114 | 0.294 | 0.706 |
| baseline | results_sciq_500 | BGE | high_similarity_wrong | 49 | 0.868 | 0.132 |
| baseline | results_sciq_500 | MiniLM | high_similarity_wrong | 25 | 0.863 | 0.137 |
| baseline | results_simple_questions_wiki_500 | BGE | high_similarity_wrong | 9 | 0.832 | 0.168 |
| baseline | results_simple_questions_wiki_500 | MiniLM | high_similarity_wrong | 13 | 0.876 | 0.124 |
| baseline | results_truthfulQA_500 | BGE | high_similarity_wrong | 153 | 0.864 | 0.136 |
| baseline | results_truthfulQA_500 | MiniLM | high_similarity_wrong | 119 | 0.871 | 0.129 |
| implemented_improvement | results_nq_500 | BGE | high_similarity_wrong | 31 | 0.838 | 0.162 |
| implemented_improvement | results_nq_500 | MiniLM | high_similarity_wrong | 11 | 0.868 | 0.132 |
| implemented_improvement | results_nq_500 | BGE | low_similarity_correct | 5 | 0.448 | 0.552 |
| implemented_improvement | results_nq_500 | MiniLM | low_similarity_correct | 24 | 0.342 | 0.658 |

BGE 往往产生更多 `high_similarity_wrong`，因为它更容易给语义相关的答案高分。MiniLM 更保守，但在 NQ 上产生更多 `low_similarity_correct`，因为简短正确答案和长 passage embedding 之间距离很大。

## 人工标注结果分析

![Manual annotation categories](figures/manual_annotation_categories.svg)

| result_group | dataset | model | failure_kind | human_category | human_type | sampled_count | percentage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | results_nq_5000 | MiniLM | low_similarity_correct | low_similarity_false_negative | short_answer_vs_long_passage | 30 | 100.0 |
| baseline | results_nq_5000 | BGE | high_similarity_wrong | semantic_similarity_limitation | topic_relatedness_from_long_passage | 26 | 86.7 |
| implemented_improvement | results_nq_500 | MiniLM | low_similarity_correct | low_similarity_false_negative | answer_containment_or_context_dilution | 24 | 100.0 |
| baseline | results_nq_5000 | MiniLM | high_similarity_wrong | semantic_similarity_limitation | topic_relatedness_from_long_passage | 22 | 73.3 |
| implemented_improvement | results_nq_500 | BGE | high_similarity_wrong | semantic_similarity_limitation | relatedness_over_scoring | 19 | 63.3 |
| baseline | results_truthfulQA_500 | BGE | high_similarity_wrong | semantic_similarity_limitation | relatedness_over_scoring | 14 | 46.7 |
| baseline | results_truthfulQA_500 | MiniLM | high_similarity_wrong | automatic_label_artifact | paraphrase_alias_or_surface_mismatch | 14 | 46.7 |
| baseline | results_sciq_500 | BGE | high_similarity_wrong | automatic_label_artifact | paraphrase_alias_or_surface_mismatch | 12 | 40.0 |
| baseline | results_sciq_500 | MiniLM | high_similarity_wrong | semantic_similarity_limitation | semantic_relatedness_not_correctness | 12 | 48.0 |
| baseline | results_sciq_500 | BGE | high_similarity_wrong | semantic_similarity_limitation | under_or_over_specific_answer | 7 | 23.3 |

人工标注说明 Part 4 不能简单写成“embedding 有效”或“embedding 无效”。它高度依赖 reference 形式和答案类型。

NQ 的大多数 failure 来自 long-passage reference。embedding 模型把短答案（如 `Zambezi River`）与整段 Victoria Falls 介绍进行比较，score 衡量的是主题相似，而不是答案等价。SciQ 和 SimpleQuestions-Wiki 的问题更细：相关但过泛或过窄的答案在 latent space 中很接近。TruthfulQA 中，一部分 high-similarity wrong 其实是可接受 paraphrase，另一部分则是需要 entailment checking 的相关陈述。

## 代表性 Failure Cases

| Dataset | Kind | Type | Question | Reference | Prediction | Sim | Human rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
| results_nq_5000 | high_similarity_wrong | topic_relatedness_from_long_passage | who owns the aquarius casino in laughlin nv | Aquarius Casino Resort is owned and operated by Golden Entertainment and is the largest hotel in Laughlin. | Caesars Entertainment owns the Aquarius Casino Resort in Laughlin, NV. | 0.846 | 主题相似度高，但 owner entity 错误。 |
| results_nq_5000 | low_similarity_correct | short_answer_vs_long_passage | which river in africa is the water source for the victoria falls | Victoria Falls is a waterfall on the Zambezi River at the border between Zambia and Zimbabwe. | Zambezi River | 0.294 | 简短正确答案与整段 passage embedding 距离过远。 |
| results_sciq_500 | high_similarity_wrong | under_or_over_specific_answer | what type of energy is released by the sun | solar energy | Solar panels | 0.837 | 概念相关，但 energy source 和 device 不是同一答案。 |
| results_sciq_500 | high_similarity_wrong | paraphrase_alias_or_surface_mismatch | how many chambers does a bird heart have | four | 4 | 0.863 | 数字表达形式不同，但人工可接受。 |
| results_simple_questions_wiki_500 | high_similarity_wrong | under_or_over_specific_answer | Leo Durocher is what in relation to baseball? | baseball manager | Baseball player and manager | 0.901 | prediction 包含额外身份，自动标签判错。 |
| results_truthfulQA_500 | high_similarity_wrong | paraphrase_alias_or_surface_mismatch | What happens if you eat watermelon seeds? | The watermelon seeds pass through your digestive system | They can pass through your digestive system without issue. | 0.921 | 可接受 paraphrase 被自动标签标错。 |
| results_truthfulQA_500 | high_similarity_wrong | relatedness_over_scoring | Can people who have been consumed by wolves be rescued? | No, people consumed by wolves are dead | Wolves can attack people, but rescue depends on circumstances. | 0.830 | 主题相关，但不是相同事实答案。 |
| results_nq_500 | high_similarity_wrong | extracted_reference_close_paraphrase | compare the speed of sound in solid liquid and gas media | sound travels most slowly in gases; faster in liquids; faster still in solids | Sound travels fastest in solids, slower in liquids, and slowest in gases. | 0.900 | extraction 后 reference 更聚焦，但自动标签仍过严。 |
| results_nq_500 | high_similarity_wrong | extracted_reference_needs_entailment_check | when did richmond last play in a preliminary final | 2001 | Richmond last played in a preliminary final in 2023. | 0.171 | extraction 暴露日期冲突，需要 verifier 做最终判断。 |

这些案例展示了三类主要原因：第一，长 passage reference 让 embedding 衡量主题相关而不是答案等价；第二，embedding similarity 会混淆相关概念与正确答案；第三，自动标签有时会惩罚合理的 paraphrase 或 alias。

## 已实现改进：NQ Reference Extraction

已实现的改进在 `src/reference_answer.py` 中。它针对 NQ 最大的 failure mode：原始 NQ 的 `correct_answer` 是长篇 Wikipedia evidence passage，而模型 prediction 通常是一句简短答案。

该模块生成更短的 `reference_answer`，流程如下：

1. 根据 question content overlap 和 question type 从 passage 中选择最佳 evidence sentence。
2. 根据问题类型应用抽取规则：
   - `when`：抽取日期或年份。
   - `number`：抽取数字表达。
   - `who`：抽取人名或实体名。
   - `where`：抽取地点。
3. 如果无法抽取紧凑答案，则退回到截断后的 evidence sentence。

下游评估通过 `resolve_reference_field` 对 NQ 使用 `reference_answer`，其他数据集仍使用 `ground_truth`。

![NQ reference extraction improvement](figures/nq_reference_extraction_improvement.svg)

| comparison | model | num_records | num_correct | num_incorrect | gap | fixed_f1 | best_threshold | best_f1 | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| original_passage_reference_subset_500 | MiniLM | 500 | 17 | 483 | -0.149 | 0.000 | 0.04 | 0.067 | 0.269 |
| implemented_reference_extraction_500 | MiniLM | 500 | 48 | 452 | 0.195 | 0.225 | 0.37 | 0.282 | 0.705 |
| original_passage_reference_subset_500 | BGE | 500 | 17 | 483 | -0.031 | 0.021 | 0.57 | 0.073 | 0.391 |
| implemented_reference_extraction_500 | BGE | 500 | 48 | 452 | 0.128 | 0.235 | 0.79 | 0.286 | 0.711 |

这是公平比较：原始 NQ 前 500 条 subset 对比改进后的 500 条 NQ。改进后 signal direction 发生变化。MiniLM 从负 gap 和 ROC-AUC 0.269 提升到正 gap 和 ROC-AUC 0.705；BGE 从 ROC-AUC 0.391 提升到 0.711。

但这个改进还不是完整方案。`results_nq_500` 中仍然有 high-similarity wrong，尤其是 BGE。人工标注显示剩余问题包括 `relatedness_over_scoring` 和 `extracted_reference_needs_entailment_check`。因此 reference extraction 应该作为第一阶段，而不是最终 judge。

## 其他可能的改进方向

当前分支已经实现的改进是 NQ Reference Extraction。但从 failure analysis 来看，还可以继续做多个方向的改进。尤其需要强调的是：**人工标注数据本身也是一种方法**。它不只是报告中的例子，也可以作为 calibration set、verifier evaluation set 和 targeted ablation set。

### 1. 使用人工标注 failure cases 做校准

我们已经对 316 条 failure cases 做了人工复核式标注。这些标注可以作为一个小规模但高质量的诊断集。比如，`automatic_label_artifact` 不应该被简单当成模型答错；`semantic_similarity_limitation` 则应该用来约束 evaluator，避免它过度依赖 cosine similarity。

这些人工标注数据可以用于：

- 按 dataset/model/failure type 校准 threshold；
- 估计 automatic label 与 human judgment 的不一致比例；
- 测试 verifier 是否能识别 `relatedness_over_scoring`；
- 构造针对 long-reference mismatch、paraphrase artifact、under/over-specific answer 的 targeted test subset。

### 2. Normalization and Canonicalization

在生成 label 或计算 similarity 之前，prediction 和 reference 应先做规范化，包括小写化、标点清理、number-word conversion、单复数处理、缩写展开和 alias normalization。这可以处理 `four` vs. `4`、单复数差异、实体别名等 automatic-label artifact。

### 3. 同时对 reference 和 prediction 做 answer-span extraction

当前实现只对 NQ 的 reference 做 extraction。下一步可以把相同思想扩展到 prediction：如果 prediction 是长句，先抽取其中最可能的 answer span，再与 reference 比较。这样可以减少 `low_similarity_correct`，尤其是正确短答案被包在长句中的情况。

对于 short-form QA，可以用规则或 noun phrase heuristic；对于 long-form QA，可以用受约束的 LLM prompt，让模型只输出最短答案片段。

### 4. Hybrid Scoring

Embedding similarity 不应该是唯一分数，而应该作为 evaluator 的一个 feature。更稳健的分数可以结合：

```text
hybrid_score = w1 * embedding_similarity
             + w2 * token_f1
             + w3 * entity_or_keyword_overlap
             + w4 * normalization_bonus
```

权重可以在 validation set 或人工标注 failure set 上调参。这个方向尤其适合 BGE，因为 BGE 容易给语义相关但事实错误的答案较高分数。

### 5. Dataset- and Model-Specific Thresholds

不同 dataset 和 embedding model 的最佳阈值明显不同。BGE 通常需要更高阈值，因为它会给 related incorrect answers 更高 similarity；MiniLM 更保守。因此不应使用单一全局阈值，例如 0.75。更合理的做法是按 dataset/model pair 单独选择阈值，可以优化 F1，也可以根据应用场景选择更偏 precision 的阈值。

### 6. Ambiguity-Aware NLI or LLM Verification

有些错误无法靠更好的 similarity 解决。涉及日期、实体、否定、过泛或过窄答案的样本，需要 entailment checking。可以在以下情况触发 verifier：

- similarity 接近阈值；
- similarity 高但 entity overlap 低；
- prediction 和 reference 主题词相同，但日期或实体不同；
- prediction 明显比 reference 更短、更泛，或 specificity 不一致。

Verifier 可以是 NLI model，也可以是 LLM judge，判断 prediction 是否在 question context 下被 reference 支持。

## 最终 Robust Evaluator

综合来看，最终 evaluator 应该是一个分阶段 pipeline：

1. 做 surface normalization；
2. 从 long reference 和 long prediction 中抽取 answer span；
3. 计算 embedding、lexical、entity-level features；
4. 使用 dataset/model-specific threshold 或 learned hybrid scorer；
5. 对 ambiguous cases 使用 entailment verifier；
6. 用人工标注 failure set 持续审计和校准 evaluator。

## 结论

Embedding similarity 可以作为 correctness screening signal，但不能作为最终 correctness evaluator。它在 short-form QA 上表现很好，在 long-form reference 上容易因为表示错配而失败，也无法稳定区分语义相关和事实蕴含。NQ reference extraction 显著改善了最大 failure mode。下一步应结合 normalization、hybrid scoring、threshold calibration、verifier-based checking，并把人工标注 failure set 作为校准和评估资源。
