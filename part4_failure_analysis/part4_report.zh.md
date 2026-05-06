# Part 4 失败案例分析

英文版见：[`part4_report.md`](part4_report.md)。

## 1. 目标与定义
这里我们把 failure case 定义为：基于 similarity threshold 得到的正确/错误判断，与自动生成的 `correct_label` 不一致。

- `high_similarity_wrong`：`correct_label = 0`，但 similarity 很高。
- `low_similarity_correct`：`correct_label = 1`，但 similarity 很低。

需要注意的是，这些 failure case 是 similarity-as-classifier pipeline 的失败，不一定都是 LLM 答错。有些样例反而暴露了自动标签本身的问题。

## 2. 使用的数据与结果
我们分析了四个结果目录：`results_nq`、`results_sciq_5000`、`results_truthfulQA_500` 和 `results_wiki`。每个目录使用 evaluation table、failure-case JSONL 文件，以及 `manual_annotation_sample.csv` 中的人工标注样本。

注意：人工查看后，`results_nq` 中的样例看起来更像 short-form science QA，而不像真正的 Natural Questions long-form QA。因此这里把它作为一个额外结果目录分析，但不单独用它支撑关于 NQ long-form 的强结论。

## 3. Embedding Model 对比
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

MiniLM 在四个结果目录中都有更大的 correct-vs-incorrect similarity gap，大约为 0.45-0.46。BGE 的 ROC-AUC 与 MiniLM 接近，但它给错误答案的平均 similarity 也更高，因此最佳阈值更高。实际使用时，BGE 更 recall-friendly，但也更不保守。

## 4. 全量 Failure Case 数量
![Failure case counts](figures/failure_case_counts.svg)

| Model | High-Sim Wrong | Low-Sim Correct | Total |
| --- | --- | --- | --- |
| BGE | 1505 | 0 | 1505 |
| MiniLM | 771 | 114 | 885 |

BGE 在这些输出中没有 low-similarity-correct cases，但 high-similarity-wrong 数量明显更多。MiniLM 的 high-similarity false positives 更少，但当正确短答案嵌在更长 prediction 中时，它更容易给出较低 similarity。

## 5. Human Annotation 分析
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

人工标注样本把 failure 分成三大类：

- `automatic_label_artifact`：模型答案可能是可接受的，但自动正确性标签太严格。
- `semantic_similarity_limitation`：prediction 和 reference 语义相关，但不一定事实正确。
- `low_similarity_false_negative`：prediction 包含或表达了正确答案，但 embedding similarity 偏低。

## 6. 主要 Failure Type
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

常见的 high-similarity-wrong 包括单复数变化、数字等价、同义改写和答案过泛/过细。真正的 similarity 局限在于：embedding similarity 经常衡量的是“语义相关”，而不是“事实正确”。例如，一个更泛的答案可能和标准答案很接近，但缺少关键限定词。

low-similarity-correct 主要出现在 MiniLM 中。这类样例通常是在较长 prediction 中包含了正确短答案，额外上下文稀释了整体句向量。

## 7. 结论
- MiniLM 更保守，对正确和错误答案的分离更明显。
- BGE 的语义匹配能力更强，但更容易把相关但不完整的答案打高分。
- 很多表面上的错误其实是自动标签缺陷，而不是 LLM 真正答错。
- 单一 cosine similarity threshold 可以作为筛选信号，但不足以作为最终 correctness evaluator。

## 8. 改进方案
更稳健的 evaluator 应该结合多种检查：

1. 更强 normalization：大小写、标点/连字符、词形还原、数字词转换。
2. Answer containment 和 answer extraction，尤其是 prediction 比 reference 更长时。
3. Entity 或 keyword overlap，用来检查关键事实单元。
4. 对长答案使用 sentence-level similarity。
5. 对模糊的 high-similarity cases 使用 NLI 或 LLM judge 做事实一致性验证。

一个可实现的 hybrid pipeline 是：先做 normalization 和 exact/containment 检查；再使用 dataset/model-specific threshold 的 embedding similarity；最后把 ambiguous cases 交给 verifier。

## 9. 可复现方式
在项目根目录运行：

```bash
python part4_failure_analysis/scripts/analyze_part4_failures.py
python part4_failure_analysis/scripts/build_final_part4_report.py
```

关键统计表保存在 `part4_failure_analysis/summary_tables/`；每个数据集的报告和标注文件保存在 `part4_failure_analysis/datasets/`。
