# semantic-similarity-llm-eval

项目用于研究 **embedding latent space 中的语义相似度能否作为 LLM QA 预测正确性的指标**。它覆盖 short-form QA 和 long-form QA，包含从预测生成、文本规范化、embedding 表示、similarity 计算、threshold-based correctness evaluation、可视化、baseline/ablation 到 failure case analysis 的完整流水线。

**快速结论**
- 本项目的“评估标签”(`correct_label`)由代码生成（参见 `src/correctness_labeling.py`）。
- 原始数据中的参考答案字段为 `correct_answer`（在加载时会映射为 `ground_truth`），参见 `processed_data/*/merged_fb.json` 和 `src/data_loader.py`。
- `config.yaml` 现在默认启用 `project.auto_paths: true`：只需要修改 `data.dataset` 和 `data.sample_size`，输入文件、中间文件、结果目录会自动同步，避免把 SciQ 数据写入 NQ 结果目录。

**目录概览**
- `processed_data/`：教师提供的原始数据（例如 SciQ），包含 `correct_answer`。
- `data/predictions/`：LLM 生成的预测 JSONL（脚本 `scripts/01_generate_predictions.py` 输出）。
- `data/similarity/`：在带有 `correct_label` 的记录上追加 similarity 分数后的输出（脚本 `scripts/02_compute_similarity.py` 输出）。
- `results*/`：评估结果、失败样例和可视化图表。
- `src/`：主要实现模块（embedding / similarity / labeling / evaluate / visualize 等）。
- `scripts/`：一键运行流水线的脚本。

**主要实现点（快速指引）**
- 打标签：`src/correctness_labeling.py` 中的 `label_correctness_for_records` 根据 `prediction` 和 `ground_truth` 计算 `exact_match`、`token_f1`、`contains_ground_truth`，并合成 `correct_label`（关键逻辑在文件内）。
- 计算相似度：`src/compute_similarity.py` 中的 `add_similarity_scores` 使用 embedding 模型对 `prediction` 与 `ground_truth` 分别做 embedding，并在记录上追加 `similarity_{model_name}` 字段；`threshold_similarity` 可将相似度转成预测标签。
- 评估：`src/evaluate.py` 使用 `correct_label` 作为真值，比较 similarity-based classifier 与自动标签的表现，并导出失败用例。
- Baseline / ablation：`scripts/03_evaluate.py` 额外导出 exact match、containment、token F1、entity/token overlap、embedding similarity 和 hybrid score 的对比表。

**支持的数据集**
- Short-form QA：`sciq`、`simple_questions_wiki`
- Long-form QA：`nq`、`truthfulQA`

**如何运行（示例）**
1. 准备虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 编辑 `config.yaml`。通常只需要改：

```yaml
data:
  dataset: sciq              # sciq / simple_questions_wiki / nq / truthfulQA
  sample_size: 5000          # 小测试可设 20，全量设 null
project:
  auto_paths: true
```

启用 `auto_paths` 后，路径会自动解析为类似：

```text
data/predictions/{dataset}_{llm_run_name}_predictions_{sample_size}.jsonl
data/similarity/{dataset}_{llm_run_name}_similarity_{sample_size}.jsonl
results_{dataset}_{sample_size}/
```

3. 生成预测（示例）：

```bash
python scripts/01_generate_predictions.py
```

4. 计算 similarity 并生成 `correct_label`（脚本会先打标再追加 similarity 字段）：

```bash
python scripts/02_compute_similarity.py
```

5. 评估并导出失败用例 / 指标：

```bash
python scripts/03_evaluate.py
```

6. 生成可视化图表：

```bash
python scripts/04_visualize_results.py
```

生成图表包括 similarity distribution、ROC、PR 和 similarity-token-F1 correlation plot。

**配置要点**
- `config.yaml` 控制数据集、embedding 模型、相似度阈值和 I/O 路径。默认评估标签字段为 `correct_label`（`evaluation.label_field`）。
- 如果 `project.auto_paths: true`，脚本会忽略手写的 `prediction.input_file` / `similarity.input_file` / `output.*`，改为根据 `data.dataset` 自动生成一致路径。
- 如果关闭 `auto_paths`，代码会校验 `data.dataset` 与 `prediction.input_file` 是否匹配；不匹配会直接报错，防止 NQ/SciQ 结果混用。
- 原始教师数据样式：`processed_data/{dataset}/merged_fb.json` 的每行 JSON 包含 `question` 和 `correct_answer`，加载时会创建 `ground_truth` 字段（参见 `src/data_loader.py`）。

**输出与故障样例**
- 相似度输出示例字段：`similarity_sentence_transformers_all_MiniLM_L6_v2` 或 `similarity_BAAI_bge_base_en_v1_5`（`/` 和 `-` 会被替换成 `_`）。
- 失败样例会保存到 `results*/failure_cases/`，并且评估表格存于 `results*/tables/evaluation_results.csv`。
- `results*/tables/dataset_statistics.csv`：数据规模、正确标签比例、答案长度等。
- `results*/tables/baseline_ablation_results.csv`：baseline、embedding 和 hybrid method 对比。
- `results*/tables/case_studies.csv`：报告/海报可直接使用的 representative failure cases。
- `results*/tables/run_metadata.json`：记录本次实验实际使用的数据集、输入文件、输出目录、LLM 和 embedding 模型。

**NQ/SciQ 结果混用问题**
旧配置允许 `data.dataset` 写成 NQ 或 Wiki，但 `prediction.input_file` 仍指向 `processed_data/sciq/merged_fb.json`，这会导致 `results_nq` 里出现 SciQ failure cases。现在 `load_config()` 会自动解析并校验路径；真实 NQ 实验应重新生成到 `results_nq_5000/`，不要继续使用旧的 `results_nq/` 作为 Natural Questions 结论来源。

**与 guideline 的对应关系**
- Part 1 Prediction and Representation：`scripts/01_generate_predictions.py`、`scripts/02_compute_similarity.py`
- Part 2 Similarity Analysis：`scripts/03_evaluate.py`、`scripts/04_visualize_results.py`
- Part 3 Empirical Study：`results*/tables/evaluation_results.csv`、`baseline_ablation_results.csv` 和所有 figures
- Part 4 Failure Analysis and Improvement：`results*/failure_cases/`、`case_studies.csv`、`part4_failure_analysis/`
