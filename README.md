# semantic-similarity-llm-eval

项目用于评估短回答类 LLM 预测的“正确性”与预测-参考之间的语义相似度，包含从数据准备、生成预测、计算 embedding/similarity 到评估与可视化的完整流水线。

**快速结论**
- 本项目的“评估标签”(`correct_label`)由代码生成（参见 `src/correctness_labeling.py`）。
- 原始数据中的参考答案字段为 `correct_answer`（在加载时会映射为 `ground_truth`），参见 `processed_data/*/merged_fb.json` 和 `src/data_loader.py`。

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
- 评估：`src/evaluate.py` 使用 `correct_label` 作为真值，比较 similarity-based classifier 与真实标签的表现，并导出失败用例。

**如何运行（示例）**
1. 准备虚拟环境并安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 编辑 `config.yaml`（数据路径 / 模型 / 采样大小 / 输出目录等）。

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

**配置要点**
- `config.yaml` 控制数据集、embedding 模型、相似度阈值和 I/O 路径。默认评估标签字段为 `correct_label`（`evaluation.label_field`）。
- 原始教师数据样式：`processed_data/{dataset}/merged_fb.json` 的每行 JSON 包含 `question` 和 `correct_answer`，加载时会创建 `ground_truth` 字段（参见 `src/data_loader.py`）。

**输出与故障样例**
- 相似度输出示例字段：`similarity_sentence_transformers_all_MiniLM_L6_v2` 或 `similarity_BAAI_bge_base_en_v1_5`（`/` 和 `-` 会被替换成 `_`）。
- 失败样例会保存到 `results*/failure_cases/`，并且评估表格存于 `results*/tables/evaluation_results.csv`。

**扩展建议 / 下步**
- 若想用相似度直接作为 pseudo-label，可以用 `src/compute_similarity.threshold_similarity` 把相似度转换成预测标签，并和 `correct_label` 做对比。
- 我可以为你画出数据流图（processed_data -> predictions -> labeling -> similarity -> evaluation），或者添加一个小的 `Makefile` / README 内的快速运行脚本。

---

如果你希望 README 使用英文或需要我把示例命令改成更适合你的运行环境（GPU / 本地 token 设置 / 代理），告诉我我来更新。
