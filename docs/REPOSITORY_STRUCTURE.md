# 仓库结构说明

本文档面向需要快速接手该仓库的人，重点说明目录职责、主流水线、
核心模块关系、数据产物位置，以及常见修改应该从哪里入手。

## 1. 项目定位

本仓库是一个 Python 实验型 NLP 项目，用于研究：

> embedding latent space 中的语义相似度，能否作为 LLM QA 预测正确性的
> 评估指标。

项目覆盖 short-form QA 与 long-form QA，主链路包括：

1. 从教师提供的数据集中读取 `question` 与 `correct_answer`。
2. 调用 LLM 生成 `prediction`。
3. 生成自动正确性标签 `correct_label`。
4. 用 sentence-transformers 模型计算预测答案与参考答案的 embedding
   similarity。
5. 把 similarity 当作二分类分数，评估其判断回答正确性的能力。
6. 导出表格、图像、失败案例，并对失败模式做 Part 4 分析。

## 2. 总体目录树

```text
NLP/
  README.md                         # 项目说明与运行入口
  config.yaml                       # 数据集、模型、路径、输出配置
  requirements.txt                  # Python 依赖
  data/
    raw/                            # 教师提供的原始 QA 数据
    interim/                        # 预测与 similarity 中间 JSONL
  outputs/
    experiments/results_*/          # 各数据集实验结果
    analysis/failures_analysis_and_improvement/
                                      # Part 4 失败分析与改进产物
  docs/                             # 结构说明、上手文档、计划与参考 PDF
  scripts/
    pipeline/                       # 主流水线脚本
    analysis/                       # 分析/报告脚本
  src/                              # 可复用实现模块
  tests/                            # 单元测试
```

当前工作区没有发现 `report/` 目录；报告相关 Markdown 产物主要位于
`outputs/analysis/failures_analysis_and_improvement/`。

## 3. 核心执行链路

该项目不是 Web 服务，也没有数据库。它的主要入口是 `config.yaml` 和
`scripts/` 下的四个流水线脚本。

```text
data/raw/{dataset}/merged_fb.json
  |
  v
scripts/pipeline/01_generate_predictions.py
  |
  v
data/interim/predictions/*_predictions_*.jsonl
  |
  v
scripts/pipeline/02_compute_similarity.py
  |
  v
data/interim/similarity/*_similarity_*.jsonl
或 outputs/experiments/results_*/runs/{run_id}/similarity/*.jsonl
  |
  v
scripts/pipeline/03_evaluate.py
  |
  v
outputs/experiments/results_*/tables/
outputs/experiments/results_*/failure_cases/
  |
  v
scripts/pipeline/04_visualize_results.py
  |
  v
outputs/experiments/results_*/figures/
```

Part 4 分析另有一条汇总链路：

```text
outputs/experiments/results_*/
  |
  v
scripts/analysis/analyze_part4_strict.py
  |
  v
outputs/analysis/failures_analysis_and_improvement/
  summary_tables/
  figures/
  part4_report.md
  part4_report.zh.md
```

## 4. 顶层文件职责

| 路径 | 职责 |
| --- | --- |
| `README.md` | 项目背景、运行方式、配置说明和输出解释。 |
| `docs/ONBOARDING.md` | 面向新贡献者的项目导览。 |
| `docs/REPOSITORY_STRUCTURE.md` | 仓库结构与模块关系说明。 |
| `config.yaml` | 控制数据集、LLM、embedding 模型、阈值和 I/O 路径。 |
| `requirements.txt` | 声明运行实验所需 Python 依赖。 |
| `.gitignore` | 忽略 `.venv/`、`__pycache__/`、`.pytest_cache/` 和 `*.pyc`。 |

`config.yaml` 是实际运行时最重要的配置入口。当前默认设置启用：

| 配置项 | 含义 |
| --- | --- |
| `project.auto_paths: true` | 根据 `data.dataset` 和 `sample_size` 自动生成输入输出路径。 |
| `project.preserve_runs: true` | 将 similarity/evaluation 输出写入独立 run 目录，避免覆盖。 |
| `project.run_id: auto` | similarity 阶段生成新 run，evaluation/visualization 读取 latest run。 |
| `evaluation.reference_field: auto` | NQ 默认用 `reference_answer`，其他数据集用 `ground_truth`。 |

## 5. 数据目录

| 路径 | 内容 | 由谁读取或生成 |
| --- | --- | --- |
| `data/raw/{dataset}/merged_fb.json` | 教师提供的 JSONL 原始数据，每行含 `question` 与 `correct_answer`。 | `scripts/pipeline/01_generate_predictions.py` |
| `data/interim/predictions/` | LLM 预测输出，每条记录追加 `prompt` 与 `prediction`。 | 由 `01_generate_predictions.py` 生成，由 `02_compute_similarity.py` 读取 |
| `data/interim/similarity/` | 旧式固定路径 similarity 输出。 | 由 `02_compute_similarity.py` 生成，由 `03_evaluate.py` 读取 |
| `outputs/experiments/results_*/runs/{run_id}/similarity/` | 启用 `preserve_runs` 后的 similarity 输出。 | 由 `02_compute_similarity.py` 生成 |
| `outputs/experiments/results_*/tables/` | 指标、ablation、case study、metadata 等 CSV/JSON。 | 由 `03_evaluate.py` 生成 |
| `outputs/experiments/results_*/failure_cases/` | 高相似但错误、低相似但正确的 JSONL 失败样例。 | 由 `03_evaluate.py` 生成 |
| `outputs/experiments/results_*/figures/` | ROC、PR、similarity distribution、correlation 图。 | 由 `04_visualize_results.py` 生成 |

支持的数据集由 `src/utils.py` 和 `src/data_loader.py` 共同约束：

| 数据集 | 任务类型 |
| --- | --- |
| `sciq` | `short_form` |
| `simple_questions_wiki` | `short_form` |
| `nq` | `long_form` |
| `truthfulQA` | `long_form` |

## 6. 脚本层结构

`scripts/` 是人直接运行的入口层。脚本负责读取配置、组织 I/O、串联
`src/` 中的实现模块。

| 脚本 | 作用 | 主要调用 |
| --- | --- | --- |
| `scripts/pipeline/01_generate_predictions.py` | 读取数据并调用 LLM 生成预测。 | `src.generate_predictions`、`src.utils` |
| `scripts/pipeline/02_compute_similarity.py` | 准备 reference、打标签、抽取 span/factual units、计算 similarity。 | `src.reference_answer`、`src.correctness_labeling`、`src.compute_similarity` |
| `scripts/pipeline/03_evaluate.py` | 把 similarity 当作分类分数，导出指标、ablation、失败样例和 metadata。 | `src.evaluate`、`src.entity_overlap`、`src.factual_units` |
| `scripts/pipeline/04_visualize_results.py` | 基于 evaluation 输入生成图表。 | `src.visualize` |
| `scripts/analysis/analyze_part4_strict.py` | 汇总多组结果，生成 Part 4 分析表格、图像和报告。 | 直接读取 `outputs/experiments/results_*` 与写入 `outputs/analysis/failures_analysis_and_improvement/` |

运行顺序通常是：

```bash
python scripts/pipeline/01_generate_predictions.py
python scripts/pipeline/02_compute_similarity.py
python scripts/pipeline/03_evaluate.py
python scripts/pipeline/04_visualize_results.py
```

如果只重做 Part 4 汇总分析，运行：

```bash
python scripts/analysis/analyze_part4_strict.py
```

## 7. `src/` 模块分层

`src/` 是实际业务逻辑层。可以按职责分成五类：

### 7.1 配置、I/O 与数据加载

| 文件 | 职责 |
| --- | --- |
| `src/utils.py` | 加载并解析 `config.yaml`、自动路径解析、run id、JSON/JSONL I/O、文本规范化。 |
| `src/data_loader.py` | 读取教师处理后的数据，统一生成 `ground_truth` 字段，构造 QA prompt。 |

关键关系：

- `load_config()` 会调用 `resolve_config()`，在 `auto_paths` 开启时自动改写
  `prediction`、`similarity`、`evaluation` 和 `output` 路径。
- `validate_config()` 会防止 `data.dataset` 与输入文件不匹配。
- `validate_records_dataset()` 会防止读取别的数据集产物。

### 7.2 LLM 预测生成

| 文件 | 职责 |
| --- | --- |
| `src/generate_predictions.py` | 定义 `DummyLLMClient` 与 OpenAI-compatible 客户端，批量生成 `prediction`。 |

当前 `config.yaml` 中的 LLM provider 是 `qwen`，使用 OpenAI-compatible SDK 访问
DashScope compatible endpoint。真实预测生成需要环境变量
`DASHSCOPE_API_KEY`。

### 7.3 Reference、label 与答案粒度处理

| 文件 | 职责 |
| --- | --- |
| `src/reference_answer.py` | 为 NQ 从长 evidence passage 中抽取较短 `reference_answer` 与 v2 reference 字段。 |
| `src/correctness_labeling.py` | 基于 EM、token F1、containment 生成 `correct_label`。 |
| `src/answer_span.py` | 按 question type 从预测文本中抽取 `prediction_answer_span`。 |
| `src/factual_units.py` | 抽取日期、数字、实体等 factual units，并检测匹配与冲突。 |
| `src/entity_overlap.py` | 计算 token/entity overlap 与 factual-conflict-adjusted score。 |

这里有两个容易混淆的字段：

| 字段 | 含义 |
| --- | --- |
| `ground_truth` | 从原始 `correct_answer` 映射得到的项目内参考答案。 |
| `reference_answer` | 实际用于评估/相似度的 reference；NQ 会从长 passage 中抽取。 |

`correct_label` 不是人工标签，而是由 `src/correctness_labeling.py` 自动生成。
long-form 数据集额外允许 `prediction` 被 reference 包含时判为正确。

### 7.4 Embedding 与 similarity

| 文件 | 职责 |
| --- | --- |
| `src/compute_embeddings.py` | 包装 sentence-transformers 模型并批量编码文本。 |
| `src/compute_similarity.py` | 计算 cosine similarity、blended similarity 和阈值预测。 |
| `src/sentence_level_similarity.py` | 提供句子切分与句子级最大相似度。 |
| `src/multi_view_similarity.py` | 计算 span-level 与 multi-view similarity 字段。 |

当前默认 embedding 模型来自 `config.yaml`：

| 模型 |
| --- |
| `sentence-transformers/all-MiniLM-L6-v2` |
| `BAAI/bge-base-en-v1.5` |

模型名会被转换成安全字段名，例如：

| 原模型名 | 字段后缀 |
| --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` | `sentence_transformers_all_MiniLM_L6_v2` |
| `BAAI/bge-base-en-v1.5` | `BAAI_bge_base_en_v1.5` |

### 7.5 评估与可视化

| 文件 | 职责 |
| --- | --- |
| `src/evaluate.py` | 计算 accuracy、precision、recall、F1、ROC-AUC、PR-AUC、best threshold 和 failure cases。 |
| `src/visualize.py` | 生成 similarity distribution、ROC、PR、correlation 图。 |

`scripts/pipeline/03_evaluate.py` 在 `src/evaluate.py` 基础上增加了很多实验产物：

| 输出 | 含义 |
| --- | --- |
| `evaluation_results.csv` | embedding cosine 主指标表。 |
| `baseline_ablation_results.csv` | baseline、hybrid、multi-view 等方法对比。 |
| `multi_view_ablation_results.csv` | multi-view 相关 stage 的子集。 |
| `question_type_metrics.csv` | 按 question type 的指标。 |
| `case_studies.csv` | 可直接写入报告的代表性样例。 |
| `reference_quality_report.csv` | reference 抽取质量统计。 |
| `prediction_span_report.csv` | prediction answer span 抽取统计。 |
| `factual_unit_report.csv` | factual units 匹配/冲突统计。 |
| `label_change_audit.csv` | 启用 v2 label 时的标签变更审计。 |
| `run_metadata.json` | 本次实验使用的数据集、模型、路径、run id 等元信息。 |

## 8. 测试结构

`tests/` 使用 `unittest` 风格组织。测试文件基本按模块映射：

| 测试文件 | 覆盖模块 |
| --- | --- |
| `tests/test_utils.py` | `src/utils.py` 的配置解析与路径校验。 |
| `tests/test_compute_similarity.py` | similarity blending 等逻辑。 |
| `tests/test_evaluate.py` | 指标计算、阈值、ablation 与报告行生成。 |
| `tests/test_reference_answer.py` | NQ reference 抽取与校验。 |
| `tests/test_answer_span.py` | prediction answer span 抽取。 |
| `tests/test_factual_units.py` | factual unit 抽取、规范化和冲突判断。 |
| `tests/test_multi_view_similarity.py` | multi-view/span-level similarity。 |

运行全部测试：

```bash
python -m unittest discover -s tests
```

## 9. 产物目录命名规则

当 `project.auto_paths: true` 时，路径由 `data.dataset`、
`data.sample_size` 和 `llm.run_name` 自动生成。

示例配置：

```yaml
data:
  dataset: sciq
  sample_size: 500
llm:
  run_name: qwen25_7b_instruct
```

会得到类似路径：

```text
data/interim/predictions/sciq_qwen25_7b_instruct_predictions_500.jsonl
outputs/experiments/results_sciq_500/runs/{run_id}/similarity/
outputs/experiments/results_sciq_500/runs/{run_id}/tables/
outputs/experiments/results_sciq_500/runs/{run_id}/failure_cases/
```

如果 `preserve_runs` 关闭，则会回到较旧的固定路径：

```text
data/interim/similarity/sciq_qwen25_7b_instruct_similarity_500.jsonl
outputs/experiments/results_sciq_500/tables/
outputs/experiments/results_sciq_500/failure_cases/
outputs/experiments/results_sciq_500/figures/
```

## 10. Part 4 相关结构

`outputs/analysis/failures_analysis_and_improvement/` 是 Part 4 的集中产物目录。

```text
outputs/analysis/failures_analysis_and_improvement/
  part4_report.md                  # 英文分析报告
  part4_report.zh.md               # 中文分析报告
  summary_tables/                  # 汇总 CSV
  figures/                         # 汇总 SVG 图
  datasets/                        # 拷贝后的分数据集分析输入/输出
```

`scripts/analysis/analyze_part4_strict.py` 会读取多个 `outputs/experiments/results_*` 目录中的指标、
failure cases 和 NQ improvement runs，并生成上面的汇总产物。

`docs/plans/multi-view-embedding-evaluator-plan.md` 记录了 NQ-focused
multi-view evaluator 的设计计划。它解释了为什么引入：

- `reference_answer_v2`
- `prediction_answer_span`
- `span_max_similarity_*`
- `multi_view_score_*`
- factual-unit conflict adjustment
- question-type-aware reporting

## 11. 常见修改入口

| 你要改什么 | 优先看哪里 |
| --- | --- |
| 换数据集、样本量、模型或输出目录 | `config.yaml`、`src/utils.py` |
| 修改 prompt 或 LLM 调用方式 | `src/data_loader.py`、`src/generate_predictions.py`、`scripts/pipeline/01_generate_predictions.py` |
| 修改自动正确性标签逻辑 | `src/correctness_labeling.py`、`tests/test_evaluate.py` |
| 修改 NQ reference 抽取 | `src/reference_answer.py`、`tests/test_reference_answer.py` |
| 修改预测答案 span 抽取 | `src/answer_span.py`、`tests/test_answer_span.py` |
| 修改 embedding similarity 计算 | `src/compute_embeddings.py`、`src/compute_similarity.py` |
| 修改 multi-view 方法 | `src/multi_view_similarity.py`、`scripts/pipeline/02_compute_similarity.py` |
| 修改评估指标或 ablation 表 | `src/evaluate.py`、`scripts/pipeline/03_evaluate.py`、`tests/test_evaluate.py` |
| 修改图表输出 | `src/visualize.py`、`scripts/pipeline/04_visualize_results.py` |
| 修改 Part 4 报告生成 | `scripts/analysis/analyze_part4_strict.py`、`outputs/analysis/failures_analysis_and_improvement/` |

## 12. 阅读建议

如果你只想理解主流程，按这个顺序读：

1. `README.md`
2. `config.yaml`
3. `scripts/pipeline/01_generate_predictions.py`
4. `scripts/pipeline/02_compute_similarity.py`
5. `scripts/pipeline/03_evaluate.py`
6. `src/utils.py`
7. `src/reference_answer.py`
8. `src/evaluate.py`

如果你要改 NQ improvement 或 Part 4，优先读：

1. `docs/plans/multi-view-embedding-evaluator-plan.md`
2. `src/reference_answer.py`
3. `src/answer_span.py`
4. `src/factual_units.py`
5. `src/multi_view_similarity.py`
6. `scripts/analysis/analyze_part4_strict.py`
7. `outputs/analysis/failures_analysis_and_improvement/part4_report.zh.md`
