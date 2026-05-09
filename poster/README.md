# Poster

This folder contains the A1 project poster.

- `poster.tex`: LaTeX source.
- `poster.pdf`: compiled poster.

Compile from this folder with:

```bash
pdflatex -interaction=nonstopmode -halt-on-error poster.tex
```

The poster summarizes the whole project pipeline: LLM prediction, automatic correctness labeling, embedding similarity evaluation, failure analysis, and the implemented NQ reference-extraction improvement.

Main data sources:

- `results_sciq_500/`
- `results_simple_questions_wiki_500/`
- `results_nq_5000/`
- `results_truthfulQA_500/`
- `results_nq_500/`
- `failures_analysis_and_improvement/summary_tables/`
