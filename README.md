# Ko-RL-QR

Ko-RL-QR is an experimental skeleton for analyzing reward-guided query rewriting on Korean hard retrieval cases in RAG pipelines.

The project is not only about improving retrieval scores. Its main goal is to inspect why original Korean queries fail, which rewrite strategies recover those failures, and whether BM25, Dense, and Hybrid retrievers react differently.

## Project Questions

1. Which Korean RAG queries become hard retrieval cases with the original question?
2. Do BM25, Dense, and Hybrid retrievers respond differently to rewrite strategies?
3. Can a retrieval reward select rewritten queries that recover hard cases better?

## Installation

```bash
pip install -r requirements.txt
```

For offline smoke tests, keep `dense_backend: lexical` in `configs/default.yaml`.
For real dense retrieval experiments, install `sentence-transformers` and `faiss-cpu`, then set:

```yaml
dense_backend: sentence_transformers
model_name: intfloat/multilingual-e5-base
```

## Data Preparation

Place a KorQuAD or KLUE-MRC style JSON file under `data/raw/`.
The default config expects:

```text
data/raw/KorQuAD_v1.0_dev.json
```

Then run:

```bash
python scripts/01_build_dataset.py
```

This creates:

- `data/processed/corpus.jsonl`
- `data/processed/qa_pairs.jsonl`

Each QA record contains:

- `qid`
- `question`
- `answer`
- `gold_doc_id`
- `gold_passage`

If the raw file is missing, the script writes a tiny Korean dummy dataset so the pipeline can run once.

## Full Pipeline

```bash
python scripts/01_build_dataset.py
python scripts/02_run_original_retrieval.py
python scripts/03_build_hard_cases.py
python scripts/04_generate_rewrites.py
python scripts/05_evaluate_rewrites.py
python scripts/06_reward_selection.py
python scripts/07_build_analysis_tables.py
```

Optional follow-up experiments:

```bash
python scripts/08_run_sentence_dense_retrieval.py
python scripts/09_sweep_hybrid_alpha.py
```

## Script Guide

- `01_build_dataset.py`: Converts KorQuAD/KLUE-MRC style QA data into `corpus.jsonl` and `qa_pairs.jsonl`.
- `02_run_original_retrieval.py`: Runs original-query baseline retrieval with BM25, Dense, and Hybrid.
- `03_build_hard_cases.py`: Saves cases where the gold document is missing from top-10 retrieval results.
- `04_generate_rewrites.py`: Generates rule-based rewrite candidates: `original`, `keyword`, `prompt`, `expanded`, `structured`.
- `05_evaluate_rewrites.py`: Evaluates each rewrite candidate with every retriever and writes per-candidate metrics.
- `06_reward_selection.py`: Selects the best rewritten query per hard case and retriever using retrieval reward.
- `07_build_analysis_tables.py`: Builds final CSV tables for strategy, failure type, retriever, and reward ablation analysis.
- `08_run_sentence_dense_retrieval.py`: Forces `sentence_transformers` dense retrieval and writes a separate baseline table.
- `09_sweep_hybrid_alpha.py`: Sweeps Hybrid BM25/Dense weights and finds a better alpha setting.

## Main Outputs

- `data/outputs/original_results.csv`: Original-query baseline metrics.
- `data/outputs/hard_cases.jsonl`: Hard retrieval cases for later manual failure-type annotation.
- `data/outputs/rewrite_candidates.jsonl`: Rewrite candidates per hard case.
- `data/outputs/rewrite_results.jsonl`: Per-candidate retrieval metrics and reward.
- `data/outputs/best_queries.jsonl`: Reward-selected query per hard case and retriever.
- `data/outputs/hard_case_recovery.csv`: Recovery@10 summary after reward-guided selection.
- `data/outputs/main_results.csv`: Strategy-level aggregate results.
- `data/outputs/failure_type_analysis.csv`: Failure-type-aware aggregate results.
- `data/outputs/retriever_specific_results.csv`: Retriever-specific strategy comparison.
- `data/outputs/reward_ablation_results.csv`: Reward weight ablation table.
- `data/outputs/sentence_dense_original_results.csv`: Baseline results with real sentence-transformers dense retrieval.
- `data/outputs/hybrid_alpha_sweep.csv`: Hybrid alpha sweep results.

## Example Result Table

| retriever | strategy | recall@1 | recall@5 | recall@10 | mrr | reward |
|---|---|---:|---:|---:|---:|---:|
| bm25 | original | 0.42 | 0.66 | 0.74 | 0.53 | 0.69 |
| bm25 | keyword | 0.45 | 0.70 | 0.79 | 0.56 | 0.73 |
| dense | expanded | 0.31 | 0.58 | 0.68 | 0.44 | 0.61 |
| hybrid | structured | 0.47 | 0.72 | 0.81 | 0.59 | 0.76 |

## Failure Type Policy

The current policy is a placeholder:

- `expression_mismatch`: `keyword`, `expanded`
- `ellipsis`: `structured`, `expanded`
- `compound_noun`: `keyword`
- `colloquial_mismatch`: `prompt`
- `abbreviation`: `expanded`
- `unlabeled`: `original`, `keyword`, `expanded`

`failure_type` is initially saved as `unlabeled` so it can be manually annotated later.

## TODO / Future Work

- Add Korean morphological tokenization with Kiwi, Mecab, or Okt.
- Add LLM-based query rewrite generation.
- Add manual failure-type annotation workflow.
- Add answer F1 and faithfulness terms to the reward.
- Add confidence intervals and statistical significance tests.
- Extend reward-guided selection into a trained policy or RL-based query rewriting module.
