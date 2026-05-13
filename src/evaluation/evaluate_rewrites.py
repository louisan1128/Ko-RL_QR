import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import mrr, recall_at_k
from src.evaluation.reward import RewardCalculator
from src.rewriting.candidate_generator import RewriteCandidateGenerator
from src.rewriting.policy import select_strategies
from src.utils.io import ensure_dir


def evaluate_rewrites(
    hard_cases: list[dict[str, Any]],
    retrievers: dict[str, Any],
    reward_calculator: RewardCalculator,
    top_k: int = 10,
    out_path: str | None = None,
    candidate_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    generator = RewriteCandidateGenerator()
    candidates_by_qid = {record["qid"]: record.get("candidates", {}) for record in candidate_records or []}
    rewrite_results = []
    for hard_case in hard_cases:
        question = hard_case["question"]
        gold_doc_id = hard_case["gold_doc_id"]
        candidates = candidates_by_qid.get(hard_case["qid"]) or generator.generate(question)
        recommended_strategies = select_strategies(hard_case.get("failure_type", "unlabeled"))

        for strategy, candidate_query in candidates.items():
            for retriever_name, retriever in retrievers.items():
                retrieved = retriever.retrieve(candidate_query, top_k=top_k)
                recall10 = recall_at_k(retrieved, gold_doc_id, 10)
                score_mrr = mrr(retrieved, gold_doc_id)
                reward = reward_calculator.compute_reward(recall10, score_mrr, candidate_query)
                result = {
                    "qid": hard_case["qid"],
                    "failure_type": hard_case.get("failure_type", "unlabeled"),
                    "retriever": retriever_name,
                    "strategy": strategy,
                    "policy_recommended": strategy in recommended_strategies,
                    "query": candidate_query,
                    "recall@1": recall_at_k(retrieved, gold_doc_id, 1),
                    "recall@5": recall_at_k(retrieved, gold_doc_id, 5),
                    "recall@10": recall10,
                    "mrr": score_mrr,
                    "reward": reward,
                }
                rewrite_results.append(result)

    if out_path:
        ensure_dir(Path(out_path).parent)
        with Path(out_path).open("w", encoding="utf-8") as fout:
            for record in rewrite_results:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rewrite_results
