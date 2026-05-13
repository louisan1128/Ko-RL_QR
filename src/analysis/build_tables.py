from collections import defaultdict

from src.evaluation.reward import RewardCalculator
from src.utils.io import write_csv


METRIC_FIELDS = ["recall@1", "recall@5", "recall@10", "mrr", "reward"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_mean(records: list[dict], keys: list[str], metrics: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(key, "") for key in keys)].append(record)

    rows = []
    for group_key, items in sorted(grouped.items()):
        row = {key: value for key, value in zip(keys, group_key)}
        for metric in metrics:
            row[metric] = _mean([float(item.get(metric, 0.0)) for item in items])
        row["num_records"] = len(items)
        rows.append(row)
    return rows


def build_main_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    # TODO: Add confidence intervals once experiments run on full datasets.
    rows = _group_mean(rewrite_results, ["retriever", "strategy"], METRIC_FIELDS)
    write_csv(rows, out_path, fieldnames=["retriever", "strategy", *METRIC_FIELDS, "num_records"])
    return rows


def build_failure_type_analysis(rewrite_results: list[dict], out_path: str) -> list[dict]:
    rows = _group_mean(rewrite_results, ["failure_type", "retriever", "strategy"], ["recall@10", "mrr", "reward"])
    write_csv(rows, out_path, fieldnames=["failure_type", "retriever", "strategy", "recall@10", "mrr", "reward", "num_records"])
    return rows


def build_retriever_specific_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    rows = _group_mean(rewrite_results, ["retriever", "strategy", "policy_recommended"], ["recall@10", "mrr", "reward"])
    write_csv(rows, out_path, fieldnames=["retriever", "strategy", "policy_recommended", "recall@10", "mrr", "reward", "num_records"])
    return rows


def build_reward_ablation_results(rewrite_results: list[dict], out_path: str) -> list[dict]:
    settings = [
        {"alpha": 1.0, "beta": 0.0, "lambda_": 0.0},
        {"alpha": 1.0, "beta": 0.5, "lambda_": 0.0},
        {"alpha": 1.0, "beta": 0.5, "lambda_": 0.05},
        {"alpha": 0.5, "beta": 1.0, "lambda_": 0.05},
    ]
    rows = []
    for setting in settings:
        calculator = RewardCalculator(**setting)
        rescored = []
        for record in rewrite_results:
            reward = calculator.compute_reward(float(record["recall@10"]), float(record["mrr"]), record["query"])
            rescored.append({**record, "reward": reward})
        for row in _group_mean(rescored, ["retriever", "strategy"], ["reward", "recall@10", "mrr"]):
            row["alpha"] = setting["alpha"]
            row["beta"] = setting["beta"]
            row["lambda"] = setting["lambda_"]
            rows.append(row)
    write_csv(
        rows,
        out_path,
        fieldnames=["retriever", "strategy", "reward", "recall@10", "mrr", "num_records", "alpha", "beta", "lambda"],
    )
    return rows
