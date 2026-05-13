import json
import sys
import os
from collections import defaultdict
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.utils.io import read_jsonl, read_yaml, ensure_dir, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    rewrite_results = read_jsonl(data_config["rewrite_results_path"])
    if not rewrite_results:
        print(f"No rewrite results found at {data_config['rewrite_results_path']}. Run scripts/05_evaluate_rewrites.py first.")
        write_csv([], data_config["recovery_path"], fieldnames=["retriever", "recovered_at_10", "total_hard_cases", "recovery@10"])
        return

    best_by_retriever = {}
    for record in rewrite_results:
        key = (record["qid"], record["retriever"])
        current = best_by_retriever.get(key)
        if current is None or record["reward"] > current["reward"]:
            best_by_retriever[key] = record

    grouped = defaultdict(dict)
    for record in best_by_retriever.values():
        grouped[record["qid"]][record["retriever"]] = {
            "strategy": record["strategy"],
            "query": record["query"],
            "reward": record["reward"],
            "recall@10": record["recall@10"],
            "mrr": record["mrr"],
        }

    best_records = []
    for qid, best_query_by_retriever in grouped.items():
        sample = next(record for record in rewrite_results if record["qid"] == qid)
        best_records.append(
            {
                "qid": qid,
                "failure_type": sample.get("failure_type", "unlabeled"),
                "best_query_by_retriever": best_query_by_retriever,
            }
        )

    output_path = Path(data_config["best_queries_path"])
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in best_records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    flat_best = list(best_by_retriever.values())
    totals = defaultdict(int)
    recovery_counts = defaultdict(int)
    for record in flat_best:
        totals[record["retriever"]] += 1
        if record["recall@10"] > 0:
            recovery_counts[record["retriever"]] += 1

    recovery_rows = []
    for retriever_name in sorted(totals):
        count = recovery_counts[retriever_name]
        total = totals[retriever_name]
        rate = count / total if total else 0.0
        recovery_rows.append(
            {
                "retriever": retriever_name,
                "recovered_at_10": count,
                "total_hard_cases": total,
                "recovery@10": rate,
            }
        )
        print(f"{retriever_name} recovery@10: {count}/{total} = {rate:.4f}")

    recovery_path = Path(data_config["recovery_path"])
    write_csv(recovery_rows, recovery_path)
    print(f"Saved best query selections to {output_path}")
    print(f"Saved recovery summary to {recovery_path}")


if __name__ == "__main__":
    main()
