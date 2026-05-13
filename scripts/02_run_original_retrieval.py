import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_original import evaluate_named_retrievers, load_qa_pairs
from src.retrievers.factory import build_retrievers
from src.utils.io import read_yaml, write_csv


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    top_k = config.get("top_k", 10)
    qa_pairs = load_qa_pairs(data_config["qa_path"])

    retrievers = build_retrievers(config)
    results = evaluate_named_retrievers(retrievers, qa_pairs, top_k=top_k)
    for record in results:
        record["hybrid_alpha"] = config["hybrid"]["alpha"] if record["retriever"] == "hybrid" else ""

    output_path = Path(data_config["original_results_path"])
    write_csv(results, output_path)
    print(f"Saved original baseline results to {output_path}")


if __name__ == "__main__":
    main()
