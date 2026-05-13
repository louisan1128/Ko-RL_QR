import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.evaluation.evaluate_original import load_qa_pairs, save_hard_cases
from src.retrievers.factory import build_retrievers
from src.utils.io import read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    qa_pairs = load_qa_pairs(data_config["qa_path"])
    retrievers = build_retrievers(config)
    hard_cases = save_hard_cases(retrievers, qa_pairs, data_config["hard_cases_path"], top_k=config.get("top_k", 10))
    print(f"Saved {len(hard_cases)} hard cases to {data_config['hard_cases_path']}")


if __name__ == "__main__":
    main()
