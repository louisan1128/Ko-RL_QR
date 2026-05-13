import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.rewriting.candidate_generator import RewriteCandidateGenerator
from src.utils.io import read_jsonl, write_jsonl, read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    hard_cases = read_jsonl(data_config["hard_cases_path"])
    generator = RewriteCandidateGenerator()

    candidates = []
    for hard_case in hard_cases:
        candidate_queries = generator.generate(hard_case["question"])
        candidates.append({
            "qid": hard_case["qid"],
            "question": hard_case["question"],
            "failure_type": hard_case.get("failure_type", "unlabeled"),
            "candidates": candidate_queries,
        })

    output_path = Path(data_config["rewrite_candidates_path"])
    write_jsonl(candidates, output_path)
    print(f"Saved rewrite candidates to {output_path}")


if __name__ == "__main__":
    main()
