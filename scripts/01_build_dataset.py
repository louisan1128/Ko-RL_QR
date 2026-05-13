import sys
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.append(str(root))

from src.preprocessing.build_dataset import build_dataset
from src.utils.io import read_yaml


def main():
    config = read_yaml(root / "configs" / "default.yaml")
    data_config = config["data"]
    build_dataset(
        raw_path=data_config["raw_path"],
        corpus_out=data_config["corpus_path"],
        qa_out=data_config["qa_path"],
        limit=data_config.get("limit"),
    )


if __name__ == "__main__":
    main()
