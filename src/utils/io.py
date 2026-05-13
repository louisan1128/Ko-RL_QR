import json
from pathlib import Path
from typing import Iterable

import yaml


def ensure_dir(directory: Path | str) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path | str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path | str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: Iterable[dict], path: Path | str, fieldnames: list[str] | None = None) -> None:
    import csv

    rows = list(records)
    path = Path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def read_yaml(path: Path | str):
    path = Path(path)
    with path.open("r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)
