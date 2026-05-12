import json
from pathlib import Path


def load_qa_pairs(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")

    qa_pairs = []
    with path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            qa_pairs.append(json.loads(line))

    return qa_pairs


def evaluate_retriever(retriever, qa_pairs, k_values=None, top_k=10):
    if k_values is None:
        k_values = [1, 5, 10]

    metrics = {f"recall@{k}": 0.0 for k in k_values}
    metrics["mrr"] = 0.0
    num_questions = len(qa_pairs)

    for qa in qa_pairs:
        question = qa["question"]
        gold_doc_id = qa["gold_doc_id"]
        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [item["doc_id"] for item in retrieved]

        for k in k_values:
            if gold_doc_id in retrieved_ids[:k]:
                metrics[f"recall@{k}"] += 1.0

        if gold_doc_id in retrieved_ids:
            rank = retrieved_ids.index(gold_doc_id) + 1
            metrics["mrr"] += 1.0 / rank

    if num_questions > 0:
        for k in k_values:
            metrics[f"recall@{k}"] /= num_questions
        metrics["mrr"] /= num_questions

    metrics["num_questions"] = num_questions
    return metrics


def save_hard_cases(retriever, qa_pairs, out_path, top_k=10):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hard_cases = []
    with out_path.open("w", encoding="utf-8") as writer:
        for qa in qa_pairs:
            question = qa["question"]
            gold_doc_id = qa["gold_doc_id"]
            retrieved = retriever.retrieve(question, top_k=top_k)
            retrieved_ids = [item["doc_id"] for item in retrieved]

            if gold_doc_id not in retrieved_ids:
                hard_case = {
                    "qid": qa["qid"],
                    "question": question,
                    "answer": qa.get("answer", ""),
                    "gold_doc_id": gold_doc_id,
                    "retrieved_top10": retrieved_ids,
                }
                writer.write(json.dumps(hard_case, ensure_ascii=False) + "\n")
                hard_cases.append(hard_case)

    return hard_cases
