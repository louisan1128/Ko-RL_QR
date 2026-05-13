

def recall_at_k(retrieved: list[dict], gold_doc_id: str, k: int) -> int:
    retrieved_ids = [item["doc_id"] for item in retrieved[:k]]
    return 1 if gold_doc_id in retrieved_ids else 0


def mrr(retrieved: list[dict], gold_doc_id: str) -> float:
    retrieved_ids = [item["doc_id"] for item in retrieved]
    if gold_doc_id in retrieved_ids:
        rank = retrieved_ids.index(gold_doc_id) + 1
        return 1.0 / rank
    return 0.0
