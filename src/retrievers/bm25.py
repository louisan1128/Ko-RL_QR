import json
from pathlib import Path

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, corpus_path: str):
        self.corpus_path = Path(corpus_path)
        self.documents = self._load_corpus()
        self.tokenized_corpus = [self.tokenize(doc["text"]) for doc in self.documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _load_corpus(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        documents = []
        with self.corpus_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                documents.append({"doc_id": item["doc_id"], "text": item["text"]})

        if not documents:
            raise ValueError("Corpus file is empty or invalid.")

        return documents

    def tokenize(self, text: str) -> list[str]:
        return text.strip().split()

    def retrieve(self, question: str, top_k: int = 10) -> list[dict]:
        if not question or not question.strip():
            return []

        tokens = self.tokenize(question)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            [
                {
                    "doc_id": doc["doc_id"],
                    "score": float(score),
                    "text": doc["text"],
                }
                for doc, score in zip(self.documents, scores)
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k]
