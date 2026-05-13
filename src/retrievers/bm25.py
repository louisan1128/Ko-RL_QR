import json
import math
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from src.retrievers.base import BaseRetriever
from src.utils.text import tokenize


class SimpleBM25Okapi:
    """Small fallback BM25 implementation for dependency-light smoke tests."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self._initialize()

    def _initialize(self) -> None:
        nd = {}
        for document in self.corpus:
            frequencies = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.doc_freqs.append(frequencies)
            for token in frequencies:
                nd[token] = nd.get(token, 0) + 1

        total_docs = len(self.corpus)
        for token, freq in nd.items():
            self.idf[token] = math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for idx, frequencies in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_len[idx] or 1
            for token in query_tokens:
                if token not in frequencies:
                    continue
                tf = frequencies[token]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avgdl or 1.0))
                score += self.idf.get(token, 0.0) * tf * (self.k1 + 1) / denom
            scores.append(score)
        return scores


class BM25Retriever(BaseRetriever):
    def __init__(self, corpus_path: str):
        super().__init__(corpus_path)
        self.documents = self._load_corpus()
        self.tokenized_corpus = [self.tokenize(doc["text"]) for doc in self.documents]
        # TODO: Replace whitespace tokenization with a Korean morphological analyzer.
        bm25_cls = BM25Okapi or SimpleBM25Okapi
        self.bm25 = bm25_cls(self.tokenized_corpus)

    def _load_corpus(self) -> list[dict]:
        path = Path(self.corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        docs = []
        with path.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                docs.append({"doc_id": item["doc_id"], "text": item["text"]})

        if not docs:
            raise ValueError("Corpus file is empty.")
        return docs

    def tokenize(self, text: str) -> list[str]:
        return tokenize(text)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        query = query or ""
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
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
