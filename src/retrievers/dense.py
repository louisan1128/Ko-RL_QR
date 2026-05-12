import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, corpus_path: str, model_name: str = "intfloat/multilingual-e5-base", embedding_dir: str = "embeddings"):
        self.corpus_path = Path(corpus_path)
        self.embedding_dir = Path(embedding_dir)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        self.corpus = self._load_corpus()
        self.doc_ids = [item["doc_id"] for item in self.corpus]
        self.doc_texts = [item["text"] for item in self.corpus]
        self.index = None

        self.index_path = self.embedding_dir / "faiss.index"
        self.embedding_path = self.embedding_dir / "doc_embeddings.npy"
        self.doc_ids_path = self.embedding_dir / "doc_ids.npy"

        self._build_or_load_index()

    def _load_corpus(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.corpus_path}")

        corpus = []
        with self.corpus_path.open("r", encoding="utf-8") as reader:
            for line in reader:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                corpus.append({"doc_id": item["doc_id"], "text": item["text"]})

        if not corpus:
            raise ValueError("Corpus file is empty or invalid.")

        return corpus

    def _is_e5_model(self):
        return "e5" in self.model_name.lower()

    def _prepare_passage(self, text: str) -> str:
        return f"passage: {text}" if self._is_e5_model() else text

    def _prepare_query(self, question: str) -> str:
        return f"query: {question}" if self._is_e5_model() else question

    def _build_or_load_index(self):
        if self.index_path.exists() and self.embedding_path.exists() and self.doc_ids_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.doc_embeddings = np.load(self.embedding_path)
                self.doc_ids = np.load(self.doc_ids_path, allow_pickle=True).tolist()
                return
            except Exception:
                pass

        self._build_index()

    def _build_index(self):
        passages = [self._prepare_passage(text) for text in self.doc_texts]
        embeddings = self.model.encode(passages, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        embeddings = self._normalize_embeddings(embeddings)

        self.doc_embeddings = embeddings.astype(np.float32)
        self.index = faiss.IndexFlatIP(self.doc_embeddings.shape[1])
        self.index.add(self.doc_embeddings)

        faiss.write_index(self.index, str(self.index_path))
        np.save(self.embedding_path, self.doc_embeddings)
        np.save(self.doc_ids_path, np.array(self.doc_ids, dtype=object))

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def retrieve(self, question: str, top_k: int = 10) -> list[dict]:
        if not question or not question.strip():
            return []

        query_text = self._prepare_query(question)
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        query_embedding = self._normalize_embeddings(query_embedding.astype(np.float32))

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.doc_ids):
                continue
            results.append(
                {
                    "doc_id": self.doc_ids[idx],
                    "score": float(score),
                    "text": self.doc_texts[idx],
                }
            )

        return results
