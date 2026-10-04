from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return " ".join(str(text).replace("_", " ").split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = []
        for chunk in self.documents:
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            tokens = [token for token in segment_vietnamese(text).split() if token]
            self.corpus_tokens.append(tokens)

        if self.corpus_tokens:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []

        tokenized_query = [token for token in segment_vietnamese(query).split() if token]
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            chunk = self.documents[idx]
            text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
            metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            results.append(SearchResult(text=text, score=score, metadata=metadata, method="bm25"))
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return

        from qdrant_client.models import Distance, VectorParams, PointStruct

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )

        texts = [chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict)]
        if not texts:
            return

        vectors = self._get_encoder().encode(texts, show_progress_bar=False, normalize_embeddings=True)
        points = [
            PointStruct(
                id=i,
                vector=v.tolist(),
                payload={**chunk.get("metadata", {}), "text": chunk["text"]},
            )
            for i, (chunk, v) in enumerate(zip(chunks, vectors))
            if isinstance(chunk, dict)
        ]
        if points:
            self.client.upsert(collection_name=collection, points=points)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query:
            return []

        query_vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        points = getattr(response, "points", [])
        results = []
        for pt in points:
            payload = getattr(pt, "payload", {}) or {}
            text = payload.get("text", "")
            score = float(getattr(pt, "score", 0.0) or 0.0)
            if not text:
                continue
            results.append(SearchResult(text=text, score=score, metadata=payload, method="dense"))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}

    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = result.text
            if key not in rrf_scores:
                rrf_scores[key] = {"score": 0.0, "result": result}
            rrf_scores[key]["score"] += 1.0 / (k + rank + 1)

    if not rrf_scores:
        return []

    merged = []
    for _, info in sorted(rrf_scores.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]:
        result = info["result"]
        result.score = float(info["score"])
        result.method = "hybrid"
        merged.append(result)
    return merged


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
