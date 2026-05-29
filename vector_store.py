from dataclasses import dataclass
from typing import Any, cast, Optional, Dict, List

import chromadb
from chromadb.utils import embedding_functions

from parser import CodeChunk
import json
import os
from bm25 import BM25


@dataclass
class SearchResult:
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    score: float


def _format_embedding_text(chunk: CodeChunk) -> str:
    doc = chunk.docstring or ""
    return f"{chunk.name}\n{doc}\n{chunk.content}"


class VectorStore:
    def __init__(
        self, persist_path: str = "./chroma_db", collection_name: str = "codelens"
    ):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_path)

        # Chromadb embedding func has a complex type signature
        # Cast to Any to keep mypy quiet while preserving runtime behavior.
        self.embedding_function = cast(
            Any,
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="paraphrase-multilingual-MiniLM-L12-v2"
            ),
        )

        # Cast embedding_function to Any to avoid typing mismatch with chromadb stubs
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=cast(Any, self.embedding_function),
        )
        self.bm25: Optional[BM25] = None
        self._load_bm25()

    def _load_bm25(self) -> None:
        bm25_path = os.path.join(self.persist_path, "bm25.json")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.bm25 = BM25.from_dict(data)
            except Exception:
                self._rebuild_bm25()
        else:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        try:
            data: Any = self.collection.get()
        except Exception:
            data = {"metadatas": []}

        if data and data.get("metadatas"):
            corpus = [m.get("content", "") for m in data["metadatas"]]
            self.bm25 = BM25(corpus)
            bm25_path = os.path.join(self.persist_path, "bm25.json")
            try:
                with open(bm25_path, "w", encoding="utf-8") as f:
                    json.dump(self.bm25.to_dict(), f)
            except Exception:
                pass
        else:
            self.bm25 = BM25()

    def add_chunks(self, chunks: list[CodeChunk]) -> None:
        if not chunks:
            return

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(_format_embedding_text(chunk))
            metadatas.append(
                {
                    "file_path": chunk.file_path,
                    "type": chunk.type,
                    "name": chunk.name,
                    "start_line": chunk.start_line,
                    "docstring": chunk.docstring or "",
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,  # storing content in metadata to retrieve it easily
                }
            )

        # Upsert allows adding or updating, preventing duplicate errors
        # To avoid issues with too large batches, we can batch them if needed,
        # but for small/medium codebases one batch is fine.
        batch_size = 5461  # ChromaDB default max batch size
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=cast(Any, metadatas[i : i + batch_size]),
            )

        # Rebuild BM25 index after adding new chunks
        self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []

        results: Any = self.collection.query(query_texts=[query], n_results=top_k)
        all_data: Any = self.collection.get()
        if isinstance(all_data, dict):
            order_map = {
                str(chunk_id): idx
                for idx, chunk_id in enumerate(all_data.get("ids", []))
            }
        else:
            order_map = {}

        search_results: List[SearchResult] = []
        if not results.get("ids") or not results["ids"][0]:
            return []

        # results is a dict with lists of lists (batch queries)
        ids = results["ids"][0]
        distances = (
            results.get("distances", [[]])[0]
            if results.get("distances")
            else [0.0] * len(ids)
        )
        metadatas = results.get("metadatas", [[{}]])[0]

        # If all scores are effectively tied, fall back to collection order so
        # identical embeddings produce deterministic results.
        if distances and max(distances) - min(distances) < 1e-9:
            search_results: List[SearchResult] = []
            for i, doc_id in enumerate(all_data.get("ids", [])[:top_k]):
                meta: Any = (
                    all_data.get("metadatas", [])[i]
                    if i < len(all_data.get("metadatas", []))
                    else {}
                )
                content = str(meta.get("content", "")) if hasattr(meta, "get") else ""
                search_results.append(
                    SearchResult(
                        chunk_id=str(doc_id),
                        content=content,
                        metadata=cast(Dict[str, Any], meta),
                        score=float(distances[0]),
                    )
                )
            return search_results

        for i in range(len(ids)):
            meta: Any = metadatas[i] if i < len(metadatas) else {}
            content = str(meta.get("content", "")) if hasattr(meta, "get") else ""
            search_results.append(
                SearchResult(
                    chunk_id=str(ids[i]),
                    content=content,
                    metadata=cast(Dict[str, Any], meta),
                    score=float(distances[i]) if i < len(distances) else 0.0,
                )
            )

        search_results.sort(
            key=lambda res: (
                res.score,
                order_map.get(res.chunk_id, len(order_map)),
            )
        )

        return search_results

    def hybrid_search(
        self, query: str, top_k: int = 5, alpha: float = 0.5
    ) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []

        # Vector search (get up to 50 for fusion)
        n_fetch = min(50, self.collection.count())
        results: Any = self.collection.query(query_texts=[query], n_results=n_fetch)

        vector_ids = results.get("ids", [[]])[0] if results.get("ids") else []
        vector_distances = (
            results.get("distances", [[]])[0] if results.get("distances") else []
        )

        max_dist = max(vector_distances) if vector_distances else 1.0
        # avoid divide by zero
        eps = 1e-8
        if max_dist == 0:
            max_dist = eps

        vector_scores: Dict[str, float] = {}
        for i, vid in enumerate(vector_ids):
            # Normalize distance to similarity score (clamped)
            try:
                vector_scores[vid] = max(0.0, 1.0 - (vector_distances[i] / max_dist))
            except Exception:
                vector_scores[vid] = 0.0

        # BM25 search
        all_data: Any = self.collection.get()
        all_ids = all_data.get("ids", []) if isinstance(all_data, dict) else []
        all_metas = all_data.get("metadatas", []) if isinstance(all_data, dict) else []

        if not self.bm25 or self.bm25.corpus_size != len(all_ids):
            self._rebuild_bm25()

        assert self.bm25 is not None
        bm25_scores = self.bm25.get_scores(query)
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 == 0:
            max_bm25 = eps

        combined: List[Dict[str, Any]] = []
        for i, doc_id in enumerate(all_ids):
            vs = vector_scores.get(doc_id, 0.0)
            bs = (bm25_scores[i] / max_bm25) if i < len(bm25_scores) else 0.0

            score = alpha * vs + (1.0 - alpha) * bs
            if score > 0:
                combined.append(
                    {
                        "chunk_id": doc_id,
                        "score": score,
                        "metadata": all_metas[i] if i < len(all_metas) else {},
                    }
                )

        # Sort by combined score descending, then by collection order for ties
        order_map = {str(chunk_id): idx for idx, chunk_id in enumerate(all_ids)}
        combined.sort(
            key=lambda x: (
                -x["score"],
                order_map.get(str(x["chunk_id"]), len(order_map)),
            )
        )

        search_results: List[SearchResult] = []
        for res in combined[:top_k]:
            meta = res.get("metadata", {})
            content = str(meta.get("content", "")) if hasattr(meta, "get") else ""
            search_results.append(
                SearchResult(
                    chunk_id=str(res["chunk_id"]),
                    content=content,
                    metadata=cast(Dict[str, Any], meta),
                    score=float(res["score"]),  # Note: higher is better here
                )
            )

        return search_results

    def clear(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except ValueError:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
