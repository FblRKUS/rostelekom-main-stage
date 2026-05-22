from dataclasses import dataclass
from typing import Any

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
    def __init__(self, persist_path: str = "./chroma_db", collection_name: str = "codelens"):
        self.persist_path = persist_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_path)
        
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        self.bm25 = None
        self._load_bm25()

    def _load_bm25(self) -> None:
        bm25_path = os.path.join(self.persist_path, "bm25.json")
        if os.path.exists(bm25_path):
            try:
                with open(bm25_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.bm25 = BM25.from_dict(data)
            except Exception:
                self._rebuild_bm25()
        else:
            self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        try:
            data = self.collection.get()
        except Exception:
            data = {"metadatas": []}
            
        if data and data.get("metadatas"):
            corpus = [m.get("content", "") for m in data["metadatas"]]
            self.bm25 = BM25(corpus)
            bm25_path = os.path.join(self.persist_path, "bm25.json")
            try:
                with open(bm25_path, 'w', encoding='utf-8') as f:
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
                metadatas=metadatas[i : i + batch_size],
            )
            
        # Rebuild BM25 index after adding new chunks
        self._rebuild_bm25()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
        )

        search_results = []
        if not results["ids"] or not results["ids"][0]:
            return []

        # results is a dict with lists of lists (batch queries)
        ids = results["ids"][0]
        distances = results["distances"][0] if results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

        for i in range(len(ids)):
            meta = metadatas[i]
            content = meta.get("content", "")
            search_results.append(
                SearchResult(
                    chunk_id=ids[i],
                    content=content,
                    metadata=meta,
                    score=distances[i],
                )
            )

        return search_results

    def hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[SearchResult]:
        if self.collection.count() == 0:
            return []

        # Vector search (get up to 50 for fusion)
        n_fetch = min(50, self.collection.count())
        results = self.collection.query(query_texts=[query], n_results=n_fetch)
        
        vector_ids = results["ids"][0] if results["ids"] else []
        vector_distances = results["distances"][0] if results["distances"] else []
        
        max_dist = max(vector_distances) if vector_distances else 1.0
        if max_dist == 0: max_dist = 1.0

        vector_scores = {}
        for i, vid in enumerate(vector_ids):
            # Normalize distance to similarity score
            vector_scores[vid] = 1.0 - (vector_distances[i] / max_dist)

        # BM25 search
        all_data = self.collection.get()
        all_ids = all_data["ids"]
        all_metas = all_data["metadatas"]
        
        if not self.bm25 or self.bm25.corpus_size != len(all_ids):
            self._rebuild_bm25()
            
        bm25_scores = self.bm25.get_scores(query)
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0
        if max_bm25 == 0: max_bm25 = 1.0
        
        combined = []
        for i, doc_id in enumerate(all_ids):
            vs = vector_scores.get(doc_id, 0.0)
            bs = bm25_scores[i] / max_bm25
            
            score = alpha * vs + (1.0 - alpha) * bs
            if score > 0:
                combined.append({
                    "chunk_id": doc_id,
                    "score": score,
                    "metadata": all_metas[i]
                })
                
        # Sort by combined score descending
        combined.sort(key=lambda x: x["score"], reverse=True)
        
        search_results = []
        for res in combined[:top_k]:
            search_results.append(SearchResult(
                chunk_id=res["chunk_id"],
                content=res["metadata"].get("content", ""),
                metadata=res["metadata"],
                score=res["score"]  # Note: higher is better here
            ))
            
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
