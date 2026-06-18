import hashlib
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from index import index_repository
from vector_store import VectorStore
from rag_pipeline import RAGAnswerGenerator

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "codelens")
_ACTIVE_FILE = os.path.join(_CACHE_DIR, "active")


def _project_store_path(key: str) -> str:
    return os.path.join(_CACHE_DIR, hashlib.md5(key.encode()).hexdigest())

mcp = FastMCP("CodeLens")

# Default hybrid-search fusion weight (vector vs. BM25), tuned empirically.
_DEFAULT_ALPHA = 0.75
# Upper bound on results an agent can request, to avoid pathological scans.
_MAX_TOP_K = 20

# Shared, lazily-initialised store so every tool sees the same indexed data
# and ChromaDB is only loaded from disk once per process.
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        store_path = (
            Path(_ACTIVE_FILE).read_text().strip()
            if os.path.exists(_ACTIVE_FILE)
            else _project_store_path("default")
        )
        _store = VectorStore(persist_path=store_path)
    return _store


@mcp.tool()
def index_codebase(path: str | None = None, github_url: str | None = None) -> str:
    """
    Indexes a local directory or a GitHub repository (https://github.com/owner/repo).
    Provide EITHER path OR github_url.
    Note: The indexing may take up to 60 seconds depending on the codebase size.
    """
    global _store
    if bool(path) == bool(github_url):
        return "Error: Provide exactly one of path or github_url."
    key = str(Path(path).resolve()) if path else (github_url or "")
    store_path = _project_store_path(key)
    os.makedirs(_CACHE_DIR, exist_ok=True)
    Path(_ACTIVE_FILE).write_text(store_path)
    _store = None  # force reinit with new project's store
    try:
        return index_repository(path=path, github=github_url, persist_path=store_path)
    except Exception as e:
        return f"Error during indexing: {e}"


@mcp.tool()
def search_code(query: str, top_k: int = 5) -> str:
    """
    Search the indexed codebase using hybrid search.
    Returns the raw code chunks that match the query.
    Use this if you just want to find relevant code to read.
    """
    top_k = max(1, min(top_k, _MAX_TOP_K))
    try:
        results = _get_store().hybrid_search(query, top_k=top_k, alpha=_DEFAULT_ALPHA)

        if not results:
            return "No results found in the codebase."

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"--- Chunk {i} ---")
            output.append(f"File: {r.metadata.get('file_path', 'unknown')}")
            output.append(f"Content:\n{r.content}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Error during search: {e}"


@mcp.tool()
def ask_codebase(query: str) -> str:
    """
    Asks a natural language question about the codebase.
    Uses local Ollama to generate an answer based on the retrieved code chunks.
    Use this to get an aggregated, explained answer instead of raw code.
    """
    try:
        chunks = _get_store().hybrid_search(query, top_k=5, alpha=_DEFAULT_ALPHA)

        if not chunks:
            return "No relevant code found to answer the question."

        generator = RAGAnswerGenerator(use_llm=True)
        return generator.generate(query, chunks)
    except Exception as e:
        return f"Error during generation: {e}"


if __name__ == "__main__":
    mcp.run()
