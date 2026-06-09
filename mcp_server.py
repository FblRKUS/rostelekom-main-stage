from mcp.server.fastmcp import FastMCP
from index import index_repository
from vector_store import VectorStore
from rag_pipeline import RAGAnswerGenerator

mcp = FastMCP("CodeLens")

@mcp.tool()
def index_codebase(path: str | None = None, github_url: str | None = None) -> str:
    """
    Indexes a local directory or a GitHub repository (https://github.com/owner/repo).
    Provide EITHER path OR github_url.
    Note: The indexing may take up to 60 seconds depending on the codebase size.
    """
    if bool(path) == bool(github_url):
        return "Error: Provide exactly one of path or github_url."
    try:
        return index_repository(path=path, github=github_url)
    except Exception as e:
        return f"Error during indexing: {str(e)}"

@mcp.tool()
def search_code(query: str, top_k: int = 5) -> str:
    """
    Search the indexed codebase using hybrid search.
    Returns the raw code chunks that match the query.
    Use this if you just want to find relevant code to read.
    """
    try:
        if not hasattr(search_code, "_store"):
            search_code._store = VectorStore()
        store = search_code._store
        results = store.hybrid_search(query, top_k=top_k, alpha=0.75)
        
        output = []
        for i, r in enumerate(results, 1):
            output.append(f"--- Chunk {i} ---")
            output.append(f"File: {r.metadata.get('file_path', 'unknown')}")
            output.append(f"Content:\n{r.content}\n")
        
        if not output:
            return "No results found in the codebase."
        return "\n".join(output)
    except Exception as e:
        return f"Error during search: {str(e)}"

@mcp.tool()
def ask_codebase(query: str) -> str:
    """
    Asks a natural language question about the codebase.
    Uses local Ollama to generate an answer based on the retrieved code chunks.
    Use this to get an aggregated, explained answer instead of raw code.
    """
    try:
        store = VectorStore()
        chunks = store.hybrid_search(query, top_k=5, alpha=0.75)
        
        if not chunks:
            return "No relevant code found to answer the question."
            
        generator = RAGAnswerGenerator(use_llm=True)
        return generator.generate(query, chunks)
    except Exception as e:
        return f"Error during generation: {str(e)}"

if __name__ == "__main__":
    mcp.run()
