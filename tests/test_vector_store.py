import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from parser import CodeChunk
from vector_store import SearchResult, VectorStore


@pytest.fixture
def temp_db_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield str(Path(temp_dir) / "chroma_db")


@pytest.fixture
def mock_embedding_function():
    class MockEmbedding:
        def __call__(self, input: list[str]):
            return [[0.1] * 384 for _ in input]
            
        def embed_query(self, input: list[str]):
            return self(input)
            
        def embed_documents(self, input: list[str]):
            return self(input)
            
        def name(self):
            return "mock_embedding"
            
    return MockEmbedding()


@pytest.fixture
def sample_chunks():
    return [
        CodeChunk(
            file_path="src/utils.py",
            type="function",
            name="handle_error",
            content="def handle_error(e):\n    print('error', e)",
            docstring="как обработать ошибку",
            start_line=10,
        ),
        CodeChunk(
            file_path="src/db.py",
            type="function",
            name="save_to_db",
            content="def save_to_db(data):\n    db.insert(data)",
            docstring="сохранение в БД",
            start_line=20,
        ),
        CodeChunk(
            file_path="src/math.py",
            type="function",
            name="add",
            content="def add(a, b):\n    return a + b",
            docstring="нерелевантный",
            start_line=1,
        ),
    ]


@patch("vector_store.embedding_functions.SentenceTransformerEmbeddingFunction")
def test_add_and_search_chunks(mock_st, temp_db_path, mock_embedding_function, sample_chunks):
    mock_st.return_value = mock_embedding_function
    
    # Init store
    store = VectorStore(persist_path=temp_db_path)
    
    # Add chunks
    store.add_chunks(sample_chunks)
    
    # Search
    results = store.search("как обработать ошибку", top_k=2)
    
    assert len(results) == 2
    # Mock embedding is identical for all, so order is preserved.
    assert results[0].chunk_id == "src/utils.py:handle_error:10"
    
    # Test hybrid search
    hybrid_results = store.hybrid_search("как обработать ошибку", top_k=2, alpha=0.5)
    assert len(hybrid_results) == 2
    assert hybrid_results[0].chunk_id == "src/utils.py:handle_error:10"
    
    # Check metadata
    for res in results:
        assert res.chunk_id in ["src/utils.py:handle_error:10", "src/db.py:save_to_db:20", "src/math.py:add:1"]
        assert "file_path" in res.metadata
        assert "type" in res.metadata


@patch("vector_store.embedding_functions.SentenceTransformerEmbeddingFunction")
def test_search_empty_db(mock_st, temp_db_path, mock_embedding_function):
    mock_st.return_value = mock_embedding_function
    store = VectorStore(persist_path=temp_db_path)
    results = store.search("query", top_k=5)
    assert len(results) == 0


@patch("vector_store.embedding_functions.SentenceTransformerEmbeddingFunction")
def test_duplicate_chunks_handled(mock_st, temp_db_path, mock_embedding_function, sample_chunks):
    mock_st.return_value = mock_embedding_function
    store = VectorStore(persist_path=temp_db_path)
    
    store.add_chunks(sample_chunks)
    # Add again
    store.add_chunks(sample_chunks)
    
    # Should not throw and collection size should be 3
    collection = store.client.get_collection(store.collection_name)
    assert collection.count() == 3


def test_embedding_text_format():
    chunk = CodeChunk(
        file_path="src/utils.py",
        type="function",
        name="handle_error",
        content="def handle_error(e):\n    pass",
        docstring="Doc",
        start_line=10,
    )
    from vector_store import _format_embedding_text
    
    text = _format_embedding_text(chunk)
    assert text == "handle_error\nDoc\ndef handle_error(e):\n    pass"
    
    chunk_no_doc = CodeChunk(
        file_path="src/utils.py",
        type="function",
        name="func",
        content="def func():\n    pass",
        docstring=None,
        start_line=10,
    )
    text_no_doc = _format_embedding_text(chunk_no_doc)
    assert text_no_doc == "func\n\ndef func():\n    pass"
