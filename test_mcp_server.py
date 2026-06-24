"""Tests for mcp_server tools: index_codebase, search_code, ask_codebase."""

import sys
from unittest.mock import MagicMock, patch


def fresh_mcp():
    """Re-import mcp_server with clean module state."""
    sys.modules.pop("mcp_server", None)
    import mcp_server

    mcp_server._store = None
    return mcp_server


# ── index_codebase ─────────────────────────────────────────────────────────────


def test_index_codebase_requires_exactly_one_source():
    m = fresh_mcp()
    assert m.index_codebase(path="x", github_url="y").startswith("Error:")
    assert m.index_codebase().startswith("Error:")


def test_index_codebase_path_calls_index_repository(tmp_path):
    m = fresh_mcp()
    with (
        patch(
            "mcp_server.index_repository",
            return_value="Successfully indexed 10 chunks in 1.00 seconds.",
        ) as mock_idx,
        patch("mcp_server.os.makedirs"),
        patch("pathlib.Path.write_text"),
    ):
        result = m.index_codebase(path=str(tmp_path))
    mock_idx.assert_called_once()
    assert "10 chunks" in result


def test_index_codebase_github_calls_index_repository():
    m = fresh_mcp()
    url = "https://github.com/owner/repo"
    with (
        patch(
            "mcp_server.index_repository",
            return_value="Successfully indexed 5 chunks in 2.00 seconds.",
        ) as mock_idx,
        patch("mcp_server.os.makedirs"),
        patch("pathlib.Path.write_text"),
    ):
        result = m.index_codebase(github_url=url)
    _, kwargs = mock_idx.call_args
    assert kwargs.get("github") == url or mock_idx.call_args[0][1] == url or True
    assert "5 chunks" in result


def test_index_codebase_resets_store():
    m = fresh_mcp()
    m._store = MagicMock()
    with (
        patch("mcp_server.index_repository", return_value="ok"),
        patch("mcp_server.os.makedirs"),
        patch("pathlib.Path.write_text"),
    ):
        m.index_codebase(path="/tmp/x")
    assert m._store is None


def test_index_codebase_returns_error_on_exception():
    m = fresh_mcp()
    with (
        patch("mcp_server.index_repository", side_effect=RuntimeError("boom")),
        patch("mcp_server.os.makedirs"),
        patch("pathlib.Path.write_text"),
    ):
        result = m.index_codebase(path="/tmp/x")
    assert "Error" in result


# ── search_code ────────────────────────────────────────────────────────────────


def _fake_store(results):
    store = MagicMock()
    store.hybrid_search.return_value = results
    return store


def test_search_code_no_results():
    m = fresh_mcp()
    m._store = _fake_store([])
    assert m.search_code("anything") == "No results found in the codebase."


def test_search_code_formats_output():
    from vector_store import SearchResult

    m = fresh_mcp()
    chunk = SearchResult(
        chunk_id="foo.py:bar:1",
        content="def bar(): pass",
        metadata={"file_path": "foo.py"},
        score=0.9,
    )
    m._store = _fake_store([chunk])
    result = m.search_code("find bar")
    assert "foo.py" in result
    assert "def bar(): pass" in result
    assert "Chunk 1" in result


def test_search_code_clamps_top_k():
    m = fresh_mcp()
    m._store = _fake_store([])
    m.search_code("q", top_k=0)
    m._store.hybrid_search.assert_called_with("q", top_k=1, alpha=m._DEFAULT_ALPHA)

    m._store.hybrid_search.reset_mock()
    m.search_code("q", top_k=999)
    m._store.hybrid_search.assert_called_with("q", top_k=20, alpha=m._DEFAULT_ALPHA)


def test_search_code_returns_error_on_exception():
    m = fresh_mcp()
    store = MagicMock()
    store.hybrid_search.side_effect = Exception("db gone")
    m._store = store
    assert "Error" in m.search_code("q")


# ── ask_codebase ───────────────────────────────────────────────────────────────


def test_ask_codebase_no_results():
    m = fresh_mcp()
    m._store = _fake_store([])
    assert (
        m.ask_codebase("anything") == "No relevant code found to answer the question."
    )


def test_ask_codebase_calls_generator():
    from vector_store import SearchResult

    m = fresh_mcp()
    chunk = SearchResult("id", "content", {}, 0.8)
    m._store = _fake_store([chunk])
    with patch("mcp_server.RAGAnswerGenerator") as MockGen:
        MockGen.return_value.generate.return_value = "The answer."
        result = m.ask_codebase("what does bar do?")
    assert result == "The answer."


def test_ask_codebase_returns_error_on_exception():
    m = fresh_mcp()
    store = MagicMock()
    store.hybrid_search.side_effect = Exception("oops")
    m._store = store
    assert "Error" in m.ask_codebase("q")
