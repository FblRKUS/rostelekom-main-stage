from unittest.mock import patch

from rag_pipeline import RAGAnswerGenerator
from vector_store import SearchResult


def test_rag_messages_includes_code_and_question():
    generator = RAGAnswerGenerator(use_llm=True)

    question = "How to handle errors?"
    chunks = [
        SearchResult(
            chunk_id="utils.py:handle_error:10",
            content="def handle_error(): pass",
            metadata={"file_path": "utils.py"},
            score=0.1,
        )
    ]

    messages = generator._build_messages(question, chunks)

    # _build_messages returns a list of dicts. The system prompt is the first message.
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "def handle_error(): pass" in messages[0]["content"]
    assert "utils.py" in messages[0]["content"]

    # The last message is the user question
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "How to handle errors?"


@patch("rag_pipeline.ollama")
def test_rag_generation_with_ollama(mock_ollama):
    mock_response = {"message": {"content": "Mock answer"}}
    mock_ollama.chat.return_value = mock_response

    generator = RAGAnswerGenerator(use_llm=True)
    question = "Test?"
    chunks = []

    answer = generator.generate(question, chunks)
    assert answer == "Mock answer"


@patch("rag_pipeline.ollama")
def test_rag_generate_stream_with_ollama(mock_ollama):
    # Mock stream returning chunks
    mock_stream = [
        {"message": {"content": "Mock "}},
        {"message": {"content": "stream "}},
        {"message": {"content": "answer"}},
    ]
    mock_ollama.chat.return_value = mock_stream

    generator = RAGAnswerGenerator(use_llm=True)
    question = "Test?"
    chunks = []

    stream = generator.generate_stream(question, chunks)
    result = "".join(list(stream))

    assert result == "Mock stream answer"


def test_rag_generate_stream_without_ollama():
    generator = RAGAnswerGenerator(use_llm=False)
    question = "Test?"
    chunks = [
        SearchResult(
            chunk_id="utils.py:handle_error:10",
            content="def handle_error(): pass",
            metadata={"file_path": "utils.py"},
            score=0.1,
        )
    ]
    stream = generator.generate_stream(question, chunks)
    result = "".join(list(stream))
    assert "LLM is disabled" in result
    assert "utils.py:handle_error:10" in result


def test_rag_without_ollama():
    generator = RAGAnswerGenerator(use_llm=False)

    question = "Test?"
    chunks = [
        SearchResult(
            chunk_id="utils.py:handle_error:10",
            content="def handle_error(): pass",
            metadata={"file_path": "utils.py"},
            score=0.1,
        )
    ]

    answer = generator.generate(question, chunks)
    assert "LLM is disabled" in answer
    assert "utils.py:handle_error:10" in answer
