import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rag_pipeline import RAGAnswerGenerator
from vector_store import SearchResult


def test_rag_prompt_includes_code_and_question():
    generator = RAGAnswerGenerator(use_llm=True)
    
    question = "How to handle errors?"
    chunks = [
        SearchResult(
            chunk_id="utils.py:handle_error:10",
            content="def handle_error(): pass",
            metadata={"file_path": "utils.py"},
            score=0.1
        )
    ]
    
    prompt = generator._build_prompt(question, chunks)
    
    assert "How to handle errors?" in prompt
    assert "def handle_error(): pass" in prompt
    assert "utils.py" in prompt


@patch("rag_pipeline.ollama")
def test_rag_generation_with_ollama(mock_ollama):
    mock_response = {"response": "Mock answer"}
    mock_ollama.generate.return_value = mock_response
    
    generator = RAGAnswerGenerator(use_llm=True)
    question = "Test?"
    chunks = []
    
    answer = generator.generate(question, chunks)
    assert answer == "Mock answer"


def test_rag_without_ollama():
    generator = RAGAnswerGenerator(use_llm=False)
    
    question = "Test?"
    chunks = [
        SearchResult(
            chunk_id="utils.py:handle_error:10",
            content="def handle_error(): pass",
            metadata={"file_path": "utils.py"},
            score=0.1
        )
    ]
    
    answer = generator.generate(question, chunks)
    assert "LLM is disabled" in answer
    assert "utils.py:handle_error:10" in answer


def test_precision_with_line_tolerance():
    from score import chunks_match
    
    # Exact match
    assert chunks_match("a.py:func:10", "a.py:func:10")
    # Within tolerance (+2)
    assert chunks_match("a.py:func:12", "a.py:func:10")
    # Within tolerance (-2)
    assert chunks_match("a.py:func:8", "a.py:func:10")
    # Outside tolerance (+3)
    assert not chunks_match("a.py:func:13", "a.py:func:10")
    # Different name
    assert not chunks_match("a.py:func_other:10", "a.py:func:10")


def test_precision_calculation():
    from score import score_question
    
    correct = ["a:func:10", "b:func:20"]
    
    # 2 references, 2 found
    assert score_question(["a:func:10", "b:func:20"], correct) == 1.0
    
    # 2 references, 1 found
    assert score_question(["a:func:10", "c:func:30"], correct) == 0.5
    
    # 2 references, 0 found
    assert score_question(["c:func:10", "d:func:20"], correct) == 0.0
