from score import chunks_match, score_question, parse_chunk_id


def test_parse_chunk_id():
    assert parse_chunk_id("file.py:func:10") == ("file.py", "func", 10)
    assert parse_chunk_id("file.py:func") is None
    assert parse_chunk_id("file.py:func:abc") is None


def test_precision_with_line_tolerance():
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
    # Different path
    assert not chunks_match("b.py:func:10", "a.py:func:10")


def test_precision_calculation():
    correct = ["a:func:10", "b:func:20"]

    # 2 references, 2 found
    assert score_question(["a:func:10", "b:func:20"], correct) == 1.0

    # 2 references, 1 found
    assert score_question(["a:func:10", "c:func:30"], correct) == 0.5

    # 2 references, 0 found
    assert score_question(["c:func:10", "d:func:20"], correct) == 0.0

    # 0 references
    assert score_question(["c:func:10"], []) == 0.0
