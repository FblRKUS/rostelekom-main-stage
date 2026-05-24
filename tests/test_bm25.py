from bm25 import BM25, tokenize


def test_tokenize():
    text = "Hello, world! This is a test_function."
    tokens = tokenize(text)
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens
    assert "function" in tokens


def test_bm25_scoring():
    corpus = [
        "How to handle errors in Python",
        "How to add numbers",
        "Python error handling and exceptions",
        "Just some random text",
    ]

    bm25 = BM25(corpus)

    query = "python error"
    scores = bm25.get_scores(query)

    assert len(scores) == 4

    # The first and third docs should have higher scores than second and fourth
    assert scores[0] > scores[1]
    assert scores[2] > scores[3]
    assert scores[0] > 0
    assert scores[2] > 0
    assert scores[1] == 0
    assert scores[3] == 0


def test_bm25_empty():
    bm25 = BM25([])
    assert bm25.get_scores("test") == []


def test_bm25_serialization():
    corpus = ["test doc", "another doc"]
    bm25 = BM25(corpus)

    data = bm25.to_dict()
    assert "idf" in data
    assert "doc_len" in data

    bm25_loaded = BM25.from_dict(data)
    assert bm25_loaded.corpus_size == 2
    assert bm25_loaded.doc_len == bm25.doc_len
    assert bm25_loaded.get_scores("test") == bm25.get_scores("test")
