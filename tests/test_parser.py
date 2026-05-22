import os
import tempfile
from pathlib import Path
from parser import CodeChunk, CodeIndexer

import pytest


@pytest.fixture
def temp_workspace():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_parse_simple_function(temp_workspace):
    file_path = temp_workspace / "test.py"
    file_path.write_text(
        "def hello(name):\n"
        '    """Say hello"""\n'
        "    print(f'Hello {name}')\n",
        encoding="utf-8",
    )
    
    indexer = CodeIndexer()
    chunks = indexer.parse_file(file_path, temp_workspace)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.name == "hello"
    assert chunk.type == "function"
    assert chunk.docstring == "Say hello"
    assert chunk.start_line == 1
    assert chunk.file_path == "test.py"
    assert chunk.chunk_id == "test.py:hello:1"
    assert "print(f'Hello {name}')" in chunk.content


def test_parse_class_with_methods(temp_workspace):
    file_path = temp_workspace / "test_cls.py"
    file_path.write_text(
        "class Calculator:\n"
        '    """A simple calculator"""\n'
        "    def add(self, a, b):\n"
        "        return a + b\n"
        "\n"
        "    def sub(self, a, b):\n"
        "        return a - b\n",
        encoding="utf-8",
    )
    
    indexer = CodeIndexer()
    chunks = indexer.parse_file(file_path, temp_workspace)
    
    assert len(chunks) == 3
    
    # Class chunk
    cls_chunk = next(c for c in chunks if c.name == "Calculator")
    assert cls_chunk.type == "class"
    assert cls_chunk.docstring == "A simple calculator"
    
    # Method chunks
    add_chunk = next(c for c in chunks if c.name == "Calculator.add")
    assert add_chunk.type == "function"
    assert "return a + b" in add_chunk.content
    
    sub_chunk = next(c for c in chunks if c.name == "Calculator.sub")
    assert sub_chunk.type == "function"
    assert "return a - b" in sub_chunk.content


def test_parse_async_function(temp_workspace):
    file_path = temp_workspace / "test_async.py"
    file_path.write_text(
        "async def fetch_data():\n"
        "    await asyncio.sleep(1)\n"
        "    return 42\n",
        encoding="utf-8",
    )
    
    indexer = CodeIndexer()
    chunks = indexer.parse_file(file_path, temp_workspace)
    
    assert len(chunks) == 1
    assert chunks[0].name == "fetch_data"
    assert chunks[0].type == "function"


def test_parse_error_handling(temp_workspace, capsys):
    file_path = temp_workspace / "test_error.py"
    file_path.write_text("def invalid_syntax(\n", encoding="utf-8")
    
    indexer = CodeIndexer()
    chunks = indexer.parse_file(file_path, temp_workspace)
    
    assert len(chunks) == 0


def test_recursive_directory_scan(temp_workspace):
    (temp_workspace / "dir1").mkdir()
    (temp_workspace / "dir2").mkdir()
    
    (temp_workspace / "dir1" / "f1.py").write_text("def f1(): pass\n")
    (temp_workspace / "dir2" / "f2.py").write_text("def f2(): pass\n")
    (temp_workspace / "f3.py").write_text("def f3(): pass\n")
    
    indexer = CodeIndexer()
    chunks = indexer.scan_directory(temp_workspace)
    
    assert len(chunks) == 3
    names = {c.name for c in chunks}
    assert names == {"f1", "f2", "f3"}


def test_java_bonus(temp_workspace):
    file_path = temp_workspace / "Test.java"
    file_path.write_text(
        "public class Test {\n"
        "    public void doSomething() {\n"
        "        System.out.println(\"Hello\");\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    
    indexer = CodeIndexer()
    chunks = indexer.parse_file(file_path, temp_workspace)
    
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.name == "Test.doSomething"
    assert chunk.type == "function"
    assert chunk.file_path == "Test.java"
