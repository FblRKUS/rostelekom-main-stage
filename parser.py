import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    file_path: str
    type: str
    name: str
    content: str
    docstring: str | None
    start_line: int

    @property
    def chunk_id(self) -> str:
        return f"{self.file_path}:{self.name}:{self.start_line}"


class _Visitor(ast.NodeVisitor):
    def __init__(self, source_lines: list[str], file_path: str):
        self.source_lines = source_lines
        self.file_path = file_path
        self.chunks: list[CodeChunk] = []
        self._current_class: str | None = None

    def _extract_content(self, node: ast.AST) -> str:
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        return "\n".join(self.source_lines[start:end])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record_function(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        content = self._extract_content(node)
        self.chunks.append(
            CodeChunk(
                file_path=self.file_path,
                type="class",
                name=node.name,
                content=content,
                docstring=ast.get_docstring(node),
                start_line=node.lineno,
            )
        )
        prev_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = prev_class

    def _record_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        name = f"{self._current_class}.{node.name}" if self._current_class else node.name
        content = self._extract_content(node)
        self.chunks.append(
            CodeChunk(
                file_path=self.file_path,
                type="function",
                name=name,
                content=content,
                docstring=ast.get_docstring(node),
                start_line=node.lineno,
            )
        )


class CodeIndexer:
    def parse_file(self, file_path: Path, base_path: Path) -> list[CodeChunk]:
        try:
            source = file_path.read_text(encoding="utf-8", errors="replace")
            rel_path = file_path.relative_to(base_path).as_posix()
            
            if file_path.suffix == ".py":
                return self._parse_python(source, rel_path)
            elif file_path.suffix == ".java":
                return self._parse_java(source, rel_path)
            return []
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}")
            return []

    def _parse_python(self, source: str, rel_path: str) -> list[CodeChunk]:
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {rel_path}: {e}")
            return []

        source_lines = source.splitlines()
        visitor = _Visitor(source_lines, rel_path)
        visitor.visit(tree)
        return visitor.chunks

    def _parse_java(self, source: str, rel_path: str) -> list[CodeChunk]:
        chunks = []
        lines = source.splitlines()
        
        class_name = "Unknown"
        # Extremely primitive Java method extraction
        class_pattern = re.compile(r'class\s+(\w+)')
        method_pattern = re.compile(r'(?:public|protected|private)\s+(?:static\s+)?[A-Za-z0-9_<>\.\[\]]+\s+(\w+)\s*\(')
        
        for i, line in enumerate(lines):
            cls_match = class_pattern.search(line)
            if cls_match:
                class_name = cls_match.group(1)
                continue
                
            method_match = method_pattern.search(line)
            if method_match:
                method_name = method_match.group(1)
                chunks.append(CodeChunk(
                    file_path=rel_path,
                    type="function",
                    name=f"{class_name}.{method_name}",
                    content=line.strip(),
                    docstring=None,
                    start_line=i + 1
                ))
                
        return chunks

    def scan_directory(self, directory: Path) -> list[CodeChunk]:
        all_chunks = []
        # Sort paths to ensure stable order
        py_files = sorted(directory.rglob("*.py"))
        java_files = sorted(directory.rglob("*.java"))
        
        for p in py_files + java_files:
            if not p.is_file():
                continue
            all_chunks.extend(self.parse_file(p, directory))
            
        return all_chunks
