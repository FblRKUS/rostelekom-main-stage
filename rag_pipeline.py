import logging
from typing import Optional, List, Dict, Any

try:
    import ollama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from vector_store import SearchResult

logger = logging.getLogger(__name__)


class RAGAnswerGenerator:
    def __init__(self, use_llm: bool = True, model: str = "mistral:7b"):
        self.use_llm = use_llm
        self.model = model
        if self.use_llm and not OLLAMA_AVAILABLE:
            logger.warning("Ollama module is not installed. LLM generation disabled.")
            self.use_llm = False

    def _build_messages(
        self,
        question: str,
        chunks: list[SearchResult],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> list[dict]:
        context = ""
        for i, chunk in enumerate(chunks, 1):
            context += f"--- Fragment {i} ({chunk.metadata.get('file_path')}) ---\n"
            context += chunk.content + "\n\n"

        system_prompt = (
            "You are a helpful coding assistant. CRITICAL RULE: You MUST answer the user's question in the EXACT SAME LANGUAGE as the question itself. "
            "If the question is in Russian, you MUST write your answer in Russian. Если вопрос на русском, отвечай СТРОГО на русском языке.\n"
            "Answer based ONLY on the provided code fragments. If the code fragments don't contain the answer, say so clearly (e.g. 'В предоставленном коде нет информации об этом').\n\n"
            "Code fragments:\n"
            f"{context}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    def generate(
        self,
        question: str,
        chunks: list[SearchResult],
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.use_llm:
            chunk_list = "\n".join([f"- {c.chunk_id}" for c in chunks])
            return f"LLM is disabled. Found {len(chunks)} relevant fragments:\n{chunk_list}"

        messages = self._build_messages(question, chunks, history)

        try:
            response = ollama.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to generate answer with Ollama: {e}")
            return f"Error connecting to LLM: {e}"

    def generate_stream(
        self,
        question: str,
        chunks: list[SearchResult],
        history: Optional[List[Dict[str, Any]]] = None,
    ):
        if not self.use_llm:
            yield self.generate(question, chunks)
            return

        messages = self._build_messages(question, chunks, history)

        try:
            stream = ollama.chat(model=self.model, messages=messages, stream=True)
            for chunk in stream:
                yield chunk["message"]["content"]
        except Exception as e:
            logger.error(f"Failed to stream answer with Ollama: {e}")
            yield f"Error connecting to LLM: {e}"
