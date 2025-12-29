from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Callable

from .document_loaders import load_document


@dataclass
class DocumentChunk:
    id: str
    source: str
    text: str
    # 可选的向量表示；只有在配置了 embedder 时才会被填充
    embedding: List[float] | None = None


class KnowledgeBase:
    """极简文档知识库。

    为避免引入重依赖，这里使用非常简单的基于词重叠的检索策略，
    主要目的是演示 RAG pipeline 结构，方便后续替换为向量检索等方案。

    如果调用 ``set_embedder()`` 注册了文本向量函数，``query()`` 会优先尝试
    使用向量相似度进行检索，在没有嵌入可用时回退到当前的词重叠策略。
    """

    def __init__(self) -> None:
        self._chunks: List[DocumentChunk] = []
        self._embedder: Callable[[str], List[float]] | None = None

    # -------- 文档导入 --------
    def ingest_documents(self, paths: List[Path], chunk_size: int = 400) -> None:
        for path in paths:
            text = load_document(path)
            self._add_document(path, text, chunk_size=chunk_size)

    def _add_document(self, path: Path, text: str, chunk_size: int) -> None:
        words = text.split()
        source = str(path)
        chunk_id = 0
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i : i + chunk_size]
            if not chunk_words:
                continue
            chunk_id += 1
            chunk_text = " ".join(chunk_words)

            embedding: List[float] | None = None
            if self._embedder is not None:
                try:
                    embedding = self._embedder(chunk_text)
                except Exception:
                    embedding = None

            self._chunks.append(
                DocumentChunk(
                    id=f"{path.name}-{chunk_id}",
                    source=source,
                    text=chunk_text,
                    embedding=embedding,
                )
            )

    # -------- 可选向量检索挂钩 --------
    def set_embedder(self, embedder: Callable[[str], List[float]]) -> None:
        """注册文本 embedding 函数，用于可选的向量检索。

        embedder 接收一段文本，返回一个浮点向量。例如可以在外部封装
        sentence-transformers、OpenAI / ModelScope embeddings 等后传入。
        """
        self._embedder = embedder

        # 对已有 chunk 立即补 embeddings，避免新旧数据不一致
        for chunk in self._chunks:
            try:
                chunk.embedding = embedder(chunk.text)
            except Exception:
                chunk.embedding = None

    # -------- 简单 / 向量检索 --------
    def query(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        if not self._chunks:
            return []

        # 若配置了 embedder，优先使用向量相似度
        if self._embedder is not None:
            try:
                q_vec = self._embedder(query)
            except Exception:
                q_vec = []

            if q_vec:
                scored_vec: List[tuple[float, DocumentChunk]] = []
                for chunk in self._chunks:
                    if chunk.embedding is None:
                        continue
                    score = _cosine_similarity(q_vec, chunk.embedding)
                    if score > 0.0:
                        scored_vec.append((score, chunk))

                if scored_vec:
                    scored_vec.sort(key=lambda x: x[0], reverse=True)
                    return [c for _, c in scored_vec[:top_k]]

        # 回退到当前的基于词重叠的简单检索
        q_tokens = _tokenize(query)
        scored: List[tuple[int, DocumentChunk]] = []
        for chunk in self._chunks:
            score = _overlap_score(q_tokens, _tokenize(chunk.text))
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def as_context(self, query: str, top_k: int = 5) -> str:
        chunks = self.query(query, top_k=top_k)
        if not chunks:
            return ""
        lines = []
        for c in chunks:
            lines.append(f"[Source: {c.source} | Chunk: {c.id}]\n{c.text}")
        return "\n\n".join(lines)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split() if t.strip()]


def _overlap_score(q_tokens: List[str], d_tokens: List[str]) -> int:
    q_set = set(q_tokens)
    d_set = set(d_tokens)
    return len(q_set & d_set)


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na ** 0.5 * nb ** 0.5)
