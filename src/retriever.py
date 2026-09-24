from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retriever with deterministic document IDs."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    @staticmethod
    def _document_id(content: str, source: str) -> str:
        return hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()

    async def initialize(self) -> None:
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)
        vector_path = Path(self.settings.vector_db_path).expanduser()
        vector_path.mkdir(parents=True, exist_ok=True)

        try:
            store = Chroma(
                collection_name=self.settings.rag_collection,
                persist_directory=str(vector_path),
                embedding_function=embeddings,
            )
            ids = [
                self._document_id(doc.page_content, str(doc.metadata.get("source", "unknown.md")))
                for doc in documents
            ]
            existing = store.get(ids=ids)
            existing_ids = set(existing.get("ids", []))
            missing_docs: list = []
            missing_ids: list[str] = []
            for doc, doc_id in zip(documents, ids, strict=True):
                if doc_id not in existing_ids:
                    missing_docs.append(doc)
                    missing_ids.append(doc_id)
            if missing_docs:
                store.add_documents(missing_docs, ids=missing_ids)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise RuntimeError(f"Knowledge index initialization failed: {exc}") from exc

        self._store = store

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if not query.strip():
            return []
        if self._store is None:
            raise ComponentNotReadyError("Knowledge retriever is not ready")

        k = max(1, limit or self.settings.rag_top_k)
        try:
            results = self._store.similarity_search_with_relevance_scores(query.strip(), k=k)
        except Exception as exc:  # pragma: no cover - backend-specific failure
            raise RuntimeError(f"Knowledge retrieval failed: {exc}") from exc

        normalized: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for document, score in results:
            if score < self.settings.rag_min_relevance:
                continue
            source = Path(str(document.metadata.get("source", "unknown.md"))).name
            content = document.page_content.strip()
            if not content:
                continue
            key = (source, content)
            if key in seen:
                continue
            seen.add(key)
            normalized.append({"content": content, "source": source})
        return normalized[:k]
