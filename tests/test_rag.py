from pathlib import Path

import pytest

from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


@pytest.mark.asyncio
async def test_uninitialized_retriever_rejects_search(tmp_path: Path) -> None:
    from src.config import Settings

    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vectors"),
        embedding_model="test-embedding",
        rag_collection="test-support",
        rag_top_k=3,
        rag_min_relevance=0.35,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )
    retriever = KnowledgeRetriever(settings, tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("shipping")
