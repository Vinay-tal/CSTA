from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled configuration for the support agent."""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    rag_min_relevance: float
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int


def _env(name: str, default: str) -> str:
    """Read environment variables, including Streamlit Cloud secrets."""
    try:
        import streamlit as st

        if name in st.secrets:
            value = str(st.secrets[name]).strip()
            if value:
                return value
    except Exception:
        pass
    return os.getenv(name, default)


def load_settings() -> Settings:
    """Load provider-neutral settings from environment variables or Streamlit Secrets."""
    load_dotenv()
    return Settings(
        # Hugging Face's OpenAI-compatible router is a hosted inference option;
        # local Ollama remains the documented local-development default.
        llm_base_url=_env("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=_env("LLM_API_KEY", "not-required"),
        llm_model=_env("LLM_MODEL", "qwen2.5:3b"),
        vector_db_path=_env("VECTOR_DB_PATH", ".data/vector_db"),
        embedding_model=_env(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_collection=_env("RAG_COLLECTION", "customer-support"),
        rag_top_k=int(_env("RAG_TOP_K", "3")),
        rag_min_relevance=float(_env("RAG_MIN_RELEVANCE", "0.35")),
        api_host=_env("API_HOST", "127.0.0.1"),
        api_port=int(_env("API_PORT", "8000")),
        streamlit_host=_env("STREAMLIT_HOST", "0.0.0.0"),
        streamlit_port=int(_env("STREAMLIT_PORT", "8501")),
    )
