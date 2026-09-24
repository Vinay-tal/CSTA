from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.workflow = None
        self.ready = False

    async def initialize(self) -> None:
        """Initialize shared retrieval and workflow components once."""
        await self.retriever.initialize()
        self.workflow = build_support_workflow(
            self.model, self.retriever, self.sessions, self.tickets
        )
        self.ready = True

    async def process(self, session_id: str, message: str) -> ChatResponse:
        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError("Support pipeline is not ready")

        session_id = session_id.strip()
        message = message.strip()
        if not session_id:
            raise AgentProcessingError("session_id must not be blank")
        if not message:
            raise AgentProcessingError("message must not be blank")

        session = self.sessions.get_or_create(session_id)
        state = {
            "session_id": session_id,
            "customer_message": message,
            "messages": session.history.copy(),
        }
        try:
            result = await self.workflow.ainvoke(state)
            response_text = str(result.get("response_text", "")).strip()
            if not response_text:
                raise AgentProcessingError("The agent returned an empty response")

            sources = [str(source).split("/")[-1].split("\\")[-1] for source in result.get("sources", [])]
            sources = list(dict.fromkeys(sources))
            ticket_id = result.get("ticket_id")
            if ticket_id and self.tickets.get(str(ticket_id)) is None:
                raise AgentProcessingError("The agent returned an unknown ticket identifier")

            session.history.append({"role": "user", "content": message})
            session.history.append({"role": "assistant", "content": response_text})
            return ChatResponse(
                success=True,
                session_id=session_id,
                response=response_text,
                sources=sources,
                ticket_id=str(ticket_id) if ticket_id else None,
            )
        except AgentProcessingError:
            raise
        except Exception as exc:
            raise AgentProcessingError(str(exc)) from exc
