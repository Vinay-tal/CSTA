from __future__ import annotations

from collections.abc import Iterable
from threading import RLock

from langchain_core.tools import StructuredTool

from src.models import Ticket, TicketCreate


class TicketRepository:
    """In-memory mock ticket service with session-level idempotency."""

    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._session_ticket: dict[str, str] = {}
        self._lock = RLock()

    def create(self, session_id: str, data: TicketCreate) -> Ticket:
        with self._lock:
            existing_id = self._session_ticket.get(session_id)
            if existing_id:
                return self._tickets[existing_id]

            ticket_id = f"CST-2026-{len(self._tickets) + 1:04d}"
            ticket = Ticket(ticket_id=ticket_id, **data.model_dump())
            self._tickets[ticket_id] = ticket
            self._session_ticket[session_id] = ticket_id
            return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        with self._lock:
            return self._tickets.get(ticket_id)

    def all(self) -> Iterable[Ticket]:
        with self._lock:
            return tuple(self._tickets.values())


def create_ticket_tool(repository: TicketRepository, session_id: str) -> StructuredTool:
    """Bind the supplied mock repository to one session."""

    def create_ticket(
        customer_name: str,
        customer_email: str,
        issue_description: str,
        category: str,
        summary: str,
    ) -> str:
        request = TicketCreate(
            customer_name=customer_name.strip(),
            customer_email=customer_email.strip(),
            issue_description=issue_description.strip(),
            category=category,  # type: ignore[arg-type]
            summary=summary.strip(),
        )
        return repository.create(session_id, request).ticket_id

    return StructuredTool.from_function(
        func=create_ticket,
        name="create_support_ticket",
        description="Create one support ticket only after all required customer fields are validated.",
    )
