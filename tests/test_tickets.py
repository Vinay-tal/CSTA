from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


def request() -> TicketCreate:
    return TicketCreate(
        customer_name="Test User",
        customer_email="test@example.com",
        issue_description="Payment was charged twice",
        category="payment",
        summary="Payment support issue: charged twice",
    )


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    repository = TicketRepository()
    first = repository.create("session-1", request())
    second = repository.create("session-1", request())
    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_repository_uses_unique_ids_across_sessions() -> None:
    repository = TicketRepository()
    first = repository.create("session-1", request())
    second = repository.create("session-2", request())
    assert first.ticket_id != second.ticket_id


def test_tool_returns_repository_issued_id() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-tool")
    ticket_id = tool.invoke(request().model_dump())
    assert ticket_id == "CST-2026-0001"
    assert repository.get(ticket_id) is not None
