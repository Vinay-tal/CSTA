import pytest

from src.llm.workflow import IntentAndFields, build_support_workflow
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository


class FakeStructuredModel:
    def __init__(self, extraction: IntentAndFields, answer: str = "Grounded answer"):
        self.extraction = extraction
        self.answer = answer

    def with_structured_output(self, _schema):
        return self

    async def ainvoke(self, messages):
        first = messages[0]
        if "Classify the customer's latest message" in str(first.content):
            return self.extraction
        class Result:
            def __init__(self, content):
                self.content = content
        return Result(self.answer)


class FakeRetriever:
    def __init__(self, chunks=None):
        self.chunks = chunks or []

    async def search(self, _query):
        return self.chunks


@pytest.mark.asyncio
async def test_policy_answer_uses_retrieved_source():
    model = FakeStructuredModel(IntentAndFields(route="answer"))
    retriever = FakeRetriever([{"content": "Standard delivery takes 3-5 business days.", "source": "shipping.md"}])
    graph = build_support_workflow(model, retriever, SessionStore(), TicketRepository())

    result = await graph.ainvoke({
        "session_id": "policy-1",
        "customer_message": "How long does shipping take?",
        "messages": [],
    })

    assert result["response_text"] == "Grounded answer"
    assert result["sources"] == ["shipping.md"]


@pytest.mark.asyncio
async def test_ticket_fields_are_collected_across_turns():
    sessions = SessionStore()
    repository = TicketRepository()
    retriever = FakeRetriever([])

    model = FakeStructuredModel(
        IntentAndFields(
            route="ticket",
            customer_name="Asha",
            customer_email="asha@example.com",
            issue_description="Payment was charged twice",
            category="payment",
        )
    )
    graph = build_support_workflow(model, retriever, sessions, repository)

    result = await graph.ainvoke({
        "session_id": "ticket-1",
        "customer_message": "My name is Asha, email is asha@example.com. My payment was charged twice, payment category.",
        "messages": [],
    })

    assert result["ticket_id"] == "CST-2026-0001"
    assert sessions.get_or_create("ticket-1").ticket_id == "CST-2026-0001"
    assert len(list(repository.all())) == 1


@pytest.mark.asyncio
async def test_missing_ticket_field_gets_one_focused_followup():
    sessions = SessionStore()
    repository = TicketRepository()
    model = FakeStructuredModel(IntentAndFields(route="ticket", customer_name="Asha"))
    graph = build_support_workflow(model, FakeRetriever([]), sessions, repository)

    result = await graph.ainvoke({
        "session_id": "ticket-2",
        "customer_message": "I need to report an issue. My name is Asha.",
        "messages": [],
    })

    assert result["response_text"] == "What email address should we use for the ticket?"
    assert result.get("ticket_id") is None
    assert len(list(repository.all())) == 0
