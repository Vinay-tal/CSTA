from __future__ import annotations

import json
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ValidationError

from src.llm.prompts import ANSWER_TEMPLATE, EXTRACTION_SYSTEM_PROMPT, SYSTEM_PROMPT
from src.models import TicketCreate
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


class IntentAndFields(BaseModel):
    route: Literal["answer", "ticket"]
    customer_name: str | None = None
    customer_email: str | None = None
    issue_description: str | None = None
    category: Literal["order", "payment", "account", "technical", "other"] | None = None


class SupportWorkflowState(TypedDict, total=False):
    session_id: str
    customer_message: str
    messages: Annotated[list, add_messages]
    retrieved_chunks: list[dict[str, str]]
    route: str
    extracted_fields: dict[str, str]
    response_text: str
    sources: list[str]
    ticket_id: str | None


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_SENSITIVE_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _safe_text(value: str) -> str:
    return _SENSITIVE_RE.sub("[redacted payment-card number]", value).strip()


def _history_text(history: list[dict[str, str]], limit: int = 8) -> str:
    rows = history[-limit:]
    if not rows:
        return "(no previous turns)"
    return "\n".join(f"{row['role']}: {row['content']}" for row in rows)


def _summary(issue: str, category: str) -> str:
    text = " ".join(issue.split())
    prefix = f"{category.title()} support issue: "
    remaining = max(5, 160 - len(prefix))
    return (prefix + text[:remaining]).strip()[:160]


def build_support_workflow(
    model: BaseChatModel,
    retriever,
    sessions: SessionStore,
    repository: TicketRepository,
):
    """Build the LangGraph workflow used by the API pipeline."""

    async def retrieve(state: SupportWorkflowState) -> SupportWorkflowState:
        chunks = await retriever.search(state["customer_message"])
        return {"retrieved_chunks": chunks}

    async def decide(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        history = _history_text(session.history)
        prompt = (
            f"Latest customer message:\n{state['customer_message']}\n\n"
            f"Conversation so far:\n{history}"
        )
        try:
            # Prefer LangChain structured output when the selected hosted/local
            # model supports it. Some OpenAI-compatible open-source endpoints do
            # not expose JSON-schema output, so fall back to ordinary JSON text.
            try:
                structured_model = model.with_structured_output(IntentAndFields)
                result = await structured_model.ainvoke(
                    [
                        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )
                if not isinstance(result, IntentAndFields):
                    result = IntentAndFields.model_validate(result)
            except Exception:
                extraction_prompt = (
                    f"{EXTRACTION_SYSTEM_PROMPT}\n\n"
                    "Return ONLY a JSON object with keys route, customer_name, "
                    "customer_email, issue_description, and category. Use null "
                    "for values not explicitly present in the latest message.\n\n"
                    f"{prompt}"
                )
                raw = await model.ainvoke(
                    [HumanMessage(content=extraction_prompt)]
                )
                raw_text = raw.content if isinstance(raw.content, str) else str(raw.content)
                raw_text = raw_text.strip()
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                if not match:
                    raise ValueError("The model did not return a JSON classification")
                result = IntentAndFields.model_validate(json.loads(match.group(0)))
        except Exception as exc:
            raise RuntimeError(f"Unable to classify the customer request: {exc}") from exc

        extracted: dict[str, str] = {}
        for field in ("customer_name", "customer_email", "issue_description", "category"):
            value = _clean(getattr(result, field))
            if value:
                extracted[field] = _safe_text(value)
        return {"route": result.route, "extracted_fields": extracted}

    async def answer(state: SupportWorkflowState) -> SupportWorkflowState:
        chunks = state.get("retrieved_chunks", [])
        sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))
        context = "\n\n".join(
            f"[{chunk['source']}]\n{chunk['content']}" for chunk in chunks
        ) or "(No relevant knowledge-base content was retrieved.)"
        session = sessions.get_or_create(state["session_id"])
        prompt = ANSWER_TEMPLATE.format(
            context=context,
            session=_history_text(session.history),
            message=state["customer_message"],
        )
        try:
            result = await model.ainvoke(
                [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            text = result.content if isinstance(result.content, str) else str(result.content)
            text = text.strip()
        except Exception as exc:
            raise RuntimeError(f"Unable to generate the support response: {exc}") from exc
        if not text:
            raise RuntimeError("The support model returned an empty response")
        return {"response_text": text, "sources": sources}

    async def collect_or_create(state: SupportWorkflowState) -> SupportWorkflowState:
        session = sessions.get_or_create(state["session_id"])
        extracted = state.get("extracted_fields", {})

        for field in ("customer_name", "customer_email", "issue_description", "category"):
            value = _clean(extracted.get(field))
            if not value:
                continue
            if field == "customer_email" and not _EMAIL_RE.match(value):
                # Keep invalid input out of persistent session state.
                continue
            if field == "category" and value not in {"order", "payment", "account", "technical", "other"}:
                continue
            setattr(session, field, value)

        if session.ticket_id:
            ticket = repository.get(session.ticket_id)
            if ticket is None:
                session.ticket_id = None
            else:
                return {
                    "response_text": f"Your support ticket is already created as **{ticket.ticket_id}**.",
                    "sources": [],
                    "ticket_id": ticket.ticket_id,
                }

        missing = session.missing_ticket_fields()
        prompts = {
            "customer_name": "What name should we put on the support ticket?",
            "customer_email": "What email address should we use for the ticket?",
            "issue_description": "Please describe the issue you need help with.",
            "category": "Which category fits the issue: order, payment, account, technical, or other?",
        }

        # Detect an explicitly invalid email separately so the user gets a useful correction.
        raw_email = extracted.get("customer_email")
        if raw_email and not _EMAIL_RE.match(raw_email):
            return {
                "response_text": "That email address does not look valid. Please provide a valid email address such as name@example.com.",
                "sources": [],
                "ticket_id": None,
            }

        if missing:
            return {"response_text": prompts[missing[0]], "sources": [], "ticket_id": None}

        try:
            request = TicketCreate(
                customer_name=session.customer_name or "",
                customer_email=session.customer_email or "",
                issue_description=_safe_text(session.issue_description or ""),
                category=session.category,  # type: ignore[arg-type]
                summary=_summary(session.issue_description or "", session.category or "other"),
            )
        except ValidationError as exc:
            # Validation is deliberately kept inside the application boundary so
            # invalid data can never reach the repository/tool side effect.
            fields = {str(error["loc"][0]) for error in exc.errors() if error.get("loc")}
            field = next(iter(fields), "required ticket details")
            return {"response_text": f"I still need a valid {field.replace('_', ' ')} before I can create the ticket.", "sources": [], "ticket_id": None}

        tool = create_ticket_tool(repository, state["session_id"])
        try:
            ticket_id = await tool.ainvoke(request.model_dump())
        except Exception as exc:
            raise RuntimeError(f"Unable to create the support ticket: {exc}") from exc
        session.ticket_id = str(ticket_id)
        return {
            "response_text": f"Your support ticket has been created successfully. Ticket ID: **{ticket_id}**.",
            "sources": [],
            "ticket_id": str(ticket_id),
        }

    def select_route(state: SupportWorkflowState) -> str:
        route = state.get("route")
        if route not in {"answer", "ticket"}:
            raise RuntimeError("The agent selected an invalid route")
        return route

    graph = StateGraph(SupportWorkflowState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("decide", decide)
    graph.add_node("answer", answer)
    graph.add_node("ticket", collect_or_create)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "decide")
    graph.add_conditional_edges("decide", select_route, {"answer": "answer", "ticket": "ticket"})
    graph.add_edge("answer", END)
    graph.add_edge("ticket", END)
    return graph.compile()
