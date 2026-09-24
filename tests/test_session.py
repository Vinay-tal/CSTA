from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")
    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_sessions_are_isolated() -> None:
    store = SessionStore()
    first = store.get_or_create("first")
    first.customer_name = "Asha"
    second = store.get_or_create("second")
    assert second.customer_name is None


def test_history_preserves_turns() -> None:
    store = SessionStore()
    state = store.get_or_create("demo")
    state.history.append({"role": "user", "content": "Hello"})
    state.history.append({"role": "assistant", "content": "Hi"})
    assert [item["role"] for item in state.history] == ["user", "assistant"]
