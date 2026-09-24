from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path


def _find_project_root() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [here, Path.cwd()]
    for base in list(candidates):
        candidates.extend(base.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "src").is_dir() and (candidate / "knowledge_base").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return candidate

    # Handle a nested repository/project layout.
    for base in (here, Path.cwd()):
        try:
            for src_dir in base.glob("**/src"):
                root = src_dir.parent.resolve()
                if (root / "knowledge_base").is_dir():
                    if str(root) not in sys.path:
                        sys.path.insert(0, str(root))
                    return root
        except OSError:
            pass

    raise RuntimeError(
        f"Could not locate the project root containing src/ and knowledge_base/. "
        f"The app is running from {here}."
    )


PROJECT_ROOT = _find_project_root()

import httpx
import streamlit as st
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


# Define the FastAPI application here instead of importing src.api.server.
# This avoids deployment failures when the api package is not included in a
# Streamlit Cloud checkout, while preserving the required FastAPI endpoints.
settings = load_settings()
pipeline = SupportPipeline(settings, PROJECT_ROOT / "knowledge_base")


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        await pipeline.initialize()
    except Exception as exc:
        pipeline.ready = False
        raise RuntimeError(f"Support pipeline startup failed: {exc}") from exc
    yield
    pipeline.ready = False


api_app = FastAPI(
    title="Customer Support Ticket Agent",
    version="1.0.0",
    lifespan=lifespan,
)


@api_app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@api_app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@api_app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}").rstrip("/")
CHAT_TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "90"))


@st.cache_resource(show_spinner="Starting the FastAPI support backend...")
def start_embedded_api() -> str:
    config = uvicorn.Config(
        api_app,
        host=API_HOST,
        port=API_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="fastapi-backend", daemon=True)
    thread.start()

    deadline = time.monotonic() + 180
    last_error = "backend did not become ready"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{API_BASE_URL}/health", timeout=3.0)
            if response.status_code == 200:
                return API_BASE_URL
            last_error = response.text[:1000]
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(0.5)

    raise RuntimeError(f"FastAPI backend did not become ready: {last_error}")


st.set_page_config(
    page_title="Customer Support Ticket Agent",
    page_icon="🎧",
    layout="centered",
)

st.title("Customer Support Ticket Agent")
st.caption(
    "Answers are grounded in the supplied support knowledge base. "
    "Tickets are created only after required details are validated."
)

try:
    API_BASE_URL = start_embedded_api()
    backend_ready = True
except Exception as exc:
    backend_ready = False
    st.error("The FastAPI support backend could not start.")
    with st.expander("Startup details", expanded=True):
        st.code(f"{type(exc).__name__}: {exc}")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Session: `{st.session_state.session_id}`")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption(f"Backend: `{API_BASE_URL}`")


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role):
        st.markdown(message.get("content", ""))
        sources = message.get("sources") or []
        if sources:
            st.caption("Sources: " + ", ".join(sorted(set(sources))))
        ticket_id = message.get("ticket_id")
        if ticket_id:
            st.success(f"Ticket created: {ticket_id}")


for message in st.session_state.messages:
    render_message(message)


if prompt := st.chat_input("How can we help?", disabled=not backend_ready):
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Working on your request..."):
                response = httpx.post(
                    f"{API_BASE_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=httpx.Timeout(CHAT_TIMEOUT, connect=10.0),
                )

            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "Backend request failed")
                except ValueError:
                    detail = "Backend request failed"
                st.error(str(detail))
            else:
                data = response.json()
                assistant_message = {
                    "role": "assistant",
                    "content": str(data.get("response", "The agent returned no response.")),
                    "sources": list(data.get("sources") or []),
                    "ticket_id": data.get("ticket_id"),
                }
                st.session_state.messages.append(assistant_message)
                st.markdown(assistant_message["content"])
                if assistant_message["sources"]:
                    st.caption(
                        "Sources: "
                        + ", ".join(sorted(set(assistant_message["sources"])))
                    )
                if assistant_message["ticket_id"]:
                    st.success(f"Ticket created: {assistant_message['ticket_id']}")
    except httpx.ConnectError:
        st.error("Could not connect to the FastAPI backend. Please restart the app.")
    except httpx.TimeoutException:
        st.error("The backend took too long to respond. Check the configured model service.")
    except httpx.HTTPError:
        st.error("The backend request failed unexpectedly. Please try again.")
