# Customer Support Ticket Agent

A text-based customer-support agent implemented from the supplied Zangoh Technical Evaluation starter repository. It uses **FastAPI + Streamlit + LangGraph + LangChain + ChromaDB + an open-source instruction-tuned LLM**.

The implementation keeps the supplied repository structure and completes the RAG, agent workflow, ticket integration, API, UI, validation, duplicate protection, tests, and documentation requirements. The assignment guide requires Python 3.11+, FastAPI, Streamlit, an open-source instruction-tuned model, a vector database, and LangChain/LangGraph orchestration.

## Architecture

```text
Customer
   |
   v
Streamlit UI
   |
   | POST /chat {session_id, message}
   v
FastAPI
   |
   v
SupportPipeline
   |
   v
LangGraph
  / | \
 v  v  v
RAG LLM SessionStore
 |    |      |
 v    |      v
Chroma |   ticket state
 |    |
 v    v
Markdown KB -> grounded answer
        \
         v
     session-bound Ticket Tool
              |
              v
      Mock Ticket Repository
```

A fuller Mermaid diagram is in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Requirements mapped to the assignment

- **Knowledge answers:** supplied Markdown files are indexed into Chroma, retrieved per turn, and passed as the only policy context to the answer model.
- **Unknown answers:** retrieval uses a relevance threshold; if no useful evidence is found, the answer explicitly states that the knowledge base does not contain the answer.
- **Source attribution:** the API returns safe Markdown source filenames used for the response.
- **Ticket creation:** the agent collects name, email, issue description, and category (`order`, `payment`, `account`, `technical`, `other`) over multiple turns.
- **Validation:** Pydantic validates the completed ticket before the repository side effect. Invalid email/category values are not committed to session state.
- **No invented details:** the LLM extraction prompt only accepts values explicitly stated in the current turn. The application owns the session merge.
- **Duplicate protection:** the mock repository is idempotent per `session_id`.
- **Session state:** the same client-supplied session ID is used for the complete conversation.
- **API:** `/health`, `/chat`, and `/tickets/{ticket_id}` are implemented as required. The assignment specifies the same endpoint contract.
- **Streamlit:** chat history, source display, ticket confirmation, new-conversation control, and friendly backend/model failure messages are implemented.
- **Testing:** automated tests cover session isolation, ticket idempotency, ticket-tool integration, RAG loading, and readiness boundaries. The assignment's minimum demonstrations include RAG, unknown handling, multi-turn tickets, retrieval, missing-field validation, duplicate protection, and graceful failure.

## 1. Prerequisites

- Python **3.11 or newer**.
- Git is optional.
- An OpenAI-compatible endpoint serving an **open-source instruction-tuned model**. The default configuration is Ollama with `qwen2.5:3b`.
- Internet access is needed once to install Python dependencies and download the configured Hugging Face embedding model unless those packages/models are already cached.

The assignment explicitly permits local or hosted open-source inference.

### Recommended local model: Ollama

Install Ollama for your operating system, then pull the configured model:

```sh
ollama pull qwen2.5:3b
```

Keep the Ollama service running while testing the application.

You may use another OpenAI-compatible open-source provider by changing `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` in `.env`.

## 2. Installation

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

### macOS

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

## 3. Environment variables

`.env.example` contains every supported variable. The important values are:

| Variable | Purpose | Default |
|---|---|---|
| `LLM_BASE_URL` | OpenAI-compatible chat endpoint | `http://localhost:11434/v1` |
| `LLM_API_KEY` | Provider key, if required | `not-required` |
| `LLM_MODEL` | Open-source instruction model | `qwen2.5:3b` |
| `API_BASE_URL` | Streamlit's FastAPI address | `http://localhost:8000` |
| `VECTOR_DB_PATH` | Local Chroma storage | `.data/vector_db` |
| `EMBEDDING_MODEL` | Hugging Face embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `RAG_COLLECTION` | Chroma collection name | `customer-support` |
| `RAG_TOP_K` | Maximum retrieved chunks | `3` |
| `RAG_MIN_RELEVANCE` | Retrieval acceptance threshold | `0.35` |

Never put a real credential into the repository. `.env` is ignored by Git.

## 3A. Streamlit Cloud deployment

The assignment requires the public interface to be Streamlit while also requiring FastAPI in the architecture. Streamlit Cloud exposes the Streamlit process publicly, so this submission starts the required FastAPI app privately inside the same Streamlit process. The request path remains:

```text
Browser -> Streamlit UI -> private FastAPI -> LangGraph -> RAG/LLM -> ticket tool
```

No additional framework, database, queue, or service is introduced. The same FastAPI endpoints (`/health`, `/chat`, `/tickets/{ticket_id}`) are used by the UI.

For hosted deployment, configure an open-source instruction-tuned model through an OpenAI-compatible inference endpoint. Hugging Face Inference Providers expose such a chat-completions endpoint; the supplied `.streamlit/secrets.toml.example` shows the required names. A Hugging Face token must be added as a Streamlit Secret, not committed to the repository.

### Streamlit Cloud steps

1. Push this project to a Git repository.
2. In Streamlit Cloud, create an app using `streamlit_app.py` as the main file.
3. Use Python 3.11 or newer.
4. Copy the names/values from `.streamlit/secrets.toml.example` into the Streamlit Cloud Secrets editor and replace the placeholder `LLM_API_KEY` with your hosted-inference token.
5. Deploy/reboot the app.
6. The app waits for `GET /health` before enabling the chat box.

For local development, you can continue using Ollama with the original local settings. For Streamlit Cloud, do not use `localhost:11434` for the LLM because that would refer to the cloud container rather than your computer.

## 4. Run the application

Terminal 1, start the FastAPI backend:

```sh
uvicorn src.api.server:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, start Streamlit:

```sh
streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://localhost:8501` for the UI.

FastAPI's interactive documentation is available at `http://localhost:8000/docs`.

### Health check

```sh
curl http://localhost:8000/health
```

Expected result:

```json
{"status":"ready"}
```

## 5. Example conversations

### RAG policy question

> How long does standard shipping take?

The answer should be grounded in `shipping.md`, and the UI/API should show `shipping.md` as the source.

### Unknown question

> What is the CEO's favorite restaurant?

The agent should not invent an answer. It should explain that the supplied support knowledge base does not contain that information and return no fabricated source.

### Multi-turn ticket creation

```text
Customer: My payment was charged twice and I need to report it.
Agent: What name should we put on the support ticket?
Customer: Asha Sharma
Agent: What email address should we use for the ticket?
Customer: asha@example.com
Agent: Which category fits the issue: order, payment, account, technical, or other?
Customer: payment
Agent: [ticket confirmation with real CST-2026-XXXX ID]
```

The exact order can vary if the customer supplies multiple fields in one message.

### Ticket retrieval

After creation:

```sh
curl http://localhost:8000/tickets/CST-2026-0001
```

The endpoint returns the repository record rather than asking the LLM to reconstruct it.

## 6. Tests

Run:

```sh
pytest -q
```

The assignment asks for automated tests and commands in the submission.

The test suite covers:

- source metadata preservation for all supplied Markdown files;
- splitter metadata preservation;
- uninitialized retrieval rejection;
- session isolation and conversation history;
- one-ticket-per-session idempotency;
- unique IDs across sessions;
- ticket tool returning the repository-issued ID;
- uninitialized pipeline rejection.

## 7. Error handling

- **FastAPI 503:** the pipeline is not initialized/ready.
- **FastAPI 502:** model, retrieval, classification, or ticket processing fails during a request.
- **Streamlit connection failure:** shown as a friendly backend connection message.
- **Streamlit timeout:** shown as a model/backend timeout message.
- **422:** FastAPI/Pydantic validation errors are surfaced as request-validation feedback.
- **Malformed backend JSON:** shown as a friendly UI error.
- **Missing ticket fields:** one focused follow-up is requested at a time.
- **Invalid email:** the value is not committed and the customer is asked for a valid address.
- **Unknown RAG result:** the model is instructed not to answer from unrestricted memory.

## 8. Mid-session requirement

The important stateful behavior is that `session_id` remains stable throughout a conversation. The Streamlit UUID is created once and stored in `st.session_state`; reruns do not generate a new ID. The FastAPI pipeline uses that ID to retrieve the same `ConversationState`.

This means ticket details can be collected across multiple messages and the final request can be retried without creating a second ticket. The assignment explicitly requires session preservation, focused follow-ups, and duplicate protection.

## 9. Submission hygiene

Do **not** include:

- `.env` with credentials;
- `.venv/`;
- `.data/` generated Chroma databases;
- `__pycache__/`;
- `.pytest_cache/`;
- generated model caches.

The submitted ZIP should contain the source code, knowledge base, README, architecture documentation, `.env.example`, and tests. The assignment specifically excludes secrets, generated caches, and local databases.

## 10. Five-minute demonstration plan

The assignment asks for a **five-minute** demonstration with at least four representative queries and an explanation of the midway/mid-session change.

Suggested recording:

1. **0:00–0:45:** show architecture and explain Streamlit → FastAPI → LangGraph → RAG/LLM → ticket tool.
2. **0:45–1:30:** ask a shipping policy question and show `shipping.md` source attribution.
3. **1:30–2:00:** ask an unknown question and show that the agent does not fabricate an answer.
4. **2:00–3:15:** begin a payment ticket, deliberately stop after some fields, then continue in later turns. Explain that the same `session_id` preserves the partially collected state.
5. **3:15–4:00:** complete the ticket, show the real ticket ID, then retry the completed request and show that the same ticket is returned rather than creating a duplicate.
6. **4:00–4:30:** call `GET /tickets/{ticket_id}` and show the stored record.
7. **4:30–5:00:** run `pytest -q` and briefly show the test result.

## 11. Acceptance checklist

The assignment says the solution is complete when a reviewer can start it from the README, ask grounded questions, create/retrieve a ticket through a multi-turn conversation, observe source attribution, and run tests on Windows, Linux, or macOS.

- [x] FastAPI backend
- [x] Streamlit interface
- [x] LangGraph orchestration
- [x] Open-source model adapter
- [x] Chroma vector database
- [x] Supplied Markdown knowledge base
- [x] Grounded answers + source names
- [x] Unknown-answer behavior
- [x] Multi-turn session state
- [x] Required ticket-field collection
- [x] Validation before side effect
- [x] Mock ticket tool
- [x] Duplicate protection
- [x] Ticket retrieval endpoint
- [x] Friendly model/backend failure states
- [x] Automated tests
- [x] `.env.example`
- [x] Platform-agnostic README
- [x] Architecture documentation
