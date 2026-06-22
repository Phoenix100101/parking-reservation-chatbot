# RAG Chatbot — Parking Reservation Assistant

A chatbot for a parking facility built on **LangGraph**. The bot answers questions
about the facility, reports live space availability and operating hours, and runs
multi-turn reservations (book / cancel / modify).

The architecture is a state-machine graph: input guardrail → intent router → one
of four agents → output guardrail.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Step-by-Step Setup & Run](#step-by-step-setup--run)
  - [Step 1. Prerequisites](#step-1-prerequisites)
  - [Step 2. Clone & environment variables](#step-2-clone--environment-variables)
  - [Step 3. Set up the Python environment (uv)](#step-3-set-up-the-python-environment-uv)
  - [Step 4. Start the databases in Docker](#step-4-start-the-databases-in-docker)
  - [Step 5. Initialize and seed PostgreSQL](#step-5-initialize-and-seed-postgresql)
  - [Step 6. Seed Weaviate](#step-6-seed-weaviate)
  - [Step 7. Run the application](#step-7-run-the-application)
- [Environment Variables](#environment-variables)
- [Testing](#testing)
- [Useful Commands](#useful-commands)
- [Security](#security)

---

## Architecture

```
                    [User Input]
                         │
                         ▼
                 [Input Guardrail]            ← blocks prompt-injection and sensitive PII (cards, SSNs)
                         │
            input_blocked? ──yes──► END
                         │ no
                         ▼
                  [Router Agent]              ← LLM intent classification
                         │
     ┌───────────────────┼───────────────────┬───────────────────┐
     ▼                   ▼                   ▼                   ▼
[RAG Agent]       [Dynamic Agent]     [Reservation Agent]   [Out of Scope]
 Weaviate           PostgreSQL          multi-turn booking     refusal
 (info_query)      (dynamic_query)      (reservation)         (out_of_scope)
     │                   │                   │                   │
     └───────────────────┴─────────┬─────────┴───────────────────┘
                                    ▼
                           [Output Guardrail]   ← masks PII leaked into the response
                                    │
                                    ▼
                           [Response to User]
```

**The router's four intents:**

| Intent          | Agent             | Data source        | Example queries |
|-----------------|-------------------|--------------------|-----------------|
| `info_query`    | RAG Agent         | Weaviate (vector search) | "where is the entrance?", "what zones are there?" |
| `dynamic_query` | Dynamic Agent     | PostgreSQL (live)  | "how many free spots on floor 2?", "when are you open?" |
| `reservation`   | Reservation Agent | PostgreSQL (writes)| "book a spot for tomorrow", "cancel my booking" |
| `out_of_scope`  | Out-of-Scope Node | —                  | "write me a poem", "what's the weather?" |

Conversation state (`ChatState`) is persisted across turns via LangGraph's
`MemorySaver` checkpointer keyed by `thread_id`, so a multi-turn reservation
(collecting time, vehicle plate, email, confirmation) is not lost between turns.

---

## Tech Stack

- **Language:** Python 3.13+
- **Package manager:** [uv](https://docs.astral.sh/uv/)
- **Orchestration:** LangGraph + LangChain
- **LLM / embeddings:** OpenAI (`gpt-4o` / `gpt-4o-mini`, `text2vec-openai`)
- **Vector DB:** Weaviate
- **Relational DB:** PostgreSQL (`parking` schema)
- **Guardrails / PII:** spaCy (`en_core_web_sm`) + regex
- **Code quality:** ruff, mypy, pytest

---

## Project Structure

```
rag-chatbot-reservation/
├── docker-compose.yml          # Postgres + Weaviate
├── pyproject.toml              # project dependencies
├── main.py                     # entry point — console chat
├── src/
│   ├── config/
│   │   └── configuration.py    # Settings (pydantic-settings) from .env
│   ├── core/
│   │   ├── graph/graph.py      # LangGraph graph assembly
│   │   ├── nodes/              # router / retrieval / dynamic / reservation / out_of_scope
│   │   └── state.py            # ChatState schema
│   ├── data/
│   │   ├── vector_store/weaviate_client.py
│   │   └── sql_store/postgres_client.py
│   ├── guardrails/             # input_filter / output_filter / pii_detector
│   ├── evaluation/             # retrieval_eval / performance_eval
│   └── seed_data/              # SQL schema, seed data, DB seeding scripts
│       ├── sql_schema_script.sql
│       ├── postgres_seed.sql
│       ├── seed_postgresql.py
│       ├── seed_data_weaviate.py
│       ├── weaviate_facility_info.json
│       └── weaviate_parking_details.json
└── CLAUDE.md
```

---

## Step-by-Step Setup & Run

### Step 1. Prerequisites

Install:

- **Python 3.13+**
- **uv** — package manager:
  ```bash
  # Linux / macOS
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # Windows (PowerShell)
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Docker** + **Docker Compose** ([Docker Desktop](https://www.docker.com/products/docker-desktop/) on Windows/macOS)
- **OpenAI API key** — required for LLM classification, responses, and for Weaviate vectorization (`text2vec-openai`).

Verify the installation:

```bash
uv --version
docker --version
docker compose version
```

### Step 2. Clone & environment variables

```bash
git clone <repo-url>
cd rag-chatbot-reservation
```

Create a `.env` file in the project root (the local-dev values match the
defaults in `docker-compose.yml`):

```env
# --- OpenAI ---
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_MINI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.0

# --- PostgreSQL ---
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=mysecretpassword
POSTGRES_DATABASE=postgres
POSTGRES_SCHEMA=parking
POSTGRES_POOL_MIN_SIZE=2
POSTGRES_POOL_MAX_SIZE=5
POSTGRES_POOL_TIMEOUT=5.0

# --- Weaviate ---
WEAVIATE_HOST=localhost
WEAVIATE_HTTP_PORT=8080
WEAVIATE_GRPC_PORT=50051

# --- Guardrails (optional) ---
GUARDRAIL_ENABLED=true
GUARDRAIL_INJECTION_CHECK=true
GUARDRAIL_SPACY_MODEL=en_core_web_sm
GUARDRAIL_MASK_LABELS=PHONE,CREDIT_CARD,SSN,IP_ADDRESS
GUARDRAIL_BLOCK_INPUT_LABELS=CREDIT_CARD,SSN

# --- Logging (optional) ---
LOG_LEVEL=INFO
```

> ⚠️ `.env` must not be committed to git (it is already in `.gitignore`).

### Step 3. Set up the Python environment (uv)

```bash
# Create a virtual environment and install dependencies from uv.lock
uv sync

# Download the spaCy language model used by the PII detector
uv run python -m spacy download en_core_web_sm
```

### Step 4. Start the databases in Docker

The project root contains a `docker-compose.yml` that brings up **PostgreSQL** and
**Weaviate** (with the `text2vec-openai` module enabled).

```bash
# Start both containers in the background
docker compose up -d

# Check status
docker compose ps
```

Expected ports:

| Service  | Container          | Port(s)            |
|----------|--------------------|--------------------|
| Postgres | `parking-postgres` | `5432`             |
| Weaviate | `parking-weaviate` | `8080` (HTTP), `50051` (gRPC) |

Readiness checks:

```bash
# Weaviate
curl http://localhost:8080/v1/.well-known/ready

# Postgres
docker exec parking-postgres pg_isready -U postgres
```

> Weaviate reads the OpenAI key from the `OPENAI_API_KEY` variable (passed into
> the container via `docker-compose.yml`). Make sure it is set in `.env`
> **before** running `docker compose up`, or exported in your environment.

### Step 5. Initialize and seed PostgreSQL

The script creates the `parking` schema, tables (`spaces`, `operating_hours`,
`reservations`), indexes, and the enum type, then loads the seed data.

```bash
uv run python src/seed_data/seed_postgresql.py
```

What the script does (`drop → create → seed`):

1. Drops existing tables/types (idempotent).
2. Applies `sql_schema_script.sql` — schema, indexes, the `reservation_status` enum.
3. Applies `postgres_seed.sql` — parking spaces and operating hours.

### Step 6. Seed Weaviate

The script recreates the `FacilityInfo` and `ParkingDetails` collections and loads
data into them from the JSON files (with automatic vectorization via OpenAI).

```bash
uv run python src/seed_data/seed_data_weaviate.py
```

What the script does (`drop → create → seed → read-test`):

1. Drops existing collections.
2. Creates `FacilityInfo` and `ParkingDetails` with `text2vec-openai`.
3. Loads `weaviate_facility_info.json` and `weaviate_parking_details.json`.
4. Prints a sample of the stored objects.

### Step 7. Run the application

```bash
uv run python main.py
```

This starts an interactive console chat. Sample queries to exercise every branch:

```
You: where is the entrance?            (info_query   → RAG agent / Weaviate)
You: how many free spots on floor 2?   (dynamic_query → Dynamic agent / Postgres)
You: I want to book a spot for tomorrow (reservation → multi-turn booking)
You: write me a poem                    (out_of_scope → refusal)
```

Exit with `quit`, `exit`, or `q`.

> **Note on imports.** Modules under `src/` are imported from the package root
> (`from core...`, `from config...`). `main.py` expects `src/` to be on
> `PYTHONPATH`. If you hit a `ModuleNotFoundError` on startup, set the path
> explicitly:
> ```bash
> # Linux / macOS
> PYTHONPATH=src uv run python main.py
> # Windows (PowerShell)
> $env:PYTHONPATH="src"; uv run python main.py
> ```

---

## Environment Variables

Settings are read via `pydantic-settings` (`src/config/configuration.py`) from
`.env`. Prefixes correspond to the setting groups.

| Variable                    | Description                                      | Default             |
|-----------------------------|--------------------------------------------------|---------------------|
| `OPENAI_API_KEY`            | OpenAI key (LLM + Weaviate embeddings)           | — (required)        |
| `OPENAI_MODEL`              | Primary LLM model (router, retrieval, dynamic, reservation) | `gpt-4o`            |
| `OPENAI_MINI_MODEL`         | Lighter LLM model for low-stakes nodes (e.g. out-of-scope) | `gpt-4o-mini`       |
| `OPENAI_TEMPERATURE`        | Generation temperature                           | `0.0`               |
| `POSTGRES_HOST`             | Postgres host                                    | `localhost`         |
| `POSTGRES_PORT`             | Postgres port                                    | `5432`              |
| `POSTGRES_USER`             | User                                             | `postgres`          |
| `POSTGRES_PASSWORD`         | Password                                         | `mysecretpassword`  |
| `POSTGRES_DATABASE`         | Database name                                    | `postgres`          |
| `POSTGRES_SCHEMA`           | Schema pinned on the connection `search_path`    | `parking`           |
| `POSTGRES_POOL_MIN_SIZE`    | Connection pool — idle connection floor          | `2`                 |
| `POSTGRES_POOL_MAX_SIZE`    | Connection pool — hard connection cap            | `5`                 |
| `POSTGRES_POOL_TIMEOUT`     | Seconds to wait for a free connection            | `5.0`               |
| `WEAVIATE_HOST`             | Weaviate host                                    | `localhost`         |
| `WEAVIATE_HTTP_PORT`        | HTTP port                                        | `8080`              |
| `WEAVIATE_GRPC_PORT`        | gRPC port                                        | `50051`             |
| `GUARDRAIL_ENABLED`         | Enable guardrails                                | `true`              |
| `GUARDRAIL_INJECTION_CHECK` | Check for prompt injection                       | `true`              |
| `GUARDRAIL_SPACY_MODEL`     | spaCy model used by the PII detector             | `en_core_web_sm`    |
| `GUARDRAIL_MASK_LABELS`     | PII labels to mask in the response (comma-separated) | `PHONE,CREDIT_CARD,SSN,IP_ADDRESS` |
| `GUARDRAIL_BLOCK_INPUT_LABELS` | PII labels that block the input               | `CREDIT_CARD,SSN`   |
| `LOG_LEVEL`                 | Root log level (`DEBUG` shows the per-node trace) | `INFO`          |

---

## Testing

Tests live under `tests/` (mirroring `src/`) and are split into `unit/` and
`integration/`. The suite is configured in `pyproject.toml` (`src` on the
pythonpath, discovery scoped to `tests/`).

```bash
# Run the whole suite
uv run pytest

# Verbose, with coverage of src/
uv run pytest -v --cov=src

# A single file
uv run pytest tests/unit/test_retrieval_eval.py -v
```

**Integration tests** are marked with `@pytest.mark.integration` and require a
**running, seeded Weaviate** plus a valid `OPENAI_API_KEY` (see Steps 4 & 6). If
that environment is unavailable they skip rather than fail.

```bash
# Fast unit-only run (skip integration)
uv run pytest -m "not integration"

# Integration tests only
uv run pytest -m integration
```

---

## Useful Commands

```bash
# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Type checking
uv run mypy src/

# Tests
uv run pytest
uv run pytest --cov=src

# Stop and remove the containers (with their data)
docker compose down -v

# Database logs
docker compose logs -f weaviate
docker compose logs -f postgres
```

---

## Security

- The OpenAI key must be read from the environment (`OPENAI_API_KEY`), never
  hardcoded in source. The Weaviate client passes it via the
  `X-OpenAI-Api-Key` header, e.g.:
  ```python
  import os
  headers = {"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}
  ```
- Make sure `.env` and any real keys are never committed. If a key is ever
  exposed in the git history, revoke it in the OpenAI dashboard and rotate it.