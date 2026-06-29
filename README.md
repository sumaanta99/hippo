# Hippo Terminal

Your external memory. Stores tiny information you'd otherwise forget.

Hippo is a **reusable memory engine** with thin interface adapters. The CLI, API, and future WhatsApp integration all call the same `HippoEngine` — no duplicate logic.

## Quick Start

```bash
cp .env.example .env   # add OpenAI API key
pip install -r requirements.txt
python3 app/main.py
```

## Using the engine directly

```python
from engine import HippoEngine

engine = HippoEngine()
await engine.initialize()

result = await engine.chat("where is my passport?", session_id="user-123")
print(result.response)        # natural language reply
print(result.intent)          # "recall"
print(result.search_results)  # structured memory snapshots
```

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for diagrams, module map, and future interface plans.

## What It Does

- **Save:** Type anything naturally → Hippo extracts and stores it
- **Query:** Ask about it → Hippo searches semantically and answers
- **Update:** Correct or modify → Hippo updates without duplicates
- **Delete:** Forget something → Hippo removes it
- **Shopping:** Add/remove items → Separate shopping list
- **Chat:** Greet Hippo → Responds conversationally

## Quick Start

1. Clone the repo
2. `cp .env.example .env` and add your OpenAI API key
3. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Run the CLI:

```bash
python3 app/main.py
```

5. Type naturally:

```
you> hair clip on the bookshelf
hippo> I'll remember that.
```

## Architecture

```
User Input
    ↓
Intent Classifier (9 intents via LLM)
    ↓
Route to Handler (save / query / update / delete / shopping / chat)
    ↓
Business Logic (memory_service, shopping_service, retriever)
    ↓
Database (SQLite: memories, shopping_items)
    ↓
Semantic Search (embeddings + LLM re-ranking)
    ↓
Response (concise, personality-driven)
```

## Examples

**Save**
```
you> hair clip on the bookshelf
hippo> I'll remember that.
```

**Query**
```
you> where's my hair clip
hippo> Found it. Hair clip is on the bookshelf.
```

**Update**
```
you> hair clip is in the drawer now
hippo> Got it. I've updated that.
```

**Shopping**
```
you> buy eggs
hippo> Added eggs to your shopping list.
```

**Delete**
```
you> forget hair clip
hippo> Removed hair clip from memory.
```

## Project Structure

```
app/
├── engine/          # HippoEngine — the core product
├── models/          # ChatResponse, MemorySnapshot, etc.
├── services/        # Memory, shopping, chat business logic
├── database/        # SQLite stores + repositories
├── llm/             # OpenAI client, classifier, embeddings
├── search/          # Semantic retrieval + reranking
├── cli/             # Terminal client (display only)
├── api/             # FastAPI scaffold
└── main.py          # CLI entry point
```

## Key Features

- **Two-stage retrieval:** Semantic search + LLM re-ranking
- **Entity-aware:** `chirag` finds all chirag-related memories
- **Smart deduplication:** Corrects old info instead of duplicating
- **Versioning:** Tracks memory versions on every update
- **Personality:** Warm, minimal responses — never robotic
- **Local persistence:** All data in `hippo.db` (SQLite)
- **Structured logging:** JSON events for latency, intents, and errors

## Dependencies

- Python 3.11+
- OpenAI API key (`gpt-4o`, `text-embedding-3-small`)
- SQLite (included with Python)

## Configuration

`.env` options:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `OPENAI_MODEL` | Chat model | `gpt-4o` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `DATABASE_PATH` | SQLite file path | `hippo.db` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `TIMEOUT_SECONDS` | LLM timeout | `30` |

## Testing

```bash
pytest app/tests/ -v
pytest app/tests/ --cov=app --cov-report=term-missing
pytest app/tests/ --cov=app --cov-report=html
```

72 tests cover intent classification, repositories, services, retrieval, shopping, edge cases, and logging. Tests use mocked LLM responses and temporary SQLite databases — no live API key required.

Coverage focuses on orchestration and business logic (`services/`, `classifier.py`, `hippo.py`, repositories). Infrastructure modules with live OpenAI calls (`llm_client.py`, `embeddings.py`, `memory.py`, `retriever.py`) are exercised via integration-style unit tests but excluded from the default coverage denominator.

## Limitations

- CLI only (no web UI)
- Single local user
- Requires OpenAI (no offline mode)
- No multi-device sync
- Text-only (no attachments)

## Next Steps

- WhatsApp integration (FastAPI + Cloud API)
- Offline semantic search (local embeddings)
- Rich output formatting
- Reminders and proactive recall
