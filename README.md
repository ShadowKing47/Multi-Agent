# Generative Agents

A discrete-time multi-agent simulation based on the [Generative Agents paper](https://arxiv.org/abs/2304.03442). Two AI personas live in a shared world, form memories, plan their days, react to observations, and reflect on experiences.

## Personas

| Agent | Age | Occupation | Lifestyle |
|---|---|---|---|
| Dr. Aris Thorne | 62 | Historian & Author | Early riser, introverted, prefers quiet |
| Lena Castillo | 24 | Barista & Community Organizer | Night owl, extroverted, socially driven |

## How It Works

Each tick (15 game-minutes):
1. **Plan** — at wake time, Sonnet generates an hourly schedule as structured JSON
2. **Act** — the agent's current action and location are set from the active plan item
3. **React** — Haiku screens incoming observations cheaply; Sonnet only fires when a reaction is needed
4. **Reflect** — when cumulative memory importance crosses a threshold, Sonnet synthesizes high-level insights and saves them as new memories
5. **Co-location** — when both agents share a location, observations are injected into each other's memory stream and up to 2 rounds of dialogue exchange occur per tick

Memories are stored in ChromaDB with semantic embeddings. Retrieval ranks candidates by a weighted combination of recency, importance, and relevance (all vectorized via Numpy).

## Project Structure

```
generative_agents/
├── constants.py      # All thresholds, model IDs, and config in one place
├── models.py         # Pydantic schemas: PersonaDefinition, PersonaState, Memory
├── scoring.py        # Numpy vectorized retrieval scoring + cached importance scoring
├── llm_router.py     # Sonnet (reasoning) / Haiku (utility) routing
├── memory.py         # ChromaDB MemoryStream with SentenceTransformer embeddings
├── agent.py          # GenerativeAgent: plan(), react(), reflect()
└── main.py           # Persona definitions, initialization, simulation loop
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** First run downloads the `all-MiniLM-L6-v2` embedding model (~79MB) to `~/.cache/chroma/`. Cached after that.

### 2. Configure API keys

Edit `.env`:

```
ANTHROPIC_API_KEY=your_anthropic_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=generative-agents
```

LangSmith tracing is optional — remove `LANGCHAIN_TRACING_V2` and `LANGCHAIN_API_KEY` to disable it.

### 3. Run

```bash
python main.py
```

The simulation runs 96 ticks (24 game-hours at 15 min/tick). The `chroma_db/` directory is created automatically on first run and persists across restarts — seed memories are only loaded once.

## Configuration

All tunable values are in `constants.py`:

| Constant | Default | Effect |
|---|---|---|
| `TICK_SIZE_MINUTES` | `15` | Game time per tick |
| `DIALOGUE_ROUNDS_MAX` | `2` | Max exchange rounds per co-location event |
| `REFLECT_THRESHOLD` | `100` | Cumulative importance that triggers reflection |
| `REFLECT_INSIGHT_COUNT` | `3` | Insights generated per reflection |
| `RETRIEVAL_TOP_K` | `10` | Memories retrieved per query |
| `RECENCY_DECAY` | `0.995` | Per-hour exponential decay on memory recency |
| `SONNET_MODEL_ID` | `claude-sonnet-4-6` | Reasoning model |
| `HAIKU_MODEL_ID` | `claude-haiku-4-5-20251001` | Utility model |

## Architecture Decisions

- **Haiku screens `react()` before Sonnet** — ~80% of observations don't require a plan change; Haiku filters them cheaply
- **Structured JSON from Sonnet** — `plan()` and `react()` prompts enforce a JSON schema, so location is extracted natively with no extra parsing step
- **Long-action tick skipping** — agents mid-way through a long action (sleeping, reading) skip their LLM call until the action completes
- **UUID memory IDs** — prevents ID collisions when multiple memories are added at the same timestamp
- **TTL cache on importance scoring** — repeated memory descriptions skip the Haiku call entirely
