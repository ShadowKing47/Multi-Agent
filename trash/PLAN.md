# Generative Agents — Refined Implementation Plan

## What We Are Building

A discrete-time multi-agent simulation. Two personas (Dr. Aris Thorne, Lena Castillo) live in a shared world. Each agent: stores memories in ChromaDB, scores and retrieves them via Numpy, plans their day with Sonnet, reacts to observations, and reflects when cumulative importance crosses a threshold. Co-location in the simulation loop triggers cross-agent observation injection and a fixed 2-3 round dialogue exchange per tick.

---

## All Design Decisions (Locked)

| Decision | Choice | Reason |
|---|---|---|
| Tick size | **15 minutes** | Fine enough to catch co-location at the cafe; coarse enough to avoid LLM call per minute |
| Long action handling | **Skip LLM call until action completes** | Avoids redundant calls for "still sleeping", "still reading" |
| Location parsing | **Structured JSON from main Sonnet call** | No extra Haiku call (cost), no regex (brittle). Prompt enforces `{"action": "...", "location": "..."}` |
| Dialogue depth | **2 rounds max per tick** | Prevents infinite loops and cascading token cost; resumes naturally next tick |
| ChromaDB mode | **`PersistentClient(path="./chroma_db")`** | Production system — memory survives restarts |

---

## Engineering Constraints (Non-Negotiable)

| Rule | Applied As |
|---|---|
| Dependency injection | Every lower-level function/class receives dependencies as arguments — no global lookups |
| Consistent naming | Variable and function names frozen from original plan — no renames across edits |
| Cache frequently used objects | LLMRouter and ChromaDB client initialized once at startup; importance scoring behind TTLCache |
| Numpy for math | All retrieval scoring is vectorized — no Python loops over score arrays |
| Haiku for utility, Sonnet for reasoning | Importance scoring → Haiku; plan/react/reflect → Sonnet |
| No downtime | Changes applied in strict dependency order (Phase ladder below) |
| Outsource to 3rd party | See package audit below |
| Reduce lines without changing behavior | Helpers extracted, repeated patterns collapsed |
| Structured JSON output | plan() and react() prompts enforce JSON schema — no regex parsing |

---

## Package Audit — What to Outsource

| Concern | Package | Replaces |
|---|---|---|
| LLM calls + tracing | `langchain-anthropic`, `langsmith` | Manual HTTP calls |
| Vector store + ANN | `chromadb` | Manual embedding DB |
| Local embeddings (no extra API key) | `sentence-transformers` (`all-MiniLM-L6-v2`) via ChromaDB EF | Voyage AI / OpenAI |
| Pydantic validation | `pydantic>=2` | Manual type checks |
| Vectorized math | `numpy` | Python loops |
| TTL caching | `cachetools` | Manual dict cache |
| LLM retry on transient errors | `tenacity` | Manual try/except retry loops |
| Env var management | `python-dotenv` | `os.environ` scattered calls |
| Structured logging | `loguru` | `print()` statements |

**`requirements.txt`:**
```
anthropic>=0.30
langchain-anthropic>=0.2
langchain-core>=0.2
langsmith>=0.1
chromadb>=0.5
sentence-transformers>=3.0
numpy>=1.26
pydantic>=2.0
cachetools>=5.3
tenacity>=8.2
python-dotenv>=1.0
loguru>=0.7
```

---

## Known Bugs in Original Plan (Fixed Here)

| Location | Bug | Fix |
|---|---|---|
| `models.py` | `typical_wake_time: "06:00"` fails Pydantic `time` validation | Use `"06:00:00"` (HH:MM:SS) |
| `memory.py` | `List` used but never imported | Add `from typing import List` |
| `llm_router.py` | `BaseCallbackHandler()` instantiated directly — abstract class, will throw | Use LangSmith env-var tracing instead (zero-code approach) |
| `llm_router.py` | Model IDs are outdated (`claude-3-5-sonnet-20240620`) | Update to current IDs from `constants.py` |
| `scoring.py` | `@cached` takes `llm_client` as arg — unhashable, breaks cache key | Separate cache key (description only) from client; client injected |
| `memory.py` | ChromaDB default embedding is not semantic | Inject `SentenceTransformerEmbeddingFunction` at collection creation |
| `main.py` | Simulation loop is `pass` | Implement discrete 15-minute tick loop with co-location detection |

---

## File Structure (Final)

```
generative_agents/
├── .env                    # API keys — never committed
├── requirements.txt
├── constants.py            # NEW: all magic numbers, model IDs, thresholds
├── models.py               # Pydantic schemas — fix time format, extend PersonaState
├── scoring.py              # Numpy vectorized scoring + fixed cache
├── llm_router.py           # Sonnet/Haiku routing, tenacity retry, env-var LangSmith
├── memory.py               # ChromaDB MemoryStream, SentenceTransformer embeddings
├── agent.py                # NEW (core): GenerativeAgent.plan(), .react(), .reflect()
└── main.py                 # Persona definitions, init, 15-min tick simulation loop
```

---

## Phase Ladder — Zero Downtime, Minimal → Maximal

Each phase is independently testable before the next begins.

---

### Phase 1 — Foundation
**Files:** `.env`, `requirements.txt`, `constants.py`

`constants.py` centralizes every magic value. One edit here changes behavior across all files.

```python
# constants.py
SONNET_MODEL_ID   = "claude-sonnet-4-6"
HAIKU_MODEL_ID    = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

REFLECT_THRESHOLD     = 100     # cumulative_importance that triggers reflection
REFLECT_INSIGHT_COUNT = 3       # how many insights reflect() generates
RETRIEVAL_TOP_K       = 10

RECENCY_DECAY               = 0.995   # per-hour exponential decay
IMPORTANCE_CACHE_TTL        = 3600    # seconds
IMPORTANCE_CACHE_SIZE       = 1000
RETRIEVAL_WEIGHT_RECENCY    = 1.0
RETRIEVAL_WEIGHT_IMPORTANCE = 1.0
RETRIEVAL_WEIGHT_RELEVANCE  = 1.0

TICK_SIZE_MINUTES   = 15
DIALOGUE_ROUNDS_MAX = 2
CHROMA_PERSIST_PATH = "./chroma_db"
```

`.env`:
```
ANTHROPIC_API_KEY=your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=generative-agents
```

---

### Phase 2 — Data Layer
**Files:** `models.py`

Changes from original:
- `typical_wake_time` / `typical_sleep_time`: Pydantic v2 coerces `"HH:MM:SS"` strings to `time` automatically — fixed in persona definitions
- Extend `PersonaState` with `current_location` and `current_action` (required by simulation loop)
- `current_action_end_time`: tracks when a long-running action finishes so ticks can be skipped

```python
# models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import time, datetime

class PersonaDefinition(BaseModel):
    agent_id: str
    name: str
    age: int
    occupation: str
    core_traits: List[str]
    lifestyle: str
    seed_memories: List[str]
    starting_location: str
    typical_wake_time: time
    typical_sleep_time: time

class PersonaState(BaseModel):
    current_plan: List[dict] = []          # list of {"time": "09:00", "action": "...", "location": "..."}
    cumulative_importance: float = 0.0
    current_location: str = ""
    current_action: str = "idle"
    current_action_end_time: Optional[datetime] = None   # skip ticks until this passes

class Memory(BaseModel):
    description: str
    creation_timestamp: datetime
    last_access_timestamp: datetime
    importance_score: int
```

---

### Phase 3 — Scoring Utilities
**Files:** `scoring.py`

Changes from original:
- Fix `@cached` bug: cache key is `memory_description` only; LLM client injected separately
- Pull all constants from `constants.py`
- Add `tenacity` retry on the LLM call
- Inline `_norm` helper reduces duplication in `compute_retrieval_scores`

```python
# scoring.py
import numpy as np
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from constants import (
    IMPORTANCE_CACHE_SIZE, IMPORTANCE_CACHE_TTL,
    RETRIEVAL_WEIGHT_RECENCY, RETRIEVAL_WEIGHT_IMPORTANCE, RETRIEVAL_WEIGHT_RELEVANCE,
)

_score_cache: TTLCache = TTLCache(maxsize=IMPORTANCE_CACHE_SIZE, ttl=IMPORTANCE_CACHE_TTL)

def compute_retrieval_scores(
    recency_scores: np.ndarray,
    importance_scores: np.ndarray,
    relevance_scores: np.ndarray,
) -> np.ndarray:
    def _norm(arr: np.ndarray) -> np.ndarray:
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-5)
    return (
        RETRIEVAL_WEIGHT_RECENCY    * _norm(recency_scores)
        + RETRIEVAL_WEIGHT_IMPORTANCE * _norm(importance_scores)
        + RETRIEVAL_WEIGHT_RELEVANCE  * _norm(relevance_scores)
    )

def get_importance_score_cached(haiku_model, memory_description: str) -> int:
    if memory_description in _score_cache:
        return _score_cache[memory_description]
    score = _call_importance_llm(haiku_model, memory_description)
    _score_cache[memory_description] = score
    return score

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _call_importance_llm(haiku_model, memory_description: str) -> int:
    prompt = (
        "On a scale of 1 to 10, rate the long-term importance of this memory. "
        f"Return only the integer.\nMemory: {memory_description}"
    )
    return max(1, min(10, int(haiku_model.invoke(prompt).content.strip())))
```

---

### Phase 4 — LLM Router
**Files:** `llm_router.py`

Changes from original:
- Remove `BaseCallbackHandler()` (wrong usage)
- LangSmith tracing via env vars only — zero code in hot path
- Model IDs from `constants.py`
- `tenacity` retry wrapper on `.invoke()` at the router level

```python
# llm_router.py
from langchain_anthropic import ChatAnthropic
from constants import SONNET_MODEL_ID, HAIKU_MODEL_ID

class LLMRouter:
    def __init__(self, anthropic_api_key: str):
        # LangSmith tracing activated by LANGCHAIN_TRACING_V2 env var — no callbacks needed
        self.sonnet = ChatAnthropic(model=SONNET_MODEL_ID, api_key=anthropic_api_key)
        self.haiku  = ChatAnthropic(model=HAIKU_MODEL_ID,  api_key=anthropic_api_key)

    def get_reasoning_model(self): return self.sonnet   # plan, react, reflect
    def get_utility_model(self):   return self.haiku    # importance scoring, parsing
```

---

### Phase 5 — Memory Layer
**Files:** `memory.py`

Changes from original:
- Inject `SentenceTransformerEmbeddingFunction` — fixes semantic search gap
- Fix missing `List` import
- Pull `RETRIEVAL_TOP_K`, `RECENCY_DECAY` from `constants.py`
- Implement bulk access-timestamp update after retrieval (was a TODO comment)
- `tenacity` retry on `collection.add` and `collection.query`

```python
# memory.py — key structure

from typing import List
from datetime import datetime
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tenacity import retry, stop_after_attempt, wait_exponential
from models import Memory
from scoring import get_importance_score_cached, compute_retrieval_scores
from constants import RETRIEVAL_TOP_K, RECENCY_DECAY, EMBEDDING_MODEL

class MemoryStream:
    def __init__(self, chroma_client: chromadb.Client, llm_router, agent_id: str):
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
        self.collection = chroma_client.get_or_create_collection(
            name=f"memories_{agent_id}",
            embedding_function=ef
        )
        self.llm_router = llm_router
        self.agent_id = agent_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def add_memory(self, description: str, current_time: datetime, is_seed: bool = False) -> None:
        importance = 9 if is_seed else get_importance_score_cached(
            self.llm_router.get_utility_model(), description
        )
        self.collection.add(
            documents=[description],
            metadatas=[{
                "creation_ts": current_time.isoformat(),
                "access_ts": current_time.isoformat(),
                "importance": importance,
            }],
            ids=[f"{self.agent_id}_{current_time.timestamp()}"],
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def retrieve(self, query: str, current_time: datetime, top_k: int = RETRIEVAL_TOP_K) -> List[str]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if not results["documents"][0]:
            return []

        metadatas = results["metadatas"][0]

        recency_scores = np.array([
            RECENCY_DECAY ** ((current_time - datetime.fromisoformat(m["access_ts"])).total_seconds() / 3600)
            for m in metadatas
        ])
        importance_scores = np.array([float(m["importance"]) for m in metadatas])
        relevance_scores  = np.array([1.0 / (1.0 + d) for d in results["distances"][0]])

        final_scores = compute_retrieval_scores(recency_scores, importance_scores, relevance_scores)
        top_indices  = np.argsort(final_scores)[::-1][:top_k]

        # Bulk-update access timestamps for retrieved memories
        retrieved_ids = [results["ids"][0][i] for i in top_indices]
        self.collection.update(
            ids=retrieved_ids,
            metadatas=[{"access_ts": current_time.isoformat()} for _ in retrieved_ids],
        )

        return [results["documents"][0][i] for i in top_indices]
```

---

### Phase 6 — Core Agent (Primary Deliverable)
**Files:** `agent.py`

#### Structured JSON Contract

All Sonnet calls in `plan()` and `react()` enforce a JSON schema in the prompt.

`plan()` returns:
```json
[
  {"time": "09:00", "action": "Walk to City Archive", "location": "City Archive", "duration_minutes": 15},
  {"time": "09:15", "action": "Begin research on Chapter 4", "location": "City Archive", "duration_minutes": 120}
]
```

`react()` returns:
```json
{"react": true, "action": "Greet Lena and ask about the poetry night", "location": "The Daily Grind Cafe"}
```

#### `plan()` — Daily Schedule Generation

```
Prompt:
  You are {name}, age {age}, {occupation}. Traits: {core_traits}. {lifestyle}.
  Current time: {current_time}. You sleep at {sleep_time}.

  Your recent memories:
  {retrieved_memories}

  Generate your schedule from now until sleep time.
  Return a JSON array of objects with keys: "time" (HH:MM), "action" (string),
  "location" (string), "duration_minutes" (int).
  Return only the JSON array. No commentary.
```

Parsed with `json.loads()` → stored in `state.current_plan`. Location extracted directly from JSON — no extra parsing.

#### `react()` — Two-Stage Observation Handler

Stage 1 uses Haiku (fast, cheap) to screen whether the observation warrants action. Stage 2 only fires if Haiku says yes.

```
Stage 1 prompt (Haiku):
  You are {name}. You are currently: {current_action} at {current_location}.
  Observation: "{observation}"
  Does this require you to change what you are doing right now? Answer JSON:
  {"react": true/false}

Stage 2 prompt (Sonnet, only if react=true):
  You are {name}. You are currently: {current_action} at {current_location}.
  Observation: "{observation}"
  Relevant memories: {retrieved_memories}
  What do you do next? Return JSON:
  {"react": true, "action": "one sentence", "location": "location name"}
```

Returns `dict` with action+location if reacting, `None` if not.

#### `reflect()` — Insight Synthesis

Triggered when `state.cumulative_importance >= REFLECT_THRESHOLD`. Saves insights as new memories. Resets counter.

```
Prompt (Sonnet):
  You are {name}. Here are your recent memories:
  {last_N_memories}

  What are {REFLECT_INSIGHT_COUNT} high-level insights you can draw from these experiences?
  Return a JSON array of {REFLECT_INSIGHT_COUNT} strings.
  Return only the JSON array.
```

#### Long-Action Skip Logic

```python
def should_skip_tick(self, current_time: datetime) -> bool:
    """True if agent is mid-action and current_time hasn't reached action_end_time."""
    return (
        self.state.current_action_end_time is not None
        and current_time < self.state.current_action_end_time
    )
```

Called by simulation loop before triggering any LLM call for this agent on this tick.

#### Full `agent.py` structure

```python
# agent.py
import json
from typing import List, Optional
from datetime import datetime
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from models import PersonaDefinition, PersonaState
from memory import MemoryStream
from llm_router import LLMRouter
from constants import REFLECT_THRESHOLD, REFLECT_INSIGHT_COUNT, RETRIEVAL_TOP_K

class GenerativeAgent:
    def __init__(
        self,
        definition: PersonaDefinition,
        state: PersonaState,
        memory: MemoryStream,
        llm_router: LLMRouter,
    ):
        self.definition = definition
        self.state = state
        self.memory = memory
        self.llm_router = llm_router

    def plan(self, current_time: datetime) -> List[dict]: ...
    def react(self, observation: str, current_time: datetime) -> Optional[dict]: ...
    def reflect(self, current_time: datetime) -> List[str]: ...

    def should_skip_tick(self, current_time: datetime) -> bool: ...
    def _add_memory_and_check_reflect(self, description: str, current_time: datetime): ...
```

---

### Phase 7 — Simulation Loop
**Files:** `main.py`

Changes from original:
- Fix `typical_wake_time` strings to `"HH:MM:SS"`
- Replace `pass` with 15-minute tick loop
- Co-location detection with 2-round dialogue exchange
- Long-action skip via `should_skip_tick()`

#### Tick Loop

```python
# main.py — simulation loop sketch

from constants import TICK_SIZE_MINUTES, DIALOGUE_ROUNDS_MAX
from datetime import timedelta

def run_simulation(aris: GenerativeAgent, lena: GenerativeAgent, start_time: datetime, total_ticks: int):
    current_time = start_time

    for _ in range(total_ticks):
        current_time += timedelta(minutes=TICK_SIZE_MINUTES)

        for agent in [aris, lena]:
            if agent.should_skip_tick(current_time):
                continue
            # Check if it's wake time and plan is empty
            wake = agent.definition.typical_wake_time
            if current_time.time().hour == wake.hour and current_time.time().minute == wake.minute:
                if not agent.state.current_plan:
                    agent.plan(current_time)
            # Advance current action from plan
            _apply_current_plan_item(agent, current_time)

        # Co-location check and dialogue exchange
        if aris.state.current_location == lena.state.current_location and aris.state.current_location:
            _handle_co_location(aris, lena, current_time)

        logger.info(f"[{current_time}] Aris: {aris.state.current_action} @ {aris.state.current_location}")
        logger.info(f"[{current_time}] Lena: {lena.state.current_action} @ {lena.state.current_location}")

def _handle_co_location(aris: GenerativeAgent, lena: GenerativeAgent, current_time: datetime):
    aris_obs = f"{lena.definition.name} is here at {aris.state.current_location}."
    lena_obs = f"{aris.definition.name} is here at {lena.state.current_location}."

    for _ in range(DIALOGUE_ROUNDS_MAX):
        aris_response = aris.react(lena_obs, current_time)
        lena_response = lena.react(aris_obs, current_time)
        if not aris_response and not lena_response:
            break
        # Feed each agent's output into the other's next observation
        if aris_response:
            lena_obs = f"{aris.definition.name} says/does: {aris_response['action']}"
        if lena_response:
            aris_obs = f"{lena.definition.name} says/does: {lena_response['action']}"
```

#### `main()` Init Order (Frequently Used → First)

```python
def main():
    load_dotenv()
    # 1. Shared infrastructure (initialized once, cached by Python module system)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    llm_router    = LLMRouter(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))

    # 2. Per-agent memory streams (SentenceTransformer loaded once per collection)
    aris_memory = MemoryStream(chroma_client, llm_router, ARIS_DEFINITION.agent_id)
    lena_memory = MemoryStream(chroma_client, llm_router, LENA_DEFINITION.agent_id)

    # 3. Agent state
    aris_state = PersonaState(current_location=ARIS_DEFINITION.starting_location)
    lena_state = PersonaState(current_location=LENA_DEFINITION.starting_location)

    # 4. Agents
    aris = GenerativeAgent(ARIS_DEFINITION, aris_state, aris_memory, llm_router)
    lena = GenerativeAgent(LENA_DEFINITION, lena_state, lena_memory, llm_router)

    # 5. Seed memories (only if collection is empty)
    start_time = datetime.now()
    if aris_memory.collection.count() == 0:
        for mem in ARIS_DEFINITION.seed_memories:
            aris_memory.add_memory(mem, start_time, is_seed=True)
    if lena_memory.collection.count() == 0:
        for mem in LENA_DEFINITION.seed_memories:
            lena_memory.add_memory(mem, start_time, is_seed=True)

    # 6. Run
    run_simulation(aris, lena, start_time, total_ticks=96)  # 96 ticks = 24 game hours at 15min/tick
```

---

## Optimization Summary

| Optimization | Where | Impact |
|---|---|---|
| `constants.py` centralization | everywhere | One edit changes any threshold/model |
| TTL-cached importance scoring (description key only) | `scoring.py` | Eliminates redundant Haiku calls |
| LLMRouter initialized once | `main.py` | Single HTTP session shared across all calls |
| ChromaDB PersistentClient initialized once | `main.py` | Single DB handle, no reconnect per tick |
| SentenceTransformer loaded once into ChromaDB EF | `memory.py` | Model in RAM once, reused across all embeds |
| Haiku screens react() before Sonnet | `agent.py` | ~80% of observations skip the expensive Sonnet call |
| Long-action tick skip | `agent.py` | Zero LLM calls during "sleeping", "reading for 2 hours" |
| Numpy vectorized scoring | `scoring.py` | O(1) wall-clock vs Python loop for retrieval scoring |
| Bulk access-timestamp update | `memory.py` | Single DB write per retrieve vs N writes |
| Structured JSON from Sonnet | `agent.py` | No extra location-parsing Haiku call |
| LangSmith via env vars only | `.env` | Zero-code tracing, no callback in hot path |
| Seed memory guard (`count() == 0`) | `main.py` | No duplicate seed memories on restart |
| `tenacity` retry on LLM + DB | `scoring.py`, `memory.py` | Handles transient failures without manual code |

---

## Dependency Graph (Build Order = Phase Order)

```
constants.py          ← Phase 1 (no deps)
    ↓
models.py             ← Phase 2
    ↓
scoring.py            ← Phase 3 (imports constants)
    ↓
llm_router.py         ← Phase 4 (imports constants)
    ↓
memory.py             ← Phase 5 (imports scoring, llm_router, models, constants)
    ↓
agent.py              ← Phase 6 (imports memory, llm_router, models, constants)
    ↓
main.py               ← Phase 7 (imports all)
```

No circular imports. Each phase testable in isolation with mocked dependencies before the next begins.

---

## What Does NOT Change From Original Plan

- Persona definitions: `ARIS_DEFINITION`, `LENA_DEFINITION` — variable names and seed memories unchanged
- ChromaDB collection naming: `memories_{agent_id}`
- Pydantic model field names across `Memory`, `PersonaDefinition`, `PersonaState`
- `LLMRouter` method names: `get_reasoning_model()`, `get_utility_model()`
- `MemoryStream` method names: `add_memory()`, `retrieve()`
- `GenerativeAgent` method names: `plan()`, `react()`, `reflect()`
- Retrieval formula: recency + importance + relevance, equal weights
