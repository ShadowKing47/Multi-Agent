# Poetry Workshop — Generative Agents + RLAIF

A multi-agent poetry workshop built on the [Generative Agents](https://arxiv.org/abs/2304.03442) architecture, evolving toward a full RLAIF (Reinforcement Learning from AI Feedback) training pipeline.

Two agents with persistent memory, reflection, and a structured debate loop. The poet runs on a locally fine-tuned open-weights model. The critic runs on Claude. The long-term goal is to close the loop: use debate outcomes as preference data, train a reward model, and fine-tune the poet via DPO.

---

## Agents

| Agent | Model | Role |
|---|---|---|
| Eliot Vane | Qwen 2.5-3B (local, SFT fine-tuned) | Poet — composes, argues, decides to revise |
| Dr. Mara Chen | Claude Sonnet 4.6 (API) | Critic — critiques, rebuts, will become reward model |

---

## How a Workshop Session Works

```
Poet composes poem
        ↓
Critic critiques
        ↓
Debate loop (up to 5 rounds)
  → Poet argues
  → Critic rebuts + optionally concedes
  → Poet checks conviction after each round
        ↓
Poet decides autonomously to revise or stand by the work
        ↓
(repeat up to 5 refinement rounds)
```

Both agents maintain a persistent memory stream (ChromaDB). Memories are retrieved by a weighted combination of recency × importance × relevance. When cumulative memory importance crosses a threshold, the agent reflects and distils insights.

---

## Project Structure

```
persona/
├── constants.py        # All config — model IDs, thresholds, paths
├── models.py           # Pydantic schemas: Poem, CritiqueNote, DebateRound, etc.
├── scoring.py          # Numpy vectorized retrieval scoring + TTL-cached importance
├── llm_router.py       # Routes poet → local model, critic → Claude API
├── local_llm.py        # LocalLLMClient: loads Qwen on MPS, merges LoRA adapter
├── memory.py           # ChromaDB MemoryStream with SentenceTransformer embeddings
├── agent.py            # GenerativeAgent: compose, critique, argue, reflect
├── main.py             # Persona definitions + workshop orchestration loop
├── sft_train.py        # Phase 0.2 — SFT baseline training for the poet model
├── cleanup.py          # Remove downloaded models, caches, and training artefacts
├── requirements.txt
├── tests/
│   ├── test_parsers.py  # Unit tests for all plain-text output parsers
│   └── test_scoring.py  # Unit tests for retrieval scoring + cache behaviour
└── trash/               # Planning documents and roadmaps
    ├── rlaif_roadmap.md
    ├── updated_RLAIF_roadmap.md
    └── doubts.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_anthropic_key
```

LangSmith tracing is optional — uncomment the relevant lines in `.env` if you have a key.

### 3. Train the SFT baseline (poet model)

Teaches Qwen 2.5-3B to write in Eliot Vane's voice and output format:

```bash
python sft_train.py
```

- Downloads `Qwen/Qwen2.5-3B-Instruct` (~6 GB, cached after first run)
- Loads 3 poetry datasets, samples 5000 examples (seed=42)
- Trains with LoRA on MPS — ~15–25 min on M-series Apple Silicon
- Saves adapter to `./poet-sft-lora/`

Skip this step to run the workshop with the base model (no persona fine-tuning).

### 4. Run the workshop

```bash
python main.py
```

`LLMRouter` auto-detects `./poet-sft-lora/` and merges the adapter on load.

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## Cleanup

Remove accumulated model downloads and caches:

```bash
python cleanup.py              # shows sizes, asks before deleting
python cleanup.py --dry-run    # shows sizes only, deletes nothing
python cleanup.py --yes        # skips confirmation prompt
python cleanup.py --keep-chroma  # preserves agent memories
```

---

## Configuration

All tunable values in `constants.py`:

| Constant | Default | Effect |
|---|---|---|
| `LOCAL_MODEL_ID` | `Qwen/Qwen2.5-3B-Instruct` | Poet model — swap to 1B for 8 GB RAM |
| `LOCAL_MODEL_DEVICE` | `mps` | Change to `cpu` if MPS unavailable |
| `SFT_ADAPTER_PATH` | `./poet-sft-lora` | Where the LoRA adapter is saved/loaded |
| `DEBATE_ROUNDS_MAX` | `5` | Max poet-critic debate rounds per critique |
| `MAX_REFINEMENT_ROUNDS` | `5` | Max full critique-debate-revise cycles |
| `REFLECT_THRESHOLD` | `100` | Cumulative importance that triggers reflection |
| `REFLECT_INSIGHT_COUNT` | `3` | Insights generated per reflection |
| `RETRIEVAL_TOP_K` | `10` | Memories retrieved per query |
| `RECENCY_DECAY` | `0.995` | Per-hour exponential decay on memory recency |

---

## RLAIF Roadmap

The workshop is Phase 0 of a larger pipeline toward reinforcement learning from AI feedback.

| Phase | Goal | Status |
|---|---|---|
| 0 | Local inference + SFT baseline | Done |
| 1 | Episode logging + arbitrated preference pairs | Not started |
| 2 | Reward model (embedding + MLP ensemble) | Not started |
| 3 | Best-of-N sampling with uncertainty gating | Not started |
| 4 | DPO fine-tuning | Not started |
| 5 | Human-in-the-loop labelling UI | Not started |

See `trash/updated_RLAIF_roadmap.md` for the full plan.

---

## Architecture Notes

- **Poet runs locally, critic stays on Claude** — critic will eventually become the reward model; keeping it on a strong model maintains critique quality during data collection
- **Plain-text LLM outputs** — all structured responses use labelled delimiters (`TITLE:`, `DECISION:`, `CONVINCED:`) parsed with simple string operations; no JSON, no schema enforcement at the LLM layer
- **Prompt injection defence** — XML delimiters on all peer content, regex sanitization, immutable security rule in every system prompt
- **Prompt caching** — system prompt sent with `cache_control: ephemeral` on the Claude path for warm starts
- **UUID memory IDs** — prevents collisions when multiple memories are added at the same timestamp
- **TTL cache on importance scoring** — repeated memory descriptions skip the Haiku call; 1-hour TTL, 1000-entry cap
- **Reflection recursion guard** — `_reflecting` flag prevents insights from triggering another reflection cascade
