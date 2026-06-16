import os
from datetime import datetime

import chromadb
from loguru import logger

from constants import CHROMA_PERSIST_PATH, MAX_REFINEMENT_ROUNDS, DEBATE_ROUNDS_MAX
from typing import Dict
from models import PersonaDefinition, PersonaState, Poem, DebateRound
from llm_router import LLMRouter
from memory import MemoryStream
from agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

POET_DEFINITION = PersonaDefinition(
    agent_id="eliot_vane",
    name="Eliot Vane",
    age=34,
    occupation="Poet",
    core_traits=["introspective", "passionate", "stubborn about his artistic vision"],
    lifestyle="writes late into the night, draws from personal grief and wonder",
    seed_memories=[
        "Eliot Vane is a poet known for spare, imagistic verse about loss and the natural world.",
        "Eliot believes a poem should feel inevitable — every word earns its place or gets cut.",
        "Eliot has been told his work is too obscure, but he distrusts poetry that explains itself.",
        "Eliot's last collection was praised for its restraint and criticized for its coldness.",
        "Eliot is protective of his drafts and does not revise lightly.",
    ],
    starting_location="Vane Studio",
    typical_wake_time="09:00:00",
    typical_sleep_time="02:00:00",
)

CRITIC_DEFINITION = PersonaDefinition(
    agent_id="dr_mara_chen",
    name="Dr. Mara Chen",
    age=47,
    occupation="Literary Critic and Professor of Poetics",
    core_traits=["precise", "demanding", "deeply well-read", "fair but unsparing"],
    lifestyle="reads widely, values clarity without simplicity, dislikes sentimentality",
    seed_memories=[
        "Dr. Mara Chen teaches contemporary poetics and has reviewed work for literary journals for 20 years.",
        "Mara believes the critic's job is to hold the work to the highest standard the poet is capable of.",
        "Mara values compression, surprise, and earned emotion — she is impatient with decoration.",
        "Mara has reviewed Eliot Vane's previous collection and found it promising but uneven.",
        "Mara gives praise sparingly but means it when she does.",
    ],
    starting_location="Chen Office",
    typical_wake_time="07:00:00",
    typical_sleep_time="23:00:00",
)


# ---------------------------------------------------------------------------
# Workshop loop
# ---------------------------------------------------------------------------

def run_poetry_workshop(
    poet: GenerativeAgent,
    critic: GenerativeAgent,
    theme: str,
    current_time: datetime,
) -> Poem:
    logger.info(f"=== Poetry Workshop | Theme: '{theme}' ===")

    poem = poet.compose_poem(theme, current_time)
    logger.info(f"\n--- {poem.title} (v{poem.version}) ---\n{poem.body}\n")

    for refinement_round in range(1, MAX_REFINEMENT_ROUNDS + 1):
        logger.info(f"=== Critique & Debate Round {refinement_round} ===")

        # Step 1: Critic reads the poem
        critique = critic.critique_poem(poem, current_time)
        logger.info(f"\n[{critic.definition.name}]:\n{critique.critique_text}\n")

        # Step 2: Debate — poet argues, critic rebuts, up to DEBATE_ROUNDS_MAX rounds
        debate_history: Dict[int, DebateRound] = {}
        poet_convinced = False

        for debate_round in range(1, DEBATE_ROUNDS_MAX + 1):
            logger.info(f"--- Debate Round {debate_round}/{DEBATE_ROUNDS_MAX} ---")

            argument = poet.argue_critique(poem, critique, debate_history, current_time)
            logger.info(f"[{poet.definition.name}] Argues:\n{argument}\n")

            rebuttal_data = critic.rebut_argument(poem, argument, critique, debate_history, current_time)
            logger.info(f"[{critic.definition.name}] Rebuts:\n{rebuttal_data.rebuttal}\n")
            if rebuttal_data.conceded_points:
                logger.info(f"[{critic.definition.name}] Concedes: {rebuttal_data.conceded_points}")

            debate_history[debate_round] = DebateRound(
                round_number=debate_round,
                poet_argument=argument,
                critic_rebuttal=rebuttal_data.rebuttal,
            )

            # Poet honestly checks if he's been genuinely convinced
            poet_convinced = poet.check_conviction(poem, debate_history, current_time)
            if poet_convinced:
                logger.info(f"[{poet.definition.name}] Convinced after {debate_round} debate round(s).")
                break

        if not poet_convinced:
            logger.info(
                f"[{poet.definition.name}] Not convinced after {DEBATE_ROUNDS_MAX} debate rounds. "
                f"Standing by the work."
            )

        # Step 3: Final decision — poet decides with full debate context
        refined, poem = poet.decide_to_refine(poem, critique, debate_history, current_time)

        if not refined:
            logger.info(
                f"[{poet.definition.name}] Final decision: no revision. "
                f"Poem stands as '{poem.title}' v{poem.version}."
            )
            break

        logger.info(f"\n--- {poem.title} (v{poem.version}) ---\n{poem.body}\n")

    logger.info(f"=== Workshop Complete | Final: '{poem.title}' v{poem.version} ===")
    return poem


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    # load_local=True downloads/caches the open-weights model for the poet
    # set load_local=False to keep both agents on Claude (pre-Phase 0 behaviour)
    llm_router    = LLMRouter(anthropic_api_key=os.environ["ANTHROPIC_API_KEY"], load_local=True)

    poet_memory   = MemoryStream(chroma_client, llm_router, POET_DEFINITION.agent_id)
    critic_memory = MemoryStream(chroma_client, llm_router, CRITIC_DEFINITION.agent_id)

    poet_state   = PersonaState()
    critic_state = PersonaState()

    # Poet runs on the local open-weights model (Phase 0)
    # Critic stays on Claude — it will become the reward model in Phase 2
    poet   = GenerativeAgent(POET_DEFINITION,   poet_state,   poet_memory,   llm_router, use_local=True)
    critic = GenerativeAgent(CRITIC_DEFINITION, critic_state, critic_memory, llm_router, use_local=False)

    start_time = datetime.now()
    if poet_memory.collection.count() == 0:
        for mem in POET_DEFINITION.seed_memories:
            poet_memory.add_memory(mem, start_time, is_seed=True)
        logger.info(f"{POET_DEFINITION.name} seed memories loaded.")
    if critic_memory.collection.count() == 0:
        for mem in CRITIC_DEFINITION.seed_memories:
            critic_memory.add_memory(mem, start_time, is_seed=True)
        logger.info(f"{CRITIC_DEFINITION.name} seed memories loaded.")

    run_poetry_workshop(poet, critic, theme="the silence after rain", current_time=start_time)


if __name__ == "__main__":
    main()
