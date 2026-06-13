import os
from datetime import datetime, timedelta

import chromadb
from loguru import logger

from constants import (
    CHROMA_PERSIST_PATH,
    DIALOGUE_ROUNDS_MAX,
    TICK_SIZE_MINUTES,
)
from models import PersonaDefinition, PersonaState
from llm_router import LLMRouter
from memory import MemoryStream
from agent import GenerativeAgent


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

ARIS_DEFINITION = PersonaDefinition(
    agent_id="dr_aris_thorne",
    name="Dr. Aris Thorne",
    age=62,
    occupation="Historian and Author",
    core_traits=["meticulous", "introverted", "intellectual"],
    lifestyle="early riser, prefers quiet environments",
    seed_memories=[
        "Dr. Aris Thorne is a historian who specializes in medieval European economics.",
        "Aris lives alone in his apartment with his cat, Empress.",
        "Aris spends most of his time at the City Archive researching his new book.",
        "Aris finds loud environments distracting and avoids the local cafe during peak hours.",
        "Aris knows Lena Castillo, the barista, because she saves a quiet corner table for him in the mornings.",
    ],
    starting_location="Thorne Apartment: Study",
    typical_wake_time="06:00:00",
    typical_sleep_time="22:00:00",
)

LENA_DEFINITION = PersonaDefinition(
    agent_id="lena_castillo",
    name="Lena Castillo",
    age=24,
    occupation="Barista and Community Organizer",
    core_traits=["energetic", "extroverted", "scattered"],
    lifestyle="night owl, thrives on social interaction",
    seed_memories=[
        "Lena Castillo works as a barista at The Daily Grind Cafe.",
        "Lena is organizing a community poetry night for this upcoming Friday.",
        "Lena lives with two roommates in a downtown loft.",
        "Lena gets easily distracted by conversations and often loses track of time.",
        "Lena knows Dr. Aris Thorne as a regular customer who likes quiet and black coffee.",
    ],
    starting_location="Castillo Loft: Kitchen",
    typical_wake_time="08:30:00",
    typical_sleep_time="01:00:00",
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _apply_current_plan_item(agent: GenerativeAgent, current_time: datetime) -> None:
    """Set agent's current_action and current_location from the active plan item."""
    if not agent.state.current_plan:
        return
    time_str = current_time.strftime("%H:%M")
    for item in reversed(agent.state.current_plan):
        if item["time"] <= time_str:
            agent.state.current_action   = item["action"]
            agent.state.current_location = item["location"]
            end = current_time + timedelta(minutes=item.get("duration_minutes", TICK_SIZE_MINUTES))
            agent.state.current_action_end_time = end
            return


def _handle_co_location(
    aris: GenerativeAgent,
    lena: GenerativeAgent,
    current_time: datetime,
) -> None:
    """Inject cross-agent observations and run up to DIALOGUE_ROUNDS_MAX exchange rounds."""
    aris_obs = f"{lena.definition.name} is here at {aris.state.current_location}."
    lena_obs = f"{aris.definition.name} is here at {lena.state.current_location}."

    for _ in range(DIALOGUE_ROUNDS_MAX):
        aris_response = aris.react(aris_obs, current_time)
        lena_response = lena.react(lena_obs, current_time)

        if not aris_response and not lena_response:
            break

        if aris_response:
            lena_obs = f"{aris.definition.name} says/does: {aris_response['action']}"
        if lena_response:
            aris_obs = f"{lena.definition.name} says/does: {lena_response['action']}"


def run_simulation(
    aris: GenerativeAgent,
    lena: GenerativeAgent,
    start_time: datetime,
    total_ticks: int,
) -> None:
    current_time = start_time

    for tick in range(total_ticks):
        current_time += timedelta(minutes=TICK_SIZE_MINUTES)
        t_str = current_time.strftime("%H:%M")

        for agent in [aris, lena]:
            if agent.should_skip_tick(current_time):
                continue

            wake = agent.definition.typical_wake_time
            at_wake = (
                current_time.time().hour   == wake.hour
                and current_time.time().minute == wake.minute
            )
            if at_wake and not agent.state.current_plan:
                agent.plan(current_time)

            _apply_current_plan_item(agent, current_time)

        # Co-location triggers dialogue exchange
        shared_location = (
            aris.state.current_location
            and aris.state.current_location == lena.state.current_location
        )
        if shared_location:
            logger.info(f"[{t_str}] Co-location: {aris.state.current_location}")
            _handle_co_location(aris, lena, current_time)

        logger.info(
            f"[{t_str}] "
            f"Aris: {aris.state.current_action} @ {aris.state.current_location} | "
            f"Lena: {lena.state.current_action} @ {lena.state.current_location}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Shared infrastructure — initialized once
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    llm_router    = LLMRouter(anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])

    # 2. Per-agent memory streams
    aris_memory = MemoryStream(chroma_client, llm_router, ARIS_DEFINITION.agent_id)
    lena_memory = MemoryStream(chroma_client, llm_router, LENA_DEFINITION.agent_id)

    # 3. Agent state — seed location from definition
    aris_state = PersonaState(current_location=ARIS_DEFINITION.starting_location)
    lena_state = PersonaState(current_location=LENA_DEFINITION.starting_location)

    # 4. Agents
    aris = GenerativeAgent(ARIS_DEFINITION, aris_state, aris_memory, llm_router)
    lena = GenerativeAgent(LENA_DEFINITION, lena_state, lena_memory, llm_router)

    # 5. Seed memories — guard prevents duplicates on restart
    start_time = datetime.now()
    if aris_memory.collection.count() == 0:
        for mem in ARIS_DEFINITION.seed_memories:
            aris_memory.add_memory(mem, start_time, is_seed=True)
        logger.info("Aris seed memories loaded.")
    if lena_memory.collection.count() == 0:
        for mem in LENA_DEFINITION.seed_memories:
            lena_memory.add_memory(mem, start_time, is_seed=True)
        logger.info("Lena seed memories loaded.")

    # 6. Run — 96 ticks = 24 game-hours at 15 min/tick
    run_simulation(aris, lena, start_time, total_ticks=96)


if __name__ == "__main__":
    main()
