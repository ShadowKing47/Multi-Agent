import json
from typing import List, Optional
from datetime import datetime, timedelta

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from models import PersonaDefinition, PersonaState
from memory import MemoryStream
from llm_router import LLMRouter
from constants import REFLECT_THRESHOLD, REFLECT_INSIGHT_COUNT, RETRIEVAL_TOP_K, TICK_SIZE_MINUTES


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

    # ------------------------------------------------------------------
    # Public simulation interface
    # ------------------------------------------------------------------

    def plan(self, current_time: datetime) -> List[dict]:
        """Generate hourly schedule from current_time until sleep time."""
        memories = self.memory.retrieve("today's schedule and goals", current_time)
        context  = "\n".join(f"- {m}" for m in memories)
        sleep_time = self.definition.typical_sleep_time.strftime("%H:%M")

        prompt = (
            f"You are {self.definition.name}, age {self.definition.age}, "
            f"{self.definition.occupation}. "
            f"Traits: {', '.join(self.definition.core_traits)}. "
            f"Lifestyle: {self.definition.lifestyle}.\n"
            f"Current time: {current_time.strftime('%H:%M')}. You sleep at {sleep_time}.\n\n"
            f"Your recent memories:\n{context}\n\n"
            f"Generate your schedule from now until sleep time as a JSON array. "
            f"Each item must have keys: "
            f"\"time\" (HH:MM), \"action\" (string), \"location\" (string), \"duration_minutes\" (int). "
            f"Return only the JSON array. No commentary."
        )

        raw = self._invoke_reasoning(prompt)
        plan = json.loads(_extract_json(raw))
        self.state.current_plan = plan
        logger.info(f"[{self.definition.name}] Plan generated: {len(plan)} items")
        return plan

    def react(self, observation: str, current_time: datetime) -> Optional[dict]:
        """
        Two-stage reaction. Haiku screens; Sonnet acts only if needed.
        Returns dict with action+location, or None if no reaction.
        """
        screen_prompt = (
            f"You are {self.definition.name}. "
            f"You are currently: {self.state.current_action} at {self.state.current_location}.\n"
            f"Observation: \"{observation}\"\n"
            f"Does this require you to change what you are doing right now? "
            f"Return only JSON: {{\"react\": true}} or {{\"react\": false}}"
        )
        screen_raw = self.llm_router.get_utility_model().invoke(screen_prompt).content.strip()
        should_react = json.loads(_extract_json(screen_raw)).get("react", False)

        self._add_memory_and_check_reflect(observation, current_time)

        if not should_react:
            return None

        memories = self.memory.retrieve(observation, current_time)
        context  = "\n".join(f"- {m}" for m in memories)

        act_prompt = (
            f"You are {self.definition.name}. "
            f"You are currently: {self.state.current_action} at {self.state.current_location}.\n"
            f"Observation: \"{observation}\"\n"
            f"Relevant memories:\n{context}\n\n"
            f"What do you do next? Return only JSON: "
            f"{{\"react\": true, \"action\": \"one sentence\", \"location\": \"location name\"}}"
        )
        act_raw    = self._invoke_reasoning(act_prompt)
        response   = json.loads(_extract_json(act_raw))

        self.state.current_action   = response["action"]
        self.state.current_location = response["location"]
        self.state.current_action_end_time = current_time + timedelta(minutes=TICK_SIZE_MINUTES)

        logger.info(f"[{self.definition.name}] Reacted: {response['action']} @ {response['location']}")
        return response

    def reflect(self, current_time: datetime) -> List[str]:
        """Generate high-level insights from recent memories and save them back."""
        memories = self.memory.retrieve("recent experiences and observations", current_time, top_k=20)
        context  = "\n".join(f"- {m}" for m in memories)

        prompt = (
            f"You are {self.definition.name}. Here are your recent memories:\n{context}\n\n"
            f"What are {REFLECT_INSIGHT_COUNT} high-level insights or realizations you can draw "
            f"from these experiences? "
            f"Return only a JSON array of {REFLECT_INSIGHT_COUNT} strings."
        )
        raw      = self._invoke_reasoning(prompt)
        insights = json.loads(_extract_json(raw))

        for insight in insights:
            self._add_memory_and_check_reflect(insight, current_time)

        logger.info(f"[{self.definition.name}] Reflected: {len(insights)} insights saved")
        return insights

    def should_skip_tick(self, current_time: datetime) -> bool:
        """True while agent is mid-action and action_end_time has not passed."""
        return (
            self.state.current_action_end_time is not None
            and current_time < self.state.current_action_end_time
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _add_memory_and_check_reflect(self, description: str, current_time: datetime) -> None:
        """Add memory and fire reflect() if cumulative importance crosses threshold."""
        from scoring import get_importance_score_cached
        importance = get_importance_score_cached(
            self.llm_router.get_utility_model(), description
        )
        self.memory.add_memory(description, current_time)
        self.state.cumulative_importance += importance
        if self.state.cumulative_importance >= REFLECT_THRESHOLD:
            self.state.cumulative_importance = 0.0
            self.reflect(current_time)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _invoke_reasoning(self, prompt: str) -> str:
        return self.llm_router.get_reasoning_model().invoke(prompt).content.strip()


def _extract_json(text: str) -> str:
    """Strip markdown code fences if the LLM wraps JSON in ```json ... ```."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    return text.strip()
