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
    current_plan: List[dict] = []
    cumulative_importance: float = 0.0
    current_location: str = ""
    current_action: str = "idle"
    current_action_end_time: Optional[datetime] = None


class Memory(BaseModel):
    description: str
    creation_timestamp: datetime
    last_access_timestamp: datetime
    importance_score: int
