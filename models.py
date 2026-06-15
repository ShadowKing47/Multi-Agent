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
    cumulative_importance: float = 0.0
    current_action: str = "idle"


class Memory(BaseModel):
    description: str
    creation_timestamp: datetime
    last_access_timestamp: datetime
    importance_score: int


class Poem(BaseModel):
    title: str
    body: str
    theme: str
    author_id: str
    version: int = 1


class CritiqueNote(BaseModel):
    critique_text: str
    critic_id: str
    poem_version: int


class DebateRound(BaseModel):
    round_number: int
    poet_argument: str
    critic_rebuttal: str


class RebuttalResult(BaseModel):
    rebuttal: str
    conceded_points: List[str] = []


class ConvictionResult(BaseModel):
    convinced: bool
    reasoning: str


class RefineResult(BaseModel):
    refine: bool
    reasoning: str
    refined_title: Optional[str] = None
    refined_body: Optional[str] = None
