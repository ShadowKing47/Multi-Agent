import numpy as np
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from constants import (
    IMPORTANCE_CACHE_SIZE,
    IMPORTANCE_CACHE_TTL,
    RETRIEVAL_WEIGHT_RECENCY,
    RETRIEVAL_WEIGHT_IMPORTANCE,
    RETRIEVAL_WEIGHT_RELEVANCE,
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
        "On a scale of 1 to 10, rate the long-term importance of this memory for an agent's life. "
        f"Return only the integer.\nMemory: {memory_description}"
    )
    return max(1, min(10, int(haiku_model.invoke(prompt).content.strip())))
