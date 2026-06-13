import uuid
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
            embedding_function=ef,
        )
        self.llm_router = llm_router
        self.agent_id = agent_id

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
    def add_memory(self, description: str, current_time: datetime, is_seed: bool = False) -> None:
        importance = 9 if is_seed else get_importance_score_cached(
            self.llm_router.get_utility_model(), description
        )
        memory = Memory(
            description=description,
            creation_timestamp=current_time,
            last_access_timestamp=current_time,
            importance_score=importance,
        )
        self.collection.add(
            documents=[memory.description],
            metadatas=[{
                "creation_ts": current_time.isoformat(),
                "access_ts": current_time.isoformat(),
                "importance": importance,
            }],
            ids=[f"{self.agent_id}_{uuid.uuid4().hex}"],
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

        retrieved_ids = [results["ids"][0][i] for i in top_indices]
        self.collection.update(
            ids=retrieved_ids,
            metadatas=[{"access_ts": current_time.isoformat()} for _ in retrieved_ids],
        )

        return [results["documents"][0][i] for i in top_indices]
