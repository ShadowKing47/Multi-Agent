from dotenv import load_dotenv
load_dotenv()

SONNET_MODEL_ID   = "claude-sonnet-4-6"
HAIKU_MODEL_ID    = "claude-haiku-4-5-20251001"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

REFLECT_THRESHOLD     = 100
REFLECT_INSIGHT_COUNT = 3
RETRIEVAL_TOP_K       = 10

RECENCY_DECAY               = 0.995
IMPORTANCE_CACHE_TTL        = 3600
IMPORTANCE_CACHE_SIZE       = 1000
RETRIEVAL_WEIGHT_RECENCY    = 1.0
RETRIEVAL_WEIGHT_IMPORTANCE = 1.0
RETRIEVAL_WEIGHT_RELEVANCE  = 1.0

MAX_REFINEMENT_ROUNDS = 5
DEBATE_ROUNDS_MAX     = 5
CHROMA_PERSIST_PATH   = "./chroma_db"

# Phase 0 — local inference
# For 8 GB unified memory use "meta-llama/Llama-3.2-1B-Instruct"
# For 16 GB+ use "meta-llama/Meta-Llama-3.1-8B-Instruct"
LOCAL_MODEL_ID     = "Qwen/Qwen2.5-3B-Instruct"
LOCAL_MODEL_DEVICE = "mps"
SFT_ADAPTER_PATH   = "./poet-sft-lora"
SFT_DATASET_ID     = "merve/poetry"
