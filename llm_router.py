from typing import Optional

import anthropic
from langchain_anthropic import ChatAnthropic

from constants import SONNET_MODEL_ID, HAIKU_MODEL_ID, LOCAL_MODEL_ID, LOCAL_MODEL_DEVICE, SFT_ADAPTER_PATH


class LLMRouter:
    def __init__(self, anthropic_api_key: str, load_local: bool = False):
        # LangSmith tracing activated by LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY env vars
        self.sonnet = ChatAnthropic(model=SONNET_MODEL_ID, api_key=anthropic_api_key)
        self.haiku  = ChatAnthropic(model=HAIKU_MODEL_ID,  api_key=anthropic_api_key)
        # Raw SDK client for cache_control support (prompt caching)
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

        # Phase 0: local open-weights model for the poet
        # Loaded once here, shared across agents that opt in via use_local=True
        self.local_client: Optional[object] = None
        if load_local:
            from local_llm import LocalLLMClient
            import os
            adapter = SFT_ADAPTER_PATH if os.path.exists(SFT_ADAPTER_PATH) else None
            self.local_client = LocalLLMClient(
                model_id=LOCAL_MODEL_ID,
                adapter_path=adapter,
                device=LOCAL_MODEL_DEVICE,
            )

    def get_reasoning_model(self):
        return self.sonnet

    def get_utility_model(self):
        return self.haiku
