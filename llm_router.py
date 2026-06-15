import anthropic
from langchain_anthropic import ChatAnthropic
from constants import SONNET_MODEL_ID, HAIKU_MODEL_ID


class LLMRouter:
    def __init__(self, anthropic_api_key: str):
        # LangSmith tracing activated by LANGCHAIN_TRACING_V2 + LANGCHAIN_API_KEY env vars
        self.sonnet = ChatAnthropic(model=SONNET_MODEL_ID, api_key=anthropic_api_key)
        self.haiku  = ChatAnthropic(model=HAIKU_MODEL_ID,  api_key=anthropic_api_key)
        # Raw SDK client for cache_control support (prompt caching)
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)

    def get_reasoning_model(self):
        """Sonnet — for importance scoring and fast screening."""
        return self.sonnet

    def get_utility_model(self):
        """Haiku — for importance scoring and fast screening."""
        return self.haiku
