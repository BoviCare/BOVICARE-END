from enum import Enum, auto
from langchain_openai import ChatOpenAI


class LLMProvider(Enum):
    OPEN_AI = auto()
    GEMINI = auto()
    GROK = auto()
    HUGGING_FACE = auto()
    DEEP_SEEK = auto()


class DiagnoseContext:
    def __init__(self, llm_provider, llm_provider_model, llm_provider_key):
        self.llm_provider = LLMProvider[llm_provider]
        self.llm_provider_model = llm_provider_model
        self.llm_provider_key = llm_provider_key
        self.llm = self.get_llm()

    def get_llm(self):
        if self.llm_provider == LLMProvider.OPEN_AI:
            return ChatOpenAI(
                api_key=self.llm_provider_key,
                model=self.llm_provider_model,
                temperature=0,
                cache=None,
            )
        else:
            raise NotImplementedError(
                "We still not support LLM providers other than OPENAI"
            )
