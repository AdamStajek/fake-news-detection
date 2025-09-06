from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLM(ABC):
    """Abstract base class for Language Models."""

    @classmethod
    @abstractmethod
    def get_chat_model(cls, model_name: str) -> BaseChatModel:
        """Get the chat model instance."""
        raise NotImplementedError
