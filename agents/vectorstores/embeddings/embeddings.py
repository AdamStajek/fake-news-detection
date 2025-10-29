from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings


class CustomEmbeddings(ABC):
    """Abstract base class for embedding models."""

    @classmethod
    @abstractmethod
    def get_embedding_model(cls, model_name: str) -> Embeddings:
        """Get the embedding model instance."""
        raise NotImplementedError
