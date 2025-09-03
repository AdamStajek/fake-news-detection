from langchain.embeddings.base import Embeddings

from sentence_transformers import SentenceTransformer
from settings import get_settings


settings = get_settings()


class StEmbeddings(Embeddings):
    """
    Custom SentenceTransformer embeddings class to be used with LangChain.
    Args:
        Embeddings: Base class for embeddings in LangChain.
    """

    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents.

        Args:
            documents (list[str]): List of documents to be embedded.

        Returns:
            list[list[float]]: _List of embeddings for each document.
        """
        return [self.model.encode(d).tolist() for d in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            query (str): The query string to be embedded.
        Returns:
            list[float]: _The embedding for the query.
        """
        return self.model.encode([text])[0].tolist()
