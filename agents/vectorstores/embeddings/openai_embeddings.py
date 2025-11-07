import os

from langchain_openai import OpenAIEmbeddings

from agents.vectorstores.embeddings.embeddings import CustomEmbeddings


class OpenAIEmbeddingsWrapper(CustomEmbeddings):
    """Wrapper around OpenAI embeddings for embedding text."""

    @classmethod
    def get_embedding_model(cls, model_name: str =
                       "text-embedding-3-small") -> OpenAIEmbeddings:
        """Get the OpenAI embedding model instance."""
        if "OPENAI_API_KEY" not in os.environ:
            msg = "OPENAI_API_KEY environment variable not set."
            raise ValueError(msg)
        return OpenAIEmbeddings(model=model_name)
