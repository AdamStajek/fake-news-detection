from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agents.settings import get_settings
from agents.vectorstores.embeddings.openai_embeddings import OpenAIEmbeddingsWrapper

settings = get_settings()
embedding_model = OpenAIEmbeddingsWrapper.get_embedding_model()

class Vectorstore:
    """Wrapper around Chroma vector store for adding and retrieving documents."""

    def __init__(self, collection_name: str,
                 embedding_function: Embeddings = embedding_model):
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=settings.vectorstore_path,
        )
        self.docs_retrieved = settings.documents_retrieved

    def is_empty(self) -> bool:
        """Check if the vectorstore is empty.

        Returns:
        bool: True if empty, False otherwise.

        """
        return self.vectorstore._collection.count() == 0

    def add_documents(self, documents: list[Document]) -> None:
        """Add documents to the vectorstore.

        Args:
            documents (list): List of documents to add.

        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)

    def get_context(self, query: str) -> list[Document]:
        """Get context documents similar to the query.

        Args:
            query (str): The query string.

        Returns:
            list: List of similar documents.

        """
        return self.vectorstore.similarity_search(query, k=self.docs_retrieved)
