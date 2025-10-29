from uuid import uuid4

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from agents.settings import get_settings

settings = get_settings()




class Vectorstore:
    """Wrapper around Chroma vector store for adding and retrieving documents."""

    def __init__(self, collection_name: str):
        embedding_model = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            collection_name=settings.vectorstore_col_name,
            embedding_function=embedding_model,
            persist_directory=settings.vectorstore_path,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self.docs_retrieved = settings.documents_retrieved

    def is_empty(self) -> bool:
        """Check if the vectorstore is empty.

        Returns:
        bool: True if empty, False otherwise.

        """
        return self.vectorstore._collection.count() == 0

    def add_documents(self, documents) -> None:
        """Add documents to the vectorstore.

        Args:
            documents (list): List of documents to add.

        """
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vectorstore.add_documents(documents=documents, ids=uuids)

    def get_context(self, query: str) -> list:
        """Get context documents similar to the query.

        Args:
            query (str): The query string.
            k (int, optional): Number of similar documents to retrieve. Defaults to 4.

        Returns:
            list: List of similar documents.

        """
        return self.vectorstore.similarity_search(query, k=self.docs_retrieved)
