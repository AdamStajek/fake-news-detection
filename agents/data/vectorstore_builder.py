import os

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from settings import get_settings
from db.db import Vectorstore


settings = get_settings()


class VectorstoreBuilder:
    """Class to build and populate the vectorstore with documents from a specified directory."""

    def __init__(self, vectorstore: Vectorstore):
        self.splitter = MarkdownTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        self.vectorstore = vectorstore
        self.docs_path = settings.docs_path
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def build_vectorstore(self) -> None:
        """Build and populate the vectorstore with documents from the specified directory."""
        docs = self._load_docs()
        self.vectorstore.add_documents(docs)

    def _load_docs(self) -> list:
        """Load and split documents from the specified directory.

        Returns:
            list: List of split documents.

        """
        splits = []
        doc_paths = self._get_docs_filepaths()
        for doc_path in doc_paths:
            loader = TextLoader(doc_path)
            doc = loader.load()
            split = self.splitter.split_documents(doc)
            splits.extend(split)
        return splits

    def _get_docs_filepaths(self) -> list:
        """Get file paths of all .txt documents in the specified directory.

        Returns:
            list: List of file paths.

        """
        docs_paths = []
        for filename in os.listdir(self.docs_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.docs_path, filename)
                docs_paths.append(file_path)
        return docs_paths
