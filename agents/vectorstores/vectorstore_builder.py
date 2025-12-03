import argparse
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from agents.logger.logger import get_logger
from agents.settings import get_settings
from agents.vectorstores.vectorstore import Vectorstore

settings = get_settings()
logger = get_logger()


class VectorstoreBuilder:
    """Class to build and populate the vectorstore with documents."""

    def __init__(self, vectorstore: Vectorstore, docs_path: str) -> None:
        """Initialize VectorstoreBuilder with vectorstore and documents path.

        Args:
            vectorstore (Vectorstore): The vectorstore instance to populate.
            docs_path (str): Path to the directory containing documents.

        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        )
        self.vectorstore = vectorstore
        self.docs_path = docs_path
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def build_vectorstore(self) -> None:
        """Build and populate vectorstore with documents from directory."""
        docs = self._load_docs()
        docs_slices = self._split_into_slices(docs, max_len=5000)
        logger.info(f"Adding {len(docs)} documents to vectorstore.")
        for docs in tqdm(docs_slices):
            self.vectorstore.add_documents(docs)

    def _load_docs(self) -> list[Document]:
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

    @staticmethod
    def _split_into_slices(container: list, max_len: int = 5000) -> list[list]:
        return [container[i:i + max_len] for i in range(0, len(container), max_len)]

    def _get_docs_filepaths(self) -> list:
        """Get file paths of all .txt documents in the specified directory.

        Returns:
            list: List of file paths.

        """
        docs_path = Path(self.docs_path)
        return [
            str(file_path)
            for file_path in docs_path.iterdir()
            if file_path.suffix == ".txt"
        ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--collection_name",
        type=str,
        help="Name of the vectorstore collection.",
    )

    parser.add_argument(
        "--docs_path",
        type=str,
        help="Path to the directory containing documents.",
    )
    args = parser.parse_args()

    vectorstore = Vectorstore(collection_name=args.collection_name)
    builder = VectorstoreBuilder(vectorstore, docs_path=args.docs_path)
    builder.build_vectorstore()
