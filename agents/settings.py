from dataclasses import dataclass


def get_settings():
    """
    Returns:
        Settings: An instance of the Settings dataclass containing configuration values.
    """
    return Settings()


@dataclass
class Settings:
    """Configuration settings for the application."""

    docs_path: str = "./docs"
    vectorstore_path: str = "./vectorstore"
    vectorstore_col_name = "main"
    chunk_size: int = 512
    chunk_overlap: int = 64
    documents_retrieved: int = 10
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    llm: str = "speakleash/Bielik-4.5B-v3.0-Instruct"
    conversations_dir: str = "./conversations"
