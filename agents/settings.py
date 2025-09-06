from dataclasses import dataclass


@dataclass
class Settings:
    """Configuration settings for the application."""

    docs_path: str = "./docs"
    vectorstore_path: str = "./vectorstore"
    vectorstore_col_name = "main"
    chunk_size: int = 512
    chunk_overlap: int = 64
    documents_retrieved: int = 10


def get_settings() -> Settings:
    """Retrieve the application settings.

    Returns:
    Settings: An instance of the Settings dataclass containing configuration values.

    """
    return Settings()
