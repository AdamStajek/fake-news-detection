"""Tests for the settings module."""

from agents.settings import Settings, get_settings


class TestSettings:
    """Test cases for the Settings dataclass."""

    def test_settings_defaults(self) -> None:
        """Test that Settings initializes with correct default values."""
        settings = Settings()

        assert settings.vectorstore_path == "./knowledge_base/vectorstore"
        assert settings.vectorstore_col_name == "main"
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 64
        assert settings.documents_retrieved == 10

    def test_settings_custom_values(self) -> None:
        """Test that Settings can be initialized with custom values."""
        settings = Settings(
            vectorstore_path="/custom/path",
            chunk_size=1024,
            chunk_overlap=128,
            documents_retrieved=20,
        )

        assert settings.vectorstore_path == "/custom/path"
        assert settings.chunk_size == 1024
        assert settings.chunk_overlap == 128
        assert settings.documents_retrieved == 20

    def test_get_settings(self) -> None:
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.vectorstore_path == "./knowledge_base/vectorstore"
