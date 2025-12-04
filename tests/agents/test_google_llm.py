"""Tests for the Google LLM module."""
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.chatbot.llms.google import GoogleLLM


class TestGoogleLLM:
    """Test cases for the GoogleLLM class."""

    def test_get_chat_model_without_api_key_raises_error(self) -> None:
        """Test that GoogleLLM raises error when API key is not set."""
        # Remove API key if it exists
        original_key = os.environ.get("GOOGLE_API_KEY")
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        try:
            with pytest.raises(ValueError, match="GOOGLE_API_KEY environment variable not set"):
                GoogleLLM.get_chat_model()
        finally:
            # Restore original key
            if original_key:
                os.environ["GOOGLE_API_KEY"] = original_key

    @patch("agents.chatbot.llms.google.ChatGoogleGenerativeAI")
    def test_get_chat_model_with_api_key(self, mock_chat_google) -> None:
        """Test that GoogleLLM creates chat model when API key is set."""
        # Set API key
        os.environ["GOOGLE_API_KEY"] = "test-key"

        mock_model = MagicMock()
        mock_chat_google.return_value = mock_model

        model = GoogleLLM.get_chat_model("gemini-2.5-flash")

        assert model == mock_model
        mock_chat_google.assert_called_once_with(model="gemini-2.5-flash", max_tokens=1024)

    @patch("agents.chatbot.llms.google.ChatGoogleGenerativeAI")
    def test_get_chat_model_default_model(self, mock_chat_google) -> None:
        """Test that GoogleLLM uses default model name."""
        os.environ["GOOGLE_API_KEY"] = "test-key"

        mock_model = MagicMock()
        mock_chat_google.return_value = mock_model

        GoogleLLM.get_chat_model()

        mock_chat_google.assert_called_once_with(model="gemini-2.5-flash", max_tokens=1024)

    @patch("agents.chatbot.llms.google.ChatGoogleGenerativeAI")
    def test_get_chat_model_custom_model(self, mock_chat_google) -> None:
        """Test that GoogleLLM accepts custom model name."""
        os.environ["GOOGLE_API_KEY"] = "test-key"

        mock_model = MagicMock()
        mock_chat_google.return_value = mock_model

        GoogleLLM.get_chat_model("gemini-2.5-pro")

        mock_chat_google.assert_called_once_with(model="gemini-2.5-pro", max_tokens=1024)
