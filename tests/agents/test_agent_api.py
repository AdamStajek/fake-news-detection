"""Tests for the agent_api module."""
from unittest.mock import MagicMock, patch

import pytest

from agents.agent_api import (
    _get_model,
    create_chatbot,
    get_response,
    stream_response,
)


class TestCreateChatbot:
    """Test cases for the create_chatbot function."""

    @patch("agents.agent_api.GoogleLLM")
    @patch("agents.agent_api.PlainChatbot")
    @patch("agents.agent_api.get_detector_prompt")
    def test_create_plain_chatbot(self, mock_prompt, mock_plain_chatbot, mock_google_llm) -> None:
        """Test creating a plain chatbot."""
        mock_model = MagicMock()
        mock_google_llm.get_chat_model.return_value = mock_model
        mock_prompt.return_value = MagicMock()

        create_chatbot(
            chatbot_type="plain",
            model_name="Gemini 2.5 Flash",
        )

        mock_google_llm.get_chat_model.assert_called_once_with("gemini-2.5-flash")
        mock_plain_chatbot.assert_called_once()

    @patch("agents.agent_api.GoogleLLM")
    @patch("agents.agent_api.AgentChatbot")
    @patch("agents.agent_api.get_detector_prompt_as_str")
    @patch("agents.agent_api.get_tools")
    def test_create_agent_chatbot(
        self, mock_get_tools, mock_prompt, mock_agent_chatbot, mock_google_llm,
    ) -> None:
        """Test creating an agent chatbot."""
        mock_model = MagicMock()
        mock_google_llm.get_chat_model.return_value = mock_model
        mock_prompt.return_value = "Test prompt"
        mock_get_tools.return_value = []

        create_chatbot(
            chatbot_type="agent",
            model_name="Gemini 2.5 Flash",
            vectorstore_collection_name="test_collection",
            selected_tools=["tool1"],
        )

        mock_google_llm.get_chat_model.assert_called_once_with("gemini-2.5-flash")
        mock_get_tools.assert_called_once_with(["tool1"])
        mock_agent_chatbot.assert_called_once()

    def test_create_agent_without_vectorstore_raises_error(self) -> None:
        """Test that creating agent chatbot without vectorstore raises ValueError."""
        with pytest.raises(ValueError, match="vectorstore_collection_name must be provided"):
            create_chatbot(
                chatbot_type="agent",
                model_name="Gemini 2.5 Flash",
            )

    def test_create_chatbot_unknown_type_raises_error(self) -> None:
        """Test that unknown chatbot type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chatbot type"):
            create_chatbot(
                chatbot_type="unknown_type",
                model_name="Gemini 2.5 Flash",
            )


class TestGetModel:
    """Test cases for the _get_model function."""

    @patch("agents.agent_api.GoogleLLM")
    def test_get_google_model(self, mock_google_llm) -> None:
        """Test getting a Google model."""
        mock_model = MagicMock()
        mock_google_llm.get_chat_model.return_value = mock_model

        model = _get_model("Gemini 2.5 Flash")

        assert model == mock_model
        mock_google_llm.get_chat_model.assert_called_once_with("gemini-2.5-flash")

    def test_get_model_unknown_raises_error(self) -> None:
        """Test that unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            _get_model("Unknown Model")


class TestGetResponse:
    """Test cases for the get_response function."""

    def test_get_response(self) -> None:
        """Test getting response from chatbot."""
        mock_chatbot = MagicMock()
        mock_chatbot.chat.return_value = "Test response"

        response = get_response(mock_chatbot, "Test input")

        assert response == "Test response"
        mock_chatbot.chat.assert_called_once_with("Test input")


class TestStreamResponse:
    """Test cases for the stream_response function."""

    def test_stream_response(self) -> None:
        """Test streaming response from chatbot."""
        mock_chatbot = MagicMock()
        mock_chatbot.stream_chat.return_value = iter(["Hello", " ", "World"])

        result = list(stream_response(mock_chatbot, "Test input"))

        assert result == ["Hello", " ", "World"]
        mock_chatbot.stream_chat.assert_called_once_with("Test input")
