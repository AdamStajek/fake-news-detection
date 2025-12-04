"""Tests for the agent chatbot module."""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from agents.chatbot.agent import AgentChatbot
from agents.models.detector_model import DetectorModel


class TestAgentChatbot:
    """Test cases for the AgentChatbot class."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock language model."""
        return MagicMock()

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        return [MagicMock(), MagicMock()]

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_initialization(self, mock_saver, mock_create_agent, mock_model) -> None:
        """Test that AgentChatbot can be initialized."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=None,
            tools=[],
            id_="test-id",
        )

        assert chatbot.model == mock_model
        assert chatbot.schema is None
        assert chatbot.tools == []
        assert chatbot.id == "test-id"
        mock_create_agent.assert_called_once()

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_initialization_with_schema(
        self, mock_saver, mock_create_agent, mock_model,
    ) -> None:
        """Test that AgentChatbot can be initialized with schema."""
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=DetectorModel,
            tools=[],
            id_="test-id",
        )

        assert chatbot.schema == DetectorModel

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_chat(self, mock_saver, mock_create_agent, mock_model) -> None:
        """Test that AgentChatbot can generate a response."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Test response")],
            "structured_response": None,
        }
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=None,
            tools=[],
            id_="test-id",
        )

        response = chatbot.chat("Test input")

        assert response == "Test response"
        mock_agent.invoke.assert_called_once()

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_chat_with_schema(self, mock_saver, mock_create_agent, mock_model) -> None:
        """Test that AgentChatbot returns structured response when schema is provided."""
        mock_detector = DetectorModel(label="fake", explanation="Test explanation")
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Test")],
            "structured_response": mock_detector,
        }
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=DetectorModel,
            tools=[],
            id_="test-id",
        )

        response = chatbot.chat("Test input")

        assert response == mock_detector
        assert response.label == "fake"

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_chat_retry_on_empty_response(
        self, mock_saver, mock_create_agent, mock_model,
    ) -> None:
        """Test that AgentChatbot retries on empty response."""
        mock_agent = MagicMock()
        # First two calls return empty, third returns valid response
        mock_agent.invoke.side_effect = [
            {"messages": [AIMessage(content="")], "structured_response": None},
            {"messages": [AIMessage(content="")], "structured_response": None},
            {"messages": [AIMessage(content="Valid response")], "structured_response": None},
        ]
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=None,
            tools=[],
            id_="test-id",
        )

        response = chatbot.chat("Test input")

        assert response == "Valid response"
        assert mock_agent.invoke.call_count == 3

    @patch("agents.chatbot.agent.create_agent")
    @patch("agents.chatbot.agent.InMemorySaver")
    def test_agent_chatbot_chat_raises_on_max_retries(
        self, mock_saver, mock_create_agent, mock_model,
    ) -> None:
        """Test that AgentChatbot raises RuntimeError after max retries."""
        mock_agent = MagicMock()
        # All calls return empty response
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="")],
            "structured_response": None,
        }
        mock_create_agent.return_value = mock_agent

        chatbot = AgentChatbot(
            model=mock_model,
            prompt="Test prompt",
            schema=None,
            tools=[],
            id_="test-id",
        )

        with pytest.raises(RuntimeError, match="Failed to get response from agent"):
            chatbot.chat("Test input")

        assert mock_agent.invoke.call_count == 3
