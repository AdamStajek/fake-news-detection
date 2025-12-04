"""Tests for the chatbot interface module."""
from unittest.mock import MagicMock

import pytest

from agents.chatbot.chatbot_interface import ChatbotInterface


class TestChatbotInterface:
    """Test cases for the ChatbotInterface abstract class."""

    def test_chatbot_interface_is_abstract(self) -> None:
        """Test that ChatbotInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ChatbotInterface()

    def test_chatbot_interface_requires_implementation(self) -> None:
        """Test that subclasses must implement abstract methods."""
        class IncompleteChatbot(ChatbotInterface):
            pass

        with pytest.raises(TypeError):
            IncompleteChatbot()

    def test_chatbot_interface_subclass_with_implementation(self) -> None:
        """Test that proper subclass can be instantiated."""
        class CompleteChatbot(ChatbotInterface):
            def __init__(self, model, prompt, schema=None, id_="test") -> None:
                self.model = model
                self.prompt = prompt
                self.schema = schema
                self.id = id_

            def chat(self, user_input: str) -> str:
                return "response"

            def stream_chat(self, user_input: str):
                yield "response"

        mock_model = MagicMock()
        chatbot = CompleteChatbot(mock_model, "test prompt")

        assert chatbot.chat("test") == "response"
        assert list(chatbot.stream_chat("test")) == ["response"]
