"""Pytest configuration and fixtures for the agents tests."""

import os
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

os.environ["GOOGLE_API_KEY"] = "test-google-api-key"
os.environ["OPENAI_API_KEY"] = "test-openai-api-key"


@pytest.fixture
def mock_google_api_key() -> None:
    original_key = os.environ.get("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = "test-google-api-key"
    yield
    if original_key:
        os.environ["GOOGLE_API_KEY"] = original_key
    else:
        del os.environ["GOOGLE_API_KEY"]


@pytest.fixture
def mock_chat_model() -> MagicMock:
    mock = MagicMock()
    mock.invoke.return_value = AIMessage(content="Test response")
    mock.with_structured_output.return_value = mock
    return mock


@pytest.fixture
def sample_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}"),
        ],
    )


@pytest.fixture
def sample_detector_schema():
    from agents.models.detector_model import DetectorModel

    return DetectorModel

