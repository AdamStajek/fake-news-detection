from __future__ import annotations

from collections.abc import Generator
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agents.chatbot.agent import AgentChatbot
from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.chatbot.llms.anthropic import AnthropicLLM
from agents.chatbot.llms.google import GoogleLLM
from agents.chatbot.llms.openai import OpenAILLM
from agents.chatbot.llms.prompts.prompts import (
    get_detector_prompt,
    get_detector_prompt_as_str,
)
from agents.chatbot.plain_chatbot import PlainChatbot
from agents.chatbot.tools import get_tools
from agents.logger.logger import get_logger
from agents.models.detector_model import DetectorModel

load_dotenv(override=True)

logger = get_logger()

def create_chatbot(chatbot_type: Literal["agent", "plain"], provider: str,
                   schema: type[BaseModel] | None = DetectorModel, 
                   vectorstore_collection_name: str | None = None) -> ChatbotInterface:
    """Create and return a PlainChatbot instance.

    Args:
        provider (str): The model provider (e.g., "OpenAI", "Anthropic", "Gemini").
        schema (BaseModel | None): Optional Pydantic schema for structured output.
        Defaults to DetectorModel. If None, no schema is used.

    Returns:
        PlainChatbot: An instance of the PlainChatbot class configured
        with the OpenAI model and detector prompt.

    """
    model = _choose_provider(provider)
    if chatbot_type == "plain":
        logger.info(f"Creating plain chatbot with provider: {provider}")
        return PlainChatbot(model=model, prompt=get_detector_prompt(), schema=schema)
    if chatbot_type == "agent":
        if vectorstore_collection_name is None:
            msg = "vectorstore_collection_name must be provided for agent chatbot"
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Creating agent chatbot with provider: {provider}")
        tools = get_tools()
        return AgentChatbot(
            model=model,
            prompt=get_detector_prompt_as_str(),
            schema=schema,
            tools=tools,
        )
    msg = f"Unknown chatbot type: {chatbot_type}"
    logger.error(msg)
    raise ValueError(msg)


def _choose_provider(provider: str) -> BaseChatModel:
    """Choose the language model based on the provider.

    Args:
        provider (str): The model provider (e.g., "OpenAI", "Anthropic", "Gemini").

    Raises:
        ValueError: If the provider is unknown.

    Returns:
        BaseChatModel: The selected language model instance.

    """
    if provider == "OpenAI":
        return OpenAILLM.get_chat_model()
    if provider == "Anthropic":
        return AnthropicLLM.get_chat_model()
    if provider == "Gemini":
        return GoogleLLM.get_chat_model()
    msg = f"Unknown provider: {provider}"
    raise ValueError(msg)


def get_response(chatbot: ChatbotInterface, user_input: str) -> str:
    """Get a response from the chatbot for the given user input.

    Args:
        chatbot (PlainChatbot): The PlainChatbot instance to use for generating the response.
        user_input (str): The input message from the user.

    Returns:
        str: The content of the chatbot's response.

    """
    response = chatbot.chat(user_input)
    return response.content


def stream_response(chatbot: ChatbotInterface, user_input: str) -> Generator[str, None, None]:
    """Stream the response from the chatbot word by word.

    Args:
        chatbot (PlainChatbot): The PlainChatbot instance to use for generating the response.
        user_input (str): The input message from the user.

    Yields:
        str: The next word in the chatbot's response.

    """
    yield from chatbot.stream_chat(user_input)
