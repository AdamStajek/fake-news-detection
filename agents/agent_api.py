from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dotenv import load_dotenv

from agents.chatbot.agent import AgentChatbot
from agents.chatbot.llms.google import GoogleLLM
from agents.chatbot.llms.prompts.prompts import (
    get_detector_prompt,
    get_detector_prompt_as_str,
)
from agents.chatbot.plain_chatbot import PlainChatbot
from agents.chatbot.tools import get_tools
from agents.logger.logger import get_logger
from agents.models.detector_model import DetectorModel

if TYPE_CHECKING:
    from collections.abc import Generator

    from langchain_core.language_models import BaseChatModel
    from pydantic import BaseModel

    from agents.chatbot.chatbot_interface import ChatbotInterface

load_dotenv(override=True)

logger = get_logger()

MODEL_MAP = {
    "Gemini 2.5 Flash": ("google", "gemini-2.5-flash"),
    "Gemini 2.5 Pro": ("google", "gemini-2.5-pro"),
}


def create_chatbot(
    chatbot_type: Literal["agent", "plain"],
    model_name: str,
    schema: type[BaseModel] | None = DetectorModel,
    vectorstore_collection_name: str | None = None,
    selected_tools: list[str] | None = None,
) -> ChatbotInterface:
    """Create and return a Chatbot instance.

    Args:
        chatbot_type: The type of chatbot ("agent" or "plain").
        model_name: The model name (e.g., "Claude Sonnet 3.7",
            "Gemini 2.5 Flash").
        schema: Optional Pydantic schema for structured output.
            Defaults to DetectorModel. If None, no schema is used.
        vectorstore_collection_name: Name of vectorstore collection
            (required for agent).
        selected_tools: List of tool names to include (only for agent
            chatbot).

    Returns:
        ChatbotInterface: An instance of the chatbot class configured
            with the specified model.

    """
    model = _get_model(model_name)
    if chatbot_type == "plain":
        logger.info(f"Creating plain chatbot with model: {model_name}")
        return PlainChatbot(
            model=model, prompt=get_detector_prompt(), schema=schema,
        )
    if chatbot_type == "agent":
        if vectorstore_collection_name is None:
            msg = (
                "vectorstore_collection_name must be provided "
                "for agent chatbot"
            )
            logger.error(msg)
            raise ValueError(msg)
        logger.info(f"Creating agent chatbot with model: {model_name}")
        tools = get_tools(selected_tools)
        return AgentChatbot(
            model=model,
            prompt=get_detector_prompt_as_str(),
            schema=schema,
            tools=tools,
        )
    msg = f"Unknown chatbot type: {chatbot_type}"
    logger.error(msg)
    raise ValueError(msg)


def _get_model(model_name: str) -> BaseChatModel:
    """Get the language model based on the model name.

    Args:
        model_name: The model name (e.g., "Claude Sonnet 3.7", "Gemini 2.5 Flash").

    Raises:
        ValueError: If the model name is unknown.

    Returns:
        BaseChatModel: The selected language model instance.

    """
    if model_name not in MODEL_MAP:
        msg = f"Unknown model: {model_name}"
        raise ValueError(msg)

    provider, model_id = MODEL_MAP[model_name]

    if provider == "google":
        return GoogleLLM.get_chat_model(model_id)

    msg = f"Unknown provider: {provider}"
    raise ValueError(msg)


def get_response(chatbot: ChatbotInterface, user_input: str) -> str | BaseModel:
    """Get a response from the chatbot for the given user input.

    Args:
        chatbot (PlainChatbot): The PlainChatbot instance to use for
            generating the response.
        user_input (str): The input message from the user.

    Returns:
        str: The content of the chatbot's response.

    """
    return chatbot.chat(user_input)


def stream_response(
    chatbot: ChatbotInterface, user_input: str,
) -> Generator[str, None, None]:
    """Stream the response from the chatbot word by word.

    Args:
        chatbot (PlainChatbot): The PlainChatbot instance to use for
            generating the response.
        user_input (str): The input message from the user.

    Yields:
        str: The next word in the chatbot's response.

    """
    yield from chatbot.stream_chat(user_input)
