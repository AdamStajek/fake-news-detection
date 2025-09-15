from collections.abc import Generator

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

from agents.chatbot.bot import Chatbot
from agents.chatbot.llms.anthropic import AnthropicLLM
from agents.chatbot.llms.google import GoogleLLM
from agents.chatbot.llms.openai import OpenAILLM
from agents.chatbot.llms.promps.prompts import get_detector_prompt

load_dotenv(override=True)


def create_chatbot(provider: str) -> Chatbot:
    """Create and return a Chatbot instance.

    Args:
        provider (str): The model provider (e.g., "OpenAI", "Anthropic", "Gemini").

    Returns:
        Chatbot: An instance of the Chatbot class configured
        with the OpenAI model and detector prompt.

    """
    model = _choose_provider(provider)
    return Chatbot(model=model, prompt=get_detector_prompt())

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


def get_response(chatbot: Chatbot, user_input: str) -> str:
    """Get a response from the chatbot for the given user input.

    Args:
        chatbot (Chatbot): The Chatbot instance to use for generating the response.
        user_input (str): The input message from the user.

    Returns:
        str: The content of the chatbot's response.

    """
    response = chatbot.chat(user_input)
    return response.content


def stream_response(chatbot: Chatbot, user_input: str) -> Generator[str, None, None]:
    """Stream the response from the chatbot word by word.

    Args:
        chatbot (Chatbot): The Chatbot instance to use for generating the response.
        user_input (str): The input message from the user.

    Yields:
        str: The next word in the chatbot's response.

    """
    yield from chatbot.stream_chat(user_input)