from collections.abc import Generator

from dotenv import load_dotenv

from agents.chatbot.bot import Chatbot
from agents.chatbot.llms.openai import OpenAILLM
from agents.chatbot.llms.prompts import get_detector_prompt

load_dotenv(override=True)


def create_chatbot() -> Chatbot:
    """Create and return a Chatbot instance.

    Returns:
        Chatbot: An instance of the Chatbot class configured
        with the OpenAI model and detector prompt.

    """
    model = OpenAILLM.get_chat_model()
    return Chatbot(model=model, prompt=get_detector_prompt(), id_="123")


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
