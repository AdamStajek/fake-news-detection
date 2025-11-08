import uuid
from abc import ABC, abstractmethod
from collections.abc import Generator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel


class ChatbotInterface(ABC):
    """An abstract base class for chatbots."""

    model: BaseChatModel

    @abstractmethod
    def __init__(
        self,
        model: BaseChatModel,
        prompt: ChatPromptTemplate | str,
        schema: type[BaseModel] | dict | None = None,
        id_: str = str(uuid.uuid4()),
    ) -> None:
            """Create a new chatbot instance.

            Args:
                model (BaseChatModel): Language model to use for generating responses.
                prompt (ChatPromptTemplate): Prompt template to use for generating
                responses.
                schema (type[BaseModel] | dict | None): Optional Pydantic model class or
                JSON schema dict to enable structured output.
                id_ (str, optional): Id used to distinguish conversations. 
                Defaults to a UUID.

            """
            msg = "__init__ method not implemented."
            raise NotImplementedError(msg)

    @abstractmethod
    def chat(self, user_input: str) -> BaseMessage:
        """Generate a response from the chatbot.

        Args:
            user_input (str): The user's input message.

        Raises:
            NotImplementedError: When the function is not implemented in the child.

        Returns:
            BaseMessage: The chatbot's last response message.

        """
        msg = "chat method not implemented."
        raise NotImplementedError(msg)

    @abstractmethod
    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Generate a response from the chatbot word by word.

        Args:
            user_input (str): The user's input message.

        Raises:
            NotImplementedError: When the function is not implemented in the child.

        Yields:
            Generator[str, None, None]: The chatbot's response message,
            one word at a time.

        """
        msg = "stream_chat method not implemented."
        raise NotImplementedError(msg)
