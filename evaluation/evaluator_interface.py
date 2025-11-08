from abc import ABC, abstractmethod

from agents.chatbot.chatbot_interface import ChatbotInterface


class EvaluatorInterface(ABC):
    """An abstract base class for chatbot evaluators."""

    chatbot: ChatbotInterface

    @abstractmethod
    def __init__(self, chatbot: ChatbotInterface) -> None:
        """Create the evaluator class instance.

        Args:
            chatbot (ChatbotInterface): chatbot to be evaluated

        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    @abstractmethod
    def evaluate(self) -> dict:
        """Evaluate the chatbot.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            dict: A dictionary containing evaluation metrics.

        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)
