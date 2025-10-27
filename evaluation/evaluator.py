from abc import ABC, abstractmethod


class Evaluator(ABC):
    """An abstract base class for chatbot evaluators."""

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
