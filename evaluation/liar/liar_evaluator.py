import json

from agents.agent_api import get_response
from agents.chatbot.chatbot_interface import ChatbotInterface
from evaluation.evaluator import Evaluator
from evaluation.liar.liar_loader import LiarLoader


class LiarEvaluator(Evaluator):
    """A class for the evaluation of a chatbot on the Liar Dataset."""

    def __init__(self, chatbot: ChatbotInterface) -> None:
        """Initialize the LiarEvaluator.

        Args:
            chatbot (Chatbot): The chatbot to be evaluated

        """
        self.chatbot = chatbot
        self.dataset = LiarLoader()

    def evaluate(self) -> dict:
        """Evaluate the chatbot on the Liar Dataset.

        Returns:
            dict: A dictionary containing the accuracy of the chatbot
            with key "accuracy".

        """
        correct = 0
        total = len(self.dataset)

        for i in range(total):
            statement, true_label = self.dataset[i]
            true_label = self._map_label(true_label)
            response = get_response(self.chatbot, statement)
            predicted_label = json.loads(response).get("label")

            if predicted_label == true_label:
                correct += 1

        accuracy = correct / total
        return {"accuracy": accuracy}

    def _map_label(self, label: int) -> str:
        match label:
            case 0 | 4 | 5:
                return "False"
            case 2 | 3:
                return "True"
            case 1:
                return "Unclear"
        msg = f"Unknown label: {label}"
        raise ValueError(msg)