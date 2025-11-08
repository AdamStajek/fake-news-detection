import json

from agents.agent_api import get_response
from agents.chatbot.chatbot_interface import ChatbotInterface
from evaluation.evaluator import Evaluator
from evaluation.mmcovid.mmcovid_loader import MMCovidLoader


class MMCovidEvaluator(Evaluator):
    """A class for the evaluation of a chatbot on the MMCovid Dataset."""

    def __init__(self, chatbot: ChatbotInterface) -> None:
        """Initialize the MMCovidEvaluator.

        Args:
            chatbot (ChatbotInterface): The chatbot to be evaluated

        """
        self.chatbot = chatbot
        self.dataset = MMCovidLoader()

    def evaluate(self) -> dict:
        """Evaluate the chatbot on the MMCovid Dataset.

        Returns:
            dict: A dictionary containing the accuracy of the chatbot
            with key "accuracy".

        """
        correct = 0
        total = len(self.dataset)

        for i in range(total):
            claim, true_label = self.dataset[i]
            response = get_response(self.chatbot, claim)
            predicted_label = json.loads(response).get("label")

            if predicted_label == true_label:
                correct += 1

        accuracy = correct / total
        return {"accuracy": accuracy}