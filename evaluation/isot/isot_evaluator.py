import json
from evaluation.evaluator_interface import EvaluatorInterface
from evaluation.isot.isot_loader import IsotLoader
from agents.agent_api import get_response
from agents.chatbot.chatbot_interface import ChatbotInterface


class IsotEvaluator(EvaluatorInterface):
    """A class for the evaluation of a chatbot on the ISOT Dataset."""
    
    def __init__(self, chatbot: ChatbotInterface, n: int = 10) -> None:
        """Initialize the IsotEvaluator.

        Args:
            chatbot: The chatbot to be evaluated

        """
        self.chatbot = chatbot
        self.dataset = IsotLoader(n)

    def evaluate(self) -> dict:
        """Evaluate the chatbot on the ISOT Dataset.

        Returns:
            dict: A dictionary containing the accuracy of the chatbot
            with key "accuracy".

        """
        correct = 0
        total = len(self.dataset)

        for i in range(total):
            article, true_label = self.dataset[i]
            response = get_response(self.chatbot, article)
            predicted_label = json.loads(response).get("label")
            if predicted_label == true_label:
                correct += 1

        accuracy = correct / total
        return {"accuracy": accuracy}
