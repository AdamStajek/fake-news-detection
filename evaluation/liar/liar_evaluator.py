
from langchain_core.messages import AIMessage

from agents.agent_api import get_response
from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.logger.logger import get_logger
from agents.models.detector_model import DetectorModel
from evaluation.evaluator_interface import EvaluatorInterface
from evaluation.liar.liar_loader import LiarLoader

logger = get_logger()


class LiarEvaluator(EvaluatorInterface):
    """A class for the evaluation of a chatbot on the Liar Dataset."""

    def __init__(self, chatbot: ChatbotInterface, n: int = 20) -> None:
        """Initialize the LiarEvaluator.

        Args:
            chatbot: The chatbot to be evaluated.
            n: Number of samples to evaluate.

        """
        self.chatbot = chatbot
        self.n = n
        self.dataset = LiarLoader(n=n)

    def evaluate(self) -> dict:
        """Evaluate the chatbot on the Liar Dataset.

        Returns:
            dict: A dictionary containing the accuracy of the chatbot
            with key "accuracy".

        """
        correct = 0
        total = len(self.dataset)

        for i in range(total):
            try:
                statement, true_label = self.dataset[i]
                true_label = self._map_label(true_label)
                response = get_response(self.chatbot, statement)
                if not response:
                    logger.info("No response received from chatbot.")
                    continue
                if isinstance(response, AIMessage):
                    response = response.content
                    if not response:
                        logger.info("No response content received from chatbot.")
                        continue
                if not isinstance(response, DetectorModel):
                    response = DetectorModel.parse_raw(response)
                predicted_label = response.label

                logger.info(f"Predicted: {predicted_label}, True: {true_label}")

                if predicted_label == true_label:
                    correct += 1
            except Exception:
                logger.exception(f"Error during evaluation of sample {i}")
                continue

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

