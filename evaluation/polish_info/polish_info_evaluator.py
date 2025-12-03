from agents.agent_api import get_response
from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.logger.logger import get_logger
from agents.models.detector_model import DetectorModel
from evaluation.evaluator_interface import EvaluatorInterface
from evaluation.polish_info.polish_info_loader import PolishInfoLoader
from langchain_core.messages import AIMessage

logger = get_logger()


class PolishInfoEvaluator(EvaluatorInterface):
    """A class for the evaluation of a chatbot on the Polish Info Dataset."""

    def __init__(self, chatbot: ChatbotInterface, n: int = 20) -> None:
        """Initialize the PolishInfoEvaluator.

        Args:
            chatbot (ChatbotInterface): The chatbot to be evaluated
            n (int): Number of samples to evaluate. Defaults to 20.

        """
        self.chatbot = chatbot
        self.dataset = PolishInfoLoader(n)

    def evaluate(self) -> dict:
        """Evaluate the chatbot on the Polish Info Dataset.

        Returns:
            dict: A dictionary containing the accuracy of the chatbot
            with key "accuracy".

        """
        correct = 0
        total = len(self.dataset)

        for i in range(total):
            try:
                statement, true_label = self.dataset[i]
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
                    try:
                        response = DetectorModel.parse_raw(response)
                    except Exception as e:
                        logger.exception(f"Failed to parse response: {e}")
                        continue
                predicted_label = response.label

                logger.info(f"Predicted: {predicted_label}, True: {true_label}")

                if predicted_label == true_label:
                    correct += 1
            except Exception as e:
                logger.exception(f"Error during evaluation of sample {i}: {e}")
                continue

        accuracy = correct / total
        return {"accuracy": accuracy}
