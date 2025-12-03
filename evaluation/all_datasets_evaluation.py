from langchain_core.language_models import BaseChatModel
from tqdm import tqdm

from agents.chatbot.agent import AgentChatbot
from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.chatbot.llms.anthropic import AnthropicLLM
from agents.chatbot.llms.prompts.multi_agent_prompts import (
    get_multi_agent_prompts,
)
from agents.chatbot.llms.prompts.prompts import (
    get_detector_prompt,
    get_detector_prompt_as_str,
)
from agents.chatbot.multi_agent import MultiAgentChatbot
from agents.chatbot.plain_chatbot import PlainChatbot
from agents.chatbot.tools import DuckDuckGoSearchRun, get_tools
from agents.logger.logger import get_logger
from agents.models.detector_model import DetectorModel
from evaluation.evaluator_interface import EvaluatorInterface
from evaluation.polish_info.polish_info_evaluator import PolishInfoEvaluator

logger = get_logger()

models: list[BaseChatModel] = [
    AnthropicLLM.get_chat_model(),
]

chatbots: list[type[ChatbotInterface]] = [PlainChatbot]

evaluators: list[type[EvaluatorInterface]] = [
    #LiarEvaluator,
    #MMCovidEvaluator,
    #IsotEvaluator,
    PolishInfoEvaluator,
]

tools: list = [DuckDuckGoSearchRun()]


def create_chatbot_instances(
    models: list[BaseChatModel],
    chatbots: list[type[ChatbotInterface]],
) -> list[ChatbotInterface]:
    """Create instances of all chatbots."""
    chatbot_instances = []
    for chatbot_class in chatbots:
        for model in models:
            if chatbot_class == AgentChatbot:
                chatbot_instance = chatbot_class(
                    model=model,
                    prompt=get_detector_prompt_as_str(),
                    schema=DetectorModel,
                    tools=get_tools(),  # type: ignore[call-arg]
                )
            elif chatbot_class == MultiAgentChatbot:
                chatbot_instance = chatbot_class(
                    model=model,
                    prompts=get_multi_agent_prompts(),  # type: ignore[call-arg]
                    schema=DetectorModel,
                    tools=get_tools(),  # type: ignore[call-arg]
                )
            else:
                chatbot_instance = chatbot_class(
                    model=model,
                    schema=DetectorModel,
                    prompt=get_detector_prompt(),
                )
            chatbot_instances.append(chatbot_instance)
    return chatbot_instances


def create_evaluators_instances(chatbots: list[ChatbotInterface],
                                evaluators: list[type[EvaluatorInterface]]) -> list:
    """Create instances of all evaluators for evaluation.

    Returns:
        list: A list of evaluator instances.

    """
    evaluator_instances = []
    for evaluator_class in evaluators:
        for chatbot in chatbots:
            evaluator_instance = evaluator_class(chatbot)
            evaluator_instances.append(evaluator_instance)
    return evaluator_instances

def evaluate(evaluators: list[EvaluatorInterface]) -> list[dict]:
    """Evaluate various chatbots on different datasets.

    Args:
        evaluators (list[EvaluatorInterface]): list of evaluators to evaluate

    Returns:
        list[dict]: A list of evaluation results, where each dict contains:
            - model_name (str): Name of the language model
            - chatbot_type (str): Type of chatbot ('agent' or 'plain')
            - dataset (str): Name of the dataset
            - metrics (dict): Evaluation metrics returned by the evaluator

    """
    results = []
    for evaluator in tqdm(evaluators):
        chatbot_type = (type(evaluator.chatbot).__name__.lower()
                        .replace("chatbot", ""))

        model_name = getattr(
            evaluator.chatbot.model,
            "model_name",
            type(evaluator.chatbot.model).__name__,
        )

        dataset = type(evaluator).__name__.replace("Evaluator", "")

        logger.info(f"Evaluating {model_name} ({chatbot_type}) on {dataset} dataset")

        metrics = evaluator.evaluate()

        result = {
            "model_name": model_name,
            "chatbot_type": chatbot_type,
            "dataset": dataset,
            "metrics": metrics,
        }
        results.append(result)

    return results

if __name__ == "__main__":
    chatbots_instances = create_chatbot_instances(models, chatbots)
    evaluators_instances = create_evaluators_instances(
        chatbots_instances, evaluators,
    )
    results = evaluate(evaluators_instances)
    logger.info(f"Evaluation results: {results}")


