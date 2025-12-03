from pathlib import Path

import yaml
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

prompts_file = Path("agents/chatbot/llms/prompts/prompts.yaml")
with prompts_file.open() as file:
    prompts = yaml.safe_load(file)


def get_detector_prompt() -> ChatPromptTemplate:
    """Load and return fake news detection prompt template."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                prompts["fake_news_detector"]["system"],
            ),
            MessagesPlaceholder(variable_name="messages"),
        ],
    )


def get_detector_prompt_as_str() -> str:
    """Load and return the fake news detection prompt template as a string."""
    return prompts["fake_news_detector"]["system"]


def get_multi_agent_prompts() -> list[str]:
    """Return the list of prompts for the 3 agents.

    Returns:
        list[str]: List containing 3 specialized prompts for fact-checker,
            bias detector, and context analyst.

    """
    return [
        prompts["multi_agent"]["fact_checker"],
        prompts["multi_agent"]["bias_detector"],
        prompts["multi_agent"]["context_analyst"],
    ]
