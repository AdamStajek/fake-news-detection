import yaml
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

with open("agents/chatbot/llms/prompts.yaml") as file:
    prompts = yaml.safe_load(file)


def get_detector_prompt() -> ChatPromptTemplate:
    """Load and return the fake news detection prompt template as a ChatPromptTemplate."""
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                prompts["fake_news_detector"]["system"],
            ),
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
