import uuid

from langchain.agents import create_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from agents.chatbot.chatbot_interface import ChatbotInterface


class AgentChatbot(ChatbotInterface):
    """A chatbot that uses an agent to generate responses."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        schema: type[BaseModel] | dict | None = None,
        id_: str = str(uuid.uuid4()),
    ) -> None:
       """Create a new chatbot instance.

       Args:
        model (BaseChatModel): Language model to use for generating responses.
        prompt (ChatPromptTemplate): Prompt template to use for generating
        responses.
        schema (type[BaseModel] | dict | None): Optional Pydantic model class or
        JSON schema dict to enable structured output.
        id_ (str, optional): Id used to distinguish conversations.
        Defaults to a UUID.

       """
       self.agent = create_agent(model, system_prompt=prompt)


