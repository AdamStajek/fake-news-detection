import uuid
from collections.abc import Generator

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver 
from pydantic import BaseModel

from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.logger.logger import get_logger

logger = get_logger()


class AgentChatbot(ChatbotInterface):
    """A chatbot that uses an agent to generate responses."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: str,
        schema: BaseModel | None = None,
        tools: list = [],
        id_: str = str(uuid.uuid4()),
    ) -> None:
       """Create a new chatbot instance.

       Args:
        model (BaseChatModel): Language model to use for generating responses.
        prompt (str): Prompt template to use for generating
        responses.
        schema (type[BaseModel] | dict | None): Optional Pydantic model class or
        JSON schema dict to enable structured output.
        id_ (str, optional): Id used to distinguish conversations.
        Defaults to a UUID.

       """
       self.agent = create_agent(model,
                                    system_prompt=prompt,
                                    tools=tools,
                                    response_format=schema,
                                    checkpointer=InMemorySaver())
       self.id = id_

    def chat(self, user_input: str) -> BaseMessage:
        logger.info(f"User input: {user_input}")
        result = self.agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        output = result["messages"][-1]
        logger.info(f"AgentChatbot response: {output.content}")
        return output


    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Generate a response from the chatbot word by word.

        Args:
            user_input (str): The user's input message.

        Raises:
            NotImplementedError: When the function is not implemented in the child.

        Yields:
            Generator[str, None, None]: The chatbot's response message,
            one word at a time.

        """
        for step in self.agent.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": self.id}},
            stream_mode="messages",
        ):
            logger.info(f"AgentChatbot streaming step: {step[0].content}")
            yield step[0].content


