import uuid
from collections.abc import Generator

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.messages.ai import AIMessageChunk
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
       self.model = model
       self.schema = schema
       self.tools = tools  # Store tools for tool call handling
       self.agent = create_agent(
           model,
           system_prompt=prompt,
           tools=tools,
           response_format=ToolStrategy(schema) if schema else None,
           checkpointer=InMemorySaver(),
       )
       self.id = id_

    def chat(self, user_input: str) -> BaseMessage:
        retries = 3
        for i in range(retries):
            logger.info(f"User input: {user_input}")
            result = self.agent.invoke({"messages": [{"role": "user", "content": user_input}]},
                                    {"configurable": {"thread_id": self.id}},)
            if self.schema:
                output = result["structured_response"]
            else:
                output = result["messages"][-1].content
            logger.info(f"AgentChatbot response: {output}")
            if output:
                return output
        msg = "Failed to get response from agent after retries."
        raise RuntimeError(msg)


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
            if isinstance(step[0], AIMessageChunk):
                if step[0].tool_calls or step[0].tool_call_chunks:
                    continue

                content = step[0].content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and block.get("text"):
                                yield block["text"]
                        elif hasattr(block, "type") and block.type == "text":
                            if hasattr(block, "text") and block.text:
                                yield block.text


