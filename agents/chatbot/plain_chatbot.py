from __future__ import annotations

import json
import uuid
from collections.abc import Generator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.logger.logger import get_logger

logger = get_logger()


class PlainChatbot(ChatbotInterface):
    """A chatbot that uses a language model to generate responses."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        schema: type[BaseModel] | dict | None = None,
        id_: str = str(uuid.uuid4()),
    ) -> None:
        """Create a new plain chatbot (without tools and rag) instance.

        Args:
            model (BaseChatModel): Language model to use for generating responses.
            prompt (ChatPromptTemplate): Prompt template to use for generating responses.
            schema (type[BaseModel] | dict | None): Optional Pydantic model class or JSON
                schema dict to enable structured output.
            id_ (str, optional): Id used to distinguish conversations. Defaults to a UUID.

        """
        self.model = model
        self.schema = schema
        self.prompt = prompt
        self.config = RunnableConfig(configurable={"thread_id": id_})
        self.memory = MemorySaver()
        self.trimmer = trim_messages(
            max_tokens=20,
            strategy="last",
            token_counter=len,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        self.app = self._initialize_workflow()

    def _call_model(self, state: MessagesState) -> dict:
        if self.schema is not None:
            chain: RunnableSequence = (
                self.trimmer
                | self.prompt
                | self.model.with_structured_output(self.schema)
            )
            response = chain.invoke(state["messages"])
            if isinstance(response, BaseModel):
                content = response.model_dump_json()
            elif isinstance(response, dict):
                content = json.dumps(response)
            else:
                content = str(response)
            ai_msg = AIMessage(content=content)
            return {"messages": [ai_msg]}

        chain = self.trimmer | self.prompt | self.model
        response = chain.invoke(state["messages"])
        if isinstance(response, list):
            return {"messages": response}
        return {"messages": [response]}

    def _initialize_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow.compile(checkpointer=self.memory)

    def chat(self, user_input: str) -> BaseMessage:
        """Generate a response from the chatbot.

        Args:
            user_input (str): The user's input message.

        Returns:
            BaseMessage: The chatbot's last response message.

        """
        logger.info(f"User input: {user_input}")
        input_message = [HumanMessage(user_input)]
        for _ in range(3):
            output = self.app.invoke({"messages": input_message}, self.config)
            output = output["messages"][-1]
            if output and output.content is not None:
                break
            logger.warning("Empty response from model, retrying...")
        logger.info(f"PlainChatbot response: {output.content}")
        return output

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Generate a response from the chatbot word by word.

        Args:
            user_input (str): The user's input message.

        Yields:
            Generator[str, None, None]: The chatbot's response message,
            one word at a time.

        """
        input_message = [HumanMessage(user_input)]
        logger.info(f"User input: {user_input}")
        logger.info("Streaming chatbot response...")
        for chunk, _ in self.app.stream(
            {"messages": input_message},
            self.config,
            stream_mode="messages",
        ):
            yield chunk
