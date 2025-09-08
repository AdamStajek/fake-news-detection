import uuid
from collections.abc import Generator

from langchain.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.runnables import RunnableConfig, RunnableSequence
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph


class Chatbot:
    """A chatbot that uses a language model to generate responses."""

    def __init__(
        self,
        model: BaseChatModel,
        prompt: ChatPromptTemplate,
        id_: str = str(uuid.uuid4()),
    ) -> None:
        """Create a new chatbot instance.

        Args:
            model (BaseChatModel): _language model to use for generating responses.
            prompt (ChatPromptTemplate):_prompt template to use for generating responses.
            id_ (str, optional): Id used to distinguish the conversations.
            Defaults to str(uuid.uuid4()).

        """
        self.model = model
        self.prompt = prompt
        self.config = RunnableConfig(configurable={"thread_id": id_})
        self.memory = MemorySaver()
        self.trimmer = trim_messages(
            max_tokens=65,
            strategy="last",
            token_counter=model,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        self.app = self._initialize_workflow()

    def _call_model(self, state: MessagesState) -> dict:
        chain: RunnableSequence = self.trimmer | self.prompt | self.model
        response = chain.invoke(state["messages"])
        return {"messages": response}

    def _initialize_workflow(self) -> CompiledStateGraph:
        workflow = StateGraph(state_schema=MessagesState)
        workflow.add_edge(START, "model")
        workflow.add_node("model", self._call_model)
        return workflow.compile(checkpointer=self.memory)

    def chat(self, user_input: str) -> HumanMessage:
        """Generate a response from the chatbot.

        Args:
            user_input (str): The user's input message.

        Returns:
            HumanMessage: The chatbot's response message.

        """
        input_message = [HumanMessage(user_input)]
        output = self.app.invoke({"messages": input_message}, self.config)
        return output["messages"][-1]

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Generate a response from the chatbot word by word.

        Args:
            user_input (str): The user's input message.

        Yields:
            Generator[str, None, None]: The chatbot's response message,
            one word at a time.

        """
        input_message = [HumanMessage(user_input)]
        for chunk, _ in self.app.stream(
            {"messages": input_message},
            self.config,
            stream_mode="messages",
        ):
            yield chunk
