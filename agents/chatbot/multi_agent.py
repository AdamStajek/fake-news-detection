import uuid
from collections import Counter
from collections.abc import Generator

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from agents.chatbot.chatbot_interface import ChatbotInterface
from agents.logger.logger import get_logger

logger = get_logger()

NUM_AGENTS = 3


class MultiAgentChatbot(ChatbotInterface):
    """A chatbot that uses multiple agents to reach a consensus."""

    def __init__(
        self,
        model: BaseChatModel,
        prompts: list[str],
        schema: BaseModel | None = None,
        tools: list | None = None,
        id_: str = str(uuid.uuid4()),
    ) -> None:
        """Create a new multi-agent chatbot instance.

        Args:
            model (BaseChatModel): Language model to use for agents.
            prompts (list[str]): List of prompt templates for each agent.
                Should have exactly 3 prompts for 3 agents.
            schema (BaseModel | None): Optional Pydantic model class to
                enable structured output.
            tools (list | None): List of tools available to the agents.
            id_ (str, optional): Id used to distinguish conversations.
                Defaults to a UUID.

        """
        if tools is None:
            tools = []
        self.model = model
        self.schema = schema
        self.tools = tools
        self.id = id_

        if len(prompts) != NUM_AGENTS:
            msg = f"Exactly {NUM_AGENTS} prompts required for {NUM_AGENTS} agents"
            raise ValueError(msg)

        self.agents = [
            create_agent(
                model,
                system_prompt=prompt,
                tools=tools,
                response_format=ToolStrategy(schema) if schema else None,
                checkpointer=InMemorySaver(),
            )
            for prompt in prompts
        ]

        logger.info(f"Initialized MultiAgentChatbot with {NUM_AGENTS} agents")

    def chat(self, user_input: str) -> BaseMessage:
        """Generate a consensus response from multiple agents.

        Args:
            user_input (str): The user's input message.

        Returns:
            BaseMessage: The consensus response from the agents.

        """
        logger.info(f"User input: {user_input}")
        logger.info(f"Querying all {NUM_AGENTS} agents...")

        responses = []
        labels = []

        for i, agent in enumerate(self.agents, start=1):
            logger.info(f"Querying Agent {i}...")
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    {
                        "configurable": {"thread_id": f"{self.id}_agent{i}"},
                        "recursion_limit": 50,
                    },
                )

                if self.schema:
                    output = result.get("structured_response")
                else:
                    output = result["messages"][-1].content

                if output:
                    responses.append(output)
                    if self.schema and hasattr(output, "label"):
                        labels.append(output.label)
                        logger.info(
                            f"Agent {i} response: "
                            f"label={output.label}, "
                            f"explanation={output.explanation[:100]}...",
                        )
                    else:
                        logger.info(f"Agent {i} response: {output}")
                else:
                    logger.warning(f"Agent {i} returned empty response")

            except Exception:
                logger.exception(f"Error getting response from Agent {i}")
                continue

        if not responses:
            msg = "Failed to get responses from any agent"
            raise RuntimeError(msg)

        if self.schema and labels:
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]
            logger.info(f"Label voting results: {dict(label_counts)}")
            logger.info(f"Consensus label: {majority_label}")

            for response in responses:
                if hasattr(response, "label") and response.label == majority_label:
                    logger.info(f"MultiAgentChatbot consensus: {response}")
                    return response

            return responses[0]

        logger.info(f"MultiAgentChatbot response: {responses[0]}")
        return responses[0]

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Generate a response from the chatbot word by word.

        Args:
            user_input (str): The user's input message.

        Yields:
            Generator[str, None, None]: The chatbot's response message,
            one word at a time.

        """
        logger.info(
            "Streaming with Agent 1 (multi-agent streaming "
            "not fully supported)",
        )
        for step in self.agents[0].stream(
            {"messages": [{"role": "user", "content": user_input}]},
            {"configurable": {"thread_id": f"{self.id}_agent1"}},
            stream_mode="messages",
        ):
            if (
                isinstance(step, tuple)
                and len(step) > 0
                and hasattr(step[0], "content")
            ):
                yield step[0].content

