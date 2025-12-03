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


class MultiAgentChatbot(ChatbotInterface):
    """A chatbot that uses multiple agents to reach a consensus."""

    def __init__(
        self,
        model: BaseChatModel,
        prompts: list[str],
        schema: BaseModel | None = None,
        tools: list = [],
        id_: str = str(uuid.uuid4()),
    ) -> None:
        """Create a new multi-agent chatbot instance.

        Args:
            model (BaseChatModel): Language model to use for agents.
            prompts (list[str]): List of prompt templates for each agent.
                Should have exactly 3 prompts for 3 agents.
            schema (BaseModel | None): Optional Pydantic model class to
                enable structured output.
            tools (list): List of tools available to the agents.
            id_ (str, optional): Id used to distinguish conversations.
                Defaults to a UUID.

        """
        self.model = model
        self.schema = schema
        self.tools = tools
        self.id = id_

        if len(prompts) != 3:
            msg = "Exactly 3 prompts required for 3 agents"
            raise ValueError(msg)

        # Create 3 agents with different prompts
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

        logger.info("Initialized MultiAgentChatbot with 3 agents")

    def chat(self, user_input: str) -> BaseMessage:
        """Generate a consensus response from multiple agents.

        Args:
            user_input (str): The user's input message.

        Returns:
            BaseMessage: The consensus response from the agents.

        """
        logger.info(f"User input: {user_input}")
        logger.info("Querying all 3 agents...")

        responses = []
        labels = []

        # Get responses from all 3 agents
        for i, agent in enumerate(self.agents, start=1):
            logger.info(f"Querying Agent {i}...")
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    {
                        "configurable": {"thread_id": f"{self.id}_agent{i}"},
                        "recursion_limit": 50,  # Increase limit to allow for tool usage
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
                            f"explanation={output.explanation[:100]}..."
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

        # Determine consensus
        if self.schema and labels:
            # Use majority voting for structured output
            label_counts = Counter(labels)
            majority_label = label_counts.most_common(1)[0][0]
            logger.info(f"Label voting results: {dict(label_counts)}")
            logger.info(f"Consensus label: {majority_label}")

            # Find a response with the majority label
            for response in responses:
                if hasattr(response, "label") and response.label == majority_label:
                    logger.info(f"MultiAgentChatbot consensus: {response}")
                    return response

            # Fallback to first response if no exact match
            return responses[0]
        else:
            # For unstructured output, return the first valid response
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
        # For multi-agent, we'll use the first agent for streaming
        logger.info(
            f"Streaming with Agent 1 (multi-agent streaming "
            f"not fully supported)"
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

