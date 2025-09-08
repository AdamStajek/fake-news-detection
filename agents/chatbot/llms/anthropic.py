import os

from langchain_anthropic import ChatAnthropic

from agents.chatbot.llms.llm import LLM


class AnthropicLLM(LLM):
    """Wrapper around Anthropic LLM for generating responses."""

    @classmethod
    def get_chat_model(cls, model_name: str = "claude-3-7-sonnet-latest") -> ChatAnthropic:
        """Get the Anthropic chat model instance."""
        if "ANTHROPIC_API_KEY" not in os.environ:
            msg = "ANTHROPIC_API_KEY environment variable not set."
            raise ValueError(msg)
        return ChatAnthropic(model=model_name)
