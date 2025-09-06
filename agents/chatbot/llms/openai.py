import os

from langchain_openai import ChatOpenAI

from agents.chatbot.llms.llm import LLM


class OpenAILLM(LLM):
    """Wrapper around OpenAI LLM for generating responses."""

    @classmethod
    def get_chat_model(cls, model_name: str = "gpt-4.1") -> ChatOpenAI:
        """Get the OpenAI chat model instance."""
        if "OPENAI_API_KEY" not in os.environ:
            msg = "OPENAI_API_KEY environment variable not set."
            raise ValueError(msg)
        return ChatOpenAI(model=model_name, temperature=0)
