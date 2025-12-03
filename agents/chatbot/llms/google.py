import os

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.chatbot.llms.llm import LLM


class GoogleLLM(LLM):
    """Wrapper around Google LLM for generating responses."""

    @classmethod
    def get_chat_model(
        cls, model_name: str = "gemini-2.5-flash",
    ) -> ChatGoogleGenerativeAI:
        """Get the Google chat model instance."""
        if "GOOGLE_API_KEY" not in os.environ:
            msg = "GOOGLE_API_KEY environment variable not set."
            raise ValueError(msg)
        return ChatGoogleGenerativeAI(model=model_name, max_tokens=1024)
