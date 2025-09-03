import os

from db.db import Vectorstore
from db.vectorstore_builder import VectorstoreBuilder
from llm.llm import LLM
from settings import get_settings

settings = get_settings()


class App:
    """Main application class for handling user queries and interactions."""

    def __init__(self):
        self.vectorstore = self._get_vectorstore()
        self.llm = LLM()

    def _get_vectorstore(self) -> Vectorstore:
        """Get or build the vectorstore if it doesn't exist.

        Returns:
            Vectorstore: The vectorstore instance.
        """
        vectorstore = Vectorstore()
        if vectorstore.is_empty():
            builder = VectorstoreBuilder(vectorstore)
            builder.build_vectorstore()
        return vectorstore

    def run(self):
        """Run the main application loop."""
        conversation = []
        conversation_filename = input(
            "Enter filename without format to save the conversation. "
            "Leave empty if you don't want it to be saved: "
        )
        conversation_path = (
            os.path.join(settings.conversations_dir, f"{conversation_filename}.txt")
            if conversation_filename
            else None
        )
        while True:
            query = input("Enter your query (or 'exit' to quit): ")
            if query.lower() == "exit":
                break
            context = self.vectorstore.get_context(query)
            context = "\n".join([doc.page_content for doc in context])
            response = self.llm.generate(query, context)
            print(f"Response: {response}\n")
            conversation.append((query, response))
        if conversation_path:
            self._save_conversation(conversation, conversation_path)

    def _save_conversation(self, conversation, path):
        with open(path, "w", encoding="utf-8") as f:
            for query, response in conversation:
                f.write(f"Q: {query}\nA: {response}\n\n")
        print(f"Conversation saved to {path}")


if __name__ == "__main__":
    app = App()
    app.run()
