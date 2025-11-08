from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from agents.vectorstores.vectorstore import Vectorstore

vectorstore = Vectorstore("mmcovid")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, list]:
    """Retrieve information to help answer a query."""
    retrieved_docs = vectorstore.get_context(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs)
    return serialized, retrieved_docs

def get_tools() -> list:
    """Return the list of available tools."""
    return [retrieve_context, DuckDuckGoSearchRun()]