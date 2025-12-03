from typing import Any
from urllib.parse import urlparse

import arxiv
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from agents.logger.logger import get_logger
from agents.vectorstores.vectorstore import Vectorstore

logger = get_logger()


class VectorstoreHolder:
    """Holder for lazy-loaded vectorstore instance."""

    _instance: Vectorstore | None = None

    @classmethod
    def get_vectorstore(cls) -> Vectorstore:
        """Get or initialize the vectorstore lazily."""
        if cls._instance is None:
            cls._instance = Vectorstore("mmcovid")
        return cls._instance


@tool(response_format="content_and_artifact")
def retrieve_context(query: str) -> tuple[str, list]:
    """Retrieve information to help answer a query."""
    logger.info(f"Tool 'retrieve_context' called with query: {query}")
    vectorstore = VectorstoreHolder.get_vectorstore()
    retrieved_docs = vectorstore.get_context(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    logger.info(f"Serialized retrieved context: {serialized}")
    logger.info(f"Number of retrieved documents: {len(retrieved_docs)}")

    return serialized, retrieved_docs


@tool
def verify_claim_sources(claim: str) -> str:
    """Search for credible sources to verify a claim.

    Use this to find recent news articles and fact-checking sources.
    """
    logger.info(f"Tool 'verify_claim_sources' called with claim: {claim}")
    search = DuckDuckGoSearchRun()
    try:
        result = search.run(f"fact check: {claim}")
        logger.info(f"Claim verification search: {result[:200]}...")
    except Exception as e:
        logger.exception("Error during claim verification search")
        error_msg = str(e)
        if "dns error" in error_msg.lower() or "connect" in error_msg.lower():
            return (
                "Unable to verify claim through web search due to network "
                "connectivity issues. Proceeding with available information."
            )
        return "Unable to verify claim through web search"
    else:
        return result


@tool
def search_research_papers(query: str, max_results: int = 10) -> str:
    """Search for academic research papers and scientific studies from arXiv.

    Use this to find peer-reviewed research papers, scientific studies,
    and academic publications related to a query. Particularly useful for
    COVID-19, health, and scientific claims.
    """
    logger.info(
        f"Tool 'search_research_papers' called with query: {query}, "
        f"max_results: {max_results}",
    )
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        client = arxiv.Client()
        results = []

        for i, result in enumerate(client.results(search), 1):
            paper_info = f"""
Paper {i}:
Title: {result.title}
Authors: {', '.join(author.name for author in result.authors)}
Published: {result.published.strftime('%Y-%m-%d')}
Abstract: {result.summary[:500]}...
URL: {result.entry_id}
---"""
            results.append(paper_info)

        if not results:
            return f"No research papers found for query: {query}"

        final_result = "\n".join(results)
        logger.info(f"Found {len(results)} papers for query: {query}")

    except Exception:
        logger.exception("Error during arXiv research paper search")
        return f"Unable to search for research papers on query: {query}"
    else:
        return final_result


@tool
def analyze_news_source(url_or_domain: str) -> str:
    """Analyze the credibility and bias of a news source.

    Use this to check if a news source or website is reliable, known for bias,
    or has a history of spreading misinformation. Provide a URL or domain name.
    """
    logger.info(
        f"Tool 'analyze_news_source' called with "
        f"url_or_domain: {url_or_domain}",
    )
    if url_or_domain.startswith("http"):
        parsed_url = urlparse(url_or_domain)
        domain = parsed_url.netloc
    else:
        domain = url_or_domain

    search = DuckDuckGoSearchRun()
    query = (
        f"{domain} news source credibility bias fact check reliability"
    )
    try:
        result = search.run(query)
        logger.info(f"Source analysis for '{domain}': {result[:200]}...")
    except Exception as e:
        logger.exception(f"Error analyzing news source: {url_or_domain}")
        error_msg = str(e)
        if "dns error" in error_msg.lower() or "connect" in error_msg.lower():
            return (
                f"Unable to analyze source: {url_or_domain} due to "
                f"network connectivity issues."
            )
        return f"Unable to analyze source: {url_or_domain}"
    else:
        return f"Analysis of {domain}:\n{result}"


def get_available_tools() -> dict[str, Any]:
    """Return a dictionary of available tools (excluding vectorstore tool).

    Returns:
        dict: A dictionary mapping tool names to tool instances.

    """
    return {
        "web_search": DuckDuckGoSearchRun(),
        "verify_claim_sources": verify_claim_sources,
        "search_research_papers": search_research_papers,
        "analyze_news_source": analyze_news_source,
    }


def get_tools(selected_tool_names: list[str] | None = None) -> list:
    """Return the list of tools based on selection.

    Args:
        selected_tool_names: List of tool names to include.
                           If None, returns all tools including retrieve_context.

    Returns:
        list: List of tool instances.

    """
    if selected_tool_names is None:
        return [
            retrieve_context,
            DuckDuckGoSearchRun(),
            verify_claim_sources,
            search_research_papers,
            analyze_news_source,
        ]

    available = get_available_tools()
    tools = [retrieve_context]
    tools.extend(
        available[tool_name]
        for tool_name in selected_tool_names
        if tool_name in available
    )

    return tools
