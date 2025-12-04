"""Tests for the tools module."""
from unittest.mock import MagicMock, patch

from agents.chatbot.tools import (
    VectorstoreHolder,
    get_tools,
    verify_claim_sources,
)


class TestVectorstoreHolder:
    """Test cases for the VectorstoreHolder class."""

    def test_vectorstore_holder_lazy_initialization(self) -> None:
        """Test that vectorstore is initialized lazily."""
        # Reset the instance
        VectorstoreHolder._instance = None

        with patch("agents.chatbot.tools.Vectorstore") as mock_vectorstore:
            mock_vs = MagicMock()
            mock_vectorstore.return_value = mock_vs

            # First call should initialize
            vs1 = VectorstoreHolder.get_vectorstore()
            assert mock_vectorstore.call_count == 1

            # Second call should reuse instance
            vs2 = VectorstoreHolder.get_vectorstore()
            assert mock_vectorstore.call_count == 1
            assert vs1 is vs2


class TestRetrieveContext:
    """Test cases for the retrieve_context tool."""

    @patch("agents.chatbot.tools.VectorstoreHolder.get_vectorstore")
    def test_retrieve_context(self, mock_get_vectorstore) -> None:
        """Test that retrieve_context returns formatted documents."""
        from agents.chatbot import tools

        mock_doc1 = MagicMock()
        mock_doc1.page_content = "Content 1"
        mock_doc1.metadata = {"source": "doc1.txt"}

        mock_doc2 = MagicMock()
        mock_doc2.page_content = "Content 2"
        mock_doc2.metadata = {"source": "doc2.txt"}

        mock_vs = MagicMock()
        mock_vs.get_context.return_value = [mock_doc1, mock_doc2]
        mock_get_vectorstore.return_value = mock_vs

        # Call the underlying function before decoration
        content, artifacts = tools.retrieve_context.func("test query")

        assert "Content 1" in content
        assert "Content 2" in content
        assert "doc1.txt" in content
        assert "doc2.txt" in content
        assert len(artifacts) == 2

    @patch("agents.chatbot.tools.VectorstoreHolder.get_vectorstore")
    def test_retrieve_context_empty_results(self, mock_get_vectorstore) -> None:
        """Test retrieve_context with no results."""
        from agents.chatbot import tools

        mock_vs = MagicMock()
        mock_vs.get_context.return_value = []
        mock_get_vectorstore.return_value = mock_vs

        content, artifacts = tools.retrieve_context.func("test query")

        assert content == ""
        assert len(artifacts) == 0


class TestVerifyClaimSources:
    """Test cases for the verify_claim_sources tool."""

    @patch("agents.chatbot.tools.DuckDuckGoSearchRun")
    def test_verify_claim_sources_success(self, mock_ddg) -> None:
        """Test that verify_claim_sources returns search results."""
        mock_search = MagicMock()
        mock_search.run.return_value = "Search results about the claim"
        mock_ddg.return_value = mock_search

        result = verify_claim_sources.invoke("test claim")

        assert result == "Search results about the claim"
        mock_search.run.assert_called_once()

    @patch("agents.chatbot.tools.DuckDuckGoSearchRun")
    def test_verify_claim_sources_exception(self, mock_ddg) -> None:
        """Test that verify_claim_sources handles exceptions."""
        mock_search = MagicMock()
        mock_search.run.side_effect = Exception("Search failed")
        mock_ddg.return_value = mock_search

        result = verify_claim_sources.invoke("test claim")

        # Check that result contains "Unable" since that's what the error handler returns
        assert "Unable" in result


class TestGetTools:
    """Test cases for the get_tools function."""

    def test_get_tools_all(self) -> None:
        """Test getting all available tools."""
        tools = get_tools(None)

        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "retrieve_context" in tool_names
        assert "verify_claim_sources" in tool_names

    def test_get_tools_selected(self) -> None:
        """Test getting specific tools."""
        tools = get_tools(["retrieve_context"])

        tool_names = [tool.name for tool in tools]
        assert "retrieve_context" in tool_names
        assert len(tools) >= 1

    def test_get_tools_empty_list(self) -> None:
        """Test getting tools with empty list returns all tools."""
        tools = get_tools([])

        assert len(tools) > 0
