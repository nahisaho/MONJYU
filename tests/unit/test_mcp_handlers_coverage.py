# MCP Server Handlers Coverage Tests
"""
Tests for monjyu.mcp_server.handlers to improve coverage from 60% to 75%+
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest

from monjyu.mcp_server.handlers import (
    json_response,
    error_response,
    handle_search,
    handle_get_document,
    handle_list_documents,
    handle_citation_chain,
    handle_find_related,
    handle_status,
    handle_get_metrics,
    dispatch_tool,
    TOOL_HANDLERS,
)
from monjyu.api import SearchMode


# --------------------------------------------------------------------------- #
# Mock Data Classes
# --------------------------------------------------------------------------- #


@dataclass
class MockCitation:
    """Mock citation for testing."""

    doc_id: str = "doc1"
    title: str = "Test Document"
    text: str = "Sample citation text for testing purposes"
    relevance_score: float = 0.95


@dataclass
class MockSearchResult:
    """Mock search result."""

    query: str = "test query"
    answer: str = "This is a test answer."
    citations: list = None
    search_mode: SearchMode = SearchMode.LAZY
    search_level: int = 1
    total_time_ms: float = 123.45
    llm_calls: int = 2
    citation_count: int = 1

    def __post_init__(self):
        if self.citations is None:
            self.citations = [MockCitation()]


@dataclass
class MockDocument:
    """Mock document for testing."""

    id: str = "doc123"
    title: str = "Test Document Title"
    authors: list = None
    year: int = 2023
    doi: str = "10.1234/test"
    abstract: str = "Test abstract content"
    chunk_count: int = 5
    citation_count: int = 10
    reference_count: int = 15
    influence_score: float = 0.75

    def __post_init__(self):
        if self.authors is None:
            self.authors = ["Author A", "Author B"]


@dataclass
class MockReference:
    """Mock reference for testing."""

    target_id: str = "ref1"


@dataclass
class MockCitationEdge:
    """Mock citation edge for testing."""

    source_id: str = "cite1"


@dataclass
class MockMetrics:
    """Mock citation metrics."""

    citation_count: int = 10
    reference_count: int = 15
    pagerank: float = 0.123456
    hub_score: float = 0.234567
    authority_score: float = 0.345678
    influence_score: float = 0.75


class MockIndexStatus:
    """Mock index status enum."""

    def __init__(self, value: str = "ready"):
        self._value = value

    @property
    def value(self):
        return self._value


class MockLevel:
    """Mock level enum."""

    def __init__(self, value: str):
        self._value = value

    @property
    def value(self):
        return self._value


@dataclass
class MockStatus:
    """Mock MONJYU status."""

    index_status: MockIndexStatus = None
    is_ready: bool = True
    index_levels_built: list = None
    document_count: int = 100
    text_unit_count: int = 500
    noun_phrase_count: int = 1000
    community_count: int = 20
    citation_edge_count: int = 300
    last_error: str = None

    def __post_init__(self):
        if self.index_status is None:
            self.index_status = MockIndexStatus()
        if self.index_levels_built is None:
            self.index_levels_built = [MockLevel("level_0"), MockLevel("level_1")]


# --------------------------------------------------------------------------- #
# Helper Function Tests
# --------------------------------------------------------------------------- #


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_json_response(self) -> None:
        """Test json_response creates proper TextContent."""
        data = {"key": "value", "number": 42}
        result = json_response(data)

        assert len(result) == 1
        assert result[0].type == "text"
        assert '"key": "value"' in result[0].text
        assert '"number": 42' in result[0].text

    def test_json_response_unicode(self) -> None:
        """Test json_response handles Unicode."""
        data = {"日本語": "テスト"}
        result = json_response(data)

        assert "日本語" in result[0].text
        assert "テスト" in result[0].text

    def test_error_response(self) -> None:
        """Test error_response creates error message."""
        result = error_response("Something went wrong")

        assert len(result) == 1
        assert '"error": "Something went wrong"' in result[0].text


# --------------------------------------------------------------------------- #
# Search Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleSearch:
    """Tests for handle_search."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.search.return_value = MockSearchResult()
        return mock

    @pytest.mark.asyncio
    async def test_search_with_query(self, mock_monjyu: MagicMock) -> None:
        """Test basic search with query."""
        args = {"query": "test query"}
        result = await handle_search(mock_monjyu, args)

        assert len(result) == 1
        assert "query" in result[0].text
        assert "answer" in result[0].text
        mock_monjyu.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_without_query(self, mock_monjyu: MagicMock) -> None:
        """Test search without query returns error."""
        args = {}
        result = await handle_search(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Query is required" in result[0].text

    @pytest.mark.asyncio
    async def test_search_with_mode(self, mock_monjyu: MagicMock) -> None:
        """Test search with specific mode."""
        args = {"query": "test", "mode": "vector"}
        await handle_search(mock_monjyu, args)

        mock_monjyu.search.assert_called_with("test", mode=SearchMode.VECTOR, top_k=10)

    @pytest.mark.asyncio
    async def test_search_with_invalid_mode(self, mock_monjyu: MagicMock) -> None:
        """Test search falls back to lazy mode on invalid mode."""
        args = {"query": "test", "mode": "invalid"}
        await handle_search(mock_monjyu, args)

        mock_monjyu.search.assert_called_with("test", mode=SearchMode.LAZY, top_k=10)

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, mock_monjyu: MagicMock) -> None:
        """Test search with custom top_k."""
        args = {"query": "test", "top_k": 20}
        await handle_search(mock_monjyu, args)

        mock_monjyu.search.assert_called_with("test", mode=SearchMode.LAZY, top_k=20)

    @pytest.mark.asyncio
    async def test_search_exception(self, mock_monjyu: MagicMock) -> None:
        """Test search handles exceptions."""
        mock_monjyu.search.side_effect = Exception("Search error")
        args = {"query": "test"}
        result = await handle_search(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Search failed" in result[0].text


# --------------------------------------------------------------------------- #
# Get Document Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleGetDocument:
    """Tests for handle_get_document."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.get_document.return_value = MockDocument()
        return mock

    @pytest.mark.asyncio
    async def test_get_document(self, mock_monjyu: MagicMock) -> None:
        """Test getting a document."""
        args = {"document_id": "doc123"}
        result = await handle_get_document(mock_monjyu, args)

        assert "doc123" in result[0].text
        assert "Test Document Title" in result[0].text

    @pytest.mark.asyncio
    async def test_get_document_without_id(self, mock_monjyu: MagicMock) -> None:
        """Test get document without ID returns error."""
        args = {}
        result = await handle_get_document(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document ID is required" in result[0].text

    @pytest.mark.asyncio
    async def test_get_document_not_found(self, mock_monjyu: MagicMock) -> None:
        """Test get document not found."""
        mock_monjyu.get_document.return_value = None
        args = {"document_id": "nonexistent"}
        result = await handle_get_document(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document not found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_document_exception(self, mock_monjyu: MagicMock) -> None:
        """Test get document handles exceptions."""
        mock_monjyu.get_document.side_effect = Exception("DB error")
        args = {"document_id": "doc123"}
        result = await handle_get_document(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Failed to get document" in result[0].text


# --------------------------------------------------------------------------- #
# List Documents Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleListDocuments:
    """Tests for handle_list_documents."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.list_documents.return_value = [MockDocument(), MockDocument(id="doc2")]
        return mock

    @pytest.mark.asyncio
    async def test_list_documents(self, mock_monjyu: MagicMock) -> None:
        """Test listing documents."""
        args = {}
        result = await handle_list_documents(mock_monjyu, args)

        assert "documents" in result[0].text
        assert "count" in result[0].text

    @pytest.mark.asyncio
    async def test_list_documents_with_limit(self, mock_monjyu: MagicMock) -> None:
        """Test listing documents with limit."""
        args = {"limit": 10}
        await handle_list_documents(mock_monjyu, args)

        mock_monjyu.list_documents.assert_called_with(limit=10)

    @pytest.mark.asyncio
    async def test_list_documents_limit_capped(self, mock_monjyu: MagicMock) -> None:
        """Test listing documents with excessive limit is capped."""
        args = {"limit": 200}
        await handle_list_documents(mock_monjyu, args)

        mock_monjyu.list_documents.assert_called_with(limit=100)

    @pytest.mark.asyncio
    async def test_list_documents_with_offset(self, mock_monjyu: MagicMock) -> None:
        """Test listing documents with offset."""
        docs = [MockDocument(id=f"doc{i}") for i in range(5)]
        mock_monjyu.list_documents.return_value = docs
        args = {"offset": 2}
        result = await handle_list_documents(mock_monjyu, args)

        assert '"offset": 2' in result[0].text

    @pytest.mark.asyncio
    async def test_list_documents_exception(self, mock_monjyu: MagicMock) -> None:
        """Test list documents handles exceptions."""
        mock_monjyu.list_documents.side_effect = Exception("DB error")
        args = {}
        result = await handle_list_documents(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Failed to list documents" in result[0].text


# --------------------------------------------------------------------------- #
# Citation Chain Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleCitationChain:
    """Tests for handle_citation_chain."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.get_document.return_value = MockDocument()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_references.return_value = [MockReference()]
        mock_citation_manager.get_citations.return_value = [MockCitationEdge()]
        mock.get_citation_network.return_value = mock_citation_manager
        return mock

    @pytest.mark.asyncio
    async def test_citation_chain(self, mock_monjyu: MagicMock) -> None:
        """Test getting citation chain."""
        args = {"document_id": "doc123"}
        result = await handle_citation_chain(mock_monjyu, args)

        assert "references" in result[0].text
        assert "cited_by" in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_without_id(self, mock_monjyu: MagicMock) -> None:
        """Test citation chain without ID returns error."""
        args = {}
        result = await handle_citation_chain(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document ID is required" in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_document_not_found(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test citation chain when document not found."""
        mock_monjyu.get_document.return_value = None
        args = {"document_id": "nonexistent"}
        result = await handle_citation_chain(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document not found" in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_with_depth(self, mock_monjyu: MagicMock) -> None:
        """Test citation chain with depth."""
        args = {"document_id": "doc123", "depth": 2}
        result = await handle_citation_chain(mock_monjyu, args)

        assert '"depth": 2' in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_depth_capped(self, mock_monjyu: MagicMock) -> None:
        """Test citation chain depth is capped at 3."""
        args = {"document_id": "doc123", "depth": 10}
        result = await handle_citation_chain(mock_monjyu, args)

        assert '"depth": 3' in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_no_citation_manager(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test citation chain when no citation manager."""
        mock_monjyu.get_citation_network.return_value = None
        args = {"document_id": "doc123"}
        result = await handle_citation_chain(mock_monjyu, args)

        # Should still return empty lists
        assert "references" in result[0].text
        assert "cited_by" in result[0].text

    @pytest.mark.asyncio
    async def test_citation_chain_exception(self, mock_monjyu: MagicMock) -> None:
        """Test citation chain handles exceptions."""
        mock_monjyu.get_document.side_effect = Exception("DB error")
        args = {"document_id": "doc123"}
        result = await handle_citation_chain(mock_monjyu, args)

        assert "error" in result[0].text


# --------------------------------------------------------------------------- #
# Find Related Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleFindRelated:
    """Tests for handle_find_related."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.get_document.return_value = MockDocument()
        mock_citation_manager = MagicMock()
        mock_citation_manager.find_co_citation_papers.return_value = [
            ("related1", 0.9),
            ("related2", 0.8),
        ]
        mock.get_citation_network.return_value = mock_citation_manager
        return mock

    @pytest.mark.asyncio
    async def test_find_related(self, mock_monjyu: MagicMock) -> None:
        """Test finding related papers."""
        args = {"document_id": "doc123"}
        result = await handle_find_related(mock_monjyu, args)

        assert "related_papers" in result[0].text
        assert "co-citation" in result[0].text

    @pytest.mark.asyncio
    async def test_find_related_without_id(self, mock_monjyu: MagicMock) -> None:
        """Test find related without ID returns error."""
        args = {}
        result = await handle_find_related(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document ID is required" in result[0].text

    @pytest.mark.asyncio
    async def test_find_related_document_not_found(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test find related when document not found."""
        mock_monjyu.get_document.return_value = None
        args = {"document_id": "nonexistent"}
        result = await handle_find_related(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document not found" in result[0].text

    @pytest.mark.asyncio
    async def test_find_related_with_top_k(self, mock_monjyu: MagicMock) -> None:
        """Test find related with custom top_k."""
        args = {"document_id": "doc123", "top_k": 5}
        await handle_find_related(mock_monjyu, args)

        mock_monjyu.get_citation_network.return_value.find_co_citation_papers.assert_called_with(
            "doc123", 5
        )

    @pytest.mark.asyncio
    async def test_find_related_top_k_capped(self, mock_monjyu: MagicMock) -> None:
        """Test find related top_k is capped at 50."""
        args = {"document_id": "doc123", "top_k": 100}
        await handle_find_related(mock_monjyu, args)

        mock_monjyu.get_citation_network.return_value.find_co_citation_papers.assert_called_with(
            "doc123", 50
        )

    @pytest.mark.asyncio
    async def test_find_related_no_citation_manager(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test find related when no citation manager."""
        mock_monjyu.get_citation_network.return_value = None
        args = {"document_id": "doc123"}
        result = await handle_find_related(mock_monjyu, args)

        assert "related_papers" in result[0].text
        assert '"count": 0' in result[0].text


# --------------------------------------------------------------------------- #
# Status Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleStatus:
    """Tests for handle_status."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.get_status.return_value = MockStatus()
        return mock

    @pytest.mark.asyncio
    async def test_get_status(self, mock_monjyu: MagicMock) -> None:
        """Test getting status."""
        args = {}
        result = await handle_status(mock_monjyu, args)

        assert "index_status" in result[0].text
        assert "is_ready" in result[0].text
        assert "statistics" in result[0].text

    @pytest.mark.asyncio
    async def test_get_status_exception(self, mock_monjyu: MagicMock) -> None:
        """Test get status handles exceptions."""
        mock_monjyu.get_status.side_effect = Exception("Status error")
        args = {}
        result = await handle_status(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Failed to get status" in result[0].text


# --------------------------------------------------------------------------- #
# Get Metrics Handler Tests
# --------------------------------------------------------------------------- #


class TestHandleGetMetrics:
    """Tests for handle_get_metrics."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.get_document.return_value = MockDocument()
        mock_citation_manager = MagicMock()
        mock_citation_manager.get_metrics.return_value = MockMetrics()
        mock.get_citation_network.return_value = mock_citation_manager
        return mock

    @pytest.mark.asyncio
    async def test_get_metrics(self, mock_monjyu: MagicMock) -> None:
        """Test getting metrics."""
        args = {"document_id": "doc123"}
        result = await handle_get_metrics(mock_monjyu, args)

        assert "metrics" in result[0].text
        assert "citation_count" in result[0].text
        assert "pagerank" in result[0].text

    @pytest.mark.asyncio
    async def test_get_metrics_without_id(self, mock_monjyu: MagicMock) -> None:
        """Test get metrics without ID returns error."""
        args = {}
        result = await handle_get_metrics(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document ID is required" in result[0].text

    @pytest.mark.asyncio
    async def test_get_metrics_document_not_found(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test get metrics when document not found."""
        mock_monjyu.get_document.return_value = None
        args = {"document_id": "nonexistent"}
        result = await handle_get_metrics(mock_monjyu, args)

        assert "error" in result[0].text
        assert "Document not found" in result[0].text

    @pytest.mark.asyncio
    async def test_get_metrics_no_citation_manager(
        self, mock_monjyu: MagicMock
    ) -> None:
        """Test get metrics when no citation manager."""
        mock_monjyu.get_citation_network.return_value = None
        args = {"document_id": "doc123"}
        result = await handle_get_metrics(mock_monjyu, args)

        # Should use document's default metrics
        assert "metrics" in result[0].text

    @pytest.mark.asyncio
    async def test_get_metrics_exception(self, mock_monjyu: MagicMock) -> None:
        """Test get metrics handles exceptions."""
        mock_monjyu.get_document.side_effect = Exception("DB error")
        args = {"document_id": "doc123"}
        result = await handle_get_metrics(mock_monjyu, args)

        assert "error" in result[0].text


# --------------------------------------------------------------------------- #
# Dispatch Tool Tests
# --------------------------------------------------------------------------- #


class TestDispatchTool:
    """Tests for dispatch_tool."""

    @pytest.fixture
    def mock_monjyu(self):
        """Create mock MONJYU instance."""
        mock = MagicMock()
        mock.search.return_value = MockSearchResult()
        mock.get_document.return_value = MockDocument()
        mock.list_documents.return_value = [MockDocument()]
        mock.get_status.return_value = MockStatus()
        mock.get_citation_network.return_value = None
        return mock

    @pytest.mark.asyncio
    async def test_dispatch_search(self, mock_monjyu: MagicMock) -> None:
        """Test dispatching to search handler."""
        result = await dispatch_tool("monjyu_search", {"query": "test"}, mock_monjyu)
        assert "query" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatch_get_document(self, mock_monjyu: MagicMock) -> None:
        """Test dispatching to get_document handler."""
        result = await dispatch_tool(
            "monjyu_get_document", {"document_id": "doc1"}, mock_monjyu
        )
        assert "doc123" in result[0].text or "title" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatch_list_documents(self, mock_monjyu: MagicMock) -> None:
        """Test dispatching to list_documents handler."""
        result = await dispatch_tool("monjyu_list_documents", {}, mock_monjyu)
        assert "documents" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatch_status(self, mock_monjyu: MagicMock) -> None:
        """Test dispatching to status handler."""
        result = await dispatch_tool("monjyu_status", {}, mock_monjyu)
        assert "index_status" in result[0].text

    @pytest.mark.asyncio
    async def test_dispatch_unknown_tool(self, mock_monjyu: MagicMock) -> None:
        """Test dispatching unknown tool returns error."""
        result = await dispatch_tool("unknown_tool", {}, mock_monjyu)
        assert "error" in result[0].text
        assert "Unknown tool" in result[0].text

    def test_tool_handlers_registry(self) -> None:
        """Test all expected handlers are registered."""
        expected_tools = [
            "monjyu_search",
            "monjyu_get_document",
            "monjyu_list_documents",
            "monjyu_citation_chain",
            "monjyu_find_related",
            "monjyu_status",
            "monjyu_get_metrics",
        ]
        for tool in expected_tools:
            assert tool in TOOL_HANDLERS
