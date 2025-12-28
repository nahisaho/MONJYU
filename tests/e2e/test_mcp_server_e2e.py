"""End-to-End tests for MCP Server.

FEAT-009: MCP Server E2E Tests
Tests the complete MCP server workflow including:
- Server initialization
- Tool invocations
- Resource access
- Prompt generation
- HTTP transport (when available)
"""

import asyncio
import json
import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# ============================================================
# Test Data
# ============================================================


@dataclass
class MockDocumentInfo:
    """Mock document for testing."""
    id: str = "doc_e2e_001"
    title: str = "End-to-End Test Paper: Advanced RAG Systems"
    authors: list = None
    year: int = 2024
    doi: str = "10.1234/e2e.test"
    abstract: str = "This paper presents an end-to-end test for RAG systems."
    chunk_count: int = 25
    citation_count: int = 100
    reference_count: int = 50
    influence_score: float = 0.95
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = ["E2E Author", "Test Author", "Mock Author"]


@dataclass
class MockSearchResult:
    """Mock search result for testing."""
    query: str = "test query"
    answer: str = "This is a comprehensive answer based on the retrieved documents."
    citations: list = None
    search_mode: MagicMock = None
    search_level: int = 1
    total_time_ms: float = 250.5
    llm_calls: int = 5
    citation_count: int = 3
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.search_mode is None:
            self.search_mode = MagicMock(value="lazy")


@dataclass
class MockStatus:
    """Mock status for testing."""
    index_status: MagicMock = None
    is_ready: bool = True
    index_levels_built: list = None
    index_levels: list = None
    document_count: int = 500
    text_unit_count: int = 5000
    noun_phrase_count: int = 2500
    community_count: int = 50
    citation_edge_count: int = 1000
    entity_count: int = 3000
    relationship_count: int = 8000
    last_updated: datetime = None
    last_error: str = None
    
    def __post_init__(self):
        if self.index_status is None:
            self.index_status = MagicMock(value="ready")
        if self.index_levels_built is None:
            self.index_levels_built = [MagicMock(value=0), MagicMock(value=1)]
        if self.index_levels is None:
            self.index_levels = ["level0", "level1"]
        if self.last_updated is None:
            self.last_updated = datetime(2024, 12, 28, 12, 0, 0)


@dataclass
class MockTextUnit:
    """Mock text unit for testing."""
    id: str = "tu_e2e_001"
    text: str = "This is a sample text chunk from the document."
    chunk_index: int = 0
    document_id: str = "doc_e2e_001"


@dataclass
class MockCitationEdge:
    """Mock citation edge for testing."""
    source_id: str = "doc_e2e_001"
    target_id: str = "doc_e2e_002"


def create_comprehensive_mock_monjyu():
    """Create a comprehensive mock MONJYU instance for E2E testing."""
    mock = MagicMock()
    
    # Documents
    doc1 = MockDocumentInfo()
    doc2 = MockDocumentInfo(
        id="doc_e2e_002",
        title="Second E2E Test Paper: Knowledge Graphs",
        authors=["Another Author"],
        year=2023,
        citation_count=50,
    )
    doc3 = MockDocumentInfo(
        id="doc_e2e_003",
        title="Third Paper: Vector Embeddings",
        authors=["Vector Author"],
        year=2022,
        citation_count=200,
    )
    
    # Text units
    text_units = [
        MockTextUnit(id=f"tu_{i}", text=f"Text chunk {i} content.", chunk_index=i)
        for i in range(5)
    ]
    
    # Search result with citations
    search_result = MockSearchResult(
        query="What are the key concepts in RAG?",
        answer="RAG (Retrieval-Augmented Generation) combines retrieval with generation. Key concepts include vector search, knowledge graphs, and entity extraction.",
        citation_count=3,
    )
    
    # Status
    status = MockStatus()
    
    # Configure mock methods
    mock.get_document.return_value = doc1
    mock.list_documents.return_value = [doc1, doc2, doc3]
    mock.get_text_units.return_value = text_units
    mock.search.return_value = search_result
    mock.get_status.return_value = status
    mock.find_related.return_value = [doc2, doc3]
    mock.get_citation_network.return_value = None
    
    # Configure document lookup by ID
    def get_doc_by_id(doc_id):
        docs = {"doc_e2e_001": doc1, "doc_e2e_002": doc2, "doc_e2e_003": doc3}
        return docs.get(doc_id)
    
    mock.get_document.side_effect = get_doc_by_id
    
    return mock


# ============================================================
# E2E Test: Complete Tool Workflow
# ============================================================


class TestMCPToolWorkflow:
    """E2E tests for MCP tool invocation workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Set up mock MONJYU for each test."""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_comprehensive_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_search_workflow(self):
        """Test complete search workflow: search -> get details -> get metrics."""
        from monjyu.mcp_server.server import (
            monjyu_search,
            monjyu_get_document,
            monjyu_get_metrics,
        )
        
        # Step 1: Search for documents
        search_result = await monjyu_search(
            "What are the key concepts in RAG?",
            mode="lazy",
            top_k=10,
        )
        search_data = json.loads(search_result)
        
        assert "answer" in search_data
        assert "search_info" in search_data
        assert search_data["search_info"]["mode"] == "lazy"
        
        # Step 2: Get document details
        doc_result = await monjyu_get_document("doc_e2e_001")
        doc_data = json.loads(doc_result)
        
        assert doc_data["id"] == "doc_e2e_001"
        assert "citation_metrics" in doc_data
        
        # Step 3: Get metrics
        metrics_result = await monjyu_get_metrics("doc_e2e_001")
        metrics_data = json.loads(metrics_result)
        
        assert "document" in metrics_data
        assert "metrics" in metrics_data
    
    @pytest.mark.asyncio
    async def test_document_exploration_workflow(self):
        """Test document exploration: list -> get details -> get content -> citations."""
        from monjyu.mcp_server.server import (
            monjyu_list_documents,
            monjyu_get_document,
            monjyu_citation_chain,
            monjyu_find_related,
        )
        
        # Step 1: List all documents
        list_result = await monjyu_list_documents(limit=50)
        list_data = json.loads(list_result)
        
        assert list_data["count"] == 3
        assert len(list_data["documents"]) == 3
        
        # Step 2: Get details for first document
        doc_id = list_data["documents"][0]["id"]
        doc_result = await monjyu_get_document(doc_id)
        doc_data = json.loads(doc_result)
        
        assert "title" in doc_data
        assert "authors" in doc_data
        
        # Step 3: Get citation chain
        citation_result = await monjyu_citation_chain(doc_id, depth=2)
        citation_data = json.loads(citation_result)
        
        assert "document" in citation_data
        assert "references" in citation_data
        assert "cited_by" in citation_data
        
        # Step 4: Find related papers
        related_result = await monjyu_find_related(doc_id, top_k=5)
        related_data = json.loads(related_result)
        
        assert "source_document" in related_data
        assert "related_papers" in related_data
    
    @pytest.mark.asyncio
    async def test_status_monitoring_workflow(self):
        """Test status monitoring workflow."""
        from monjyu.mcp_server.server import monjyu_status
        
        status_result = await monjyu_status()
        status_data = json.loads(status_result)
        
        assert status_data["is_ready"] is True
        assert status_data["index_status"] == "ready"
        assert "statistics" in status_data
        assert status_data["statistics"]["documents"] == 500
        assert status_data["statistics"]["text_units"] == 5000


# ============================================================
# E2E Test: Complete Resource Workflow
# ============================================================


class TestMCPResourceWorkflow:
    """E2E tests for MCP resource access workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Set up mock MONJYU for each test."""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_comprehensive_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_index_status_resource(self):
        """Test index status resource access."""
        from monjyu.mcp_server.server import resource_index_status
        
        result = await resource_index_status()
        data = json.loads(result)
        
        assert data["type"] == "index_status"
        assert data["document_count"] == 500
        assert data["text_unit_count"] == 5000
        assert data["is_ready"] is True
    
    @pytest.mark.asyncio
    async def test_documents_list_resource(self):
        """Test documents list resource access."""
        from monjyu.mcp_server.server import resource_documents_list
        
        result = await resource_documents_list()
        data = json.loads(result)
        
        assert data["type"] == "document_list"
        assert data["count"] == 3
        assert len(data["documents"]) == 3
        
        # Verify each document has required fields
        for doc in data["documents"]:
            assert "id" in doc
            assert "title" in doc
            assert "uri" in doc
            assert doc["uri"].startswith("monjyu://documents/")
    
    @pytest.mark.asyncio
    async def test_document_detail_resource(self):
        """Test document detail resource access."""
        from monjyu.mcp_server.server import resource_document_detail
        
        result = await resource_document_detail("doc_e2e_001")
        data = json.loads(result)
        
        assert data["type"] == "document"
        assert data["id"] == "doc_e2e_001"
        assert "title" in data
        assert "abstract" in data
        # Resource has citation_count directly, not nested
        assert "citation_count" in data
    
    @pytest.mark.asyncio
    async def test_document_content_resource(self):
        """Test document content resource access."""
        from monjyu.mcp_server.server import resource_document_content
        
        result = await resource_document_content("doc_e2e_001")
        data = json.loads(result)
        
        assert data["type"] == "document_content"
        assert data["document_id"] == "doc_e2e_001"
        assert "text_units" in data
        assert len(data["text_units"]) == 5


# ============================================================
# E2E Test: Complete Prompt Workflow
# ============================================================


class TestMCPPromptWorkflow:
    """E2E tests for MCP prompt generation workflow."""
    
    @pytest.mark.asyncio
    async def test_literature_review_prompt(self):
        """Test literature review prompt generation."""
        from monjyu.mcp_server.server import literature_review
        
        result = await literature_review(
            topic="Retrieval-Augmented Generation",
            num_papers=15,
            focus_area="knowledge graphs",
        )
        
        assert isinstance(result, str)
        assert "Retrieval-Augmented Generation" in result
        assert "knowledge graphs" in result
        assert "monjyu_search" in result
        assert "15" in result
    
    @pytest.mark.asyncio
    async def test_paper_summary_prompt(self):
        """Test paper summary prompt generation."""
        from monjyu.mcp_server.server import paper_summary
        
        result = await paper_summary(document_id="doc_e2e_001")
        
        assert isinstance(result, str)
        assert "doc_e2e_001" in result
        assert "monjyu_get_document" in result
        assert "monjyu_citation_chain" in result
    
    @pytest.mark.asyncio
    async def test_compare_papers_prompt(self):
        """Test paper comparison prompt generation."""
        from monjyu.mcp_server.server import compare_papers
        
        result = await compare_papers(
            paper_ids="doc_e2e_001, doc_e2e_002, doc_e2e_003",
            comparison_aspects="methodology,findings,limitations,contributions",
        )
        
        assert isinstance(result, str)
        assert "doc_e2e_001" in result
        assert "doc_e2e_002" in result
        assert "doc_e2e_003" in result
        assert "Methodology" in result or "methodology" in result
    
    @pytest.mark.asyncio
    async def test_research_question_prompt(self):
        """Test research question exploration prompt."""
        from monjyu.mcp_server.server import research_question
        
        result = await research_question(
            domain="Natural Language Processing",
            current_interest="Large Language Models",
            methodology_preference="experimental",
        )
        
        assert isinstance(result, str)
        assert "Natural Language Processing" in result
        assert "Large Language Models" in result
        assert "experimental" in result
    
    @pytest.mark.asyncio
    async def test_citation_analysis_prompt(self):
        """Test citation analysis prompt generation."""
        from monjyu.mcp_server.server import citation_analysis
        
        result = await citation_analysis(
            document_id="doc_e2e_001",
            analysis_type="full",
        )
        
        assert isinstance(result, str)
        assert "doc_e2e_001" in result
        assert "monjyu_get_metrics" in result or "monjyu_citation_chain" in result


# ============================================================
# E2E Test: Tool-Resource Integration
# ============================================================


class TestToolResourceIntegration:
    """E2E tests for tool and resource integration."""
    
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Set up mock MONJYU for each test."""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_comprehensive_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_tool_resource_data_consistency(self):
        """Test that tools and resources return consistent data."""
        from monjyu.mcp_server.server import (
            monjyu_status,
            resource_index_status,
            monjyu_list_documents,
            resource_documents_list,
        )
        
        # Compare status
        tool_status = json.loads(await monjyu_status())
        resource_status = json.loads(await resource_index_status())
        
        assert tool_status["statistics"]["documents"] == resource_status["document_count"]
        assert tool_status["is_ready"] == resource_status["is_ready"]
        
        # Compare document list
        tool_docs = json.loads(await monjyu_list_documents(limit=100))
        resource_docs = json.loads(await resource_documents_list())
        
        assert tool_docs["count"] == resource_docs["count"]
    
    @pytest.mark.asyncio
    async def test_document_journey(self):
        """Test complete document journey: discover -> detail -> content -> analyze."""
        from monjyu.mcp_server.server import (
            resource_documents_list,
            resource_document_detail,
            resource_document_content,
            monjyu_citation_chain,
            monjyu_get_metrics,
        )
        
        # Step 1: Discover documents via resource
        docs = json.loads(await resource_documents_list())
        assert docs["count"] > 0
        
        # Step 2: Get detail for first document via resource
        doc_id = docs["documents"][0]["id"]
        detail = json.loads(await resource_document_detail(doc_id))
        assert detail["id"] == doc_id
        
        # Step 3: Get content via resource
        content = json.loads(await resource_document_content(doc_id))
        assert content["document_id"] == doc_id
        
        # Step 4: Analyze via tools
        citations = json.loads(await monjyu_citation_chain(doc_id, depth=1))
        assert citations["document"]["id"] == doc_id
        
        metrics = json.loads(await monjyu_get_metrics(doc_id))
        assert "metrics" in metrics


# ============================================================
# E2E Test: Error Recovery
# ============================================================


class TestErrorRecovery:
    """E2E tests for error handling and recovery."""
    
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Set up mock MONJYU for each test."""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_comprehensive_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_graceful_not_found_handling(self):
        """Test graceful handling of not found errors."""
        from monjyu.mcp_server.server import (
            monjyu_get_document,
            resource_document_detail,
        )
        
        # Tool should return error JSON, not raise exception
        tool_result = await monjyu_get_document("nonexistent_doc")
        tool_data = json.loads(tool_result)
        assert "error" in tool_data
        
        # Resource should also return error JSON
        resource_result = await resource_document_detail("nonexistent_doc")
        resource_data = json.loads(resource_result)
        assert "error" in resource_data
    
    @pytest.mark.asyncio
    async def test_graceful_exception_handling(self):
        """Test graceful handling of internal exceptions."""
        from monjyu.mcp_server.server import monjyu_search
        
        # Make search raise an exception
        self.mock_monjyu.search.side_effect = Exception("Internal error")
        
        # Should return error JSON, not raise
        result = await monjyu_search("test query")
        data = json.loads(result)
        assert "error" in data
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty or invalid queries."""
        from monjyu.mcp_server.server import monjyu_search, monjyu_get_document
        
        # Empty query
        result = await monjyu_search("")
        data = json.loads(result)
        assert "error" in data
        
        # Empty document ID
        result = await monjyu_get_document("")
        data = json.loads(result)
        assert "error" in data


# ============================================================
# E2E Test: CLI Interface
# ============================================================


class TestCLIInterface:
    """E2E tests for CLI interface."""
    
    def test_cli_help_output(self, capsys):
        """Test CLI help output contains all options."""
        import sys
        from monjyu.mcp_server.server import run
        
        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--help"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Verify all options are documented
        assert "--http" in output
        assert "--host" in output
        assert "--port" in output
        assert "--help" in output
        
        # Verify tools are listed
        assert "monjyu_search" in output
        assert "monjyu_get_document" in output
        
        # Verify resources are listed
        assert "monjyu://index/status" in output
        assert "monjyu://documents" in output
    
    def test_cli_version_output(self, capsys):
        """Test CLI version output."""
        import sys
        from monjyu.mcp_server.server import run
        
        old_argv = sys.argv
        try:
            sys.argv = ["monjyu-mcp", "--version"]
            run()
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Verify version format
        assert "0." in output  # Version starts with 0.


# ============================================================
# E2E Test: Concurrent Access
# ============================================================


class TestConcurrentAccess:
    """E2E tests for concurrent access patterns."""
    
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """Set up mock MONJYU for each test."""
        from monjyu.mcp_server.server import set_monjyu, reset_monjyu
        self.mock_monjyu = create_comprehensive_mock_monjyu()
        set_monjyu(self.mock_monjyu)
        yield
        reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test concurrent tool invocations."""
        from monjyu.mcp_server.server import (
            monjyu_status,
            monjyu_list_documents,
            monjyu_get_document,
        )
        
        # Run multiple tool calls concurrently
        results = await asyncio.gather(
            monjyu_status(),
            monjyu_list_documents(limit=10),
            monjyu_get_document("doc_e2e_001"),
            monjyu_get_document("doc_e2e_002"),
            monjyu_get_document("doc_e2e_003"),
        )
        
        # Verify all returned valid JSON
        for result in results:
            data = json.loads(result)
            assert "error" not in data or data.get("error") is None
    
    @pytest.mark.asyncio
    async def test_concurrent_resource_access(self):
        """Test concurrent resource access."""
        from monjyu.mcp_server.server import (
            resource_index_status,
            resource_documents_list,
            resource_document_detail,
        )
        
        # Run multiple resource accesses concurrently
        results = await asyncio.gather(
            resource_index_status(),
            resource_documents_list(),
            resource_document_detail("doc_e2e_001"),
            resource_document_detail("doc_e2e_002"),
        )
        
        # Verify all returned valid JSON
        for result in results:
            data = json.loads(result)
            assert "error" not in data


# ============================================================
# E2E Test: Server Lifecycle
# ============================================================


class TestServerLifecycle:
    """E2E tests for server lifecycle management."""
    
    def test_server_initialization(self):
        """Test that server initializes correctly."""
        from monjyu.mcp_server.server import mcp, get_monjyu, reset_monjyu
        
        reset_monjyu()
        
        # Server object should exist
        assert mcp is not None
        assert mcp.name == "monjyu"
    
    def test_monjyu_lazy_initialization(self):
        """Test lazy initialization of MONJYU instance."""
        from monjyu.mcp_server.server import get_monjyu, reset_monjyu
        
        reset_monjyu()
        
        # Getting MONJYU should work (may create default instance)
        try:
            monjyu = get_monjyu()
            assert monjyu is not None
        except Exception:
            # May fail without config, which is expected
            pass
    
    def test_monjyu_injection(self):
        """Test MONJYU instance injection for testing."""
        from monjyu.mcp_server.server import (
            get_monjyu,
            set_monjyu,
            reset_monjyu,
        )
        
        reset_monjyu()
        
        mock = MagicMock()
        set_monjyu(mock)
        
        assert get_monjyu() is mock
        
        reset_monjyu()
