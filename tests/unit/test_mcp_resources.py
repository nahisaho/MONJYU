"""Tests for MCP Server Resources.

FEAT-009: MCP Server Resources
"""

import json
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime


# === Test Fixtures ===

@pytest.fixture
def mock_monjyu():
    """Create a mock MONJYU instance."""
    mock = MagicMock()
    
    # Mock document
    mock_doc = MagicMock()
    mock_doc.id = "doc1"
    mock_doc.title = "Test Document"
    mock_doc.authors = ["Author 1", "Author 2"]
    mock_doc.year = 2024
    mock_doc.doi = "10.1234/test"
    mock_doc.abstract = "Test abstract"
    mock_doc.chunk_count = 10
    mock_doc.citation_count = 5
    mock_doc.reference_count = 20
    mock_doc.influence_score = 0.85
    
    # Mock text unit
    mock_tu = MagicMock()
    mock_tu.id = "tu1"
    mock_tu.text = "Test text content"
    mock_tu.chunk_index = 0
    
    # Mock status
    mock_status = MagicMock()
    mock_status.document_count = 100
    mock_status.text_unit_count = 1000
    mock_status.entity_count = 500
    mock_status.relationship_count = 2000
    mock_status.community_count = 50
    mock_status.index_levels = ["level0", "level1"]
    mock_status.is_ready = True
    mock_status.last_updated = datetime(2024, 12, 28, 12, 0, 0)
    
    # Configure mock methods
    mock.get_document.return_value = mock_doc
    mock.list_documents.return_value = [mock_doc]
    mock.get_text_units.return_value = [mock_tu]
    mock.get_status.return_value = mock_status
    mock.get_citation_network.return_value = None
    
    return mock


@pytest.fixture
def mock_citation_manager():
    """Create a mock citation manager."""
    mock = MagicMock()
    
    # Mock reference
    mock_ref = MagicMock()
    mock_ref.target_id = "ref1"
    
    # Mock citation
    mock_cite = MagicMock()
    mock_cite.source_id = "cite1"
    
    mock.get_references.return_value = [mock_ref]
    mock.get_citations.return_value = [mock_cite]
    mock.get_statistics.return_value = {
        "node_count": 100,
        "edge_count": 500,
        "density": 0.05,
        "avg_citations": 5.0,
    }
    mock.get_most_cited.return_value = [("doc1", 10)]
    
    return mock


# === Resource Index Status Tests ===

class TestResourceIndexStatus:
    """Tests for monjyu://index/status resource."""
    
    @pytest.mark.asyncio
    async def test_get_index_status(self, mock_monjyu):
        """Test getting index status resource."""
        from monjyu.mcp_server.server import (
            resource_index_status,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_index_status()
            data = json.loads(result)
            
            assert data["type"] == "index_status"
            assert data["document_count"] == 100
            assert data["text_unit_count"] == 1000
            assert data["is_ready"] is True
            assert "index_levels" in data
        finally:
            reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_index_status_error(self):
        """Test index status with error."""
        from monjyu.mcp_server.server import (
            resource_index_status,
            set_monjyu,
            reset_monjyu,
        )
        
        mock = MagicMock()
        mock.get_status.side_effect = Exception("Test error")
        
        set_monjyu(mock)
        try:
            result = await resource_index_status()
            data = json.loads(result)
            
            assert "error" in data
        finally:
            reset_monjyu()


# === Resource Documents List Tests ===

class TestResourceDocumentsList:
    """Tests for monjyu://documents resource."""
    
    @pytest.mark.asyncio
    async def test_get_documents_list(self, mock_monjyu):
        """Test getting documents list resource."""
        from monjyu.mcp_server.server import (
            resource_documents_list,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_documents_list()
            data = json.loads(result)
            
            assert data["type"] == "document_list"
            assert data["count"] == 1
            assert len(data["documents"]) == 1
            assert data["documents"][0]["id"] == "doc1"
            assert "uri" in data["documents"][0]
        finally:
            reset_monjyu()


# === Resource Document Detail Tests ===

class TestResourceDocumentDetail:
    """Tests for monjyu://documents/{id} resource."""
    
    @pytest.mark.asyncio
    async def test_get_document_detail(self, mock_monjyu):
        """Test getting document detail resource."""
        from monjyu.mcp_server.server import (
            resource_document_detail,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_document_detail("doc1")
            data = json.loads(result)
            
            assert data["type"] == "document"
            assert data["id"] == "doc1"
            assert data["title"] == "Test Document"
            assert "content_uri" in data
            assert "citations_uri" in data
        finally:
            reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_document_not_found(self, mock_monjyu):
        """Test document not found."""
        from monjyu.mcp_server.server import (
            resource_document_detail,
            set_monjyu,
            reset_monjyu,
        )
        
        mock_monjyu.get_document.return_value = None
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_document_detail("nonexistent")
            data = json.loads(result)
            
            assert "error" in data
            assert "not found" in data["error"].lower()
        finally:
            reset_monjyu()


# === Resource Document Content Tests ===

class TestResourceDocumentContent:
    """Tests for monjyu://documents/{id}/content resource."""
    
    @pytest.mark.asyncio
    async def test_get_document_content(self, mock_monjyu):
        """Test getting document content resource."""
        from monjyu.mcp_server.server import (
            resource_document_content,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_document_content("doc1")
            data = json.loads(result)
            
            assert data["type"] == "document_content"
            assert data["document_id"] == "doc1"
            assert data["text_unit_count"] == 1
            assert len(data["text_units"]) == 1
            assert "full_text" in data
        finally:
            reset_monjyu()


# === Resource Document Citations Tests ===

class TestResourceDocumentCitations:
    """Tests for monjyu://documents/{id}/citations resource."""
    
    @pytest.mark.asyncio
    async def test_get_document_citations_no_network(self, mock_monjyu):
        """Test citations without citation network."""
        from monjyu.mcp_server.server import (
            resource_document_citations,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_document_citations("doc1")
            data = json.loads(result)
            
            assert data["type"] == "document_citations"
            assert data["document_id"] == "doc1"
            assert data["references"] == []
            assert data["cited_by"] == []
        finally:
            reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_get_document_citations_with_network(
        self, mock_monjyu, mock_citation_manager
    ):
        """Test citations with citation network."""
        from monjyu.mcp_server.server import (
            resource_document_citations,
            set_monjyu,
            reset_monjyu,
        )
        
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        # Mock ref doc
        ref_doc = MagicMock()
        ref_doc.title = "Reference Doc"
        ref_doc.year = 2023
        
        # Mock cite doc
        cite_doc = MagicMock()
        cite_doc.title = "Citing Doc"
        cite_doc.year = 2025
        
        def get_doc(doc_id):
            if doc_id == "doc1":
                return mock_monjyu.get_document.return_value
            elif doc_id == "ref1":
                return ref_doc
            elif doc_id == "cite1":
                return cite_doc
            return None
        
        mock_monjyu.get_document.side_effect = get_doc
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_document_citations("doc1")
            data = json.loads(result)
            
            assert data["type"] == "document_citations"
            assert len(data["references"]) == 1
            assert len(data["cited_by"]) == 1
            assert data["references"][0]["id"] == "ref1"
            assert data["cited_by"][0]["id"] == "cite1"
        finally:
            reset_monjyu()


# === Resource Citation Network Tests ===

class TestResourceCitationNetwork:
    """Tests for monjyu://citation-network resource."""
    
    @pytest.mark.asyncio
    async def test_citation_network_not_available(self, mock_monjyu):
        """Test when citation network is not built."""
        from monjyu.mcp_server.server import (
            resource_citation_network,
            set_monjyu,
            reset_monjyu,
        )
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_citation_network()
            data = json.loads(result)
            
            assert data["type"] == "citation_network"
            assert data["available"] is False
        finally:
            reset_monjyu()
    
    @pytest.mark.asyncio
    async def test_citation_network_available(
        self, mock_monjyu, mock_citation_manager
    ):
        """Test when citation network is available."""
        from monjyu.mcp_server.server import (
            resource_citation_network,
            set_monjyu,
            reset_monjyu,
        )
        
        mock_monjyu.get_citation_network.return_value = mock_citation_manager
        
        set_monjyu(mock_monjyu)
        try:
            result = await resource_citation_network()
            data = json.loads(result)
            
            assert data["type"] == "citation_network"
            assert data["available"] is True
            assert "statistics" in data
            assert data["statistics"]["node_count"] == 100
            assert data["statistics"]["edge_count"] == 500
        finally:
            reset_monjyu()


# === Resource URI Format Tests ===

class TestResourceURIFormat:
    """Tests for resource URI format."""
    
    def test_uri_format(self):
        """Test resource URIs follow correct format."""
        expected_uris = [
            "monjyu://index/status",
            "monjyu://documents",
            "monjyu://documents/{document_id}",
            "monjyu://documents/{document_id}/content",
            "monjyu://documents/{document_id}/citations",
            "monjyu://citation-network",
        ]
        
        # Resources are defined with these URIs in server.py
        # This test documents the expected format
        for uri in expected_uris:
            assert uri.startswith("monjyu://")
            assert "//" in uri
