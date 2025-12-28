# Test External API Clients - NFR-INT-001
"""
外部APIクライアントの単体テスト
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime


# =============================================================================
# Test Data Models
# =============================================================================


class TestAuthor:
    """Authorデータモデルのテスト"""
    
    def test_author_creation(self):
        """Author作成テスト"""
        from monjyu.external.base import Author
        
        author = Author(
            name="John Doe",
            author_id="12345",
            affiliation="MIT",
            orcid="0000-0001-2345-6789",
        )
        
        assert author.name == "John Doe"
        assert author.author_id == "12345"
        assert author.affiliation == "MIT"
        assert author.orcid == "0000-0001-2345-6789"
    
    def test_author_defaults(self):
        """Authorデフォルト値テスト"""
        from monjyu.external.base import Author
        
        author = Author(name="Jane Smith")
        
        assert author.name == "Jane Smith"
        assert author.author_id is None
        assert author.affiliation is None
        assert author.orcid is None
    
    def test_author_to_dict(self):
        """to_dictテスト"""
        from monjyu.external.base import Author
        
        author = Author(name="John Doe", author_id="123")
        data = author.to_dict()
        
        assert data["name"] == "John Doe"
        assert data["author_id"] == "123"
    
    def test_author_from_dict(self):
        """from_dictテスト"""
        from monjyu.external.base import Author
        
        data = {
            "name": "John Doe",
            "author_id": "123",
            "affiliation": "Stanford",
        }
        author = Author.from_dict(data)
        
        assert author.name == "John Doe"
        assert author.affiliation == "Stanford"


class TestCitation:
    """Citationデータモデルのテスト"""
    
    def test_citation_creation(self):
        """Citation作成テスト"""
        from monjyu.external.base import Citation
        
        citation = Citation(
            paper_id="paper-123",
            title="Attention Is All You Need",
            doi="10.1234/example",
            year=2017,
            is_influential=True,
        )
        
        assert citation.paper_id == "paper-123"
        assert citation.title == "Attention Is All You Need"
        assert citation.is_influential is True
    
    def test_citation_defaults(self):
        """Citationデフォルト値テスト"""
        from monjyu.external.base import Citation
        
        citation = Citation(paper_id="p1", title="Test")
        
        assert citation.doi is None
        assert citation.year is None
        assert citation.is_influential is False
    
    def test_citation_to_dict(self):
        """to_dictテスト"""
        from monjyu.external.base import Citation
        
        citation = Citation(paper_id="p1", title="Test", year=2020)
        data = citation.to_dict()
        
        assert data["paper_id"] == "p1"
        assert data["year"] == 2020
    
    def test_citation_from_dict(self):
        """from_dictテスト"""
        from monjyu.external.base import Citation
        
        data = {"paper_id": "p1", "title": "Test", "is_influential": True}
        citation = Citation.from_dict(data)
        
        assert citation.is_influential is True


class TestPaperMetadata:
    """PaperMetadataデータモデルのテスト"""
    
    def test_paper_creation(self):
        """PaperMetadata作成テスト"""
        from monjyu.external.base import PaperMetadata, Author
        
        paper = PaperMetadata(
            paper_id="paper-123",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            abstract="We introduce a new language representation model...",
            authors=[Author(name="Jacob Devlin")],
            year=2019,
            venue="NAACL",
            doi="10.18653/v1/N19-1423",
            citation_count=50000,
        )
        
        assert paper.paper_id == "paper-123"
        assert paper.title == "BERT: Pre-training of Deep Bidirectional Transformers"
        assert len(paper.authors) == 1
        assert paper.citation_count == 50000
    
    def test_paper_defaults(self):
        """PaperMetadataデフォルト値テスト"""
        from monjyu.external.base import PaperMetadata
        
        paper = PaperMetadata(paper_id="p1", title="Test Paper")
        
        assert paper.abstract == ""
        assert paper.authors == []
        assert paper.year is None
        assert paper.citation_count == 0
        assert paper.retrieved_at is not None
    
    def test_paper_author_names(self):
        """author_namesプロパティテスト"""
        from monjyu.external.base import PaperMetadata, Author
        
        paper = PaperMetadata(
            paper_id="p1",
            title="Test",
            authors=[
                Author(name="Alice"),
                Author(name="Bob"),
                Author(name="Charlie"),
            ],
        )
        
        assert paper.author_names == ["Alice", "Bob", "Charlie"]
    
    def test_paper_has_doi(self):
        """has_doiプロパティテスト"""
        from monjyu.external.base import PaperMetadata
        
        paper_with_doi = PaperMetadata(
            paper_id="p1", title="Test", doi="10.1234/test"
        )
        paper_without_doi = PaperMetadata(paper_id="p2", title="Test2")
        
        assert paper_with_doi.has_doi is True
        assert paper_without_doi.has_doi is False
    
    def test_paper_has_arxiv(self):
        """has_arxivプロパティテスト"""
        from monjyu.external.base import PaperMetadata
        
        paper_with_arxiv = PaperMetadata(
            paper_id="p1", title="Test", arxiv_id="2301.12345"
        )
        paper_without_arxiv = PaperMetadata(paper_id="p2", title="Test2")
        
        assert paper_with_arxiv.has_arxiv is True
        assert paper_without_arxiv.has_arxiv is False
    
    def test_paper_to_dict(self):
        """to_dictテスト"""
        from monjyu.external.base import PaperMetadata, Author
        
        paper = PaperMetadata(
            paper_id="p1",
            title="Test",
            authors=[Author(name="Alice")],
            year=2023,
        )
        data = paper.to_dict()
        
        assert data["paper_id"] == "p1"
        assert data["title"] == "Test"
        assert len(data["authors"]) == 1
        assert "retrieved_at" in data
    
    def test_paper_from_dict(self):
        """from_dictテスト"""
        from monjyu.external.base import PaperMetadata
        
        data = {
            "paper_id": "p1",
            "title": "Test Paper",
            "authors": [{"name": "Alice"}],
            "year": 2023,
            "doi": "10.1234/test",
            "retrieved_at": "2024-01-01T12:00:00",
        }
        paper = PaperMetadata.from_dict(data)
        
        assert paper.paper_id == "p1"
        assert paper.year == 2023
        assert len(paper.authors) == 1


# =============================================================================
# Test Exceptions
# =============================================================================


class TestExceptions:
    """例外クラスのテスト"""
    
    def test_external_api_error(self):
        """ExternalAPIErrorテスト"""
        from monjyu.external.base import ExternalAPIError
        
        error = ExternalAPIError(
            message="Test error",
            api_name="TestAPI",
            status_code=500,
        )
        
        assert str(error) == "Test error"
        assert error.api_name == "TestAPI"
        assert error.status_code == 500
    
    def test_rate_limit_error(self):
        """RateLimitErrorテスト"""
        from monjyu.external.base import RateLimitError
        
        error = RateLimitError(
            api_name="TestAPI",
            retry_after=60,
        )
        
        assert error.status_code == 429
        assert error.retry_after == 60
    
    def test_api_response_error(self):
        """APIResponseErrorテスト"""
        from monjyu.external.base import APIResponseError
        
        error = APIResponseError(
            message="Not Found",
            api_name="TestAPI",
            status_code=404,
            response_body='{"error": "not found"}',
        )
        
        assert error.status_code == 404
        assert error.response_body == '{"error": "not found"}'


# =============================================================================
# Test Configuration
# =============================================================================


class TestConfig:
    """設定クラスのテスト"""
    
    def test_external_api_config_defaults(self):
        """ExternalAPIConfigデフォルト値テスト"""
        from monjyu.external.base import ExternalAPIConfig
        
        config = ExternalAPIConfig()
        
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert "MONJYU" in config.user_agent
    
    def test_semantic_scholar_config(self):
        """SemanticScholarConfigテスト"""
        from monjyu.external.semantic_scholar import SemanticScholarConfig
        
        config = SemanticScholarConfig(
            api_key="test-key",
            include_citations=True,
        )
        
        assert config.api_key == "test-key"
        assert config.include_citations is True
    
    def test_crossref_config(self):
        """CrossRefConfigテスト"""
        from monjyu.external.crossref import CrossRefConfig
        
        config = CrossRefConfig(mailto="test@example.com")
        
        assert config.mailto == "test@example.com"
    
    def test_unified_config(self):
        """UnifiedMetadataConfigテスト"""
        from monjyu.external.unified import UnifiedMetadataConfig
        
        config = UnifiedMetadataConfig(
            semantic_scholar_api_key="ss-key",
            crossref_mailto="test@example.com",
            prefer_semantic_scholar=False,
        )
        
        assert config.semantic_scholar_api_key == "ss-key"
        assert config.crossref_mailto == "test@example.com"
        assert config.prefer_semantic_scholar is False


# =============================================================================
# Test Semantic Scholar Client
# =============================================================================


class TestSemanticScholarClient:
    """SemanticScholarClientのテスト"""
    
    def test_client_creation(self):
        """クライアント作成テスト"""
        from monjyu.external.semantic_scholar import SemanticScholarClient
        
        client = SemanticScholarClient()
        
        assert client.api_name == "SemanticScholar"
        assert "semanticscholar.org" in client.base_url
    
    def test_client_with_config(self):
        """設定付きクライアント作成テスト"""
        from monjyu.external.semantic_scholar import (
            SemanticScholarClient,
            SemanticScholarConfig,
        )
        
        config = SemanticScholarConfig(api_key="test-key")
        client = SemanticScholarClient(config)
        
        headers = client._get_headers()
        assert headers.get("x-api-key") == "test-key"
    
    def test_parse_paper(self):
        """_parse_paperテスト"""
        from monjyu.external.semantic_scholar import SemanticScholarClient
        
        client = SemanticScholarClient()
        
        data = {
            "paperId": "abc123",
            "title": "Test Paper",
            "abstract": "This is a test abstract",
            "authors": [
                {"name": "John Doe", "authorId": "auth-1"},
                {"name": "Jane Smith", "authorId": "auth-2"},
            ],
            "year": 2023,
            "venue": "NeurIPS",
            "externalIds": {
                "DOI": "10.1234/test",
                "ArXiv": "2301.12345",
            },
            "citationCount": 100,
            "referenceCount": 50,
            "fieldsOfStudy": ["Computer Science", "Machine Learning"],
            "url": "https://example.com",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        }
        
        paper = client._parse_paper(data)
        
        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.doi == "10.1234/test"
        assert paper.arxiv_id == "2301.12345"
        assert paper.citation_count == 100
        assert paper.source == "semantic_scholar"
    
    def test_parse_paper_empty(self):
        """空データでの_parse_paperテスト"""
        from monjyu.external.semantic_scholar import SemanticScholarClient
        
        client = SemanticScholarClient()
        
        assert client._parse_paper({}) is None
        assert client._parse_paper({"title": "No ID"}) is None
    
    def test_parse_citation(self):
        """_parse_citationテスト"""
        from monjyu.external.semantic_scholar import SemanticScholarClient
        
        client = SemanticScholarClient()
        
        data = {
            "paperId": "cit-1",
            "title": "Citing Paper",
            "year": 2024,
            "externalIds": {"DOI": "10.1234/cit"},
        }
        
        citation = client._parse_citation(data, is_influential=True)
        
        assert citation.paper_id == "cit-1"
        assert citation.title == "Citing Paper"
        assert citation.is_influential is True
        assert citation.doi == "10.1234/cit"
    
    def test_factory_function(self):
        """ファクトリ関数テスト"""
        from monjyu.external.semantic_scholar import create_semantic_scholar_client
        
        client = create_semantic_scholar_client(
            api_key="test-key",
            include_citations=True,
        )
        
        assert client._config.api_key == "test-key"
        assert client._config.include_citations is True


# =============================================================================
# Test CrossRef Client
# =============================================================================


class TestCrossRefClient:
    """CrossRefClientのテスト"""
    
    def test_client_creation(self):
        """クライアント作成テスト"""
        from monjyu.external.crossref import CrossRefClient
        
        client = CrossRefClient()
        
        assert client.api_name == "CrossRef"
        assert "crossref.org" in client.base_url
    
    def test_client_with_mailto(self):
        """mailto付きクライアント作成テスト"""
        from monjyu.external.crossref import CrossRefClient, CrossRefConfig
        
        config = CrossRefConfig(mailto="test@example.com")
        client = CrossRefClient(config)
        
        params = client._add_mailto_param({})
        assert params["mailto"] == "test@example.com"
    
    def test_parse_work(self):
        """_parse_workテスト"""
        from monjyu.external.crossref import CrossRefClient
        
        client = CrossRefClient()
        
        data = {
            "DOI": "10.1234/test",
            "title": ["Test Paper Title"],
            "abstract": "<p>Test abstract</p>",
            "author": [
                {
                    "given": "John",
                    "family": "Doe",
                    "ORCID": "0000-0001-2345-6789",
                    "affiliation": [{"name": "MIT"}],
                },
            ],
            "published": {"date-parts": [[2023, 6, 15]]},
            "container-title": ["Nature"],
            "is-referenced-by-count": 50,
            "references-count": 30,
            "reference": [
                {"DOI": "10.1234/ref1", "article-title": "Reference 1"},
            ],
            "type": "journal-article",
            "publisher": "Nature Publishing Group",
        }
        
        paper = client._parse_work(data)
        
        assert paper.doi == "10.1234/test"
        assert paper.title == "Test Paper Title"
        assert "Test abstract" in paper.abstract
        assert len(paper.authors) == 1
        assert paper.authors[0].name == "John Doe"
        assert paper.year == 2023
        assert paper.venue == "Nature"
        assert paper.citation_count == 50
        assert paper.source == "crossref"
        assert len(paper.references) == 1
    
    def test_parse_work_empty(self):
        """空データでの_parse_workテスト"""
        from monjyu.external.crossref import CrossRefClient
        
        client = CrossRefClient()
        
        assert client._parse_work({}) is None
        assert client._parse_work(None) is None
    
    def test_factory_function(self):
        """ファクトリ関数テスト"""
        from monjyu.external.crossref import create_crossref_client
        
        client = create_crossref_client(mailto="test@example.com")
        
        assert client._config.mailto == "test@example.com"


# =============================================================================
# Test Unified Client
# =============================================================================


class TestUnifiedMetadataClient:
    """UnifiedMetadataClientのテスト"""
    
    def test_client_creation(self):
        """クライアント作成テスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        
        client = UnifiedMetadataClient()
        
        assert client._semantic_scholar is not None
        assert client._crossref is not None
    
    def test_client_with_config(self):
        """設定付きクライアント作成テスト"""
        from monjyu.external.unified import (
            UnifiedMetadataClient,
            UnifiedMetadataConfig,
        )
        
        config = UnifiedMetadataConfig(
            semantic_scholar_api_key="ss-key",
            crossref_mailto="test@example.com",
        )
        client = UnifiedMetadataClient(config)
        
        assert client._semantic_scholar._config.api_key == "ss-key"
        assert client._crossref._config.mailto == "test@example.com"
    
    def test_merge_papers(self):
        """_merge_papersテスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        from monjyu.external.base import PaperMetadata, Author
        
        client = UnifiedMetadataClient()
        
        paper1 = PaperMetadata(
            paper_id="p1",
            title="Test Paper",
            abstract="Abstract from source 1",
            authors=[Author(name="Author 1")],
            source="semantic_scholar",
            citation_count=100,
        )
        
        paper2 = PaperMetadata(
            paper_id="p1",
            title="Test Paper",
            abstract="",
            authors=[],
            year=2023,
            venue="NeurIPS",
            source="crossref",
            citation_count=50,
        )
        
        merged = client._merge_papers([paper1, paper2])
        
        assert merged.title == "Test Paper"
        assert merged.abstract == "Abstract from source 1"  # 空でない方
        assert len(merged.authors) == 1  # 空でない方
        assert merged.year == 2023  # 空でない方
        assert merged.citation_count == 100  # 大きい方
        assert merged.source == "unified"
    
    def test_merge_papers_single(self):
        """単一論文でのマージテスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        from monjyu.external.base import PaperMetadata
        
        client = UnifiedMetadataClient()
        
        paper = PaperMetadata(paper_id="p1", title="Test")
        merged = client._merge_papers([paper])
        
        assert merged == paper
    
    def test_merge_papers_empty(self):
        """空リストでのマージテスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        
        client = UnifiedMetadataClient()
        
        assert client._merge_papers([]) is None
        assert client._merge_papers([None, None]) is None
    
    def test_deduplicate_papers(self):
        """_deduplicate_papersテスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        from monjyu.external.base import PaperMetadata
        
        client = UnifiedMetadataClient()
        
        papers = [
            PaperMetadata(paper_id="p1", title="Paper 1", doi="10.1234/a"),
            PaperMetadata(paper_id="p2", title="Paper 2", doi="10.1234/b"),
            PaperMetadata(paper_id="p3", title="Paper 1 Dup", doi="10.1234/a"),  # 重複
        ]
        
        unique = client._deduplicate_papers(papers)
        
        assert len(unique) == 2
    
    def test_deduplicate_citations(self):
        """_deduplicate_citationsテスト"""
        from monjyu.external.unified import UnifiedMetadataClient
        from monjyu.external.base import Citation
        
        client = UnifiedMetadataClient()
        
        citations = [
            Citation(paper_id="c1", title="Citation 1", doi="10.1234/a"),
            Citation(paper_id="c2", title="Citation 2", doi="10.1234/b"),
            Citation(paper_id="c3", title="Citation 1 Dup", doi="10.1234/a"),
        ]
        
        unique = client._deduplicate_citations(citations)
        
        assert len(unique) == 2
    
    def test_factory_function(self):
        """ファクトリ関数テスト"""
        from monjyu.external.unified import create_unified_client
        
        client = create_unified_client(
            semantic_scholar_api_key="ss-key",
            crossref_mailto="test@example.com",
            prefer_semantic_scholar=False,
        )
        
        assert client.config.semantic_scholar_api_key == "ss-key"
        assert client.config.prefer_semantic_scholar is False


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """モジュールエクスポートのテスト"""
    
    def test_base_exports(self):
        """base.pyエクスポートテスト"""
        from monjyu.external.base import (
            ExternalAPIClient,
            ExternalAPIConfig,
            ExternalAPIError,
            RateLimitError,
            APIResponseError,
            PaperMetadata,
            Author,
            Citation,
        )
        
        assert ExternalAPIClient is not None
        assert ExternalAPIConfig is not None
        assert PaperMetadata is not None
    
    def test_semantic_scholar_exports(self):
        """semantic_scholar.pyエクスポートテスト"""
        from monjyu.external.semantic_scholar import (
            SemanticScholarClient,
            SemanticScholarConfig,
            create_semantic_scholar_client,
        )
        
        assert SemanticScholarClient is not None
        assert create_semantic_scholar_client is not None
    
    def test_crossref_exports(self):
        """crossref.pyエクスポートテスト"""
        from monjyu.external.crossref import (
            CrossRefClient,
            CrossRefConfig,
            create_crossref_client,
        )
        
        assert CrossRefClient is not None
        assert create_crossref_client is not None
    
    def test_unified_exports(self):
        """unified.pyエクスポートテスト"""
        from monjyu.external.unified import (
            UnifiedMetadataClient,
            UnifiedMetadataConfig,
            create_unified_client,
        )
        
        assert UnifiedMetadataClient is not None
        assert create_unified_client is not None
    
    def test_package_exports(self):
        """__init__.pyエクスポートテスト"""
        from monjyu.external import (
            # Base
            ExternalAPIClient,
            ExternalAPIConfig,
            ExternalAPIError,
            RateLimitError,
            APIResponseError,
            PaperMetadata,
            Author,
            Citation,
            # Semantic Scholar
            SemanticScholarClient,
            SemanticScholarConfig,
            create_semantic_scholar_client,
            # CrossRef
            CrossRefClient,
            CrossRefConfig,
            create_crossref_client,
            # Unified
            UnifiedMetadataClient,
            UnifiedMetadataConfig,
            create_unified_client,
        )
        
        assert ExternalAPIClient is not None
        assert SemanticScholarClient is not None
        assert CrossRefClient is not None
        assert UnifiedMetadataClient is not None
