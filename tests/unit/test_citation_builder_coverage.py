# Citation Builder Coverage Tests
"""
Tests for monjyu.citation.builder to improve coverage from 73% to 85%+
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from monjyu.citation.base import (
    CitationEdge,
    CitationGraph,
    CitationNetworkConfig,
    ReferenceMatchStatus,
)
from monjyu.citation.builder import (
    CitationGraphBuilder,
    DefaultCitationGraphBuilder,
    MockCitationGraphBuilder,
)
from monjyu.citation.resolver import ResolvedReference


# --------------------------------------------------------------------------- #
# Mock Document for Testing
# --------------------------------------------------------------------------- #
@dataclass
class MockDocument:
    """Test document mock."""

    doc_id: str
    content: str = ""
    metadata: dict[str, Any] | None = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config() -> CitationNetworkConfig:
    """Default config fixture."""
    return CitationNetworkConfig()


@pytest.fixture
def builder(config: CitationNetworkConfig) -> DefaultCitationGraphBuilder:
    """Default builder fixture."""
    return DefaultCitationGraphBuilder(config=config)


@pytest.fixture
def sample_docs() -> list[MockDocument]:
    """Sample documents for testing."""
    return [
        MockDocument(
            doc_id="doc1",
            content="Document 1 content",
            metadata={
                "title": "Document One",
                "doi": "10.1234/doc1",
                "references": ["Reference to doc2", "Reference to external"],
            },
        ),
        MockDocument(
            doc_id="doc2",
            content="Document 2 content",
            metadata={
                "title": "Document Two",
                "doi": "10.1234/doc2",
                "references": [],
            },
        ),
        MockDocument(
            doc_id="doc3",
            content="Document 3 content",
            metadata={
                "title": "Document Three",
                "doi": None,
                "references": ["Reference to doc1"],
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# DefaultCitationGraphBuilder Tests
# --------------------------------------------------------------------------- #
class TestDefaultCitationGraphBuilder:
    """Tests for DefaultCitationGraphBuilder."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        builder = DefaultCitationGraphBuilder()
        assert builder.config is not None
        assert builder.resolver is not None

    def test_init_with_custom_config(self, config: CitationNetworkConfig) -> None:
        """Test initialization with custom config."""
        builder = DefaultCitationGraphBuilder(config=config)
        assert builder.config is config

    def test_init_with_custom_resolver(self) -> None:
        """Test initialization with custom resolver."""
        mock_resolver = MagicMock()
        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)
        assert builder.resolver is mock_resolver

    def test_build_empty_documents(self, builder: DefaultCitationGraphBuilder) -> None:
        """Test building graph with empty document list."""
        graph = builder.build([])
        assert len(graph.internal_doc_ids) == 0

    def test_build_with_documents(
        self, builder: DefaultCitationGraphBuilder, sample_docs: list[MockDocument]
    ) -> None:
        """Test building graph with documents."""
        graph = builder.build(sample_docs)
        # All documents should be added
        assert "doc1" in graph.internal_doc_ids
        assert "doc2" in graph.internal_doc_ids
        assert "doc3" in graph.internal_doc_ids

    def test_build_document_metadata(
        self, builder: DefaultCitationGraphBuilder, sample_docs: list[MockDocument]
    ) -> None:
        """Test that document metadata is preserved."""
        graph = builder.build(sample_docs)
        # Check node attributes
        node_data = graph.graph.nodes.get("doc1", {})
        assert "title" in node_data
        assert node_data["title"] == "Document One"

    def test_build_with_no_metadata(self, builder: DefaultCitationGraphBuilder) -> None:
        """Test building with document having no metadata."""
        docs = [MockDocument(doc_id="doc1", metadata=None)]
        graph = builder.build(docs)
        assert "doc1" in graph.internal_doc_ids

    def test_build_with_empty_references(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test building with document having empty references."""
        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"title": "Test", "references": []},
            )
        ]
        graph = builder.build(docs)
        assert "doc1" in graph.internal_doc_ids
        assert graph.graph.number_of_edges() == 0

    def test_add_document_to_existing_graph(
        self, builder: DefaultCitationGraphBuilder, sample_docs: list[MockDocument]
    ) -> None:
        """Test adding document to existing graph."""
        graph = CitationGraph()
        graph.add_document("existing_doc", {"title": "Existing"})

        new_doc = sample_docs[0]
        result = builder.add_document(graph, new_doc)

        assert result is graph
        assert "existing_doc" in graph.internal_doc_ids
        assert "doc1" in graph.internal_doc_ids

    def test_add_document_with_no_metadata(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test adding document with no metadata."""
        graph = CitationGraph()
        doc = MockDocument(doc_id="doc1", metadata=None)
        builder.add_document(graph, doc)
        assert "doc1" in graph.internal_doc_ids

    def test_process_document_references_unresolved(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test processing unresolved references (external)."""
        # Mock resolver to return unresolved references
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="External Paper",
                status=ReferenceMatchStatus.UNRESOLVED,
                target_doc_id=None,
                confidence=0.0,
            )
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"references": ["External Paper"]},
            )
        ]

        graph = builder.build(docs)
        # Should have external reference
        assert len(graph.external_refs) > 0

    def test_process_document_references_resolved_internal(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test processing resolved internal references."""
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="Internal Paper",
                status=ReferenceMatchStatus.MATCHED_DOI,
                target_doc_id="doc2",
                confidence=0.95,
            )
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"references": ["Internal Paper"]},
            ),
            MockDocument(doc_id="doc2", metadata={}),
        ]

        graph = builder.build(docs)
        # Should have edge from doc1 to doc2
        assert graph.graph.has_edge("doc1", "doc2")

    def test_process_document_references_resolved_no_target(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test processing resolved reference with None target_doc_id."""
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="Paper",
                status=ReferenceMatchStatus.MATCHED_TITLE_FUZZY,
                target_doc_id=None,  # Resolved but no target
                confidence=0.5,
            )
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"references": ["Paper"]},
            )
        ]

        graph = builder.build(docs)
        # Should not add edge since target is None
        assert graph.graph.number_of_edges() == 0

    def test_process_document_references_non_list(
        self, builder: DefaultCitationGraphBuilder
    ) -> None:
        """Test handling non-list references in metadata."""
        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"references": "not a list"},  # Invalid type
            )
        ]
        graph = builder.build(docs)
        assert "doc1" in graph.internal_doc_ids
        # Should not crash, just skip invalid references


# --------------------------------------------------------------------------- #
# MockCitationGraphBuilder Tests
# --------------------------------------------------------------------------- #
class TestMockCitationGraphBuilder:
    """Tests for MockCitationGraphBuilder."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        builder = MockCitationGraphBuilder()
        assert builder._mock_graph is None

    def test_init_with_mock_graph(self) -> None:
        """Test initialization with mock graph."""
        mock_graph = CitationGraph()
        builder = MockCitationGraphBuilder(mock_graph=mock_graph)
        assert builder._mock_graph is mock_graph

    def test_build_returns_mock_graph(self) -> None:
        """Test build returns provided mock graph."""
        mock_graph = CitationGraph()
        mock_graph.add_document("preset_doc")
        builder = MockCitationGraphBuilder(mock_graph=mock_graph)

        docs = [MockDocument(doc_id="doc1")]
        result = builder.build(docs)

        assert result is mock_graph
        assert "preset_doc" in result.internal_doc_ids

    def test_build_creates_simple_graph(self) -> None:
        """Test build creates simple graph when no mock provided."""
        builder = MockCitationGraphBuilder()

        docs = [
            MockDocument(doc_id="doc1"),
            MockDocument(doc_id="doc2"),
        ]
        result = builder.build(docs)

        assert "doc1" in result.internal_doc_ids
        assert "doc2" in result.internal_doc_ids
        # No edges in simple graph
        assert result.graph.number_of_edges() == 0

    def test_add_document(self) -> None:
        """Test add_document adds node."""
        builder = MockCitationGraphBuilder()
        graph = CitationGraph()

        doc = MockDocument(doc_id="new_doc")
        result = builder.add_document(graph, doc)

        assert result is graph
        assert "new_doc" in graph.internal_doc_ids


# --------------------------------------------------------------------------- #
# Abstract Base Class Tests
# --------------------------------------------------------------------------- #
class TestCitationGraphBuilderABC:
    """Tests for abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CitationGraphBuilder()  # type: ignore

    def test_subclass_must_implement_build(self) -> None:
        """Test subclass must implement build method."""

        class IncompletBuilder(CitationGraphBuilder):
            def add_document(
                self, graph: CitationGraph, document: Any
            ) -> CitationGraph:
                return graph

        with pytest.raises(TypeError):
            IncompletBuilder()  # type: ignore

    def test_subclass_must_implement_add_document(self) -> None:
        """Test subclass must implement add_document method."""

        class IncompleteBuilder(CitationGraphBuilder):
            def build(self, documents: list) -> CitationGraph:
                return CitationGraph()

        with pytest.raises(TypeError):
            IncompleteBuilder()  # type: ignore


# --------------------------------------------------------------------------- #
# Edge Case Tests
# --------------------------------------------------------------------------- #
class TestBuilderEdgeCases:
    """Edge case tests for citation builder."""

    def test_document_with_special_characters_in_id(self) -> None:
        """Test document with special characters in ID."""
        builder = DefaultCitationGraphBuilder()
        docs = [MockDocument(doc_id="doc/with:special@chars")]
        graph = builder.build(docs)
        assert "doc/with:special@chars" in graph.internal_doc_ids

    def test_document_with_empty_title(self) -> None:
        """Test document with empty title."""
        builder = DefaultCitationGraphBuilder()
        docs = [MockDocument(doc_id="doc1", metadata={"title": "", "doi": None})]
        graph = builder.build(docs)
        node_data = graph.graph.nodes.get("doc1", {})
        assert node_data.get("title") == ""

    def test_large_reference_list(self) -> None:
        """Test document with many references."""
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference=f"Ref {i}",
                status=ReferenceMatchStatus.UNRESOLVED,
                target_doc_id=None,
                confidence=0.0,
            )
            for i in range(100)
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        refs = [f"Reference {i}" for i in range(100)]
        docs = [MockDocument(doc_id="doc1", metadata={"references": refs})]

        graph = builder.build(docs)
        assert len(graph.external_refs) == 100

    def test_duplicate_document_ids(self) -> None:
        """Test handling of duplicate document IDs."""
        builder = DefaultCitationGraphBuilder()
        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "First"}),
            MockDocument(doc_id="doc1", metadata={"title": "Second"}),
        ]
        # Should handle gracefully (later one may overwrite)
        graph = builder.build(docs)
        assert "doc1" in graph.internal_doc_ids

    def test_self_reference(self) -> None:
        """Test document referencing itself."""
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="Self ref",
                status=ReferenceMatchStatus.MATCHED_DOI,
                target_doc_id="doc1",  # Same as source
                confidence=1.0,
            )
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        docs = [MockDocument(doc_id="doc1", metadata={"references": ["Self ref"]})]

        graph = builder.build(docs)
        # Self-edge should be added (depending on implementation)
        assert graph.graph.has_edge("doc1", "doc1")


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #
class TestBuilderIntegration:
    """Integration tests for citation builder."""

    def test_build_then_add_document(self) -> None:
        """Test building graph then adding more documents."""
        builder = DefaultCitationGraphBuilder()

        initial_docs = [MockDocument(doc_id="doc1", metadata={"title": "First"})]
        graph = builder.build(initial_docs)

        new_doc = MockDocument(doc_id="doc2", metadata={"title": "Second"})
        builder.add_document(graph, new_doc)

        assert "doc1" in graph.internal_doc_ids
        assert "doc2" in graph.internal_doc_ids

    def test_multiple_reference_types(self) -> None:
        """Test documents with various reference resolution statuses."""
        mock_resolver = MagicMock()
        mock_resolver.resolve_batch.return_value = [
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="DOI match",
                status=ReferenceMatchStatus.MATCHED_DOI,
                target_doc_id="doc2",
                confidence=1.0,
            ),
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="Title match",
                status=ReferenceMatchStatus.MATCHED_TITLE_EXACT,
                target_doc_id="doc3",
                confidence=0.9,
            ),
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="Fuzzy match",
                status=ReferenceMatchStatus.MATCHED_TITLE_FUZZY,
                target_doc_id="doc4",
                confidence=0.7,
            ),
            ResolvedReference(
                source_doc_id="doc1",
                raw_reference="External",
                status=ReferenceMatchStatus.UNRESOLVED,
                target_doc_id=None,
                confidence=0.0,
            ),
        ]
        mock_resolver.build_index = MagicMock()

        builder = DefaultCitationGraphBuilder(resolver=mock_resolver)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"references": ["DOI", "Title", "Fuzzy", "Ext"]},
            ),
            MockDocument(doc_id="doc2"),
            MockDocument(doc_id="doc3"),
            MockDocument(doc_id="doc4"),
        ]

        graph = builder.build(docs)

        # Check internal edges
        assert graph.graph.has_edge("doc1", "doc2")
        assert graph.graph.has_edge("doc1", "doc3")
        assert graph.graph.has_edge("doc1", "doc4")
        # Check external refs
        assert len(graph.external_refs) == 1
