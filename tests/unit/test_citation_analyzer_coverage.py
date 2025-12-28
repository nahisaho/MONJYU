# Citation Analyzer Coverage Tests
"""
Tests for monjyu.citation.analyzer to improve coverage from 78% to 85%+
"""

from __future__ import annotations

import pytest
import networkx as nx

from monjyu.citation.base import (
    CitationGraph,
    CitationNetworkConfig,
    CitationPath,
    RelatedPaper,
    CitationEdge,
    ReferenceMatchStatus,
)
from monjyu.citation.analyzer import (
    CitationAnalyzer,
    DefaultCitationAnalyzer,
    MockCitationAnalyzer,
)


# --------------------------------------------------------------------------- #
# Helper Functions
# --------------------------------------------------------------------------- #
def make_edge(
    source: str,
    target: str,
    is_internal: bool = True,
    confidence: float = 1.0,
) -> CitationEdge:
    """Create a CitationEdge with required parameters."""
    return CitationEdge(
        source_id=source,
        target_id=target,
        is_internal=is_internal,
        confidence=confidence,
        reference_text="",
        match_status=ReferenceMatchStatus.MATCHED_DOI,
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config() -> CitationNetworkConfig:
    """Default config fixture."""
    return CitationNetworkConfig()


@pytest.fixture
def analyzer(config: CitationNetworkConfig) -> DefaultCitationAnalyzer:
    """Default analyzer fixture."""
    return DefaultCitationAnalyzer(config=config)


@pytest.fixture
def simple_graph() -> CitationGraph:
    """Simple graph: doc1 -> doc2 -> doc3."""
    graph = CitationGraph()
    graph.add_document("doc1", {"title": "Doc 1"})
    graph.add_document("doc2", {"title": "Doc 2"})
    graph.add_document("doc3", {"title": "Doc 3"})

    graph.add_citation_edge(make_edge("doc1", "doc2"))
    graph.add_citation_edge(make_edge("doc2", "doc3"))

    return graph


@pytest.fixture
def complex_graph() -> CitationGraph:
    """
    Complex graph for co-citation and bibliographic coupling tests.

    Structure:
        A -> C
        A -> D
        B -> C
        B -> D
        C -> E
        D -> E

    Co-citation: C and D are both cited by A and B
    Bibliographic coupling: A and B both cite C and D
    """
    graph = CitationGraph()

    for doc_id in ["A", "B", "C", "D", "E"]:
        graph.add_document(doc_id, {"title": f"Doc {doc_id}"})

    graph.add_citation_edge(make_edge("A", "C"))
    graph.add_citation_edge(make_edge("A", "D"))
    graph.add_citation_edge(make_edge("B", "C"))
    graph.add_citation_edge(make_edge("B", "D"))
    graph.add_citation_edge(make_edge("C", "E"))
    graph.add_citation_edge(make_edge("D", "E"))

    return graph


@pytest.fixture
def disconnected_graph() -> CitationGraph:
    """Graph with disconnected components."""
    graph = CitationGraph()

    # Component 1
    graph.add_document("c1_a", {})
    graph.add_document("c1_b", {})
    graph.add_citation_edge(make_edge("c1_a", "c1_b"))

    # Component 2
    graph.add_document("c2_a", {})
    graph.add_document("c2_b", {})
    graph.add_citation_edge(make_edge("c2_a", "c2_b"))

    return graph


# --------------------------------------------------------------------------- #
# DefaultCitationAnalyzer Initialization Tests
# --------------------------------------------------------------------------- #
class TestDefaultCitationAnalyzerInit:
    """Tests for DefaultCitationAnalyzer initialization."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        analyzer = DefaultCitationAnalyzer()
        assert analyzer.config is not None

    def test_init_with_custom_config(self, config: CitationNetworkConfig) -> None:
        """Test initialization with custom config."""
        analyzer = DefaultCitationAnalyzer(config=config)
        assert analyzer.config is config


# --------------------------------------------------------------------------- #
# Citation Path Tests
# --------------------------------------------------------------------------- #
class TestFindCitationPaths:
    """Tests for find_citation_paths method."""

    def test_find_direct_path(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test finding direct path between adjacent nodes."""
        paths = analyzer.find_citation_paths(simple_graph, "doc1", "doc2")
        assert len(paths) == 1
        assert paths[0].path == ("doc1", "doc2")
        assert paths[0].length == 1

    def test_find_indirect_path(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test finding path through intermediate node."""
        paths = analyzer.find_citation_paths(simple_graph, "doc1", "doc3")
        assert len(paths) == 1
        assert paths[0].path == ("doc1", "doc2", "doc3")
        assert paths[0].length == 2

    def test_no_path_exists(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test when no path exists."""
        # Reverse direction has no path
        paths = analyzer.find_citation_paths(simple_graph, "doc3", "doc1")
        assert paths == []

    def test_nonexistent_source(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent source node."""
        paths = analyzer.find_citation_paths(simple_graph, "nonexistent", "doc1")
        assert paths == []

    def test_nonexistent_target(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent target node."""
        paths = analyzer.find_citation_paths(simple_graph, "doc1", "nonexistent")
        assert paths == []

    def test_same_source_and_target(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test finding path from node to itself."""
        paths = analyzer.find_citation_paths(simple_graph, "doc1", "doc1")
        # networkx all_simple_paths includes trivial path from node to itself
        # (a path of length 0, containing just the source node)
        # So we verify the behavior: either empty or single trivial path
        if paths:
            assert paths[0].path == ("doc1",)
            assert paths[0].length == 0

    def test_max_length_limit(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test max_length parameter limits path search."""
        # Path from doc1 to doc3 has length 2
        paths = analyzer.find_citation_paths(simple_graph, "doc1", "doc3", max_length=1)
        assert paths == []

        paths = analyzer.find_citation_paths(simple_graph, "doc1", "doc3", max_length=2)
        assert len(paths) == 1

    def test_multiple_paths(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test finding multiple paths."""
        graph = CitationGraph()
        for doc_id in ["A", "B", "C", "D"]:
            graph.add_document(doc_id, {})

        # A -> B -> D
        # A -> C -> D
        graph.add_citation_edge(make_edge("A", "B"))
        graph.add_citation_edge(make_edge("B", "D"))
        graph.add_citation_edge(make_edge("A", "C"))
        graph.add_citation_edge(make_edge("C", "D"))

        paths = analyzer.find_citation_paths(graph, "A", "D")
        assert len(paths) == 2
        # Should be sorted by length
        assert all(p.length == 2 for p in paths)


# --------------------------------------------------------------------------- #
# Shortest Path Tests
# --------------------------------------------------------------------------- #
class TestFindShortestPath:
    """Tests for find_shortest_path method."""

    def test_shortest_path_direct(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test finding shortest direct path."""
        path = analyzer.find_shortest_path(simple_graph, "doc1", "doc2")
        assert path is not None
        assert path.path == ("doc1", "doc2")
        assert path.length == 1

    def test_shortest_path_indirect(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test finding shortest indirect path."""
        path = analyzer.find_shortest_path(simple_graph, "doc1", "doc3")
        assert path is not None
        assert path.path == ("doc1", "doc2", "doc3")
        assert path.length == 2

    def test_no_shortest_path(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test when no path exists."""
        path = analyzer.find_shortest_path(simple_graph, "doc3", "doc1")
        assert path is None

    def test_nonexistent_source(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent source."""
        path = analyzer.find_shortest_path(simple_graph, "nonexistent", "doc1")
        assert path is None

    def test_nonexistent_target(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent target."""
        path = analyzer.find_shortest_path(simple_graph, "doc1", "nonexistent")
        assert path is None


# --------------------------------------------------------------------------- #
# Co-citation Tests
# --------------------------------------------------------------------------- #
class TestFindCoCitations:
    """Tests for find_co_citations method."""

    def test_co_citations_found(
        self, analyzer: DefaultCitationAnalyzer, complex_graph: CitationGraph
    ) -> None:
        """Test finding co-citations."""
        # C is cited by A and B
        # D is also cited by A and B
        # So C and D are co-cited
        related = analyzer.find_co_citations(complex_graph, "C")

        # D should be in co-citations
        doc_ids = [r.doc_id for r in related]
        assert "D" in doc_ids

        # Check relation type
        for r in related:
            assert r.relation_type == "co_citation"

    def test_co_citations_score(
        self, analyzer: DefaultCitationAnalyzer, complex_graph: CitationGraph
    ) -> None:
        """Test co-citation scores."""
        related = analyzer.find_co_citations(complex_graph, "C")

        # D is co-cited with C by both A and B
        d_result = next((r for r in related if r.doc_id == "D"), None)
        assert d_result is not None
        # Score = 2 (shared) / 2 (citing papers) = 1.0
        assert d_result.similarity_score == 1.0

    def test_co_citations_min_count(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test min_count filtering."""
        graph = CitationGraph()
        for doc_id in ["A", "B", "C", "D"]:
            graph.add_document(doc_id, {})

        # Only A cites both C and D (not enough for min_count=2)
        graph.add_citation_edge(make_edge("A", "C"))
        graph.add_citation_edge(make_edge("A", "D"))

        # min_count=2 should return empty
        related = analyzer.find_co_citations(graph, "C", min_count=2)
        assert related == []

        # min_count=1 should find D
        related = analyzer.find_co_citations(graph, "C", min_count=1)
        assert len(related) > 0

    def test_co_citations_nonexistent_doc(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent document."""
        related = analyzer.find_co_citations(simple_graph, "nonexistent")
        assert related == []

    def test_co_citations_no_citing_papers(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test document with no citing papers."""
        graph = CitationGraph()
        graph.add_document("alone", {})

        related = analyzer.find_co_citations(graph, "alone")
        assert related == []


# --------------------------------------------------------------------------- #
# Bibliographic Coupling Tests
# --------------------------------------------------------------------------- #
class TestFindBibliographicCoupling:
    """Tests for find_bibliographic_coupling method."""

    def test_bibliographic_coupling_found(
        self, analyzer: DefaultCitationAnalyzer, complex_graph: CitationGraph
    ) -> None:
        """Test finding bibliographic coupling."""
        # A cites C and D
        # B also cites C and D
        # So A and B have bibliographic coupling
        related = analyzer.find_bibliographic_coupling(complex_graph, "A")

        doc_ids = [r.doc_id for r in related]
        assert "B" in doc_ids

        for r in related:
            assert r.relation_type == "bibliographic_coupling"

    def test_bibliographic_coupling_score(
        self, analyzer: DefaultCitationAnalyzer, complex_graph: CitationGraph
    ) -> None:
        """Test bibliographic coupling scores."""
        related = analyzer.find_bibliographic_coupling(complex_graph, "A")

        b_result = next((r for r in related if r.doc_id == "B"), None)
        assert b_result is not None
        # Score = 2 (shared refs) / 2 (A's refs) = 1.0
        assert b_result.similarity_score == 1.0

    def test_bibliographic_coupling_min_count(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test min_count filtering."""
        graph = CitationGraph()
        for doc_id in ["A", "B", "C"]:
            graph.add_document(doc_id, {})

        # A and B both cite C, but only share 1 reference
        graph.add_citation_edge(make_edge("A", "C"))
        graph.add_citation_edge(make_edge("B", "C"))

        # min_count=2 should return empty
        related = analyzer.find_bibliographic_coupling(graph, "A", min_count=2)
        assert related == []

    def test_bibliographic_coupling_nonexistent_doc(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent document."""
        related = analyzer.find_bibliographic_coupling(simple_graph, "nonexistent")
        assert related == []

    def test_bibliographic_coupling_no_references(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test document with no references."""
        graph = CitationGraph()
        graph.add_document("alone", {})

        related = analyzer.find_bibliographic_coupling(graph, "alone")
        assert related == []


# --------------------------------------------------------------------------- #
# Citation Context Tests
# --------------------------------------------------------------------------- #
class TestGetCitationContext:
    """Tests for get_citation_context method."""

    def test_citation_context_depth_1(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test getting citation context with depth 1."""
        context = analyzer.get_citation_context(simple_graph, "doc2", depth=1)

        # Should include doc2, doc1 (predecessor), doc3 (successor)
        assert "doc2" in context.internal_doc_ids
        assert "doc1" in context.internal_doc_ids or context.graph.has_node("doc1")
        assert "doc3" in context.internal_doc_ids or context.graph.has_node("doc3")

    def test_citation_context_depth_2(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test getting citation context with depth 2."""
        context = analyzer.get_citation_context(simple_graph, "doc1", depth=2)

        # Should include all documents
        assert context.graph.has_node("doc1")
        assert context.graph.has_node("doc2")
        assert context.graph.has_node("doc3")

    def test_citation_context_nonexistent_doc(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test with nonexistent document."""
        context = analyzer.get_citation_context(simple_graph, "nonexistent")
        # Should return empty graph
        assert context.graph.number_of_nodes() == 0

    def test_citation_context_preserves_edges(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test that edges are preserved in context."""
        context = analyzer.get_citation_context(simple_graph, "doc2", depth=1)

        # Should have edges between included nodes
        assert context.graph.has_edge("doc1", "doc2")
        assert context.graph.has_edge("doc2", "doc3")

    def test_citation_context_isolated_node(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test context for isolated node."""
        graph = CitationGraph()
        graph.add_document("isolated", {})

        context = analyzer.get_citation_context(graph, "isolated", depth=1)
        assert "isolated" in context.internal_doc_ids


# --------------------------------------------------------------------------- #
# Connected Components Tests
# --------------------------------------------------------------------------- #
class TestGetConnectedComponents:
    """Tests for get_connected_components method."""

    def test_single_component(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test graph with single connected component."""
        components = analyzer.get_connected_components(simple_graph)
        assert len(components) == 1
        assert len(components[0]) == 3

    def test_multiple_components(
        self, analyzer: DefaultCitationAnalyzer, disconnected_graph: CitationGraph
    ) -> None:
        """Test graph with multiple components."""
        components = analyzer.get_connected_components(disconnected_graph)
        assert len(components) == 2
        # Each component has 2 nodes
        assert all(len(c) == 2 for c in components)


# --------------------------------------------------------------------------- #
# Strongly Connected Components Tests
# --------------------------------------------------------------------------- #
class TestGetStronglyConnectedComponents:
    """Tests for get_strongly_connected_components method."""

    def test_no_strongly_connected(
        self, analyzer: DefaultCitationAnalyzer, simple_graph: CitationGraph
    ) -> None:
        """Test graph with no strongly connected components."""
        components = analyzer.get_strongly_connected_components(simple_graph)
        # In a DAG, each node is its own SCC
        assert len(components) == 3
        assert all(len(c) == 1 for c in components)

    def test_with_cycle(
        self, analyzer: DefaultCitationAnalyzer
    ) -> None:
        """Test graph with cycle (strongly connected)."""
        graph = CitationGraph()
        for doc_id in ["A", "B", "C"]:
            graph.add_document(doc_id, {})

        # Create cycle: A -> B -> C -> A
        graph.add_citation_edge(make_edge("A", "B"))
        graph.add_citation_edge(make_edge("B", "C"))
        graph.add_citation_edge(make_edge("C", "A"))

        components = analyzer.get_strongly_connected_components(graph)
        # Should have 1 SCC with 3 nodes
        assert len(components) == 1
        assert len(components[0]) == 3


# --------------------------------------------------------------------------- #
# MockCitationAnalyzer Tests
# --------------------------------------------------------------------------- #
class TestMockCitationAnalyzer:
    """Tests for MockCitationAnalyzer."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        analyzer = MockCitationAnalyzer()
        assert analyzer._mock_paths == {}
        assert analyzer._mock_co_citations == {}
        assert analyzer._mock_couplings == {}

    def test_init_with_mock_paths(self) -> None:
        """Test initialization with mock paths."""
        mock_path = CitationPath(path=("A", "B"), length=1)
        analyzer = MockCitationAnalyzer(
            mock_paths={("A", "B"): [mock_path]}
        )

        graph = CitationGraph()
        result = analyzer.find_citation_paths(graph, "A", "B")
        assert result == [mock_path]

    def test_init_with_mock_co_citations(self) -> None:
        """Test initialization with mock co-citations."""
        mock_related = RelatedPaper(
            doc_id="B",
            relation_type="co_citation",
            similarity_score=0.8,
            shared_papers=frozenset(["X"]),
        )
        analyzer = MockCitationAnalyzer(
            mock_co_citations={"A": [mock_related]}
        )

        graph = CitationGraph()
        result = analyzer.find_co_citations(graph, "A")
        assert result == [mock_related]

    def test_init_with_mock_couplings(self) -> None:
        """Test initialization with mock bibliographic couplings."""
        mock_related = RelatedPaper(
            doc_id="B",
            relation_type="bibliographic_coupling",
            similarity_score=0.7,
            shared_papers=frozenset(["Y"]),
        )
        analyzer = MockCitationAnalyzer(
            mock_couplings={"A": [mock_related]}
        )

        graph = CitationGraph()
        result = analyzer.find_bibliographic_coupling(graph, "A")
        assert result == [mock_related]

    def test_find_citation_paths_no_mock(self) -> None:
        """Test find_citation_paths returns empty when no mock."""
        analyzer = MockCitationAnalyzer()
        graph = CitationGraph()

        result = analyzer.find_citation_paths(graph, "A", "B")
        assert result == []

    def test_find_co_citations_no_mock(self) -> None:
        """Test find_co_citations returns empty when no mock."""
        analyzer = MockCitationAnalyzer()
        graph = CitationGraph()

        result = analyzer.find_co_citations(graph, "A")
        assert result == []

    def test_find_bibliographic_coupling_no_mock(self) -> None:
        """Test find_bibliographic_coupling returns empty when no mock."""
        analyzer = MockCitationAnalyzer()
        graph = CitationGraph()

        result = analyzer.find_bibliographic_coupling(graph, "A")
        assert result == []

    def test_get_citation_context(self) -> None:
        """Test get_citation_context returns empty graph."""
        analyzer = MockCitationAnalyzer()
        graph = CitationGraph()
        graph.add_document("A", {})

        result = analyzer.get_citation_context(graph, "A")
        assert isinstance(result, CitationGraph)
        assert result.graph.number_of_nodes() == 0


# --------------------------------------------------------------------------- #
# Abstract Base Class Tests
# --------------------------------------------------------------------------- #
class TestCitationAnalyzerABC:
    """Tests for abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        """Test that ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CitationAnalyzer()  # type: ignore


# --------------------------------------------------------------------------- #
# Edge Case Tests
# --------------------------------------------------------------------------- #
class TestAnalyzerEdgeCases:
    """Edge case tests for citation analyzer."""

    def test_empty_graph(self, analyzer: DefaultCitationAnalyzer) -> None:
        """Test operations on empty graph."""
        graph = CitationGraph()

        assert analyzer.find_citation_paths(graph, "A", "B") == []
        assert analyzer.find_co_citations(graph, "A") == []
        assert analyzer.find_bibliographic_coupling(graph, "A") == []
        assert analyzer.get_connected_components(graph) == []

    def test_single_node_graph(self, analyzer: DefaultCitationAnalyzer) -> None:
        """Test operations on single node graph."""
        graph = CitationGraph()
        graph.add_document("only", {})

        # networkx all_simple_paths may return trivial path from node to itself
        paths = analyzer.find_citation_paths(graph, "only", "only")
        # Check behavior: may be empty or trivial path
        assert all(p.path == ("only",) for p in paths)

        assert analyzer.find_co_citations(graph, "only") == []
        assert analyzer.find_bibliographic_coupling(graph, "only") == []

        components = analyzer.get_connected_components(graph)
        assert len(components) == 1
        assert "only" in components[0]

    def test_self_loop(self, analyzer: DefaultCitationAnalyzer) -> None:
        """Test graph with self-loop."""
        graph = CitationGraph()
        graph.add_document("self", {})
        graph.add_citation_edge(make_edge("self", "self"))

        # Self-loop creates a path
        paths = analyzer.find_citation_paths(graph, "self", "self")
        assert len(paths) > 0

        scc = analyzer.get_strongly_connected_components(graph)
        assert len(scc) == 1

    def test_long_chain(self, analyzer: DefaultCitationAnalyzer) -> None:
        """Test long chain of citations."""
        graph = CitationGraph()

        # Create chain: A -> B -> C -> D -> E
        nodes = ["A", "B", "C", "D", "E"]
        for node in nodes:
            graph.add_document(node, {})

        for i in range(len(nodes) - 1):
            graph.add_citation_edge(make_edge(nodes[i], nodes[i + 1]))

        # Find path from A to E
        paths = analyzer.find_citation_paths(graph, "A", "E")
        assert len(paths) == 1
        assert paths[0].length == 4

    def test_dense_graph(self, analyzer: DefaultCitationAnalyzer) -> None:
        """Test dense graph with many edges."""
        graph = CitationGraph()

        # Create complete graph (every node cites every other)
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            graph.add_document(node, {})

        for source in nodes:
            for target in nodes:
                if source != target:
                    graph.add_citation_edge(make_edge(source, target))

        # Many paths should exist
        paths = analyzer.find_citation_paths(graph, "A", "D", max_length=3)
        assert len(paths) > 1

        # Strong coupling
        related = analyzer.find_bibliographic_coupling(graph, "A", min_count=1)
        assert len(related) == 3  # B, C, D all share references with A
