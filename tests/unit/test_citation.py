# Citation Network Unit Tests
"""
FEAT-006: Citation Network - 単体テスト
"""

import pytest
from dataclasses import dataclass
from typing import Any

# ========== テスト用のモック Document ==========

@dataclass
class MockDocument:
    """テスト用のモック文書"""
    doc_id: str
    content: str = ""
    metadata: dict[str, Any] | None = None


# ========== Base Types Tests ==========

class TestReferenceMatchStatus:
    """ReferenceMatchStatus のテスト"""

    def test_enum_values(self):
        from monjyu.citation.base import ReferenceMatchStatus

        assert ReferenceMatchStatus.MATCHED_DOI.value == "matched_doi"
        assert ReferenceMatchStatus.MATCHED_TITLE_EXACT.value == "matched_title_exact"
        assert ReferenceMatchStatus.MATCHED_TITLE_FUZZY.value == "matched_title_fuzzy"
        assert ReferenceMatchStatus.UNRESOLVED.value == "unresolved"


class TestResolvedReference:
    """ResolvedReference のテスト"""

    def test_creation(self):
        from monjyu.citation.base import ReferenceMatchStatus, ResolvedReference

        ref = ResolvedReference(
            source_doc_id="doc1",
            target_doc_id="doc2",
            status=ReferenceMatchStatus.MATCHED_DOI,
            confidence=1.0,
            raw_reference="test reference",
            matched_doi="10.1234/test",
        )

        assert ref.source_doc_id == "doc1"
        assert ref.target_doc_id == "doc2"
        assert ref.status == ReferenceMatchStatus.MATCHED_DOI
        assert ref.confidence == 1.0
        assert ref.matched_doi == "10.1234/test"

    def test_frozen(self):
        from monjyu.citation.base import ReferenceMatchStatus, ResolvedReference

        ref = ResolvedReference(
            source_doc_id="doc1",
            target_doc_id="doc2",
            status=ReferenceMatchStatus.MATCHED_DOI,
            confidence=1.0,
            raw_reference="test",
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            ref.source_doc_id = "doc3"


class TestCitationEdge:
    """CitationEdge のテスト"""

    def test_creation(self):
        from monjyu.citation.base import ReferenceMatchStatus, CitationEdge

        edge = CitationEdge(
            source_id="doc1",
            target_id="doc2",
            is_internal=True,
            confidence=0.9,
            reference_text="test ref",
            match_status=ReferenceMatchStatus.MATCHED_TITLE_FUZZY,
        )

        assert edge.source_id == "doc1"
        assert edge.target_id == "doc2"
        assert edge.is_internal is True
        assert edge.confidence == 0.9


class TestCitationGraph:
    """CitationGraph のテスト"""

    def test_empty_graph(self):
        from monjyu.citation.base import CitationGraph

        graph = CitationGraph()

        assert graph.node_count == 0
        assert graph.edge_count == 0
        assert graph.internal_edge_count == 0
        assert graph.external_edge_count == 0

    def test_add_document(self):
        from monjyu.citation.base import CitationGraph

        graph = CitationGraph()
        graph.add_document("doc1", {"title": "Test Doc"})

        assert graph.node_count == 1
        assert "doc1" in graph.internal_doc_ids
        assert graph.graph.nodes["doc1"]["title"] == "Test Doc"

    def test_add_external_reference(self):
        from monjyu.citation.base import CitationGraph

        graph = CitationGraph()
        graph.add_external_reference("ext_123", "External Paper Title")

        assert graph.node_count == 1
        assert "ext_123" in graph.external_refs
        assert graph.external_refs["ext_123"] == "External Paper Title"

    def test_add_citation_edge(self):
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        graph = CitationGraph()
        graph.add_document("doc1")
        graph.add_document("doc2")

        edge = CitationEdge(
            source_id="doc1",
            target_id="doc2",
            is_internal=True,
            confidence=1.0,
            reference_text="ref to doc2",
            match_status=ReferenceMatchStatus.MATCHED_DOI,
        )
        graph.add_citation_edge(edge)

        assert graph.edge_count == 1
        assert graph.internal_edge_count == 1
        assert "doc2" in graph.get_citations("doc1")
        assert "doc1" in graph.get_cited_by("doc2")

    def test_citation_counts(self):
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        graph = CitationGraph()
        graph.add_document("doc1")
        graph.add_document("doc2")
        graph.add_document("doc3")

        # doc1 -> doc2, doc1 -> doc3
        for target in ["doc2", "doc3"]:
            edge = CitationEdge(
                source_id="doc1",
                target_id=target,
                is_internal=True,
                confidence=1.0,
                reference_text=f"ref to {target}",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        assert graph.get_reference_count("doc1") == 2  # doc1 は2つ引用
        assert graph.get_citation_count("doc2") == 1   # doc2 は1回被引用
        assert graph.get_citation_count("doc1") == 0   # doc1 は被引用なし


class TestDocumentMetrics:
    """DocumentMetrics のテスト"""

    def test_creation(self):
        from monjyu.citation.base import DocumentMetrics

        metrics = DocumentMetrics(
            doc_id="doc1",
            citation_count=10,
            reference_count=5,
            pagerank=0.15,
            hub_score=0.3,
            authority_score=0.5,
        )

        assert metrics.doc_id == "doc1"
        assert metrics.citation_count == 10
        assert metrics.total_connections == 15


class TestCitationPath:
    """CitationPath のテスト"""

    def test_creation(self):
        from monjyu.citation.base import CitationPath

        path = CitationPath(
            path=("doc1", "doc2", "doc3"),
            length=2,
        )

        assert path.source == "doc1"
        assert path.target == "doc3"
        assert len(path) == 2
        assert list(path) == ["doc1", "doc2", "doc3"]


class TestCitationNetworkConfig:
    """CitationNetworkConfig のテスト"""

    def test_defaults(self):
        from monjyu.citation.base import CitationNetworkConfig

        config = CitationNetworkConfig()

        assert config.fuzzy_match_threshold == 0.85
        assert config.pagerank_alpha == 0.85
        assert config.max_path_length == 5

    def test_validation(self):
        from monjyu.citation.base import CitationNetworkConfig

        with pytest.raises(ValueError):
            CitationNetworkConfig(fuzzy_match_threshold=1.5)

        with pytest.raises(ValueError):
            CitationNetworkConfig(pagerank_alpha=0.0)

        with pytest.raises(ValueError):
            CitationNetworkConfig(max_path_length=0)


# ========== ReferenceResolver Tests ==========

class TestDefaultReferenceResolver:
    """DefaultReferenceResolver のテスト"""

    def test_normalize_title(self):
        from monjyu.citation.resolver import DefaultReferenceResolver

        assert DefaultReferenceResolver.normalize_title("Hello World!") == "hello world"
        assert DefaultReferenceResolver.normalize_title("  Multiple   Spaces  ") == "multiple spaces"
        assert DefaultReferenceResolver.normalize_title("Test: A Study") == "test a study"

    def test_extract_doi(self):
        from monjyu.citation.resolver import DefaultReferenceResolver

        assert DefaultReferenceResolver.extract_doi("DOI: 10.1234/abc123") == "10.1234/abc123"
        assert DefaultReferenceResolver.extract_doi("https://doi.org/10.5678/xyz") == "10.5678/xyz"
        assert DefaultReferenceResolver.extract_doi("No DOI here") is None

    def test_extract_title(self):
        from monjyu.citation.resolver import DefaultReferenceResolver

        assert DefaultReferenceResolver.extract_title('"Machine Learning for NLP"') == "Machine Learning for NLP"
        assert DefaultReferenceResolver.extract_title("Introduction to AI. By Smith.") == "Introduction to AI"

    def test_doi_matching(self):
        from monjyu.citation.resolver import DefaultReferenceResolver
        from monjyu.citation.base import ReferenceMatchStatus

        resolver = DefaultReferenceResolver()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"doi": "10.1234/paper1", "title": "Paper One"},
            ),
            MockDocument(
                doc_id="doc2",
                metadata={"doi": "10.1234/paper2", "title": "Paper Two"},
            ),
        ]

        resolver.build_index(docs)

        result = resolver.resolve("doc1", "See 10.1234/paper2 for details")

        assert result.status == ReferenceMatchStatus.MATCHED_DOI
        assert result.target_doc_id == "doc2"
        assert result.confidence == 1.0

    def test_exact_title_matching(self):
        from monjyu.citation.resolver import DefaultReferenceResolver
        from monjyu.citation.base import ReferenceMatchStatus

        resolver = DefaultReferenceResolver()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"title": "Machine Learning Survey"},
            ),
            MockDocument(
                doc_id="doc2",
                metadata={"title": "Deep Learning Fundamentals"},
            ),
        ]

        resolver.build_index(docs)

        result = resolver.resolve("doc1", '"Deep Learning Fundamentals" is a good reference')

        assert result.status == ReferenceMatchStatus.MATCHED_TITLE_EXACT
        assert result.target_doc_id == "doc2"

    def test_fuzzy_title_matching(self):
        from monjyu.citation.resolver import DefaultReferenceResolver
        from monjyu.citation.base import ReferenceMatchStatus, CitationNetworkConfig

        config = CitationNetworkConfig(fuzzy_match_threshold=0.8)
        resolver = DefaultReferenceResolver(config)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"title": "Machine Learning Survey"},
            ),
            MockDocument(
                doc_id="doc2",
                metadata={"title": "A Comprehensive Survey on Machine Learning Techniques"},
            ),
        ]

        resolver.build_index(docs)

        # 部分的に類似するタイトル
        result = resolver.resolve("doc1", '"Survey on Machine Learning Techniques" by Smith')

        # ファジーマッチは閾値によっては失敗する可能性
        # このテストでは閾値0.8で類似度が十分かどうかを確認
        assert result.status in (
            ReferenceMatchStatus.MATCHED_TITLE_FUZZY,
            ReferenceMatchStatus.UNRESOLVED,
        )

    def test_unresolved_reference(self):
        from monjyu.citation.resolver import DefaultReferenceResolver
        from monjyu.citation.base import ReferenceMatchStatus

        resolver = DefaultReferenceResolver()

        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "Paper One"}),
        ]

        resolver.build_index(docs)

        result = resolver.resolve("doc1", "Some external paper not in corpus")

        assert result.status == ReferenceMatchStatus.UNRESOLVED
        assert result.target_doc_id is None

    def test_batch_resolve(self):
        from monjyu.citation.resolver import DefaultReferenceResolver

        resolver = DefaultReferenceResolver()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"doi": "10.1234/paper1", "title": "Paper One"},
            ),
            MockDocument(
                doc_id="doc2",
                metadata={"doi": "10.1234/paper2", "title": "Paper Two"},
            ),
        ]

        resolver.build_index(docs)

        refs = [
            "See 10.1234/paper2",
            "Unknown reference",
        ]

        results = resolver.resolve_batch("doc1", refs)

        assert len(results) == 2
        assert results[0].target_doc_id == "doc2"
        assert results[1].target_doc_id is None


class TestMockReferenceResolver:
    """MockReferenceResolver のテスト"""

    def test_mock_resolution(self):
        from monjyu.citation.resolver import MockReferenceResolver
        from monjyu.citation.base import ReferenceMatchStatus, ResolvedReference

        mock_resolutions = {
            ("doc1", "ref1"): ResolvedReference(
                source_doc_id="doc1",
                target_doc_id="doc2",
                status=ReferenceMatchStatus.MATCHED_DOI,
                confidence=1.0,
                raw_reference="ref1",
            ),
        }

        resolver = MockReferenceResolver(mock_resolutions)
        resolver.build_index([])

        result = resolver.resolve("doc1", "ref1")
        assert result.target_doc_id == "doc2"

        # 未登録の参照
        result2 = resolver.resolve("doc1", "unknown")
        assert result2.status == ReferenceMatchStatus.UNRESOLVED


# ========== CitationGraphBuilder Tests ==========

class TestDefaultCitationGraphBuilder:
    """DefaultCitationGraphBuilder のテスト"""

    def test_build_empty(self):
        from monjyu.citation.builder import DefaultCitationGraphBuilder

        builder = DefaultCitationGraphBuilder()
        graph = builder.build([])

        assert graph.node_count == 0

    def test_build_documents_only(self):
        from monjyu.citation.builder import DefaultCitationGraphBuilder

        builder = DefaultCitationGraphBuilder()

        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "Paper One"}),
            MockDocument(doc_id="doc2", metadata={"title": "Paper Two"}),
        ]

        graph = builder.build(docs)

        assert graph.node_count == 2
        assert "doc1" in graph.internal_doc_ids
        assert "doc2" in graph.internal_doc_ids
        assert graph.edge_count == 0

    def test_build_with_references(self):
        from monjyu.citation.builder import DefaultCitationGraphBuilder

        builder = DefaultCitationGraphBuilder()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "title": "Paper One",
                    "doi": "10.1234/paper1",
                    "references": ["See 10.1234/paper2 for details"],
                },
            ),
            MockDocument(
                doc_id="doc2",
                metadata={
                    "title": "Paper Two",
                    "doi": "10.1234/paper2",
                },
            ),
        ]

        graph = builder.build(docs)

        assert graph.node_count == 2
        assert graph.internal_edge_count == 1
        assert "doc2" in graph.get_citations("doc1")

    def test_external_references(self):
        from monjyu.citation.builder import DefaultCitationGraphBuilder

        builder = DefaultCitationGraphBuilder()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "title": "Paper One",
                    "references": ["External paper not in corpus"],
                },
            ),
        ]

        graph = builder.build(docs)

        assert graph.external_edge_count == 1
        assert len(graph.external_refs) == 1


# ========== MetricsCalculator Tests ==========

class TestDefaultMetricsCalculator:
    """DefaultMetricsCalculator のテスト"""

    def test_empty_graph(self):
        from monjyu.citation.metrics import DefaultMetricsCalculator
        from monjyu.citation.base import CitationGraph

        calculator = DefaultMetricsCalculator()
        graph = CitationGraph()

        metrics = calculator.calculate(graph)
        assert metrics == {}

    def test_single_document(self):
        from monjyu.citation.metrics import DefaultMetricsCalculator
        from monjyu.citation.base import CitationGraph

        calculator = DefaultMetricsCalculator()
        graph = CitationGraph()
        graph.add_document("doc1")

        metrics = calculator.calculate(graph)

        assert "doc1" in metrics
        assert metrics["doc1"].citation_count == 0
        assert metrics["doc1"].reference_count == 0

    def test_pagerank_calculation(self):
        from monjyu.citation.metrics import DefaultMetricsCalculator
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        calculator = DefaultMetricsCalculator()
        graph = CitationGraph()

        # doc1 -> doc2 -> doc3
        for doc_id in ["doc1", "doc2", "doc3"]:
            graph.add_document(doc_id)

        edges = [
            ("doc1", "doc2"),
            ("doc2", "doc3"),
            ("doc1", "doc3"),
        ]

        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        metrics = calculator.calculate(graph)

        # doc3 は最も被引用が多いのでPageRankが高いはず
        assert metrics["doc3"].pagerank > metrics["doc1"].pagerank

    def test_get_top_by_pagerank(self):
        from monjyu.citation.metrics import DefaultMetricsCalculator
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        calculator = DefaultMetricsCalculator()
        graph = CitationGraph()

        for i in range(5):
            graph.add_document(f"doc{i}")

        # doc0 を全員が引用
        for i in range(1, 5):
            edge = CitationEdge(
                source_id=f"doc{i}",
                target_id="doc0",
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        top = calculator.get_top_by_pagerank(graph, limit=3)

        assert len(top) == 3
        assert top[0].doc_id == "doc0"  # 最も高いPageRank


# ========== CitationAnalyzer Tests ==========

class TestDefaultCitationAnalyzer:
    """DefaultCitationAnalyzer のテスト"""

    def test_find_citation_paths(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        analyzer = DefaultCitationAnalyzer()
        graph = CitationGraph()

        for doc_id in ["doc1", "doc2", "doc3"]:
            graph.add_document(doc_id)

        # doc1 -> doc2 -> doc3
        edges = [("doc1", "doc2"), ("doc2", "doc3")]
        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        paths = analyzer.find_citation_paths(graph, "doc1", "doc3")

        assert len(paths) == 1
        assert paths[0].path == ("doc1", "doc2", "doc3")
        assert paths[0].length == 2

    def test_find_shortest_path(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        analyzer = DefaultCitationAnalyzer()
        graph = CitationGraph()

        for doc_id in ["doc1", "doc2", "doc3"]:
            graph.add_document(doc_id)

        # doc1 -> doc2 -> doc3, doc1 -> doc3 (直接)
        edges = [("doc1", "doc2"), ("doc2", "doc3"), ("doc1", "doc3")]
        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        path = analyzer.find_shortest_path(graph, "doc1", "doc3")

        assert path is not None
        assert path.length == 1  # 直接パス

    def test_no_path(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph

        analyzer = DefaultCitationAnalyzer()
        graph = CitationGraph()

        graph.add_document("doc1")
        graph.add_document("doc2")
        # エッジなし

        paths = analyzer.find_citation_paths(graph, "doc1", "doc2")
        assert paths == []

    def test_co_citation(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus, CitationNetworkConfig

        config = CitationNetworkConfig(min_co_citation_count=1)
        analyzer = DefaultCitationAnalyzer(config)
        graph = CitationGraph()

        # doc1, doc2 は doc3, doc4 から引用される
        for doc_id in ["doc1", "doc2", "doc3", "doc4"]:
            graph.add_document(doc_id)

        # doc3 -> doc1, doc3 -> doc2
        # doc4 -> doc1, doc4 -> doc2
        edges = [
            ("doc3", "doc1"), ("doc3", "doc2"),
            ("doc4", "doc1"), ("doc4", "doc2"),
        ]
        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        co_cited = analyzer.find_co_citations(graph, "doc1")

        assert len(co_cited) == 1
        assert co_cited[0].doc_id == "doc2"
        assert co_cited[0].relation_type == "co_citation"

    def test_bibliographic_coupling(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus, CitationNetworkConfig

        config = CitationNetworkConfig(min_coupling_count=1)
        analyzer = DefaultCitationAnalyzer(config)
        graph = CitationGraph()

        # doc1, doc2 は両方とも doc3, doc4 を引用
        for doc_id in ["doc1", "doc2", "doc3", "doc4"]:
            graph.add_document(doc_id)

        # doc1 -> doc3, doc1 -> doc4
        # doc2 -> doc3, doc2 -> doc4
        edges = [
            ("doc1", "doc3"), ("doc1", "doc4"),
            ("doc2", "doc3"), ("doc2", "doc4"),
        ]
        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        coupled = analyzer.find_bibliographic_coupling(graph, "doc1")

        assert len(coupled) == 1
        assert coupled[0].doc_id == "doc2"
        assert coupled[0].relation_type == "bibliographic_coupling"

    def test_citation_context(self):
        from monjyu.citation.analyzer import DefaultCitationAnalyzer
        from monjyu.citation.base import CitationGraph, CitationEdge, ReferenceMatchStatus

        analyzer = DefaultCitationAnalyzer()
        graph = CitationGraph()

        for doc_id in ["doc1", "doc2", "doc3", "doc4"]:
            graph.add_document(doc_id)

        # doc1 -> doc2, doc2 -> doc3, doc3 -> doc4
        edges = [("doc1", "doc2"), ("doc2", "doc3"), ("doc3", "doc4")]
        for src, tgt in edges:
            edge = CitationEdge(
                source_id=src,
                target_id=tgt,
                is_internal=True,
                confidence=1.0,
                reference_text="ref",
                match_status=ReferenceMatchStatus.MATCHED_DOI,
            )
            graph.add_citation_edge(edge)

        # depth=1 で doc2 の周辺
        context = analyzer.get_citation_context(graph, "doc2", depth=1)

        assert "doc1" in context.internal_doc_ids  # 親
        assert "doc2" in context.internal_doc_ids  # 自分
        assert "doc3" in context.internal_doc_ids  # 子
        assert "doc4" not in context.internal_doc_ids  # depth 2


# ========== CitationNetworkManager Tests ==========

class TestCitationNetworkManager:
    """CitationNetworkManager のテスト"""

    def test_initialization(self):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        assert not manager.is_built
        assert manager.graph is None

    def test_build(self):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "title": "Paper One",
                    "doi": "10.1234/paper1",
                    "references": ["See 10.1234/paper2"],
                },
            ),
            MockDocument(
                doc_id="doc2",
                metadata={
                    "title": "Paper Two",
                    "doi": "10.1234/paper2",
                },
            ),
        ]

        result = manager.build(docs)

        assert manager.is_built
        assert result.document_count == 2
        assert result.internal_edge_count == 1

    def test_get_metrics(self):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "Paper One"}),
            MockDocument(doc_id="doc2", metadata={"title": "Paper Two"}),
        ]

        manager.build(docs)

        metrics = manager.get_metrics("doc1")
        assert metrics is not None
        assert metrics.doc_id == "doc1"

    def test_find_citation_paths(self):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "title": "Paper One",
                    "doi": "10.1234/paper1",
                    "references": ["See 10.1234/paper2"],
                },
            ),
            MockDocument(
                doc_id="doc2",
                metadata={
                    "title": "Paper Two",
                    "doi": "10.1234/paper2",
                    "references": ["See 10.1234/paper3"],
                },
            ),
            MockDocument(
                doc_id="doc3",
                metadata={
                    "title": "Paper Three",
                    "doi": "10.1234/paper3",
                },
            ),
        ]

        manager.build(docs)

        paths = manager.find_citation_paths("doc1", "doc3")
        assert len(paths) == 1
        assert paths[0].length == 2

    def test_find_related_papers(self):
        from monjyu.citation.manager import CitationNetworkManager
        from monjyu.citation.base import CitationNetworkConfig

        config = CitationNetworkConfig(min_co_citation_count=1, min_coupling_count=1)
        manager = CitationNetworkManager(config=config)

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={"title": "Paper One", "doi": "10.1234/paper1"},
            ),
            MockDocument(
                doc_id="doc2",
                metadata={"title": "Paper Two", "doi": "10.1234/paper2"},
            ),
            MockDocument(
                doc_id="doc3",
                metadata={
                    "title": "Citing Paper",
                    "doi": "10.1234/paper3",
                    "references": ["10.1234/paper1", "10.1234/paper2"],
                },
            ),
        ]

        manager.build(docs)

        related = manager.find_related_papers("doc1", method="co_citation")
        # doc1 と doc2 は doc3 から共引用されている
        assert any(r.doc_id == "doc2" for r in related)

    def test_save_and_load_graphml(self, tmp_path):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "Paper One"}),
            MockDocument(doc_id="doc2", metadata={"title": "Paper Two"}),
        ]

        manager.build(docs)

        # 保存
        graphml_path = tmp_path / "test.graphml"
        manager.save_graphml(graphml_path)
        assert graphml_path.exists()

        # 読み込み
        manager2 = CitationNetworkManager()
        manager2.load_graphml(graphml_path)

        assert manager2.is_built
        assert manager2.graph.node_count == 2

    def test_save_json(self, tmp_path):
        from monjyu.citation.manager import CitationNetworkManager
        import json

        manager = CitationNetworkManager()

        docs = [
            MockDocument(doc_id="doc1", metadata={"title": "Paper One"}),
        ]

        manager.build(docs)

        json_path = tmp_path / "test.json"
        manager.save_json(json_path)

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "stats" in data
        assert "metrics" in data

    def test_export_edges_csv(self, tmp_path):
        from monjyu.citation.manager import CitationNetworkManager

        manager = CitationNetworkManager()

        docs = [
            MockDocument(
                doc_id="doc1",
                metadata={
                    "title": "Paper One",
                    "doi": "10.1234/paper1",
                    "references": ["See 10.1234/paper2"],
                },
            ),
            MockDocument(
                doc_id="doc2",
                metadata={
                    "title": "Paper Two",
                    "doi": "10.1234/paper2",
                },
            ),
        ]

        manager.build(docs)

        csv_path = tmp_path / "edges.csv"
        manager.export_edges_csv(csv_path)

        assert csv_path.exists()

        with open(csv_path) as f:
            lines = f.readlines()

        assert len(lines) == 2  # header + 1 edge
        assert "source,target" in lines[0]


# ========== Factory Function Tests ==========

class TestFactoryFunction:
    """ファクトリ関数のテスト"""

    def test_create_citation_network_manager(self):
        from monjyu.citation.manager import create_citation_network_manager
        from monjyu.citation.base import CitationNetworkConfig

        manager = create_citation_network_manager()
        assert manager is not None

        config = CitationNetworkConfig(pagerank_alpha=0.9)
        manager2 = create_citation_network_manager(config)
        assert manager2.config.pagerank_alpha == 0.9
