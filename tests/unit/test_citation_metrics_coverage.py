# Citation Metrics Coverage Tests
"""
Unit tests for citation metrics calculator to improve coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch
import pytest

import networkx as nx

from monjyu.citation.base import (
    CitationGraph,
    DocumentMetrics,
    CitationNetworkConfig,
    CitationEdge,
    ReferenceMatchStatus,
)
from monjyu.citation.metrics import (
    MetricsCalculator,
    DefaultMetricsCalculator,
    MockMetricsCalculator,
)


def make_edge(source: str, target: str) -> CitationEdge:
    """テスト用のCitationEdgeを作成"""
    return CitationEdge(
        source_id=source,
        target_id=target,
        is_internal=True,
        confidence=1.0,
        reference_text="",
        match_status=ReferenceMatchStatus.MATCHED_DOI,
    )


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def empty_graph():
    """空の引用グラフ"""
    graph = CitationGraph()
    return graph


@pytest.fixture
def simple_graph():
    """シンプルな引用グラフ (A -> B -> C)"""
    graph = CitationGraph()
    graph.add_document("doc_a")
    graph.add_document("doc_b")
    graph.add_document("doc_c")
    graph.add_citation_edge(make_edge("doc_a", "doc_b"))
    graph.add_citation_edge(make_edge("doc_b", "doc_c"))
    return graph


@pytest.fixture
def complex_graph():
    """複雑な引用グラフ"""
    graph = CitationGraph()
    # 5文書を追加
    for i in range(5):
        graph.add_document(f"doc_{i}")
    
    # 複数の引用関係
    edges = [
        ("doc_0", "doc_1"),
        ("doc_0", "doc_2"),
        ("doc_1", "doc_2"),
        ("doc_1", "doc_3"),
        ("doc_2", "doc_3"),
        ("doc_2", "doc_4"),
        ("doc_3", "doc_4"),
    ]
    for source, target in edges:
        graph.add_citation_edge(make_edge(source, target))
    
    return graph


@pytest.fixture
def single_node_graph():
    """ノード1つだけのグラフ"""
    graph = CitationGraph()
    graph.add_document("doc_only")
    return graph


@pytest.fixture
def default_calculator():
    """デフォルト設定の計算器"""
    return DefaultMetricsCalculator()


@pytest.fixture
def custom_config_calculator():
    """カスタム設定の計算器"""
    config = CitationNetworkConfig(
        pagerank_alpha=0.9,
        pagerank_max_iter=50,
    )
    return DefaultMetricsCalculator(config=config)


# ==============================================================================
# MetricsCalculator Abstract Base Class Tests
# ==============================================================================

class TestMetricsCalculatorABC:
    """抽象基底クラスのテスト"""
    
    def test_cannot_instantiate_abc(self):
        """抽象クラスはインスタンス化できない"""
        with pytest.raises(TypeError):
            MetricsCalculator()


# ==============================================================================
# DefaultMetricsCalculator Tests
# ==============================================================================

class TestDefaultMetricsCalculatorInit:
    """初期化テスト"""
    
    def test_init_default_config(self, default_calculator):
        """デフォルト設定で初期化"""
        assert default_calculator.config is not None
        assert default_calculator._cached_pagerank is None
        assert default_calculator._cached_hits is None
        assert default_calculator._cache_graph_hash is None
    
    def test_init_custom_config(self, custom_config_calculator):
        """カスタム設定で初期化"""
        assert custom_config_calculator.config.pagerank_alpha == 0.9
        assert custom_config_calculator.config.pagerank_max_iter == 50


class TestDefaultMetricsCalculatorCache:
    """キャッシュ機能テスト"""
    
    def test_get_graph_hash(self, default_calculator, simple_graph):
        """グラフハッシュを取得"""
        hash1 = default_calculator._get_graph_hash(simple_graph)
        hash2 = default_calculator._get_graph_hash(simple_graph)
        assert hash1 == hash2
    
    def test_different_graphs_different_hash(self, default_calculator, simple_graph, complex_graph):
        """異なるグラフは異なるハッシュ"""
        hash1 = default_calculator._get_graph_hash(simple_graph)
        hash2 = default_calculator._get_graph_hash(complex_graph)
        assert hash1 != hash2
    
    def test_ensure_cache_computes_metrics(self, default_calculator, simple_graph):
        """キャッシュ更新時にメトリクスを計算"""
        assert default_calculator._cached_pagerank is None
        default_calculator._ensure_cache(simple_graph)
        assert default_calculator._cached_pagerank is not None
        assert default_calculator._cached_hits is not None
    
    def test_cache_not_recomputed_for_same_graph(self, default_calculator, simple_graph):
        """同じグラフでは再計算しない"""
        default_calculator._ensure_cache(simple_graph)
        first_pagerank = default_calculator._cached_pagerank
        
        # 再度呼び出し
        default_calculator._ensure_cache(simple_graph)
        assert default_calculator._cached_pagerank is first_pagerank


class TestDefaultMetricsCalculatorCompute:
    """メトリクス計算テスト"""
    
    def test_compute_all_metrics_empty_graph(self, default_calculator, empty_graph):
        """空グラフでのメトリクス計算"""
        default_calculator._compute_all_metrics(empty_graph)
        assert default_calculator._cached_pagerank == {}
        assert default_calculator._cached_hits is not None
    
    def test_compute_all_metrics_single_node(self, default_calculator, single_node_graph):
        """1ノードグラフでのメトリクス計算"""
        default_calculator._compute_all_metrics(single_node_graph)
        assert "doc_only" in default_calculator._cached_pagerank
        # シングルノードでは均一なスコア
        hubs, authorities = default_calculator._cached_hits
        assert hubs.get("doc_only") == 1.0
    
    def test_compute_all_metrics_simple_graph(self, default_calculator, simple_graph):
        """シンプルグラフでのメトリクス計算"""
        default_calculator._compute_all_metrics(simple_graph)
        
        # PageRankが計算されている
        assert len(default_calculator._cached_pagerank) == 3
        for doc_id in ["doc_a", "doc_b", "doc_c"]:
            assert doc_id in default_calculator._cached_pagerank
        
        # HITSが計算されている
        hubs, authorities = default_calculator._cached_hits
        assert len(hubs) == 3
        assert len(authorities) == 3
    
    def test_compute_all_metrics_complex_graph(self, default_calculator, complex_graph):
        """複雑グラフでのメトリクス計算"""
        default_calculator._compute_all_metrics(complex_graph)
        
        # PageRankが計算されている
        assert len(default_calculator._cached_pagerank) == 5
        
        # doc_4 は最も被引用が多い → PageRank高い
        pageranks = default_calculator._cached_pagerank
        assert pageranks["doc_4"] >= pageranks["doc_0"]


class TestDefaultMetricsCalculatorCalculate:
    """calculate メソッドテスト"""
    
    def test_calculate_all(self, default_calculator, simple_graph):
        """全文書のメトリクスを計算"""
        metrics = default_calculator.calculate(simple_graph)
        
        assert len(metrics) == 3
        for doc_id in ["doc_a", "doc_b", "doc_c"]:
            assert doc_id in metrics
            assert isinstance(metrics[doc_id], DocumentMetrics)
    
    def test_calculate_returns_document_metrics(self, default_calculator, simple_graph):
        """DocumentMetrics型を返す"""
        metrics = default_calculator.calculate(simple_graph)
        
        m = metrics["doc_a"]
        assert m.doc_id == "doc_a"
        assert isinstance(m.pagerank, float)
        assert isinstance(m.hub_score, float)
        assert isinstance(m.authority_score, float)
    
    def test_calculate_citation_counts(self, default_calculator, simple_graph):
        """引用数が正しく計算されている"""
        metrics = default_calculator.calculate(simple_graph)
        
        # doc_a: 0引用される、1参照
        assert metrics["doc_a"].citation_count == 0
        assert metrics["doc_a"].reference_count == 1
        
        # doc_b: 1引用される、1参照
        assert metrics["doc_b"].citation_count == 1
        assert metrics["doc_b"].reference_count == 1
        
        # doc_c: 1引用される、0参照
        assert metrics["doc_c"].citation_count == 1
        assert metrics["doc_c"].reference_count == 0


class TestDefaultMetricsCalculatorCalculateForDocument:
    """calculate_for_document メソッドテスト"""
    
    def test_calculate_for_existing_document(self, default_calculator, simple_graph):
        """存在する文書のメトリクスを計算"""
        metrics = default_calculator.calculate_for_document(simple_graph, "doc_a")
        
        assert metrics.doc_id == "doc_a"
        assert isinstance(metrics, DocumentMetrics)
    
    def test_calculate_for_nonexistent_document(self, default_calculator, simple_graph):
        """存在しない文書のメトリクスを計算"""
        metrics = default_calculator.calculate_for_document(simple_graph, "nonexistent")
        
        # デフォルト値が返される
        assert metrics.doc_id == "nonexistent"
        assert metrics.pagerank == 0.0


class TestDefaultMetricsCalculatorTopBy:
    """get_top_by_* メソッドテスト"""
    
    def test_get_top_by_pagerank(self, default_calculator, complex_graph):
        """PageRankトップ取得"""
        top = default_calculator.get_top_by_pagerank(complex_graph, limit=3)
        
        assert len(top) == 3
        # ソートされている
        for i in range(len(top) - 1):
            assert top[i].pagerank >= top[i + 1].pagerank
    
    def test_get_top_by_authority(self, default_calculator, complex_graph):
        """Authorityトップ取得"""
        top = default_calculator.get_top_by_authority(complex_graph, limit=3)
        
        assert len(top) == 3
        # ソートされている
        for i in range(len(top) - 1):
            assert top[i].authority_score >= top[i + 1].authority_score
    
    def test_get_top_by_hub(self, default_calculator, complex_graph):
        """Hubトップ取得"""
        top = default_calculator.get_top_by_hub(complex_graph, limit=3)
        
        assert len(top) == 3
        # ソートされている
        for i in range(len(top) - 1):
            assert top[i].hub_score >= top[i + 1].hub_score
    
    def test_get_top_by_citations(self, default_calculator, complex_graph):
        """被引用数トップ取得"""
        top = default_calculator.get_top_by_citations(complex_graph, limit=3)
        
        assert len(top) == 3
        # ソートされている
        for i in range(len(top) - 1):
            assert top[i].citation_count >= top[i + 1].citation_count
    
    def test_get_top_with_small_limit(self, default_calculator, complex_graph):
        """小さいlimitで取得"""
        top = default_calculator.get_top_by_pagerank(complex_graph, limit=1)
        assert len(top) == 1
    
    def test_get_top_with_large_limit(self, default_calculator, simple_graph):
        """グラフサイズより大きいlimitで取得"""
        top = default_calculator.get_top_by_pagerank(simple_graph, limit=100)
        # 実際のノード数が上限
        assert len(top) == 3


class TestDefaultMetricsCalculatorErrorHandling:
    """エラーハンドリングテスト"""
    
    def test_pagerank_empty_graph(self, default_calculator, empty_graph):
        """空グラフでPageRank計算"""
        default_calculator._compute_all_metrics(empty_graph)
        # エラーなく完了
        assert default_calculator._cached_pagerank is not None
    
    def test_hits_with_networkx_error(self, default_calculator, simple_graph):
        """HITS計算でNetworkXエラー"""
        with patch("networkx.hits", side_effect=nx.NetworkXError("Test error")):
            default_calculator._compute_all_metrics(simple_graph)
            # エラーがキャッチされ、均一なスコアが設定される
            hubs, authorities = default_calculator._cached_hits
            assert len(hubs) == 3
    
    def test_hits_with_generic_exception(self, default_calculator, simple_graph):
        """HITS計算で一般例外"""
        with patch("networkx.hits", side_effect=Exception("Scipy error")):
            default_calculator._compute_all_metrics(simple_graph)
            # エラーがキャッチされる
            assert default_calculator._cached_hits is not None


# ==============================================================================
# MockMetricsCalculator Tests
# ==============================================================================

class TestMockMetricsCalculatorInit:
    """MockMetricsCalculator初期化テスト"""
    
    def test_init_empty(self):
        """空のモック"""
        calculator = MockMetricsCalculator()
        assert calculator._mock_metrics == {}
    
    def test_init_with_metrics(self):
        """メトリクス付きで初期化"""
        mock_metrics = {
            "doc_1": DocumentMetrics(
                doc_id="doc_1",
                citation_count=5,
                reference_count=3,
                pagerank=0.5,
                hub_score=0.3,
                authority_score=0.7,
            ),
        }
        calculator = MockMetricsCalculator(mock_metrics=mock_metrics)
        assert "doc_1" in calculator._mock_metrics


class TestMockMetricsCalculatorCalculate:
    """MockMetricsCalculatorのcalculateテスト"""
    
    def test_calculate_returns_mock(self, simple_graph):
        """モックメトリクスを返す"""
        mock_metrics = {
            "doc_a": DocumentMetrics(
                doc_id="doc_a",
                citation_count=10,
                reference_count=5,
                pagerank=0.9,
                hub_score=0.8,
                authority_score=0.7,
            ),
        }
        calculator = MockMetricsCalculator(mock_metrics=mock_metrics)
        
        result = calculator.calculate(simple_graph)
        assert result == mock_metrics
    
    def test_calculate_for_document_existing(self, simple_graph):
        """存在するドキュメントのメトリクス"""
        mock_metrics = {
            "doc_a": DocumentMetrics(
                doc_id="doc_a",
                citation_count=10,
                reference_count=5,
                pagerank=0.9,
                hub_score=0.8,
                authority_score=0.7,
            ),
        }
        calculator = MockMetricsCalculator(mock_metrics=mock_metrics)
        
        result = calculator.calculate_for_document(simple_graph, "doc_a")
        assert result.citation_count == 10
        assert result.pagerank == 0.9
    
    def test_calculate_for_document_nonexistent(self, simple_graph):
        """存在しないドキュメントはデフォルト値"""
        calculator = MockMetricsCalculator()
        
        result = calculator.calculate_for_document(simple_graph, "nonexistent")
        assert result.doc_id == "nonexistent"
        assert result.citation_count == 0
        assert result.pagerank == 0.0


class TestMockMetricsCalculatorTopBy:
    """MockMetricsCalculatorのget_top_by_*テスト"""
    
    @pytest.fixture
    def mock_calculator_with_data(self):
        """テストデータ付きモック計算器"""
        mock_metrics = {
            "doc_1": DocumentMetrics(
                doc_id="doc_1",
                citation_count=10,
                reference_count=5,
                pagerank=0.9,
                hub_score=0.3,
                authority_score=0.7,
            ),
            "doc_2": DocumentMetrics(
                doc_id="doc_2",
                citation_count=5,
                reference_count=10,
                pagerank=0.5,
                hub_score=0.8,
                authority_score=0.4,
            ),
            "doc_3": DocumentMetrics(
                doc_id="doc_3",
                citation_count=15,
                reference_count=2,
                pagerank=0.3,
                hub_score=0.5,
                authority_score=0.9,
            ),
        }
        return MockMetricsCalculator(mock_metrics=mock_metrics)
    
    def test_get_top_by_pagerank(self, mock_calculator_with_data, simple_graph):
        """PageRankトップ"""
        top = mock_calculator_with_data.get_top_by_pagerank(simple_graph, limit=2)
        
        assert len(top) == 2
        assert top[0].doc_id == "doc_1"  # pagerank=0.9
        assert top[1].doc_id == "doc_2"  # pagerank=0.5
    
    def test_get_top_by_authority(self, mock_calculator_with_data, simple_graph):
        """Authorityトップ"""
        top = mock_calculator_with_data.get_top_by_authority(simple_graph, limit=2)
        
        assert len(top) == 2
        assert top[0].doc_id == "doc_3"  # authority=0.9
        assert top[1].doc_id == "doc_1"  # authority=0.7
