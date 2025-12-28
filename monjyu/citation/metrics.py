# Metrics Calculator
"""
monjyu.citation.metrics - 引用メトリクス計算

FEAT-006: Citation Network
- PageRank アルゴリズム
- HITS アルゴリズム (Hub/Authority)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx

from monjyu.citation.base import (
    CitationGraph,
    DocumentMetrics,
    CitationNetworkConfig,
)


class MetricsCalculator(ABC):
    """メトリクス計算器の抽象基底クラス"""

    @abstractmethod
    def calculate(self, graph: CitationGraph) -> dict[str, DocumentMetrics]:
        """全文書のメトリクスを計算"""
        pass

    @abstractmethod
    def calculate_for_document(
        self,
        graph: CitationGraph,
        doc_id: str,
    ) -> DocumentMetrics:
        """特定文書のメトリクスを計算"""
        pass

    @abstractmethod
    def get_top_by_pagerank(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """PageRankスコア上位の文書を取得"""
        pass

    @abstractmethod
    def get_top_by_authority(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """Authorityスコア上位の文書を取得"""
        pass


class DefaultMetricsCalculator(MetricsCalculator):
    """デフォルトメトリクス計算器"""

    def __init__(self, config: CitationNetworkConfig | None = None):
        self.config = config or CitationNetworkConfig()
        self._cached_pagerank: dict[str, float] | None = None
        self._cached_hits: tuple[dict[str, float], dict[str, float]] | None = None
        self._cache_graph_hash: int | None = None

    def _get_graph_hash(self, graph: CitationGraph) -> int:
        """グラフのハッシュを計算（キャッシュ用）"""
        return hash((
            frozenset(graph.graph.nodes()),
            frozenset(graph.graph.edges()),
        ))

    def _ensure_cache(self, graph: CitationGraph) -> None:
        """キャッシュを更新（必要な場合）"""
        current_hash = self._get_graph_hash(graph)
        if self._cache_graph_hash != current_hash:
            self._compute_all_metrics(graph)
            self._cache_graph_hash = current_hash

    def _compute_all_metrics(self, graph: CitationGraph) -> None:
        """全メトリクスを計算してキャッシュ"""
        g = graph.graph

        # PageRank
        try:
            self._cached_pagerank = nx.pagerank(
                g,
                alpha=self.config.pagerank_alpha,
                max_iter=self.config.pagerank_max_iter,
            )
        except nx.NetworkXError:
            # グラフが空の場合など
            self._cached_pagerank = {node: 0.0 for node in g.nodes()}

        # HITS
        try:
            # HITSは最低2ノード、1エッジが必要
            if g.number_of_nodes() >= 2 and g.number_of_edges() >= 1:
                hubs, authorities = nx.hits(g, max_iter=100)
                self._cached_hits = (hubs, authorities)
            else:
                # 小規模グラフでは均等なスコアを設定
                uniform = 1.0 / max(g.number_of_nodes(), 1)
                self._cached_hits = (
                    {node: uniform for node in g.nodes()},
                    {node: uniform for node in g.nodes()},
                )
        except (nx.NetworkXError, Exception):
            # NetworkXやSciPyのエラーをキャッチ
            uniform = 1.0 / max(g.number_of_nodes(), 1)
            self._cached_hits = (
                {node: uniform for node in g.nodes()},
                {node: uniform for node in g.nodes()},
            )

    def calculate(self, graph: CitationGraph) -> dict[str, DocumentMetrics]:
        """全文書のメトリクスを計算"""
        self._ensure_cache(graph)

        metrics: dict[str, DocumentMetrics] = {}

        for doc_id in graph.internal_doc_ids:
            metrics[doc_id] = self._build_metrics(graph, doc_id)

        return metrics

    def calculate_for_document(
        self,
        graph: CitationGraph,
        doc_id: str,
    ) -> DocumentMetrics:
        """特定文書のメトリクスを計算"""
        self._ensure_cache(graph)
        return self._build_metrics(graph, doc_id)

    def _build_metrics(self, graph: CitationGraph, doc_id: str) -> DocumentMetrics:
        """DocumentMetrics オブジェクトを構築"""
        pagerank = self._cached_pagerank or {}
        hubs, authorities = self._cached_hits or ({}, {})

        return DocumentMetrics(
            doc_id=doc_id,
            citation_count=graph.get_citation_count(doc_id),
            reference_count=graph.get_reference_count(doc_id),
            pagerank=pagerank.get(doc_id, 0.0),
            hub_score=hubs.get(doc_id, 0.0),
            authority_score=authorities.get(doc_id, 0.0),
        )

    def get_top_by_pagerank(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """PageRankスコア上位の文書を取得"""
        all_metrics = self.calculate(graph)
        sorted_metrics = sorted(
            all_metrics.values(),
            key=lambda m: m.pagerank,
            reverse=True,
        )
        return sorted_metrics[:limit]

    def get_top_by_authority(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """Authorityスコア上位の文書を取得"""
        all_metrics = self.calculate(graph)
        sorted_metrics = sorted(
            all_metrics.values(),
            key=lambda m: m.authority_score,
            reverse=True,
        )
        return sorted_metrics[:limit]

    def get_top_by_hub(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """Hubスコア上位の文書を取得"""
        all_metrics = self.calculate(graph)
        sorted_metrics = sorted(
            all_metrics.values(),
            key=lambda m: m.hub_score,
            reverse=True,
        )
        return sorted_metrics[:limit]

    def get_top_by_citations(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """被引用数上位の文書を取得"""
        all_metrics = self.calculate(graph)
        sorted_metrics = sorted(
            all_metrics.values(),
            key=lambda m: m.citation_count,
            reverse=True,
        )
        return sorted_metrics[:limit]


class MockMetricsCalculator(MetricsCalculator):
    """テスト用のモックメトリクス計算器"""

    def __init__(
        self,
        mock_metrics: dict[str, DocumentMetrics] | None = None,
    ):
        self._mock_metrics = mock_metrics or {}

    def calculate(self, graph: CitationGraph) -> dict[str, DocumentMetrics]:
        """モックメトリクスを返す"""
        return self._mock_metrics

    def calculate_for_document(
        self,
        graph: CitationGraph,
        doc_id: str,
    ) -> DocumentMetrics:
        """モックメトリクスを返す"""
        if doc_id in self._mock_metrics:
            return self._mock_metrics[doc_id]

        return DocumentMetrics(
            doc_id=doc_id,
            citation_count=0,
            reference_count=0,
            pagerank=0.0,
            hub_score=0.0,
            authority_score=0.0,
        )

    def get_top_by_pagerank(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """モックメトリクスからソート"""
        sorted_metrics = sorted(
            self._mock_metrics.values(),
            key=lambda m: m.pagerank,
            reverse=True,
        )
        return sorted_metrics[:limit]

    def get_top_by_authority(
        self,
        graph: CitationGraph,
        limit: int = 10,
    ) -> list[DocumentMetrics]:
        """モックメトリクスからソート"""
        sorted_metrics = sorted(
            self._mock_metrics.values(),
            key=lambda m: m.authority_score,
            reverse=True,
        )
        return sorted_metrics[:limit]
