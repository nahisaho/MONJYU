# Citation Network Manager
"""
monjyu.citation.manager - 引用ネットワーク管理（Facade）

FEAT-006: Citation Network
- 統合インターフェース
- Parquet / GraphML 出力
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json

import networkx as nx

if TYPE_CHECKING:
    from monjyu.document.base import Document

from monjyu.citation.base import (
    CitationGraph,
    CitationPath,
    DocumentMetrics,
    RelatedPaper,
    CitationNetworkConfig,
)
from monjyu.citation.resolver import DefaultReferenceResolver, ReferenceResolver
from monjyu.citation.builder import DefaultCitationGraphBuilder, CitationGraphBuilder
from monjyu.citation.metrics import DefaultMetricsCalculator, MetricsCalculator
from monjyu.citation.analyzer import DefaultCitationAnalyzer, CitationAnalyzer


@dataclass
class CitationNetworkBuildResult:
    """引用ネットワーク構築結果"""

    graph: CitationGraph
    metrics: dict[str, DocumentMetrics]
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        """文書数"""
        return len(self.graph.internal_doc_ids)

    @property
    def internal_edge_count(self) -> int:
        """内部エッジ数"""
        return self.graph.internal_edge_count

    @property
    def external_edge_count(self) -> int:
        """外部エッジ数"""
        return self.graph.external_edge_count


class CitationNetworkManager:
    """引用ネットワーク管理（Facade）"""

    def __init__(
        self,
        config: CitationNetworkConfig | None = None,
        resolver: ReferenceResolver | None = None,
        builder: CitationGraphBuilder | None = None,
        metrics_calculator: MetricsCalculator | None = None,
        analyzer: CitationAnalyzer | None = None,
    ):
        self.config = config or CitationNetworkConfig()

        # コンポーネントの初期化
        self._resolver = resolver or DefaultReferenceResolver(self.config)
        self._builder = builder or DefaultCitationGraphBuilder(
            resolver=self._resolver,
            config=self.config,
        )
        self._metrics = metrics_calculator or DefaultMetricsCalculator(self.config)
        self._analyzer = analyzer or DefaultCitationAnalyzer(self.config)

        # 現在のグラフ
        self._graph: CitationGraph | None = None
        self._cached_metrics: dict[str, DocumentMetrics] | None = None

    @property
    def graph(self) -> CitationGraph | None:
        """現在の引用グラフ"""
        return self._graph

    @property
    def is_built(self) -> bool:
        """グラフが構築済みか"""
        return self._graph is not None

    def build(self, documents: list[Document]) -> CitationNetworkBuildResult:
        """引用ネットワークを構築"""
        # グラフを構築
        self._graph = self._builder.build(documents)

        # メトリクスを計算
        self._cached_metrics = self._metrics.calculate(self._graph)

        # 統計を計算
        stats = self._compute_stats()

        return CitationNetworkBuildResult(
            graph=self._graph,
            metrics=self._cached_metrics,
            stats=stats,
        )

    def _compute_stats(self) -> dict[str, Any]:
        """統計を計算"""
        if self._graph is None:
            return {}

        g = self._graph.graph

        stats: dict[str, Any] = {
            "node_count": self._graph.node_count,
            "edge_count": self._graph.edge_count,
            "internal_edge_count": self._graph.internal_edge_count,
            "external_edge_count": self._graph.external_edge_count,
            "document_count": len(self._graph.internal_doc_ids),
            "external_ref_count": len(self._graph.external_refs),
        }

        # 密度
        if g.number_of_nodes() > 1:
            stats["density"] = nx.density(g)
        else:
            stats["density"] = 0.0

        # 連結成分
        undirected = g.to_undirected()
        stats["connected_components"] = nx.number_connected_components(undirected)

        return stats

    # ========== 分析メソッド ==========

    def get_metrics(self, doc_id: str) -> DocumentMetrics | None:
        """文書のメトリクスを取得"""
        if self._cached_metrics is None:
            return None
        return self._cached_metrics.get(doc_id)

    def get_all_metrics(self) -> dict[str, DocumentMetrics]:
        """全文書のメトリクスを取得"""
        return self._cached_metrics or {}

    def get_top_by_pagerank(self, limit: int = 10) -> list[DocumentMetrics]:
        """PageRankスコア上位の文書"""
        if self._graph is None:
            return []
        return self._metrics.get_top_by_pagerank(self._graph, limit)

    def get_top_by_citations(self, limit: int = 10) -> list[DocumentMetrics]:
        """被引用数上位の文書"""
        if self._graph is None or self._cached_metrics is None:
            return []
        sorted_metrics = sorted(
            self._cached_metrics.values(),
            key=lambda m: m.citation_count,
            reverse=True,
        )
        return sorted_metrics[:limit]

    def find_citation_paths(
        self,
        source_id: str,
        target_id: str,
        max_length: int | None = None,
    ) -> list[CitationPath]:
        """引用パスを探索"""
        if self._graph is None:
            return []
        return self._analyzer.find_citation_paths(
            self._graph, source_id, target_id, max_length
        )

    def find_related_papers(
        self,
        doc_id: str,
        method: str = "both",
    ) -> list[RelatedPaper]:
        """
        関連論文を発見

        Args:
            doc_id: 対象文書ID
            method: "co_citation", "bibliographic_coupling", "both"
        """
        if self._graph is None:
            return []

        results: list[RelatedPaper] = []

        if method in ("co_citation", "both"):
            results.extend(
                self._analyzer.find_co_citations(self._graph, doc_id)
            )

        if method in ("bibliographic_coupling", "both"):
            results.extend(
                self._analyzer.find_bibliographic_coupling(self._graph, doc_id)
            )

        # スコアでソート
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def get_citation_context(
        self,
        doc_id: str,
        depth: int = 1,
    ) -> CitationGraph:
        """文書の引用コンテキストを取得"""
        if self._graph is None:
            return CitationGraph()
        return self._analyzer.get_citation_context(self._graph, doc_id, depth)

    # ========== 永続化メソッド ==========

    def save_graphml(self, path: Path | str) -> None:
        """GraphML形式で保存"""
        if self._graph is None:
            raise ValueError("Graph not built yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # GraphML は None をサポートしないため、コピーを作成して None を除去
        g_copy = self._graph.graph.copy()
        for node in g_copy.nodes():
            for key, val in list(g_copy.nodes[node].items()):
                if val is None:
                    g_copy.nodes[node][key] = ""
        for u, v in g_copy.edges():
            for key, val in list(g_copy.edges[u, v].items()):
                if val is None:
                    g_copy.edges[u, v][key] = ""

        nx.write_graphml(g_copy, str(path))

    def load_graphml(self, path: Path | str) -> None:
        """GraphML形式から読み込み"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GraphML file not found: {path}")

        g = nx.read_graphml(str(path))

        # CitationGraph を再構築
        self._graph = CitationGraph()
        self._graph.graph = g

        # ノードタイプから internal/external を復元
        for node, data in g.nodes(data=True):
            if data.get("node_type") == "document":
                self._graph.internal_doc_ids.add(node)
            elif data.get("node_type") == "external":
                self._graph.external_refs[node] = data.get("raw_text", "")

        # メトリクスを再計算
        self._cached_metrics = self._metrics.calculate(self._graph)

    def save_json(self, path: Path | str) -> None:
        """JSON形式でメタデータを保存"""
        if self._graph is None:
            raise ValueError("Graph not built yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "stats": self._compute_stats(),
            "metrics": {
                doc_id: {
                    "doc_id": m.doc_id,
                    "citation_count": m.citation_count,
                    "reference_count": m.reference_count,
                    "pagerank": m.pagerank,
                    "hub_score": m.hub_score,
                    "authority_score": m.authority_score,
                }
                for doc_id, m in (self._cached_metrics or {}).items()
            },
            "external_refs": self._graph.external_refs,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def export_edges_csv(self, path: Path | str) -> None:
        """エッジをCSV形式でエクスポート"""
        if self._graph is None:
            raise ValueError("Graph not built yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write("source,target,is_internal,confidence,match_status\n")
            for u, v, data in self._graph.graph.edges(data=True):
                is_internal = data.get("is_internal", False)
                confidence = data.get("confidence", 0.0)
                match_status = data.get("match_status", "unknown")
                f.write(f"{u},{v},{is_internal},{confidence},{match_status}\n")


def create_citation_network_manager(
    config: CitationNetworkConfig | None = None,
) -> CitationNetworkManager:
    """CitationNetworkManagerを作成するファクトリ関数"""
    return CitationNetworkManager(config=config)
