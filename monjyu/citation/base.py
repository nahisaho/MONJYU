# Citation Network Base Types
"""
monjyu.citation.base - 引用ネットワークの基本型定義

FEAT-006: Citation Network
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx


class ReferenceMatchStatus(Enum):
    """参照解決状態"""

    MATCHED_DOI = "matched_doi"  # DOI完全一致
    MATCHED_TITLE_EXACT = "matched_title_exact"  # タイトル完全一致
    MATCHED_TITLE_FUZZY = "matched_title_fuzzy"  # タイトル類似一致
    UNRESOLVED = "unresolved"  # 未解決（外部文書）


@dataclass(frozen=True)
class ResolvedReference:
    """解決済み参照情報"""

    source_doc_id: str
    target_doc_id: str | None  # Noneの場合は外部参照
    status: ReferenceMatchStatus
    confidence: float  # 0.0-1.0
    raw_reference: str  # 元の参照文字列
    matched_doi: str | None = None
    matched_title: str | None = None


@dataclass(frozen=True)
class CitationEdge:
    """引用エッジ"""

    source_id: str  # 引用元文書ID
    target_id: str  # 引用先文書ID（内部） or 参照キー（外部）
    is_internal: bool  # コーパス内部の引用か
    confidence: float
    reference_text: str  # 参照テキスト
    match_status: ReferenceMatchStatus


@dataclass
class CitationGraph:
    """引用グラフ"""

    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    internal_doc_ids: set[str] = field(default_factory=set)
    external_refs: dict[str, str] = field(default_factory=dict)  # ref_key -> raw_text

    @property
    def node_count(self) -> int:
        """ノード数"""
        return self.graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """エッジ数"""
        return self.graph.number_of_edges()

    @property
    def internal_edge_count(self) -> int:
        """内部エッジ数"""
        return sum(
            1
            for _, _, data in self.graph.edges(data=True)
            if data.get("is_internal", False)
        )

    @property
    def external_edge_count(self) -> int:
        """外部エッジ数"""
        return self.edge_count - self.internal_edge_count

    def add_document(self, doc_id: str, metadata: dict[str, Any] | None = None) -> None:
        """文書ノードを追加"""
        self.graph.add_node(doc_id, node_type="document", **(metadata or {}))
        self.internal_doc_ids.add(doc_id)

    def add_external_reference(self, ref_key: str, raw_text: str) -> None:
        """外部参照ノードを追加"""
        self.graph.add_node(ref_key, node_type="external", raw_text=raw_text)
        self.external_refs[ref_key] = raw_text

    def add_citation_edge(self, edge: CitationEdge) -> None:
        """引用エッジを追加"""
        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            is_internal=edge.is_internal,
            confidence=edge.confidence,
            reference_text=edge.reference_text,
            match_status=edge.match_status.value,
        )

    def get_citations(self, doc_id: str) -> list[str]:
        """指定文書が引用している文書一覧"""
        return list(self.graph.successors(doc_id))

    def get_cited_by(self, doc_id: str) -> list[str]:
        """指定文書を引用している文書一覧"""
        return list(self.graph.predecessors(doc_id))

    def get_citation_count(self, doc_id: str) -> int:
        """被引用数（in-degree）"""
        return self.graph.in_degree(doc_id)

    def get_reference_count(self, doc_id: str) -> int:
        """引用数（out-degree）"""
        return self.graph.out_degree(doc_id)


@dataclass(frozen=True)
class DocumentMetrics:
    """文書の引用メトリクス"""

    doc_id: str
    citation_count: int  # 被引用数
    reference_count: int  # 引用数
    pagerank: float
    hub_score: float
    authority_score: float

    @property
    def total_connections(self) -> int:
        """総接続数"""
        return self.citation_count + self.reference_count


@dataclass(frozen=True)
class CitationPath:
    """引用パス"""

    path: tuple[str, ...]  # 文書IDのシーケンス
    length: int

    @property
    def source(self) -> str:
        """始点"""
        return self.path[0]

    @property
    def target(self) -> str:
        """終点"""
        return self.path[-1]

    def __iter__(self):
        return iter(self.path)

    def __len__(self) -> int:
        return self.length


@dataclass(frozen=True)
class RelatedPaper:
    """関連論文"""

    doc_id: str
    relation_type: str  # co_citation, bibliographic_coupling
    similarity_score: float
    shared_papers: frozenset[str]  # 共通して引用/被引用している論文


@dataclass
class CitationNetworkConfig:
    """引用ネットワーク設定"""

    # ReferenceResolver
    fuzzy_match_threshold: float = 0.85
    enable_doi_matching: bool = True
    enable_title_matching: bool = True

    # MetricsCalculator
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 100

    # CitationAnalyzer
    max_path_length: int = 5
    min_co_citation_count: int = 2
    min_coupling_count: int = 2

    def __post_init__(self):
        """バリデーション"""
        if not 0 < self.fuzzy_match_threshold <= 1.0:
            raise ValueError("fuzzy_match_threshold must be between 0 and 1")
        if not 0 < self.pagerank_alpha < 1.0:
            raise ValueError("pagerank_alpha must be between 0 and 1")
        if self.max_path_length < 1:
            raise ValueError("max_path_length must be at least 1")
