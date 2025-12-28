# Graph Base Classes and Protocols
"""
Base classes and protocols for graph construction and community detection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class NounPhraseNode:
    """名詞句ノード
    
    Attributes:
        id: ノードID
        phrase: 名詞句テキスト
        frequency: 出現頻度
        document_ids: 出現ドキュメントIDリスト
        text_unit_ids: 出現TextUnit IDリスト
        entity_type: エンティティタイプ（該当する場合）
    """
    id: str
    phrase: str
    frequency: int
    document_ids: list[str] = field(default_factory=list)
    text_unit_ids: list[str] = field(default_factory=list)
    entity_type: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "phrase": self.phrase,
            "frequency": self.frequency,
            "document_ids": self.document_ids,
            "text_unit_ids": self.text_unit_ids,
            "entity_type": self.entity_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NounPhraseNode":
        """辞書から復元"""
        return cls(
            id=data["id"],
            phrase=data["phrase"],
            frequency=data["frequency"],
            document_ids=data.get("document_ids", []),
            text_unit_ids=data.get("text_unit_ids", []),
            entity_type=data.get("entity_type"),
        )


@dataclass
class NounPhraseEdge:
    """名詞句エッジ（共起関係）
    
    Attributes:
        source: ソースノードID
        target: ターゲットノードID
        weight: エッジの重み
        cooccurrence_count: 共起回数
        document_ids: 共起ドキュメントIDリスト
    """
    source: str
    target: str
    weight: float
    cooccurrence_count: int
    document_ids: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "cooccurrence_count": self.cooccurrence_count,
            "document_ids": self.document_ids,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NounPhraseEdge":
        """辞書から復元"""
        return cls(
            source=data["source"],
            target=data["target"],
            weight=data["weight"],
            cooccurrence_count=data["cooccurrence_count"],
            document_ids=data.get("document_ids", []),
        )


@dataclass
class Community:
    """コミュニティ
    
    Attributes:
        id: コミュニティID
        level: 階層レベル
        node_ids: 所属ノードIDリスト
        representative_phrases: 代表的な名詞句（頻度順）
        size: コミュニティサイズ
        internal_edges: 内部エッジ数
        parent_id: 親コミュニティID（階層構造）
    """
    id: str
    level: int
    node_ids: list[str] = field(default_factory=list)
    representative_phrases: list[str] = field(default_factory=list)
    size: int = 0
    internal_edges: int = 0
    parent_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "level": self.level,
            "node_ids": self.node_ids,
            "representative_phrases": self.representative_phrases,
            "size": self.size,
            "internal_edges": self.internal_edges,
            "parent_id": self.parent_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Community":
        """辞書から復元"""
        return cls(
            id=data["id"],
            level=data["level"],
            node_ids=data.get("node_ids", []),
            representative_phrases=data.get("representative_phrases", []),
            size=data.get("size", 0),
            internal_edges=data.get("internal_edges", 0),
            parent_id=data.get("parent_id"),
        )


@runtime_checkable
class GraphBuilderProtocol(Protocol):
    """グラフビルダープロトコル"""
    
    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None:
        """ノードを追加"""
        ...
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """エッジを追加"""
        ...
    
    def build_from_cooccurrence(
        self,
        documents: list[list[str]],
        window_size: int = 5,
    ) -> None:
        """共起関係からグラフを構築"""
        ...
    
    def get_nodes(self) -> list[NounPhraseNode]:
        """全ノードを取得"""
        ...
    
    def get_edges(self) -> list[NounPhraseEdge]:
        """全エッジを取得"""
        ...


@runtime_checkable
class CommunityDetectorProtocol(Protocol):
    """コミュニティ検出プロトコル"""
    
    def detect(
        self,
        graph: "nx.Graph",
        resolution: float = 1.0,
    ) -> list[Community]:
        """コミュニティを検出"""
        ...
    
    def detect_hierarchical(
        self,
        graph: "nx.Graph",
        levels: int = 3,
    ) -> list[list[Community]]:
        """階層的コミュニティを検出"""
        ...


class GraphBuilder(ABC):
    """グラフビルダー基底クラス"""
    
    @abstractmethod
    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None:
        """ノードを追加"""
        ...
    
    @abstractmethod
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """エッジを追加"""
        ...
    
    @abstractmethod
    def build_from_cooccurrence(
        self,
        documents: list[list[str]],
        window_size: int = 5,
    ) -> None:
        """共起関係からグラフを構築"""
        ...
    
    @abstractmethod
    def get_nodes(self) -> list[NounPhraseNode]:
        """全ノードを取得"""
        ...
    
    @abstractmethod
    def get_edges(self) -> list[NounPhraseEdge]:
        """全エッジを取得"""
        ...


class CommunityDetector(ABC):
    """コミュニティ検出基底クラス"""
    
    @abstractmethod
    def detect(
        self,
        graph: "nx.Graph",
        resolution: float = 1.0,
    ) -> list[Community]:
        """コミュニティを検出"""
        ...
    
    @abstractmethod
    def detect_hierarchical(
        self,
        graph: "nx.Graph",
        levels: int = 3,
    ) -> list[list[Community]]:
        """階層的コミュニティを検出"""
        ...
