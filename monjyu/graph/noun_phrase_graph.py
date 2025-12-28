# Noun Phrase Graph Builder
"""
Builder for noun phrase co-occurrence graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import networkx as nx

from monjyu.graph.base import (
    GraphBuilder,
    NounPhraseNode,
    NounPhraseEdge,
)
from monjyu.nlp.base import NLPFeatures

if TYPE_CHECKING:
    pass


class NounPhraseGraphBuilder(GraphBuilder):
    """名詞句グラフビルダー
    
    名詞句の共起関係からグラフを構築する。
    
    Example:
        >>> builder = NounPhraseGraphBuilder()
        >>> nodes, edges = builder.build_from_features(nlp_features, text_units)
        >>> graph = builder.get_networkx_graph()
    
    Attributes:
        min_frequency: 最小出現頻度
        window_size: 共起ウィンドウサイズ
    """
    
    def __init__(
        self,
        min_frequency: int = 2,
        window_size: int = 5,
    ) -> None:
        """初期化
        
        Args:
            min_frequency: 最小出現頻度（これ未満の名詞句は除外）
            window_size: 共起ウィンドウサイズ
        """
        self.min_frequency = min_frequency
        self.window_size = window_size
        
        self.graph = nx.Graph()
        self._nodes: dict[str, NounPhraseNode] = {}
        self._edges: dict[tuple[str, str], NounPhraseEdge] = {}
        self._phrase_to_id: dict[str, str] = {}
    
    def add_node(self, node_id: str, attributes: dict[str, Any]) -> None:
        """ノードを追加
        
        Args:
            node_id: ノードID
            attributes: ノード属性
        """
        self.graph.add_node(node_id, **attributes)
        
        # NounPhraseNodeも更新
        if node_id not in self._nodes:
            self._nodes[node_id] = NounPhraseNode(
                id=node_id,
                phrase=attributes.get("phrase", ""),
                frequency=attributes.get("frequency", 0),
                document_ids=attributes.get("document_ids", []),
                text_unit_ids=attributes.get("text_unit_ids", []),
                entity_type=attributes.get("entity_type"),
            )
    
    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 1.0,
    ) -> None:
        """エッジを追加
        
        Args:
            source: ソースノードID
            target: ターゲットノードID
            weight: エッジの重み
        """
        self.graph.add_edge(source, target, weight=weight)
        
        # エッジキーを正規化（小さい方を先に）
        edge_key = tuple(sorted([source, target]))
        
        if edge_key not in self._edges:
            self._edges[edge_key] = NounPhraseEdge(
                source=edge_key[0],
                target=edge_key[1],
                weight=weight,
                cooccurrence_count=int(weight),
                document_ids=[],
            )
        else:
            # 重みを更新
            self._edges[edge_key].weight = weight
            self._edges[edge_key].cooccurrence_count = int(weight)
    
    def build_from_cooccurrence(
        self,
        documents: list[list[str]],
        window_size: int = 5,
    ) -> None:
        """共起関係からグラフを構築
        
        Args:
            documents: 文書ごとの名詞句リスト
            window_size: 共起ウィンドウサイズ
        """
        # 名詞句の頻度を集計
        phrase_freq = defaultdict(int)
        for doc in documents:
            for phrase in doc:
                phrase_freq[phrase] += 1
        
        # 低頻度をフィルタリング
        valid_phrases = {
            p for p, f in phrase_freq.items()
            if f >= self.min_frequency
        }
        
        # ノード作成
        for i, phrase in enumerate(sorted(valid_phrases)):
            node_id = f"np_{i}"
            self._phrase_to_id[phrase] = node_id
            
            self.add_node(node_id, {
                "phrase": phrase,
                "frequency": phrase_freq[phrase],
            })
        
        # 共起をカウント
        cooccurrence = defaultdict(int)
        for doc in documents:
            filtered = [p for p in doc if p in valid_phrases]
            for i, p1 in enumerate(filtered):
                for j in range(i + 1, min(i + window_size + 1, len(filtered))):
                    p2 = filtered[j]
                    if p1 != p2:
                        key = tuple(sorted([p1, p2]))
                        cooccurrence[key] += 1
        
        # エッジ作成
        for (p1, p2), count in cooccurrence.items():
            source_id = self._phrase_to_id[p1]
            target_id = self._phrase_to_id[p2]
            self.add_edge(source_id, target_id, weight=count)
    
    def build_from_features(
        self,
        nlp_features_list: list[NLPFeatures],
        text_units: list[Any],
        window_size: int | None = None,
    ) -> tuple[list[NounPhraseNode], list[NounPhraseEdge]]:
        """NLP特徴量から名詞句グラフを構築
        
        Args:
            nlp_features_list: NLP特徴量リスト
            text_units: TextUnitリスト（document_id取得用）
            window_size: 共起ウィンドウサイズ
            
        Returns:
            (ノードリスト, エッジリスト)
        """
        window_size = window_size or self.window_size
        
        # 1. 名詞句の頻度と出現場所を集計
        phrase_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "frequency": 0,
                "document_ids": set(),
                "text_unit_ids": set(),
                "entity_type": None,
            }
        )
        
        for features, tu in zip(nlp_features_list, text_units, strict=True):
            doc_id = getattr(tu, "document_id", None) or ""
            tu_id = getattr(tu, "id", "") or features.text_unit_id
            
            for phrase in features.noun_phrases:
                phrase_stats[phrase]["frequency"] += 1
                phrase_stats[phrase]["document_ids"].add(doc_id)
                phrase_stats[phrase]["text_unit_ids"].add(tu_id)
            
            # エンティティタイプを記録
            for entity, entity_type in features.entities:
                entity_lower = entity.lower()
                if entity_lower in phrase_stats:
                    phrase_stats[entity_lower]["entity_type"] = entity_type
        
        # 2. 低頻度の名詞句をフィルタリング
        filtered_phrases = {
            phrase: stats
            for phrase, stats in phrase_stats.items()
            if stats["frequency"] >= self.min_frequency
        }
        
        # 3. ノード作成
        for i, (phrase, stats) in enumerate(sorted(filtered_phrases.items())):
            node_id = f"np_{i}"
            self._phrase_to_id[phrase] = node_id
            
            node = NounPhraseNode(
                id=node_id,
                phrase=phrase,
                frequency=stats["frequency"],
                document_ids=list(stats["document_ids"]),
                text_unit_ids=list(stats["text_unit_ids"]),
                entity_type=stats["entity_type"],
            )
            self._nodes[node_id] = node
            
            self.graph.add_node(
                node_id,
                phrase=phrase,
                frequency=stats["frequency"],
                document_ids=list(stats["document_ids"]),
                text_unit_ids=list(stats["text_unit_ids"]),
                entity_type=stats["entity_type"],
            )
        
        # 4. 共起関係からエッジ作成
        cooccurrence: dict[tuple[str, str], dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "document_ids": set()}
        )
        
        for features, tu in zip(nlp_features_list, text_units, strict=True):
            doc_id = getattr(tu, "document_id", None) or ""
            
            # このTextUnit内の有効な名詞句
            phrases_in_unit = [
                p for p in features.noun_phrases
                if p in self._phrase_to_id
            ]
            
            # ウィンドウ内の共起をカウント
            for i, p1 in enumerate(phrases_in_unit):
                for j in range(i + 1, min(i + window_size + 1, len(phrases_in_unit))):
                    p2 = phrases_in_unit[j]
                    if p1 != p2:
                        key = tuple(sorted([p1, p2]))
                        cooccurrence[key]["count"] += 1
                        cooccurrence[key]["document_ids"].add(doc_id)
        
        # 5. エッジ作成
        for (p1, p2), stats in cooccurrence.items():
            if stats["count"] >= 1:
                source_id = self._phrase_to_id[p1]
                target_id = self._phrase_to_id[p2]
                weight = float(stats["count"])
                
                edge = NounPhraseEdge(
                    source=source_id,
                    target=target_id,
                    weight=weight,
                    cooccurrence_count=stats["count"],
                    document_ids=list(stats["document_ids"]),
                )
                
                edge_key = tuple(sorted([source_id, target_id]))
                self._edges[edge_key] = edge
                
                self.graph.add_edge(source_id, target_id, weight=weight)
        
        return list(self._nodes.values()), list(self._edges.values())
    
    def get_nodes(self) -> list[NounPhraseNode]:
        """全ノードを取得"""
        return list(self._nodes.values())
    
    def get_edges(self) -> list[NounPhraseEdge]:
        """全エッジを取得"""
        return list(self._edges.values())
    
    def get_networkx_graph(self) -> nx.Graph:
        """NetworkXグラフを取得"""
        return self.graph
    
    def get_node_by_phrase(self, phrase: str) -> NounPhraseNode | None:
        """名詞句でノードを検索"""
        node_id = self._phrase_to_id.get(phrase)
        if node_id:
            return self._nodes.get(node_id)
        return None
    
    def get_neighbors(self, node_id: str) -> list[NounPhraseNode]:
        """隣接ノードを取得"""
        if node_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor_id in self.graph.neighbors(node_id):
            if neighbor_id in self._nodes:
                neighbors.append(self._nodes[neighbor_id])
        
        return neighbors
    
    @property
    def node_count(self) -> int:
        """ノード数"""
        return len(self._nodes)
    
    @property
    def edge_count(self) -> int:
        """エッジ数"""
        return len(self._edges)
    
    def clear(self) -> None:
        """グラフをクリア"""
        self.graph.clear()
        self._nodes.clear()
        self._edges.clear()
        self._phrase_to_id.clear()
