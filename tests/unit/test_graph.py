# Graph Unit Tests
"""
Unit tests for graph components.
"""

from __future__ import annotations

import pytest
import networkx as nx

from monjyu.graph.base import NounPhraseNode, NounPhraseEdge, Community
from monjyu.graph.noun_phrase_graph import NounPhraseGraphBuilder
from monjyu.graph.community_detector import LeidenCommunityDetector, LouvainCommunityDetector
from monjyu.nlp.base import NLPFeatures


class TestNounPhraseNode:
    """NounPhraseNodeのテスト"""
    
    def test_create_node(self):
        """ノード作成"""
        node = NounPhraseNode(
            id="np_001",
            phrase="deep learning",
            frequency=10,
            document_ids=["doc_001", "doc_002"],
            text_unit_ids=["tu_001", "tu_002", "tu_003"],
            entity_type="CONCEPT",
        )
        
        assert node.id == "np_001"
        assert node.phrase == "deep learning"
        assert node.frequency == 10
        assert len(node.document_ids) == 2
    
    def test_to_dict(self):
        """辞書変換"""
        node = NounPhraseNode(
            id="np_001",
            phrase="neural network",
            frequency=5,
        )
        
        data = node.to_dict()
        
        assert data["id"] == "np_001"
        assert data["phrase"] == "neural network"
        assert data["frequency"] == 5
    
    def test_from_dict(self):
        """辞書からの復元"""
        data = {
            "id": "np_001",
            "phrase": "attention",
            "frequency": 8,
            "document_ids": ["doc_001"],
            "text_unit_ids": ["tu_001"],
            "entity_type": "CONCEPT",
        }
        
        node = NounPhraseNode.from_dict(data)
        
        assert node.id == "np_001"
        assert node.entity_type == "CONCEPT"


class TestNounPhraseEdge:
    """NounPhraseEdgeのテスト"""
    
    def test_create_edge(self):
        """エッジ作成"""
        edge = NounPhraseEdge(
            source="np_001",
            target="np_002",
            weight=5.0,
            cooccurrence_count=5,
            document_ids=["doc_001"],
        )
        
        assert edge.source == "np_001"
        assert edge.target == "np_002"
        assert edge.weight == 5.0
    
    def test_to_dict(self):
        """辞書変換"""
        edge = NounPhraseEdge(
            source="np_001",
            target="np_002",
            weight=3.0,
            cooccurrence_count=3,
        )
        
        data = edge.to_dict()
        
        assert data["source"] == "np_001"
        assert data["weight"] == 3.0


class TestCommunity:
    """Communityのテスト"""
    
    def test_create_community(self):
        """コミュニティ作成"""
        community = Community(
            id="community_l1_0",
            level=1,
            node_ids=["np_001", "np_002", "np_003"],
            representative_phrases=["deep learning", "neural network"],
            size=3,
            internal_edges=2,
        )
        
        assert community.id == "community_l1_0"
        assert community.level == 1
        assert community.size == 3
    
    def test_to_dict(self):
        """辞書変換"""
        community = Community(
            id="community_l1_0",
            level=1,
            node_ids=["np_001"],
        )
        
        data = community.to_dict()
        
        assert data["id"] == "community_l1_0"
        assert data["level"] == 1


class TestNounPhraseGraphBuilder:
    """NounPhraseGraphBuilderのテスト"""
    
    def test_init(self):
        """初期化"""
        builder = NounPhraseGraphBuilder()
        
        assert builder.min_frequency == 2
        assert builder.window_size == 5
        assert builder.node_count == 0
        assert builder.edge_count == 0
    
    def test_add_node(self):
        """ノード追加"""
        builder = NounPhraseGraphBuilder()
        
        builder.add_node("np_001", {
            "phrase": "deep learning",
            "frequency": 5,
        })
        
        assert builder.node_count == 1
        assert "np_001" in builder.graph
    
    def test_add_edge(self):
        """エッジ追加"""
        builder = NounPhraseGraphBuilder()
        
        builder.add_node("np_001", {"phrase": "deep learning"})
        builder.add_node("np_002", {"phrase": "neural network"})
        builder.add_edge("np_001", "np_002", weight=3.0)
        
        assert builder.edge_count == 1
        assert builder.graph.has_edge("np_001", "np_002")
    
    def test_build_from_cooccurrence(self):
        """共起からグラフ構築"""
        builder = NounPhraseGraphBuilder(min_frequency=1)
        
        documents = [
            ["deep learning", "neural network", "attention"],
            ["neural network", "transformer", "attention"],
            ["deep learning", "transformer"],
        ]
        
        builder.build_from_cooccurrence(documents, window_size=5)
        
        assert builder.node_count >= 3
        assert builder.edge_count >= 1
    
    def test_build_from_features(self):
        """NLP特徴量からグラフ構築"""
        builder = NounPhraseGraphBuilder(min_frequency=1)
        
        # モックTextUnit
        class MockTextUnit:
            def __init__(self, id, document_id):
                self.id = id
                self.document_id = document_id
        
        features = [
            NLPFeatures(
                text_unit_id="tu_001",
                noun_phrases=["deep learning", "neural network"],
                entities=[("BERT", "MODEL")],
            ),
            NLPFeatures(
                text_unit_id="tu_002",
                noun_phrases=["neural network", "attention mechanism"],
            ),
        ]
        
        text_units = [
            MockTextUnit("tu_001", "doc_001"),
            MockTextUnit("tu_002", "doc_001"),
        ]
        
        nodes, edges = builder.build_from_features(features, text_units)
        
        assert len(nodes) >= 2
        # neural networkが両方に出現するので共起エッジができる
    
    def test_get_networkx_graph(self):
        """NetworkXグラフ取得"""
        builder = NounPhraseGraphBuilder()
        
        builder.add_node("np_001", {"phrase": "test"})
        
        graph = builder.get_networkx_graph()
        
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 1
    
    def test_clear(self):
        """クリア"""
        builder = NounPhraseGraphBuilder()
        
        builder.add_node("np_001", {"phrase": "test"})
        builder.clear()
        
        assert builder.node_count == 0


class TestLeidenCommunityDetector:
    """LeidenCommunityDetectorのテスト"""
    
    def test_init(self):
        """初期化"""
        detector = LeidenCommunityDetector(resolution=1.0)
        
        assert detector.resolution == 1.0
    
    def test_detect_empty_graph(self):
        """空グラフでの検出"""
        detector = LeidenCommunityDetector()
        graph = nx.Graph()
        
        communities = detector.detect(graph)
        
        assert communities == []
    
    def test_detect_simple_graph(self):
        """シンプルグラフでの検出"""
        detector = LeidenCommunityDetector()
        
        # 2つのクリークを持つグラフ
        graph = nx.Graph()
        # クリーク1
        for i in range(3):
            graph.add_node(f"np_{i}", phrase=f"phrase_{i}", frequency=1)
        graph.add_edge("np_0", "np_1", weight=1)
        graph.add_edge("np_1", "np_2", weight=1)
        graph.add_edge("np_0", "np_2", weight=1)
        
        # クリーク2
        for i in range(3, 6):
            graph.add_node(f"np_{i}", phrase=f"phrase_{i}", frequency=1)
        graph.add_edge("np_3", "np_4", weight=1)
        graph.add_edge("np_4", "np_5", weight=1)
        graph.add_edge("np_3", "np_5", weight=1)
        
        # 弱い接続
        graph.add_edge("np_2", "np_3", weight=0.1)
        
        communities = detector.detect(graph)
        
        assert len(communities) >= 1
        assert all(isinstance(c, Community) for c in communities)
    
    def test_detect_hierarchical(self):
        """階層的コミュニティ検出"""
        detector = LeidenCommunityDetector()
        
        graph = nx.Graph()
        for i in range(10):
            graph.add_node(f"np_{i}", phrase=f"phrase_{i}", frequency=1)
            if i > 0:
                graph.add_edge(f"np_{i-1}", f"np_{i}", weight=1)
        
        all_levels = detector.detect_hierarchical(graph, levels=2)
        
        assert len(all_levels) == 2
        assert all(isinstance(level, list) for level in all_levels)


class TestLouvainCommunityDetector:
    """LouvainCommunityDetectorのテスト"""
    
    def test_detect(self):
        """コミュニティ検出"""
        detector = LouvainCommunityDetector()
        
        graph = nx.Graph()
        for i in range(5):
            graph.add_node(f"np_{i}", phrase=f"phrase_{i}", frequency=1)
        graph.add_edge("np_0", "np_1", weight=1)
        graph.add_edge("np_1", "np_2", weight=1)
        graph.add_edge("np_3", "np_4", weight=1)
        
        communities = detector.detect(graph)
        
        assert len(communities) >= 1
