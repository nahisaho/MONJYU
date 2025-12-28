# Test Community Detector - FEAT-012
"""
CommunityDetector の単体テスト
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Tuple


class TestCommunityTypes:
    """Community データモデルのテスト"""
    
    def test_community_creation(self):
        """Community作成テスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(
            id="comm-001",
            level=0,
            member_ids=["node-1", "node-2", "node-3"],
        )
        
        assert comm.id == "comm-001"
        assert comm.level == 0
        assert len(comm.member_ids) == 3
        assert comm.size == 3
    
    def test_community_defaults(self):
        """Communityデフォルト値テスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(id="test")
        
        assert comm.level == 0
        assert comm.member_ids == []
        assert comm.parent_id is None
        assert comm.child_ids == []
        assert comm.title is None
        assert comm.summary is None
        assert comm.density == 0.0
        assert comm.modularity == 0.0
    
    def test_community_size_property(self):
        """sizeプロパティテスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(id="test", member_ids=["a", "b", "c", "d"])
        assert comm.size == 4
        
        comm.member_ids = []
        assert comm.size == 0
    
    def test_community_contains(self):
        """containsメソッドテスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(id="test", member_ids=["node-1", "node-2"])
        
        assert comm.contains("node-1") is True
        assert comm.contains("node-2") is True
        assert comm.contains("node-3") is False
    
    def test_community_add_member(self):
        """add_memberメソッドテスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(id="test", member_ids=["node-1"])
        
        comm.add_member("node-2")
        assert "node-2" in comm.member_ids
        assert comm.size == 2
        
        # 重複追加は無視
        comm.add_member("node-2")
        assert comm.size == 2
    
    def test_community_remove_member(self):
        """remove_memberメソッドテスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(id="test", member_ids=["node-1", "node-2"])
        
        comm.remove_member("node-1")
        assert "node-1" not in comm.member_ids
        assert comm.size == 1
        
        # 存在しないメンバー削除は無視
        comm.remove_member("node-999")
        assert comm.size == 1
    
    def test_community_to_dict(self):
        """to_dictメソッドテスト"""
        from monjyu.index.community_detector.types import Community
        
        comm = Community(
            id="comm-001",
            level=1,
            member_ids=["a", "b"],
            title="Test Community",
            density=0.8,
        )
        
        data = comm.to_dict()
        
        assert data["id"] == "comm-001"
        assert data["level"] == 1
        assert data["member_ids"] == ["a", "b"]
        assert data["title"] == "Test Community"
        assert data["density"] == 0.8
        assert data["size"] == 2
    
    def test_community_from_dict(self):
        """from_dictメソッドテスト"""
        from monjyu.index.community_detector.types import Community
        
        data = {
            "id": "comm-001",
            "level": 2,
            "member_ids": ["x", "y", "z"],
            "title": "Restored Community",
            "density": 0.5,
        }
        
        comm = Community.from_dict(data)
        
        assert comm.id == "comm-001"
        assert comm.level == 2
        assert comm.member_ids == ["x", "y", "z"]
        assert comm.title == "Restored Community"


class TestCommunityDetectionResult:
    """CommunityDetectionResult のテスト"""
    
    def test_result_creation(self):
        """Result作成テスト"""
        from monjyu.index.community_detector.types import (
            Community,
            CommunityDetectionResult,
        )
        
        communities = [
            Community(id="c1", member_ids=["a", "b"]),
            Community(id="c2", member_ids=["c", "d", "e"]),
        ]
        
        result = CommunityDetectionResult(
            communities=communities,
            algorithm="louvain",
            resolution=1.0,
            modularity=0.75,
        )
        
        assert result.community_count == 2
        assert result.total_members == 5
        assert result.success is True
    
    def test_result_with_error(self):
        """エラー結果テスト"""
        from monjyu.index.community_detector.types import CommunityDetectionResult
        
        result = CommunityDetectionResult(error="Test error")
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.community_count == 0
    
    def test_get_community_by_id(self):
        """get_community_by_idテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            CommunityDetectionResult,
        )
        
        communities = [
            Community(id="comm-001", member_ids=["a"]),
            Community(id="comm-002", member_ids=["b"]),
        ]
        result = CommunityDetectionResult(communities=communities)
        
        found = result.get_community_by_id("comm-001")
        assert found is not None
        assert found.id == "comm-001"
        
        not_found = result.get_community_by_id("comm-999")
        assert not_found is None
    
    def test_get_communities_at_level(self):
        """get_communities_at_levelテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            CommunityDetectionResult,
        )
        
        communities = [
            Community(id="c1", level=0, member_ids=["a"]),
            Community(id="c2", level=0, member_ids=["b"]),
            Community(id="c3", level=1, member_ids=["c"]),
        ]
        result = CommunityDetectionResult(communities=communities)
        
        level_0 = result.get_communities_at_level(0)
        assert len(level_0) == 2
        
        level_1 = result.get_communities_at_level(1)
        assert len(level_1) == 1
        
        level_2 = result.get_communities_at_level(2)
        assert len(level_2) == 0
    
    def test_result_to_dict(self):
        """to_dictテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            CommunityDetectionResult,
        )
        
        result = CommunityDetectionResult(
            communities=[Community(id="c1", member_ids=["a"])],
            algorithm="leiden",
            modularity=0.6,
        )
        
        data = result.to_dict()
        
        assert data["algorithm"] == "leiden"
        assert data["modularity"] == 0.6
        assert data["community_count"] == 1
        assert len(data["communities"]) == 1


class TestHierarchicalCommunities:
    """HierarchicalCommunities のテスト"""
    
    def test_hierarchical_creation(self):
        """Hierarchical作成テスト"""
        from monjyu.index.community_detector.types import (
            Community,
            HierarchicalCommunities,
        )
        
        hierarchical = HierarchicalCommunities()
        
        assert hierarchical.max_level == 0
        assert hierarchical.total_communities == 0
    
    def test_add_community(self):
        """add_communityテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            HierarchicalCommunities,
        )
        
        hierarchical = HierarchicalCommunities()
        
        hierarchical.add_community(Community(id="c1", level=0, member_ids=["a"]))
        hierarchical.add_community(Community(id="c2", level=0, member_ids=["b"]))
        hierarchical.add_community(Community(id="c3", level=1, member_ids=["a", "b"]))
        
        assert hierarchical.total_communities == 3
        assert hierarchical.max_level == 1
        assert len(hierarchical.get_level(0)) == 2
        assert len(hierarchical.get_level(1)) == 1
    
    def test_get_all_communities(self):
        """get_all_communitiesテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            HierarchicalCommunities,
        )
        
        hierarchical = HierarchicalCommunities()
        hierarchical.add_community(Community(id="c1", level=0))
        hierarchical.add_community(Community(id="c2", level=1))
        
        all_comms = hierarchical.get_all_communities()
        assert len(all_comms) == 2
    
    def test_find_community_for_node(self):
        """find_community_for_nodeテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            HierarchicalCommunities,
        )
        
        hierarchical = HierarchicalCommunities()
        hierarchical.add_community(
            Community(id="c1", level=0, member_ids=["node-1", "node-2"])
        )
        hierarchical.add_community(
            Community(id="c2", level=0, member_ids=["node-3"])
        )
        
        found = hierarchical.find_community_for_node("node-1", level=0)
        assert found is not None
        assert found.id == "c1"
        
        not_found = hierarchical.find_community_for_node("node-999", level=0)
        assert not_found is None
    
    def test_hierarchical_to_dict(self):
        """to_dictテスト"""
        from monjyu.index.community_detector.types import (
            Community,
            HierarchicalCommunities,
        )
        
        hierarchical = HierarchicalCommunities()
        hierarchical.add_community(Community(id="c1", level=0, member_ids=["a"]))
        
        data = hierarchical.to_dict()
        
        assert "levels" in data
        assert data["max_level"] == 0
        assert data["total_communities"] == 1


class TestCommunityDetectorConfig:
    """CommunityDetectorConfig のテスト"""
    
    def test_config_defaults(self):
        """デフォルト設定テスト"""
        from monjyu.index.community_detector.detector import (
            CommunityDetectorConfig,
        )
        
        config = CommunityDetectorConfig()
        
        # autoは利用可能なアルゴリズムに解決される
        assert config.algorithm in ["leiden", "louvain", "none"]
        assert config.default_resolution == 1.0
        assert config.min_community_size == 2
    
    def test_config_custom(self):
        """カスタム設定テスト"""
        from monjyu.index.community_detector.detector import (
            CommunityDetectorConfig,
            HAS_NETWORKX,
        )
        
        if HAS_NETWORKX:
            config = CommunityDetectorConfig(
                algorithm="louvain",
                default_resolution=1.5,
                min_community_size=3,
                seed=42,
            )
            
            assert config.algorithm == "louvain"
            assert config.default_resolution == 1.5
            assert config.min_community_size == 3
            assert config.seed == 42


class TestCommunityDetector:
    """CommunityDetector のテスト"""
    
    def test_detector_creation(self):
        """Detector作成テスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            detector = CommunityDetector()
            assert detector is not None
            assert len(detector.available_algorithms) > 0
    
    def test_available_algorithms(self):
        """利用可能アルゴリズムテスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            detector = CommunityDetector()
            algorithms = detector.available_algorithms
            
            if HAS_LEIDEN:
                assert "leiden" in algorithms
            if HAS_NETWORKX:
                assert "louvain" in algorithms
    
    def test_detect_empty_nodes(self):
        """空ノードでの検出テスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            detector = CommunityDetector()
            result = detector.detect(nodes=[], edges=[])
            
            assert result.success is False
            assert "No nodes" in result.error
    
    def test_detect_no_edges(self):
        """エッジなしでの検出テスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            detector = CommunityDetector()
            nodes = ["A", "B", "C"]
            result = detector.detect(nodes=nodes, edges=[])
            
            # 各ノードが個別コミュニティになる
            assert result.success is True
            assert result.community_count == 3
    
    def test_detect_simple_graph(self):
        """単純グラフでの検出テスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            # 2つの明確なクラスター
            nodes = ["A", "B", "C", "D", "E", "F"]
            edges = [
                ("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0),  # クラスター1
                ("D", "E", 1.0), ("D", "F", 1.0), ("E", "F", 1.0),  # クラスター2
            ]
            
            result = detector.detect(nodes, edges)
            
            assert result.success is True
            # 2つのコミュニティを期待
            assert result.community_count >= 1
            assert result.modularity >= 0
    
    def test_detect_with_resolution(self):
        """解像度パラメータテスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            nodes = ["A", "B", "C", "D"]
            edges = [
                ("A", "B", 1.0), ("B", "C", 1.0), ("C", "D", 1.0),
            ]
            
            # 低解像度（大きなコミュニティ）
            result_low = detector.detect(nodes, edges, resolution=0.5)
            
            # 高解像度（小さなコミュニティ）
            result_high = detector.detect(nodes, edges, resolution=2.0)
            
            assert result_low.success is True
            assert result_high.success is True


class TestCommunityDetectorHierarchical:
    """階層的コミュニティ検出のテスト"""
    
    def test_detect_hierarchical(self):
        """階層的検出テスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            nodes = ["A", "B", "C", "D", "E", "F"]
            edges = [
                ("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0),
                ("D", "E", 1.0), ("D", "F", 1.0), ("E", "F", 1.0),
                ("C", "D", 0.3),  # クラスター間の弱い接続
            ]
            
            hierarchical = detector.detect_hierarchical(
                nodes, edges, levels=2, resolution_range=(0.5, 2.0)
            )
            
            assert hierarchical.total_communities >= 1
            assert hierarchical.max_level >= 0
    
    def test_hierarchical_levels(self):
        """階層レベルテスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            nodes = ["A", "B", "C", "D"]
            edges = [
                ("A", "B", 1.0), ("C", "D", 1.0),
            ]
            
            hierarchical = detector.detect_hierarchical(
                nodes, edges, levels=3, resolution_range=(0.3, 3.0)
            )
            
            # 各レベルにコミュニティが存在
            all_comms = hierarchical.get_all_communities()
            assert len(all_comms) >= 1


class TestCreateDetector:
    """create_detectorファクトリ関数のテスト"""
    
    def test_create_detector_default(self):
        """デフォルト作成テスト"""
        from monjyu.index.community_detector import (
            create_detector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            detector = create_detector()
            assert detector is not None
    
    def test_create_detector_with_params(self):
        """パラメータ指定テスト"""
        from monjyu.index.community_detector import (
            create_detector,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_NETWORKX:
            detector = create_detector(
                algorithm="louvain",
                resolution=1.5,
                min_community_size=3,
                seed=123,
            )
            
            assert detector.config.algorithm == "louvain"
            assert detector.config.default_resolution == 1.5
            assert detector.config.min_community_size == 3
            assert detector.config.seed == 123


class TestIntegration:
    """統合テスト"""
    
    def test_full_workflow(self):
        """完全ワークフローテスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            Community,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            # 1. 検出器作成
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            # 2. グラフデータ
            nodes = ["researcher-1", "researcher-2", "method-1", 
                     "dataset-1", "dataset-2", "metric-1"]
            edges = [
                ("researcher-1", "method-1", 1.0),
                ("researcher-2", "method-1", 1.0),
                ("method-1", "dataset-1", 0.8),
                ("dataset-1", "metric-1", 0.9),
                ("dataset-1", "dataset-2", 0.7),
            ]
            
            # 3. 検出実行
            result = detector.detect(nodes, edges)
            
            assert result.success is True
            assert result.community_count >= 1
            
            # 4. コミュニティアクセス
            for comm in result.communities:
                assert comm.id is not None
                assert comm.size >= 1
            
            # 5. シリアライズ
            data = result.to_dict()
            assert "communities" in data
            assert "algorithm" in data
    
    def test_hierarchical_workflow(self):
        """階層的ワークフローテスト"""
        from monjyu.index.community_detector import (
            CommunityDetector,
            CommunityDetectorConfig,
            HAS_LEIDEN,
            HAS_NETWORKX,
        )
        
        if HAS_LEIDEN or HAS_NETWORKX:
            config = CommunityDetectorConfig(min_community_size=1)
            detector = CommunityDetector(config)
            
            # より大きなグラフ
            nodes = [f"node-{i}" for i in range(10)]
            edges = [
                # グループ1
                ("node-0", "node-1", 1.0),
                ("node-1", "node-2", 1.0),
                ("node-0", "node-2", 1.0),
                # グループ2
                ("node-3", "node-4", 1.0),
                ("node-4", "node-5", 1.0),
                ("node-3", "node-5", 1.0),
                # グループ3
                ("node-6", "node-7", 1.0),
                ("node-7", "node-8", 1.0),
                ("node-8", "node-9", 1.0),
                # グループ間接続
                ("node-2", "node-3", 0.3),
                ("node-5", "node-6", 0.3),
            ]
            
            hierarchical = detector.detect_hierarchical(
                nodes, edges, levels=2
            )
            
            assert hierarchical.total_communities >= 1
            
            # 全コミュニティにアクセス
            all_comms = hierarchical.get_all_communities()
            assert len(all_comms) >= 1
            
            # シリアライズ
            data = hierarchical.to_dict()
            assert "levels" in data
