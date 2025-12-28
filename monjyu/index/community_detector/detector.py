# Community Detector Implementation
"""
FEAT-012: CommunityDetector 実装

Leiden/Louvain アルゴリズムによるコミュニティ検出
"""

import time
import uuid
from typing import Protocol, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import leidenalg
    import igraph as ig
    HAS_LEIDEN = True
except ImportError:
    HAS_LEIDEN = False

from .types import Community, CommunityDetectionResult, HierarchicalCommunities


class CommunityDetectorProtocol(Protocol):
    """コミュニティ検出プロトコル"""
    
    def detect(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        resolution: float = 1.0,
    ) -> CommunityDetectionResult:
        """コミュニティを検出
        
        Args:
            nodes: ノードIDリスト
            edges: エッジリスト (source, target, weight)
            resolution: 解像度パラメータ（大きいほど細かいコミュニティ）
            
        Returns:
            検出結果
        """
        ...
    
    def detect_hierarchical(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        levels: int = 2,
        resolution_range: Tuple[float, float] = (0.5, 2.0),
    ) -> HierarchicalCommunities:
        """階層的コミュニティを検出
        
        Args:
            nodes: ノードIDリスト
            edges: エッジリスト
            levels: 階層レベル数
            resolution_range: 解像度の範囲
            
        Returns:
            階層的コミュニティ構造
        """
        ...


@dataclass
class CommunityDetectorConfig:
    """コミュニティ検出設定"""
    algorithm: str = "auto"  # "leiden", "louvain", "auto"
    default_resolution: float = 1.0
    min_community_size: int = 2
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.algorithm == "auto":
            if HAS_LEIDEN:
                self.algorithm = "leiden"
            elif HAS_NETWORKX:
                self.algorithm = "louvain"
            else:
                self.algorithm = "none"


class CommunityDetector:
    """コミュニティ検出器
    
    Leiden（優先）またはLouvain（フォールバック）アルゴリズムを使用。
    
    Examples:
        >>> detector = CommunityDetector()
        >>> nodes = ["A", "B", "C", "D", "E"]
        >>> edges = [("A", "B", 1.0), ("A", "C", 1.0), ("B", "C", 1.0),
        ...          ("D", "E", 1.0)]
        >>> result = detector.detect(nodes, edges)
        >>> result.community_count
        2
    """
    
    def __init__(self, config: Optional[CommunityDetectorConfig] = None):
        self.config = config or CommunityDetectorConfig()
        self._validate_dependencies()
    
    def _validate_dependencies(self) -> None:
        """依存ライブラリを検証"""
        if self.config.algorithm == "leiden" and not HAS_LEIDEN:
            raise ImportError(
                "leidenalg and igraph required for Leiden algorithm. "
                "Install with: pip install leidenalg python-igraph"
            )
        if self.config.algorithm == "louvain" and not HAS_NETWORKX:
            raise ImportError(
                "networkx required for Louvain algorithm. "
                "Install with: pip install networkx"
            )
        if self.config.algorithm == "none":
            raise ImportError(
                "No community detection library available. "
                "Install leidenalg or networkx."
            )
    
    def detect(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        resolution: Optional[float] = None,
    ) -> CommunityDetectionResult:
        """コミュニティを検出
        
        Args:
            nodes: ノードIDリスト
            edges: エッジリスト (source, target, weight)
            resolution: 解像度パラメータ
            
        Returns:
            検出結果
        """
        start_time = time.time()
        resolution = resolution or self.config.default_resolution
        
        if not nodes:
            return CommunityDetectionResult(
                algorithm=self.config.algorithm,
                resolution=resolution,
                error="No nodes provided"
            )
        
        if not edges:
            # エッジがない場合、各ノードを個別コミュニティに
            communities = [
                Community(
                    id=f"comm-{i}",
                    level=0,
                    member_ids=[node],
                )
                for i, node in enumerate(nodes)
            ]
            return CommunityDetectionResult(
                communities=communities,
                algorithm=self.config.algorithm,
                resolution=resolution,
                detection_time_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            if self.config.algorithm == "leiden":
                communities, modularity = self._detect_leiden(
                    nodes, edges, resolution
                )
            else:  # louvain
                communities, modularity = self._detect_louvain(
                    nodes, edges, resolution
                )
            
            # 小さすぎるコミュニティをフィルタ
            communities = [
                c for c in communities
                if c.size >= self.config.min_community_size
            ]
            
            detection_time = (time.time() - start_time) * 1000
            
            return CommunityDetectionResult(
                communities=communities,
                algorithm=self.config.algorithm,
                resolution=resolution,
                modularity=modularity,
                detection_time_ms=detection_time,
            )
            
        except Exception as e:
            return CommunityDetectionResult(
                algorithm=self.config.algorithm,
                resolution=resolution,
                error=str(e),
            )
    
    def _detect_leiden(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        resolution: float,
    ) -> Tuple[List[Community], float]:
        """Leidenアルゴリズムで検出"""
        # ノードIDマッピング
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        idx_to_node = {i: node for node, i in node_to_idx.items()}
        
        # igraphグラフ作成
        g = ig.Graph()
        g.add_vertices(len(nodes))
        
        edge_list = []
        weights = []
        for source, target, weight in edges:
            if source in node_to_idx and target in node_to_idx:
                edge_list.append((node_to_idx[source], node_to_idx[target]))
                weights.append(weight)
        
        g.add_edges(edge_list)
        g.es["weight"] = weights
        
        # Leidenアルゴリズム実行
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=self.config.seed,
        )
        
        modularity = partition.modularity
        
        # コミュニティ構築
        communities = []
        for i, members in enumerate(partition):
            member_ids = [idx_to_node[idx] for idx in members]
            comm = Community(
                id=f"comm-{uuid.uuid4().hex[:8]}",
                level=0,
                member_ids=member_ids,
                modularity=modularity,
            )
            # 密度計算
            if len(member_ids) > 1:
                subgraph = g.subgraph(members)
                max_edges = len(member_ids) * (len(member_ids) - 1) / 2
                comm.density = subgraph.ecount() / max_edges if max_edges > 0 else 0
            communities.append(comm)
        
        return communities, modularity
    
    def _detect_louvain(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        resolution: float,
    ) -> Tuple[List[Community], float]:
        """Louvainアルゴリズムで検出"""
        # NetworkXグラフ作成
        G = nx.Graph()
        G.add_nodes_from(nodes)
        for source, target, weight in edges:
            if source in nodes and target in nodes:
                G.add_edge(source, target, weight=weight)
        
        # Louvainアルゴリズム実行
        from networkx.algorithms.community import louvain_communities
        
        partition = louvain_communities(
            G,
            weight="weight",
            resolution=resolution,
            seed=self.config.seed,
        )
        
        # モジュラリティ計算
        from networkx.algorithms.community import modularity
        modularity_value = modularity(G, partition, weight="weight")
        
        # コミュニティ構築
        communities = []
        for i, members in enumerate(partition):
            member_ids = list(members)
            comm = Community(
                id=f"comm-{uuid.uuid4().hex[:8]}",
                level=0,
                member_ids=member_ids,
                modularity=modularity_value,
            )
            # 密度計算
            if len(member_ids) > 1:
                subgraph = G.subgraph(member_ids)
                comm.density = nx.density(subgraph)
            communities.append(comm)
        
        return communities, modularity_value
    
    def detect_hierarchical(
        self,
        nodes: List[str],
        edges: List[Tuple[str, str, float]],
        levels: int = 2,
        resolution_range: Tuple[float, float] = (0.5, 2.0),
    ) -> HierarchicalCommunities:
        """階層的コミュニティを検出
        
        異なる解像度で検出し、階層構造を構築。
        
        Args:
            nodes: ノードIDリスト
            edges: エッジリスト
            levels: 階層レベル数
            resolution_range: 解像度の範囲 (min, max)
            
        Returns:
            階層的コミュニティ構造
        """
        hierarchical = HierarchicalCommunities()
        
        # レベルごとの解像度を計算
        min_res, max_res = resolution_range
        if levels == 1:
            resolutions = [(min_res + max_res) / 2]
        else:
            step = (max_res - min_res) / (levels - 1)
            resolutions = [min_res + i * step for i in range(levels)]
        
        prev_level_communities: Optional[List[Community]] = None
        
        for level, resolution in enumerate(resolutions):
            result = self.detect(nodes, edges, resolution)
            
            if not result.success:
                continue
            
            for comm in result.communities:
                comm.level = level
                comm.id = f"L{level}-{comm.id}"
                
                # 親子関係を設定
                if prev_level_communities is not None:
                    self._establish_hierarchy(comm, prev_level_communities)
                
                hierarchical.add_community(comm)
            
            prev_level_communities = result.communities
        
        return hierarchical
    
    def _establish_hierarchy(
        self,
        child: Community,
        parents: List[Community]
    ) -> None:
        """親子関係を確立"""
        # 最も重複が大きい親を探す
        best_parent = None
        best_overlap = 0
        
        for parent in parents:
            overlap = len(set(child.member_ids) & set(parent.member_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best_parent = parent
        
        if best_parent and best_overlap > 0:
            child.parent_id = best_parent.id
            best_parent.child_ids.append(child.id)
    
    @property
    def available_algorithms(self) -> List[str]:
        """利用可能なアルゴリズム"""
        algorithms = []
        if HAS_LEIDEN:
            algorithms.append("leiden")
        if HAS_NETWORKX:
            algorithms.append("louvain")
        return algorithms


def create_detector(
    algorithm: str = "auto",
    resolution: float = 1.0,
    min_community_size: int = 2,
    seed: Optional[int] = None,
) -> CommunityDetector:
    """CommunityDetectorを作成するファクトリ関数"""
    config = CommunityDetectorConfig(
        algorithm=algorithm,
        default_resolution=resolution,
        min_community_size=min_community_size,
        seed=seed,
    )
    return CommunityDetector(config)
