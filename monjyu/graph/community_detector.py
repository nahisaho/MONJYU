# Community Detector
"""
Community detection using Leiden algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from monjyu.graph.base import Community, CommunityDetector

if TYPE_CHECKING:
    import networkx as nx


class LeidenCommunityDetector(CommunityDetector):
    """Leidenアルゴリズムによるコミュニティ検出
    
    Leidenアルゴリズムを使用してグラフのコミュニティ構造を検出する。
    Louvainアルゴリズムより高品質なコミュニティを検出可能。
    
    Example:
        >>> detector = LeidenCommunityDetector(resolution=1.0)
        >>> communities = detector.detect(graph)
        >>> print(f"Found {len(communities)} communities")
    
    Attributes:
        resolution: 解像度パラメータ（大きいほど小さなコミュニティ）
        seed: 乱数シード
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """初期化
        
        Args:
            resolution: 解像度パラメータ
            seed: 乱数シード
        """
        self.resolution = resolution
        self.seed = seed
    
    def detect(
        self,
        graph: "nx.Graph",
        resolution: float | None = None,
    ) -> list[Community]:
        """コミュニティを検出
        
        Args:
            graph: NetworkXグラフ
            resolution: 解像度パラメータ（指定なしでデフォルト使用）
            
        Returns:
            コミュニティリスト
        """
        resolution = resolution or self.resolution
        
        if graph.number_of_nodes() == 0:
            return []
        
        try:
            # leidenalgを使用
            return self._detect_with_leidenalg(graph, resolution)
        except ImportError:
            # フォールバック: NetworkXのLouvainを使用
            return self._detect_with_louvain(graph, resolution)
    
    def _detect_with_leidenalg(
        self,
        graph: "nx.Graph",
        resolution: float,
    ) -> list[Community]:
        """leidenalgを使用してコミュニティを検出"""
        import igraph as ig
        import leidenalg as la
        
        # NetworkX → igraph 変換
        ig_graph = ig.Graph.from_networkx(graph)
        
        # 重みを取得
        weights = None
        if "weight" in graph.edges(data=True):
            weights = ig_graph.es["weight"] if "weight" in ig_graph.es.attributes() else None
        
        # Leiden アルゴリズム実行
        partition = la.find_partition(
            ig_graph,
            la.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights=weights,
            seed=self.seed,
        )
        
        # Community オブジェクト作成
        communities = []
        for i, community_indices in enumerate(partition):
            node_ids = [ig_graph.vs[n]["_nx_name"] for n in community_indices]
            
            representative_phrases = self._get_representative_phrases(
                graph, node_ids, top_k=5
            )
            
            communities.append(Community(
                id=f"community_l1_{i}",
                level=1,
                node_ids=node_ids,
                representative_phrases=representative_phrases,
                size=len(node_ids),
                internal_edges=self._count_internal_edges(graph, node_ids),
            ))
        
        return communities
    
    def _detect_with_louvain(
        self,
        graph: "nx.Graph",
        resolution: float,
    ) -> list[Community]:
        """NetworkXのLouvainアルゴリズムを使用（フォールバック）"""
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
        
        # Louvain実行
        partition = louvain_communities(
            graph,
            resolution=resolution,
            seed=self.seed,
        )
        
        # Community オブジェクト作成
        communities = []
        for i, community_nodes in enumerate(partition):
            node_ids = list(community_nodes)
            
            representative_phrases = self._get_representative_phrases(
                graph, node_ids, top_k=5
            )
            
            communities.append(Community(
                id=f"community_l1_{i}",
                level=1,
                node_ids=node_ids,
                representative_phrases=representative_phrases,
                size=len(node_ids),
                internal_edges=self._count_internal_edges(graph, node_ids),
            ))
        
        return communities
    
    def detect_hierarchical(
        self,
        graph: "nx.Graph",
        levels: int = 3,
        resolution_multiplier: float = 2.0,
    ) -> list[list[Community]]:
        """階層的コミュニティを検出
        
        異なる解像度で複数回検出し、階層構造を構築する。
        
        Args:
            graph: NetworkXグラフ
            levels: 階層数
            resolution_multiplier: 解像度の倍率
            
        Returns:
            レベルごとのコミュニティリスト
        """
        all_levels: list[list[Community]] = []
        current_resolution = self.resolution
        
        for level in range(levels):
            communities = self.detect(graph, current_resolution)
            
            # レベル情報を更新
            for comm in communities:
                comm.level = level
                comm.id = f"community_l1_lv{level}_{comm.id.split('_')[-1]}"
            
            all_levels.append(communities)
            
            # 次のレベルは粗い粒度（高い解像度）
            current_resolution *= resolution_multiplier
        
        # 親子関係を設定
        self._set_parent_relationships(all_levels)
        
        return all_levels
    
    def _get_representative_phrases(
        self,
        graph: "nx.Graph",
        node_ids: list[str],
        top_k: int = 5,
    ) -> list[str]:
        """代表的な名詞句を取得（頻度順）
        
        Args:
            graph: NetworkXグラフ
            node_ids: ノードIDリスト
            top_k: 上位k件
            
        Returns:
            代表的な名詞句リスト
        """
        phrases = []
        for node_id in node_ids:
            if node_id in graph.nodes:
                phrase = graph.nodes[node_id].get("phrase", "")
                frequency = graph.nodes[node_id].get("frequency", 0)
                if phrase:
                    phrases.append((phrase, frequency))
        
        # 頻度でソート
        phrases.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in phrases[:top_k]]
    
    def _count_internal_edges(
        self,
        graph: "nx.Graph",
        node_ids: list[str],
    ) -> int:
        """コミュニティ内のエッジ数をカウント
        
        Args:
            graph: NetworkXグラフ
            node_ids: ノードIDリスト
            
        Returns:
            内部エッジ数
        """
        node_set = set(node_ids)
        count = 0
        
        for u, v in graph.edges():
            if u in node_set and v in node_set:
                count += 1
        
        return count
    
    def _set_parent_relationships(
        self,
        all_levels: list[list[Community]],
    ) -> None:
        """階層間の親子関係を設定
        
        Args:
            all_levels: レベルごとのコミュニティリスト
        """
        for level_idx in range(len(all_levels) - 1):
            child_communities = all_levels[level_idx]
            parent_communities = all_levels[level_idx + 1]
            
            for child in child_communities:
                child_nodes = set(child.node_ids)
                
                # 最も重複が大きい親を見つける
                best_parent = None
                best_overlap = 0
                
                for parent in parent_communities:
                    parent_nodes = set(parent.node_ids)
                    overlap = len(child_nodes & parent_nodes)
                    
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_parent = parent
                
                if best_parent:
                    child.parent_id = best_parent.id


class LouvainCommunityDetector(CommunityDetector):
    """Louvainアルゴリズムによるコミュニティ検出
    
    NetworkXの組み込みLouvainアルゴリズムを使用。
    leidenalgが利用できない環境向けの代替実装。
    """
    
    def __init__(
        self,
        resolution: float = 1.0,
        seed: int | None = None,
    ) -> None:
        """初期化"""
        self.resolution = resolution
        self.seed = seed
    
    def detect(
        self,
        graph: "nx.Graph",
        resolution: float | None = None,
    ) -> list[Community]:
        """コミュニティを検出"""
        from networkx.algorithms.community import louvain_communities
        
        resolution = resolution or self.resolution
        
        if graph.number_of_nodes() == 0:
            return []
        
        partition = louvain_communities(
            graph,
            resolution=resolution,
            seed=self.seed,
        )
        
        communities = []
        for i, community_nodes in enumerate(partition):
            node_ids = list(community_nodes)
            
            phrases = []
            for node_id in node_ids:
                if node_id in graph.nodes:
                    phrase = graph.nodes[node_id].get("phrase", "")
                    frequency = graph.nodes[node_id].get("frequency", 0)
                    if phrase:
                        phrases.append((phrase, frequency))
            
            phrases.sort(key=lambda x: x[1], reverse=True)
            representative_phrases = [p[0] for p in phrases[:5]]
            
            communities.append(Community(
                id=f"community_l1_{i}",
                level=1,
                node_ids=node_ids,
                representative_phrases=representative_phrases,
                size=len(node_ids),
            ))
        
        return communities
    
    def detect_hierarchical(
        self,
        graph: "nx.Graph",
        levels: int = 3,
    ) -> list[list[Community]]:
        """階層的コミュニティを検出"""
        all_levels: list[list[Community]] = []
        current_resolution = self.resolution
        
        for level in range(levels):
            communities = self.detect(graph, current_resolution)
            
            for comm in communities:
                comm.level = level
            
            all_levels.append(communities)
            current_resolution *= 2.0
        
        return all_levels
