# Community Detector Types
"""
FEAT-012: CommunityDetector データモデル定義

コミュニティの型定義とデータ構造
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Community:
    """コミュニティ
    
    グラフ内のノードのクラスターを表現。
    
    Attributes:
        id: 一意識別子
        level: 階層レベル（0=最下位）
        member_ids: メンバーノードID
        parent_id: 親コミュニティID
        child_ids: 子コミュニティID
        title: コミュニティのタイトル
        summary: コミュニティの要約
        size: メンバー数
        density: グラフ密度
        metadata: 追加メタデータ
    
    Examples:
        >>> community = Community(
        ...     id="comm-001",
        ...     level=0,
        ...     member_ids=["node-1", "node-2", "node-3"],
        ... )
        >>> community.size
        3
    """
    id: str
    level: int = 0
    member_ids: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    # コンテンツ（後で生成）
    title: Optional[str] = None
    summary: Optional[str] = None
    
    # 統計
    density: float = 0.0
    modularity: float = 0.0
    
    # 追加情報
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def size(self) -> int:
        """メンバー数"""
        return len(self.member_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "level": self.level,
            "member_ids": self.member_ids,
            "parent_id": self.parent_id,
            "child_ids": self.child_ids,
            "title": self.title,
            "summary": self.summary,
            "size": self.size,
            "density": self.density,
            "modularity": self.modularity,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Community":
        """辞書から生成"""
        return cls(
            id=data["id"],
            level=data.get("level", 0),
            member_ids=data.get("member_ids", []),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            title=data.get("title"),
            summary=data.get("summary"),
            density=data.get("density", 0.0),
            modularity=data.get("modularity", 0.0),
            metadata=data.get("metadata", {}),
        )
    
    def contains(self, node_id: str) -> bool:
        """ノードがコミュニティに含まれるか"""
        return node_id in self.member_ids
    
    def add_member(self, node_id: str) -> None:
        """メンバーを追加"""
        if node_id not in self.member_ids:
            self.member_ids.append(node_id)
    
    def remove_member(self, node_id: str) -> None:
        """メンバーを削除"""
        if node_id in self.member_ids:
            self.member_ids.remove(node_id)


@dataclass
class CommunityDetectionResult:
    """コミュニティ検出結果
    
    Attributes:
        communities: 検出されたコミュニティリスト
        algorithm: 使用アルゴリズム
        resolution: 解像度パラメータ
        modularity: 全体モジュラリティ
        detection_time_ms: 検出時間
        error: エラーメッセージ
    """
    communities: List[Community] = field(default_factory=list)
    algorithm: str = "louvain"
    resolution: float = 1.0
    modularity: float = 0.0
    detection_time_ms: float = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """検出が成功したか"""
        return self.error is None
    
    @property
    def community_count(self) -> int:
        """コミュニティ数"""
        return len(self.communities)
    
    @property
    def total_members(self) -> int:
        """総メンバー数"""
        return sum(c.size for c in self.communities)
    
    def get_community_by_id(self, community_id: str) -> Optional[Community]:
        """IDでコミュニティを取得"""
        for c in self.communities:
            if c.id == community_id:
                return c
        return None
    
    def get_communities_at_level(self, level: int) -> List[Community]:
        """特定レベルのコミュニティを取得"""
        return [c for c in self.communities if c.level == level]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "communities": [c.to_dict() for c in self.communities],
            "algorithm": self.algorithm,
            "resolution": self.resolution,
            "modularity": self.modularity,
            "detection_time_ms": self.detection_time_ms,
            "community_count": self.community_count,
            "error": self.error,
        }


@dataclass
class HierarchicalCommunities:
    """階層的コミュニティ構造
    
    複数レベルのコミュニティを管理。
    """
    levels: Dict[int, List[Community]] = field(default_factory=dict)
    max_level: int = 0
    
    def add_community(self, community: Community) -> None:
        """コミュニティを追加"""
        level = community.level
        if level not in self.levels:
            self.levels[level] = []
        self.levels[level].append(community)
        self.max_level = max(self.max_level, level)
    
    def get_level(self, level: int) -> List[Community]:
        """特定レベルのコミュニティを取得"""
        return self.levels.get(level, [])
    
    def get_all_communities(self) -> List[Community]:
        """全コミュニティを取得"""
        all_communities = []
        for level_communities in self.levels.values():
            all_communities.extend(level_communities)
        return all_communities
    
    def find_community_for_node(self, node_id: str, level: int = 0) -> Optional[Community]:
        """ノードが属するコミュニティを検索"""
        for c in self.get_level(level):
            if c.contains(node_id):
                return c
        return None
    
    @property
    def total_communities(self) -> int:
        """総コミュニティ数"""
        return sum(len(cs) for cs in self.levels.values())
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "levels": {
                level: [c.to_dict() for c in communities]
                for level, communities in self.levels.items()
            },
            "max_level": self.max_level,
            "total_communities": self.total_communities,
        }
