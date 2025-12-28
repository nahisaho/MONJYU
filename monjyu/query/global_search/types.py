"""GlobalSearch types module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GlobalSearchConfig:
    """グローバル検索設定"""
    
    community_level: int = 1
    top_k_communities: int = 10
    map_reduce_enabled: bool = True
    max_context_tokens: int = 8000
    temperature: float = 0.0
    response_language: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "community_level": self.community_level,
            "top_k_communities": self.top_k_communities,
            "map_reduce_enabled": self.map_reduce_enabled,
            "max_context_tokens": self.max_context_tokens,
            "temperature": self.temperature,
            "response_language": self.response_language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalSearchConfig":
        """辞書から作成"""
        return cls(
            community_level=data.get("community_level", 1),
            top_k_communities=data.get("top_k_communities", 10),
            map_reduce_enabled=data.get("map_reduce_enabled", True),
            max_context_tokens=data.get("max_context_tokens", 8000),
            temperature=data.get("temperature", 0.0),
            response_language=data.get("response_language", "auto"),
        )


@dataclass
class MapResult:
    """Map結果"""
    
    community_id: str
    community_title: str
    partial_answer: str
    relevance_score: float = 0.0
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "community_id": self.community_id,
            "community_title": self.community_title,
            "partial_answer": self.partial_answer,
            "relevance_score": self.relevance_score,
            "tokens_used": self.tokens_used,
        }


@dataclass
class CommunityInfo:
    """コミュニティ情報"""
    
    community_id: str
    title: str
    summary: str
    level: int = 0
    size: int = 0
    key_entities: List[str] = field(default_factory=list)
    findings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "level": self.level,
            "size": self.size,
            "key_entities": self.key_entities,
            "findings": self.findings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunityInfo":
        """辞書から作成"""
        return cls(
            community_id=data.get("community_id", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            level=data.get("level", 0),
            size=data.get("size", 0),
            key_entities=data.get("key_entities", []),
            findings=data.get("findings", []),
        )


@dataclass
class GlobalSearchResult:
    """グローバル検索結果"""
    
    query: str
    answer: str
    communities_used: List[CommunityInfo] = field(default_factory=list)
    map_results: List[MapResult] = field(default_factory=list)
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    community_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query": self.query,
            "answer": self.answer,
            "communities_used": [c.to_dict() for c in self.communities_used],
            "map_results": [m.to_dict() for m in self.map_results],
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "community_level": self.community_level,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalSearchResult":
        """辞書から作成"""
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            communities_used=[
                CommunityInfo(**c) for c in data.get("communities_used", [])
            ],
            map_results=[
                MapResult(**m) for m in data.get("map_results", [])
            ],
            processing_time_ms=data.get("processing_time_ms", 0.0),
            tokens_used=data.get("tokens_used", 0),
            community_level=data.get("community_level", 0),
        )
