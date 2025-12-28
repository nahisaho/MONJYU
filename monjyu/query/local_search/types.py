"""LocalSearch types module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LocalSearchConfig:
    """ローカル検索設定"""
    
    max_hops: int = 2
    top_k_entities: int = 10
    top_k_chunks: int = 20
    include_relationships: bool = True
    max_context_tokens: int = 8000
    temperature: float = 0.0
    response_language: str = "auto"
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "max_hops": self.max_hops,
            "top_k_entities": self.top_k_entities,
            "top_k_chunks": self.top_k_chunks,
            "include_relationships": self.include_relationships,
            "max_context_tokens": self.max_context_tokens,
            "temperature": self.temperature,
            "response_language": self.response_language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalSearchConfig":
        """辞書から作成"""
        return cls(
            max_hops=data.get("max_hops", 2),
            top_k_entities=data.get("top_k_entities", 10),
            top_k_chunks=data.get("top_k_chunks", 20),
            include_relationships=data.get("include_relationships", True),
            max_context_tokens=data.get("max_context_tokens", 8000),
            temperature=data.get("temperature", 0.0),
            response_language=data.get("response_language", "auto"),
        )


@dataclass
class EntityInfo:
    """エンティティ情報"""
    
    entity_id: str
    name: str
    entity_type: str
    description: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityInfo":
        """辞書から作成"""
        return cls(
            entity_id=data.get("entity_id", ""),
            name=data.get("name", ""),
            entity_type=data.get("entity_type", ""),
            description=data.get("description", ""),
            properties=data.get("properties", {}),
        )


@dataclass
class RelationshipInfo:
    """リレーションシップ情報"""
    
    relationship_id: str
    source_id: str
    target_id: str
    relation_type: str
    description: str = ""
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "relationship_id": self.relationship_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "description": self.description,
            "weight": self.weight,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipInfo":
        """辞書から作成"""
        return cls(
            relationship_id=data.get("relationship_id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            relation_type=data.get("relation_type", ""),
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            properties=data.get("properties", {}),
        )


@dataclass
class ChunkInfo:
    """チャンク情報"""
    
    chunk_id: str
    content: str
    paper_id: str = ""
    paper_title: str = ""
    section_type: str = ""
    relevance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section_type": self.section_type,
            "relevance_score": self.relevance_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkInfo":
        """辞書から作成"""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            content=data.get("content", ""),
            paper_id=data.get("paper_id", ""),
            paper_title=data.get("paper_title", ""),
            section_type=data.get("section_type", ""),
            relevance_score=data.get("relevance_score", 0.0),
        )


@dataclass
class EntityMatch:
    """エンティティマッチ結果"""
    
    entity: EntityInfo
    match_score: float = 0.0
    hop_distance: int = 0
    source_query_term: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "entity": self.entity.to_dict(),
            "match_score": self.match_score,
            "hop_distance": self.hop_distance,
            "source_query_term": self.source_query_term,
        }


@dataclass
class LocalSearchResult:
    """ローカル検索結果"""
    
    query: str
    answer: str
    entities_found: List[EntityMatch] = field(default_factory=list)
    relationships_used: List[RelationshipInfo] = field(default_factory=list)
    chunks_used: List[ChunkInfo] = field(default_factory=list)
    processing_time_ms: float = 0.0
    tokens_used: int = 0
    hops_traversed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query": self.query,
            "answer": self.answer,
            "entities_found": [e.to_dict() for e in self.entities_found],
            "relationships_used": [r.to_dict() for r in self.relationships_used],
            "chunks_used": [c.to_dict() for c in self.chunks_used],
            "processing_time_ms": self.processing_time_ms,
            "tokens_used": self.tokens_used,
            "hops_traversed": self.hops_traversed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LocalSearchResult":
        """辞書から作成"""
        return cls(
            query=data.get("query", ""),
            answer=data.get("answer", ""),
            entities_found=[],  # Simplified - EntityMatch requires EntityInfo
            relationships_used=[
                RelationshipInfo.from_dict(r) 
                for r in data.get("relationships_used", [])
            ],
            chunks_used=[
                ChunkInfo.from_dict(c)
                for c in data.get("chunks_used", [])
            ],
            processing_time_ms=data.get("processing_time_ms", 0.0),
            tokens_used=data.get("tokens_used", 0),
            hops_traversed=data.get("hops_traversed", 0),
        )
