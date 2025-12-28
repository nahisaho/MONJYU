"""VectorSearch types module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchHit:
    """検索ヒット"""
    
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 学術論文固有
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    section_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "content": self.content,
            "metadata": self.metadata,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section_type": self.section_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchHit":
        """辞書から作成"""
        return cls(
            chunk_id=data["chunk_id"],
            score=data.get("score", 0.0),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            paper_id=data.get("paper_id"),
            paper_title=data.get("paper_title"),
            section_type=data.get("section_type"),
        )


@dataclass
class VectorSearchConfig:
    """ベクトル検索設定"""
    
    top_k: int = 10
    min_score: float = 0.0
    include_metadata: bool = True
    rerank: bool = False
    rerank_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "top_k": self.top_k,
            "min_score": self.min_score,
            "include_metadata": self.include_metadata,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
        }


@dataclass
class VectorSearchResult:
    """ベクトル検索結果"""
    
    hits: List[SearchHit] = field(default_factory=list)
    total_count: int = 0
    processing_time_ms: float = 0.0
    query: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "hits": [h.to_dict() for h in self.hits],
            "total_count": self.total_count,
            "processing_time_ms": self.processing_time_ms,
            "query": self.query,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorSearchResult":
        """辞書から作成"""
        return cls(
            hits=[SearchHit.from_dict(h) for h in data.get("hits", [])],
            total_count=data.get("total_count", 0),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            query=data.get("query", ""),
        )


@runtime_checkable
class EmbedderProtocol(Protocol):
    """埋め込みプロトコル"""
    
    async def embed(self, text: str) -> NDArray[np.float32]:
        """テキストを埋め込みベクトルに変換"""
        ...
    
    async def embed_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """複数テキストをバッチ埋め込み"""
        ...
    
    @property
    def dimension(self) -> int:
        """埋め込み次元"""
        ...


@dataclass
class IndexedDocument:
    """インデックス済みドキュメント"""
    
    chunk_id: str
    content: str
    vector: NDArray[np.float32]
    metadata: Dict[str, Any] = field(default_factory=dict)
    paper_id: Optional[str] = None
    paper_title: Optional[str] = None
    section_type: Optional[str] = None


class VectorSearchProtocol(ABC):
    """ベクトル検索プロトコル"""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """テキストでベクトル検索"""
        ...
    
    @abstractmethod
    async def search_by_vector(
        self,
        vector: NDArray[np.float32],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """ベクトルで直接検索"""
        ...
    
    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> VectorSearchResult:
        """ハイブリッド検索（ベクトル + キーワード）"""
        ...
    
    @abstractmethod
    async def add_documents(
        self,
        documents: List[IndexedDocument],
    ) -> int:
        """ドキュメントを追加"""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """インデックス済みドキュメント数"""
        ...
