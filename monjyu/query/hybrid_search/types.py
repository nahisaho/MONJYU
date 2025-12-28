"""HybridSearch types module - REQ-QRY-005."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FusionMethod(Enum):
    """融合方式"""
    
    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # 重み付け平均
    MAX = "max"  # 最大スコア
    COMBSUM = "combsum"  # スコア合計
    COMBMNZ = "combmnz"  # CombMNZ


class SearchMethod(Enum):
    """検索方式"""
    
    VECTOR = "vector"
    LAZY = "lazy"
    KEYWORD = "keyword"
    GLOBAL = "global"
    LOCAL = "local"


@dataclass
class HybridSearchConfig:
    """ハイブリッド検索設定"""
    
    methods: List[SearchMethod] = field(
        default_factory=lambda: [SearchMethod.VECTOR, SearchMethod.LAZY]
    )
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60
    top_k: int = 10
    min_score: float = 0.0
    parallel: bool = True
    timeout_seconds: float = 30.0
    method_weights: Dict[SearchMethod, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """デフォルト重みを設定"""
        if not self.method_weights:
            # 均等重み
            weight = 1.0 / len(self.methods) if self.methods else 0.5
            self.method_weights = {m: weight for m in self.methods}
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "methods": [m.value for m in self.methods],
            "fusion": self.fusion.value,
            "rrf_k": self.rrf_k,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "parallel": self.parallel,
            "timeout_seconds": self.timeout_seconds,
            "method_weights": {m.value: w for m, w in self.method_weights.items()},
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchConfig":
        """辞書から作成"""
        methods = [SearchMethod(m) for m in data.get("methods", ["vector", "lazy"])]
        method_weights = {
            SearchMethod(k): v
            for k, v in data.get("method_weights", {}).items()
        }
        return cls(
            methods=methods,
            fusion=FusionMethod(data.get("fusion", "rrf")),
            rrf_k=data.get("rrf_k", 60),
            top_k=data.get("top_k", 10),
            min_score=data.get("min_score", 0.0),
            parallel=data.get("parallel", True),
            timeout_seconds=data.get("timeout_seconds", 30.0),
            method_weights=method_weights,
        )


@dataclass
class HybridSearchHit:
    """ハイブリッド検索ヒット"""
    
    chunk_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[str] = field(default_factory=list)  # 検索元メソッド
    
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
            "sources": self.sources,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "section_type": self.section_type,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchHit":
        """辞書から作成"""
        return cls(
            chunk_id=data["chunk_id"],
            score=data.get("score", 0.0),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            sources=data.get("sources", []),
            paper_id=data.get("paper_id"),
            paper_title=data.get("paper_title"),
            section_type=data.get("section_type"),
        )


@dataclass
class MethodSearchResult:
    """個別メソッドの検索結果"""
    
    method: SearchMethod
    hits: List[HybridSearchHit] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    search_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "method": self.method.value,
            "hits": [h.to_dict() for h in self.hits],
            "success": self.success,
            "error": self.error,
            "search_time_ms": self.search_time_ms,
        }


@dataclass
class HybridSearchResult:
    """ハイブリッド検索結果"""
    
    query: str
    hits: List[HybridSearchHit] = field(default_factory=list)
    method_results: List[MethodSearchResult] = field(default_factory=list)
    fusion_method: FusionMethod = FusionMethod.RRF
    total_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query": self.query,
            "hits": [h.to_dict() for h in self.hits],
            "method_results": [r.to_dict() for r in self.method_results],
            "fusion_method": self.fusion_method.value,
            "total_time_ms": self.total_time_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchResult":
        """辞書から作成"""
        return cls(
            query=data.get("query", ""),
            hits=[HybridSearchHit.from_dict(h) for h in data.get("hits", [])],
            fusion_method=FusionMethod(data.get("fusion_method", "rrf")),
            total_time_ms=data.get("total_time_ms", 0.0),
        )
    
    @property
    def success_count(self) -> int:
        """成功したメソッド数"""
        return sum(1 for r in self.method_results if r.success)
    
    @property
    def failed_methods(self) -> List[SearchMethod]:
        """失敗したメソッド一覧"""
        return [r.method for r in self.method_results if not r.success]
