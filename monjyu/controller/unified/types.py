"""UnifiedController types module."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from monjyu.query.router.types import QueryType, SearchMode


class SearchEngineError(Exception):
    """検索エンジンエラー"""
    pass


class EngineUnavailableError(SearchEngineError):
    """エンジン利用不可エラー"""
    pass


class SearchTimeoutError(SearchEngineError):
    """検索タイムアウトエラー"""
    pass


class EngineNotFoundError(SearchEngineError):
    """エンジン未登録エラー"""
    pass


@dataclass
class SearchContext:
    """検索コンテキスト"""
    
    query_type: Optional[QueryType] = None
    max_results: int = 10
    include_metadata: bool = True
    language: str = "auto"
    user_preference: Optional[SearchMode] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query_type": self.query_type.value if self.query_type else None,
            "max_results": self.max_results,
            "include_metadata": self.include_metadata,
            "language": self.language,
            "user_preference": self.user_preference.value if self.user_preference else None,
            "custom_params": self.custom_params,
        }


@dataclass
class SearchResultItem:
    """検索結果アイテム"""
    
    content: str
    score: float = 0.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineSearchResult:
    """エンジン検索結果"""
    
    items: List[SearchResultItem] = field(default_factory=list)
    total_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedSearchResult:
    """統合検索結果"""
    
    mode_used: SearchMode
    query_type: QueryType
    items: List[SearchResultItem] = field(default_factory=list)
    total_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    fallback_used: bool = False
    fallback_mode: Optional[SearchMode] = None
    routing_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "mode_used": self.mode_used.value,
            "query_type": self.query_type.value,
            "items": [
                {
                    "content": item.content,
                    "score": item.score,
                    "source": item.source,
                    "metadata": item.metadata,
                }
                for item in self.items
            ],
            "total_count": self.total_count,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
            "fallback_used": self.fallback_used,
            "fallback_mode": self.fallback_mode.value if self.fallback_mode else None,
            "routing_confidence": self.routing_confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UnifiedSearchResult":
        """辞書から作成"""
        return cls(
            mode_used=SearchMode(data["mode_used"]),
            query_type=QueryType(data["query_type"]),
            items=[
                SearchResultItem(
                    content=item["content"],
                    score=item.get("score", 0.0),
                    source=item.get("source"),
                    metadata=item.get("metadata", {}),
                )
                for item in data.get("items", [])
            ],
            total_count=data.get("total_count", 0),
            metadata=data.get("metadata", {}),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            fallback_used=data.get("fallback_used", False),
            fallback_mode=SearchMode(data["fallback_mode"]) if data.get("fallback_mode") else None,
            routing_confidence=data.get("routing_confidence", 0.0),
        )


@runtime_checkable
class SearchEngineProtocol(Protocol):
    """検索エンジンプロトコル"""
    
    async def search(
        self,
        query: str,
        context: Optional[SearchContext] = None,
    ) -> EngineSearchResult:
        """検索実行"""
        ...
    
    def is_available(self) -> bool:
        """利用可能チェック"""
        ...
    
    @property
    def name(self) -> str:
        """エンジン名"""
        ...


@dataclass
class UnifiedControllerConfig:
    """UnifiedController設定"""
    
    default_mode: SearchMode = SearchMode.AUTO
    enable_auto_routing: bool = True
    fallback_enabled: bool = True
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_delay_seconds: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "default_mode": self.default_mode.value,
            "enable_auto_routing": self.enable_auto_routing,
            "fallback_enabled": self.fallback_enabled,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


# フォールバック順序マッピング
FALLBACK_ORDER: Dict[SearchMode, List[SearchMode]] = {
    SearchMode.GRAPHRAG: [SearchMode.LAZY, SearchMode.HYBRID, SearchMode.VECTOR],
    SearchMode.LAZY: [SearchMode.HYBRID, SearchMode.VECTOR],
    SearchMode.HYBRID: [SearchMode.VECTOR, SearchMode.LAZY],
    SearchMode.VECTOR: [SearchMode.LAZY],
}
