"""ProgressiveController types module."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from monjyu.controller.budget import CostBudget, IndexLevel


@dataclass
class ProgressiveSearchContext:
    """Progressive 検索コンテキスト
    
    Attributes:
        budget: コスト予算
        max_results: 最大結果数
        include_metadata: メタデータを含めるか
        auto_build: 未構築レベルを自動構築するか
        custom_params: カスタムパラメータ
    """
    
    budget: CostBudget = CostBudget.STANDARD
    max_results: int = 10
    include_metadata: bool = True
    auto_build: bool = True
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "budget": self.budget.value,
            "max_results": self.max_results,
            "include_metadata": self.include_metadata,
            "auto_build": self.auto_build,
            "custom_params": self.custom_params,
        }


@dataclass
class LevelSearchResult:
    """レベル別検索結果
    
    Attributes:
        level: インデックスレベル
        items: 検索結果アイテム
        score: レベルスコア
        tokens_used: 使用トークン数
        processing_time_ms: 処理時間
    """
    
    level: IndexLevel
    items: List["ProgressiveResultItem"] = field(default_factory=list)
    score: float = 0.0
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "level": self.level.value,
            "level_name": self.level.name,
            "items": [item.to_dict() for item in self.items],
            "score": self.score,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class ProgressiveResultItem:
    """Progressive 検索結果アイテム
    
    Attributes:
        content: コンテンツ
        score: スコア
        source: ソース
        level: 取得レベル
        metadata: メタデータ
    """
    
    content: str
    score: float = 0.0
    source: Optional[str] = None
    level: Optional[IndexLevel] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "level": self.level.value if self.level is not None else None,
            "level_name": self.level.name if self.level is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class ProgressiveSearchResult:
    """Progressive 検索結果
    
    Attributes:
        query: 元のクエリ
        budget: 使用された予算
        max_level_used: 使用された最大レベル
        levels_searched: 検索されたレベル
        level_results: レベル別の結果
        merged_items: マージされた結果
        total_tokens_used: 総使用トークン数
        processing_time_ms: 総処理時間
        metadata: メタデータ
    """
    
    query: str
    budget: CostBudget
    max_level_used: IndexLevel
    levels_searched: List[IndexLevel] = field(default_factory=list)
    level_results: Dict[IndexLevel, LevelSearchResult] = field(default_factory=dict)
    merged_items: List[ProgressiveResultItem] = field(default_factory=list)
    total_tokens_used: int = 0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query": self.query,
            "budget": self.budget.value,
            "max_level_used": self.max_level_used.value,
            "max_level_name": self.max_level_used.name,
            "levels_searched": [l.value for l in self.levels_searched],
            "level_results": {
                level.value: result.to_dict()
                for level, result in self.level_results.items()
            },
            "merged_items": [item.to_dict() for item in self.merged_items],
            "total_tokens_used": self.total_tokens_used,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProgressiveSearchResult":
        """辞書から作成"""
        level_results = {}
        for level_val, result_data in data.get("level_results", {}).items():
            level = IndexLevel(int(level_val))
            level_results[level] = LevelSearchResult(
                level=level,
                items=[
                    ProgressiveResultItem(
                        content=item["content"],
                        score=item.get("score", 0.0),
                        source=item.get("source"),
                        level=IndexLevel(item["level"]) if item.get("level") is not None else None,
                        metadata=item.get("metadata", {}),
                    )
                    for item in result_data.get("items", [])
                ],
                score=result_data.get("score", 0.0),
                tokens_used=result_data.get("tokens_used", 0),
                processing_time_ms=result_data.get("processing_time_ms", 0.0),
            )
        
        return cls(
            query=data["query"],
            budget=CostBudget(data["budget"]),
            max_level_used=IndexLevel(data["max_level_used"]),
            levels_searched=[IndexLevel(l) for l in data.get("levels_searched", [])],
            level_results=level_results,
            merged_items=[
                ProgressiveResultItem(
                    content=item["content"],
                    score=item.get("score", 0.0),
                    source=item.get("source"),
                    level=IndexLevel(item["level"]) if item.get("level") is not None else None,
                    metadata=item.get("metadata", {}),
                )
                for item in data.get("merged_items", [])
            ],
            total_tokens_used=data.get("total_tokens_used", 0),
            processing_time_ms=data.get("processing_time_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@runtime_checkable
class LevelSearchEngineProtocol(Protocol):
    """レベル別検索エンジンプロトコル"""
    
    @property
    def level(self) -> IndexLevel:
        """対応するレベル"""
        ...
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> LevelSearchResult:
        """検索実行"""
        ...
    
    def is_available(self) -> bool:
        """利用可能チェック"""
        ...


@dataclass
class ProgressiveControllerConfig:
    """ProgressiveController 設定
    
    Attributes:
        default_budget: デフォルト予算
        auto_build: 未構築レベルを自動構築するか
        merge_strategy: マージ戦略 (score, rrf, level_priority)
        timeout_seconds: タイムアウト秒数
        max_retries: 最大リトライ回数
        enable_caching: キャッシュを有効にするか
    """
    
    default_budget: CostBudget = CostBudget.STANDARD
    auto_build: bool = False
    merge_strategy: str = "score"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    enable_caching: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "default_budget": self.default_budget.value,
            "auto_build": self.auto_build,
            "merge_strategy": self.merge_strategy,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "enable_caching": self.enable_caching,
        }


class ProgressiveControllerError(Exception):
    """ProgressiveController エラー"""
    pass


class LevelNotBuiltError(ProgressiveControllerError):
    """レベル未構築エラー"""
    pass


class LevelNotAllowedError(ProgressiveControllerError):
    """レベル許可なしエラー"""
    pass


class SearchTimeoutError(ProgressiveControllerError):
    """検索タイムアウトエラー"""
    pass


__all__ = [
    "ProgressiveSearchContext",
    "LevelSearchResult",
    "ProgressiveResultItem",
    "ProgressiveSearchResult",
    "LevelSearchEngineProtocol",
    "ProgressiveControllerConfig",
    "ProgressiveControllerError",
    "LevelNotBuiltError",
    "LevelNotAllowedError",
    "SearchTimeoutError",
]
