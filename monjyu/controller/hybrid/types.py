"""HybridController types module.

REQ-ARC-003: Hybrid GraphRAG Controller の型定義
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class MergeStrategy(Enum):
    """結果マージ戦略"""
    
    RRF = "rrf"  # Reciprocal Rank Fusion (デフォルト)
    SCORE = "score"  # スコアベース
    WEIGHTED = "weighted"  # 重み付け
    INTERLEAVE = "interleave"  # インターリーブ


class ExecutionMode(Enum):
    """実行モード"""
    
    PARALLEL = "parallel"  # 並列実行 (デフォルト)
    SEQUENTIAL = "sequential"  # 順次実行
    RACE = "race"  # 最初の成功結果を返す


class HybridControllerError(Exception):
    """HybridController エラー基底クラス"""
    pass


class NoEnginesRegisteredError(HybridControllerError):
    """エンジン未登録エラー"""
    pass


class AllEnginesFailedError(HybridControllerError):
    """全エンジン失敗エラー"""
    
    def __init__(self, message: str, errors: Optional[Dict[str, Exception]] = None):
        super().__init__(message)
        self.errors = errors or {}


class HybridSearchTimeoutError(HybridControllerError):
    """検索タイムアウトエラー"""
    pass


@runtime_checkable
class HybridSearchEngineProtocol(Protocol):
    """Hybrid検索エンジンプロトコル
    
    各検索エンジンはこのプロトコルを実装する必要がある。
    """
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> "EngineResult":
        """検索を実行
        
        Args:
            query: 検索クエリ
            max_results: 最大結果数
            **kwargs: 追加パラメータ
            
        Returns:
            検索結果
        """
        ...
    
    def is_available(self) -> bool:
        """エンジンが利用可能か
        
        Returns:
            利用可能であればTrue
        """
        ...
    
    @property
    def engine_name(self) -> str:
        """エンジン名を取得
        
        Returns:
            エンジン名
        """
        ...


@dataclass
class HybridSearchContext:
    """Hybrid検索コンテキスト
    
    Attributes:
        max_results: 最大結果数
        merge_strategy: マージ戦略
        execution_mode: 実行モード
        engine_weights: エンジン別重み (WEIGHTED戦略用)
        timeout_per_engine: エンジン別タイムアウト (秒)
        include_metadata: メタデータを含めるか
        fail_on_partial: 一部エンジン失敗時にエラーとするか
        custom_params: カスタムパラメータ
    """
    
    max_results: int = 10
    merge_strategy: MergeStrategy = MergeStrategy.RRF
    execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    engine_weights: Dict[str, float] = field(default_factory=dict)
    timeout_per_engine: float = 30.0
    include_metadata: bool = True
    fail_on_partial: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "max_results": self.max_results,
            "merge_strategy": self.merge_strategy.value,
            "execution_mode": self.execution_mode.value,
            "engine_weights": self.engine_weights,
            "timeout_per_engine": self.timeout_per_engine,
            "include_metadata": self.include_metadata,
            "fail_on_partial": self.fail_on_partial,
            "custom_params": self.custom_params,
        }


@dataclass
class HybridResultItem:
    """Hybrid検索結果アイテム
    
    Attributes:
        content: コンテンツ
        score: マージ後スコア
        source: ソース (ファイル名等)
        engine: 元のエンジン名
        original_score: 元のスコア
        rank: 元のランク (RRF用)
        metadata: メタデータ
    """
    
    content: str
    score: float = 0.0
    source: Optional[str] = None
    engine: Optional[str] = None
    original_score: float = 0.0
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "content": self.content,
            "score": self.score,
            "source": self.source,
            "engine": self.engine,
            "original_score": self.original_score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


@dataclass
class EngineResult:
    """エンジン別検索結果
    
    Attributes:
        engine_name: エンジン名
        items: 検索結果アイテム
        total_count: 総件数
        processing_time_ms: 処理時間
        success: 成功したか
        error: エラー (失敗時)
        metadata: メタデータ
    """
    
    engine_name: str
    items: List[HybridResultItem] = field(default_factory=list)
    total_count: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "engine_name": self.engine_name,
            "items": [item.to_dict() for item in self.items],
            "total_count": self.total_count,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class HybridSearchResult:
    """Hybrid検索結果
    
    Attributes:
        items: マージされた検索結果
        engine_results: エンジン別結果
        engines_used: 使用されたエンジン
        merge_strategy: 使用されたマージ戦略
        total_processing_time_ms: 総処理時間
        metadata: メタデータ
    """
    
    items: List[HybridResultItem] = field(default_factory=list)
    engine_results: Dict[str, EngineResult] = field(default_factory=dict)
    engines_used: List[str] = field(default_factory=list)
    merge_strategy: MergeStrategy = MergeStrategy.RRF
    total_processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_count(self) -> int:
        """総結果数"""
        return len(self.items)
    
    @property
    def successful_engines(self) -> List[str]:
        """成功したエンジン"""
        return [
            name for name, result in self.engine_results.items()
            if result.success
        ]
    
    @property
    def failed_engines(self) -> List[str]:
        """失敗したエンジン"""
        return [
            name for name, result in self.engine_results.items()
            if not result.success
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "items": [item.to_dict() for item in self.items],
            "engine_results": {
                name: result.to_dict()
                for name, result in self.engine_results.items()
            },
            "engines_used": self.engines_used,
            "merge_strategy": self.merge_strategy.value,
            "total_count": self.total_count,
            "total_processing_time_ms": self.total_processing_time_ms,
            "successful_engines": self.successful_engines,
            "failed_engines": self.failed_engines,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridSearchResult":
        """辞書から生成"""
        items = [
            HybridResultItem(
                content=item["content"],
                score=item.get("score", 0.0),
                source=item.get("source"),
                engine=item.get("engine"),
                original_score=item.get("original_score", 0.0),
                rank=item.get("rank", 0),
                metadata=item.get("metadata", {}),
            )
            for item in data.get("items", [])
        ]
        
        engine_results = {}
        for name, result_data in data.get("engine_results", {}).items():
            engine_items = [
                HybridResultItem(
                    content=item["content"],
                    score=item.get("score", 0.0),
                    source=item.get("source"),
                    engine=item.get("engine"),
                    original_score=item.get("original_score", 0.0),
                    rank=item.get("rank", 0),
                    metadata=item.get("metadata", {}),
                )
                for item in result_data.get("items", [])
            ]
            engine_results[name] = EngineResult(
                engine_name=result_data["engine_name"],
                items=engine_items,
                total_count=result_data.get("total_count", 0),
                processing_time_ms=result_data.get("processing_time_ms", 0.0),
                success=result_data.get("success", True),
                error=result_data.get("error"),
                metadata=result_data.get("metadata", {}),
            )
        
        return cls(
            items=items,
            engine_results=engine_results,
            engines_used=data.get("engines_used", []),
            merge_strategy=MergeStrategy(data.get("merge_strategy", "rrf")),
            total_processing_time_ms=data.get("total_processing_time_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HybridControllerConfig:
    """HybridController設定
    
    Attributes:
        default_merge_strategy: デフォルトマージ戦略
        default_execution_mode: デフォルト実行モード
        default_timeout: デフォルトタイムアウト (秒)
        rrf_k: RRF パラメータ k (デフォルト: 60)
        enable_caching: キャッシュを有効化
        cache_ttl_seconds: キャッシュTTL (秒)
        max_concurrent: 最大並列数
        retry_failed_engines: 失敗エンジンをリトライ
        retry_count: リトライ回数
    """
    
    default_merge_strategy: MergeStrategy = MergeStrategy.RRF
    default_execution_mode: ExecutionMode = ExecutionMode.PARALLEL
    default_timeout: float = 30.0
    rrf_k: int = 60
    enable_caching: bool = False
    cache_ttl_seconds: int = 300
    max_concurrent: int = 5
    retry_failed_engines: bool = False
    retry_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "default_merge_strategy": self.default_merge_strategy.value,
            "default_execution_mode": self.default_execution_mode.value,
            "default_timeout": self.default_timeout,
            "rrf_k": self.rrf_k,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "max_concurrent": self.max_concurrent,
            "retry_failed_engines": self.retry_failed_engines,
            "retry_count": self.retry_count,
        }
