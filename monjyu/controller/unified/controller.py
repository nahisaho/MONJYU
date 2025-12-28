"""UnifiedController implementation."""

import asyncio
import time
from typing import Dict, List, Optional

from monjyu.query.router import QueryRouter, RoutingContext
from monjyu.query.router.types import QueryType, SearchMode

from .types import (
    EngineNotFoundError,
    EngineSearchResult,
    EngineUnavailableError,
    FALLBACK_ORDER,
    SearchContext,
    SearchEngineError,
    SearchEngineProtocol,
    SearchResultItem,
    SearchTimeoutError,
    UnifiedControllerConfig,
    UnifiedSearchResult,
)


class UnifiedController:
    """統合検索コントローラ
    
    QueryRouterの決定に基づいて最適な検索エンジンを選択・実行する。
    """
    
    def __init__(
        self,
        router: Optional[QueryRouter] = None,
        config: Optional[UnifiedControllerConfig] = None,
    ):
        """初期化
        
        Args:
            router: QueryRouterインスタンス（Noneの場合は内部で作成）
            config: 設定
        """
        self.router = router or QueryRouter()
        self.config = config or UnifiedControllerConfig()
        self._engines: Dict[SearchMode, SearchEngineProtocol] = {}
    
    def register_engine(
        self,
        mode: SearchMode,
        engine: SearchEngineProtocol,
    ) -> None:
        """検索エンジンを登録
        
        Args:
            mode: 検索モード
            engine: 検索エンジン
        """
        self._engines[mode] = engine
    
    def unregister_engine(self, mode: SearchMode) -> None:
        """検索エンジンを登録解除
        
        Args:
            mode: 検索モード
        """
        if mode in self._engines:
            del self._engines[mode]
    
    def get_available_modes(self) -> List[SearchMode]:
        """利用可能な検索モードを取得
        
        Returns:
            利用可能なSearchModeのリスト
        """
        available = []
        for mode, engine in self._engines.items():
            if engine.is_available():
                available.append(mode)
        return available
    
    def has_engine(self, mode: SearchMode) -> bool:
        """エンジンが登録されているかチェック
        
        Args:
            mode: 検索モード
            
        Returns:
            登録されていればTrue
        """
        return mode in self._engines
    
    async def search(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        context: Optional[SearchContext] = None,
    ) -> UnifiedSearchResult:
        """統合検索を実行
        
        Args:
            query: 検索クエリ
            mode: 検索モード（Noneの場合はconfig.default_mode）
            context: 検索コンテキスト
            
        Returns:
            統合検索結果
            
        Raises:
            EngineNotFoundError: エンジンが見つからない
            EngineUnavailableError: エンジンが利用不可
            SearchTimeoutError: タイムアウト
        """
        start_time = time.time()
        context = context or SearchContext()
        mode = mode or self.config.default_mode
        
        # 自動ルーティング
        query_type = QueryType.UNKNOWN
        routing_confidence = 0.0
        fallback_mode = None
        
        if mode == SearchMode.AUTO and self.config.enable_auto_routing:
            routing_context = RoutingContext(
                available_modes=set(self.get_available_modes()),
                user_preference=context.user_preference,
            )
            routing = await self.router.route(query, routing_context)
            mode = routing.mode
            query_type = routing.query_type
            routing_confidence = routing.confidence
            fallback_mode = routing.fallback_mode
        
        # コンテキストにquery_typeがあれば使用
        if context.query_type:
            query_type = context.query_type
        
        # 検索実行
        try:
            result = await self._execute_search(query, mode, context)
            
            processing_time = (time.time() - start_time) * 1000
            
            return UnifiedSearchResult(
                mode_used=mode,
                query_type=query_type,
                items=result.items,
                total_count=result.total_count,
                metadata=result.metadata,
                processing_time_ms=processing_time,
                fallback_used=False,
                fallback_mode=None,
                routing_confidence=routing_confidence,
            )
            
        except SearchEngineError as e:
            # フォールバック処理
            if self.config.fallback_enabled and mode in FALLBACK_ORDER:
                return await self._execute_fallback(
                    query, mode, context, query_type, routing_confidence, start_time
                )
            raise
    
    async def search_with_retry(
        self,
        query: str,
        mode: Optional[SearchMode] = None,
        context: Optional[SearchContext] = None,
    ) -> UnifiedSearchResult:
        """リトライ付き検索を実行
        
        Args:
            query: 検索クエリ
            mode: 検索モード
            context: 検索コンテキスト
            
        Returns:
            統合検索結果
        """
        last_error: Optional[Exception] = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self.search(query, mode, context)
            except SearchEngineError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    continue
                raise
        
        # Should not reach here, but for type safety
        if last_error:
            raise last_error
        raise SearchEngineError("Unknown error during retry")
    
    async def _execute_search(
        self,
        query: str,
        mode: SearchMode,
        context: SearchContext,
    ) -> EngineSearchResult:
        """検索を実行
        
        Args:
            query: 検索クエリ
            mode: 検索モード
            context: 検索コンテキスト
            
        Returns:
            エンジン検索結果
        """
        if mode not in self._engines:
            raise EngineNotFoundError(f"Engine not found for mode: {mode.value}")
        
        engine = self._engines[mode]
        
        if not engine.is_available():
            raise EngineUnavailableError(f"Engine unavailable: {engine.name}")
        
        try:
            return await asyncio.wait_for(
                engine.search(query, context),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise SearchTimeoutError(
                f"Search timeout after {self.config.timeout_seconds}s"
            )
    
    async def _execute_fallback(
        self,
        query: str,
        original_mode: SearchMode,
        context: SearchContext,
        query_type: QueryType,
        routing_confidence: float,
        start_time: float,
    ) -> UnifiedSearchResult:
        """フォールバック検索を実行
        
        Args:
            query: 検索クエリ
            original_mode: 元の検索モード
            context: 検索コンテキスト
            query_type: クエリタイプ
            routing_confidence: ルーティング確信度
            start_time: 開始時刻
            
        Returns:
            統合検索結果
        """
        fallback_order = FALLBACK_ORDER.get(original_mode, [])
        
        for fallback_mode in fallback_order:
            if fallback_mode not in self._engines:
                continue
            
            engine = self._engines[fallback_mode]
            if not engine.is_available():
                continue
            
            try:
                result = await asyncio.wait_for(
                    engine.search(query, context),
                    timeout=self.config.timeout_seconds,
                )
                
                processing_time = (time.time() - start_time) * 1000
                
                return UnifiedSearchResult(
                    mode_used=fallback_mode,
                    query_type=query_type,
                    items=result.items,
                    total_count=result.total_count,
                    metadata=result.metadata,
                    processing_time_ms=processing_time,
                    fallback_used=True,
                    fallback_mode=fallback_mode,
                    routing_confidence=routing_confidence,
                )
            except SearchEngineError:
                continue
        
        raise EngineUnavailableError(
            f"All fallback engines failed for mode: {original_mode.value}"
        )


def create_unified_controller(
    router: Optional[QueryRouter] = None,
    config: Optional[UnifiedControllerConfig] = None,
) -> UnifiedController:
    """UnifiedControllerを作成
    
    Args:
        router: QueryRouterインスタンス
        config: 設定
        
    Returns:
        UnifiedControllerインスタンス
    """
    return UnifiedController(router=router, config=config)
