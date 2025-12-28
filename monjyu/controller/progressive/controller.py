"""ProgressiveController implementation.

REQ-ARC-002: Progressive GraphRAG Controller

段階的インデックスを使用して予算制約内で検索を実行するコントローラ。
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

from monjyu.controller.budget import (
    BudgetController,
    CostBudget,
    IndexLevel,
    TokenUsage,
)
from monjyu.index.manager import ProgressiveIndexManager

from .types import (
    LevelNotAllowedError,
    LevelNotBuiltError,
    LevelSearchEngineProtocol,
    LevelSearchResult,
    ProgressiveControllerConfig,
    ProgressiveControllerError,
    ProgressiveResultItem,
    ProgressiveSearchContext,
    ProgressiveSearchResult,
    SearchTimeoutError,
)

logger = logging.getLogger(__name__)


class ProgressiveController:
    """Progressive GraphRAG コントローラ
    
    段階的にインデックスを使用して検索を実行する。
    予算に応じて使用するレベルを決定し、各レベルの結果をマージする。
    
    Example:
        >>> controller = ProgressiveController(
        ...     index_manager=index_manager,
        ...     budget_controller=budget_controller,
        ... )
        >>> 
        >>> # 検索エンジン登録
        >>> controller.register_engine(IndexLevel.RAW, vector_search_engine)
        >>> controller.register_engine(IndexLevel.LAZY, lazy_search_engine)
        >>> 
        >>> # 検索実行
        >>> result = await controller.search(
        ...     query="What is GraphRAG?",
        ...     budget=CostBudget.STANDARD,
        ... )
        >>> print(f"Levels used: {result.levels_searched}")
    """
    
    def __init__(
        self,
        index_manager: Optional[ProgressiveIndexManager] = None,
        budget_controller: Optional[BudgetController] = None,
        config: Optional[ProgressiveControllerConfig] = None,
    ):
        """初期化
        
        Args:
            index_manager: インデックスマネージャー
            budget_controller: 予算コントローラ
            config: 設定
        """
        self.index_manager = index_manager
        self.budget_controller = budget_controller or BudgetController()
        self.config = config or ProgressiveControllerConfig()
        self._engines: Dict[IndexLevel, LevelSearchEngineProtocol] = {}
        self._cache: Dict[str, ProgressiveSearchResult] = {}
    
    def register_engine(
        self,
        level: IndexLevel,
        engine: LevelSearchEngineProtocol,
    ) -> None:
        """検索エンジンを登録
        
        Args:
            level: インデックスレベル
            engine: 検索エンジン
        """
        self._engines[level] = engine
        logger.debug(f"Registered engine for level {level.name}")
    
    def unregister_engine(self, level: IndexLevel) -> None:
        """検索エンジンを登録解除
        
        Args:
            level: インデックスレベル
        """
        if level in self._engines:
            del self._engines[level]
            logger.debug(f"Unregistered engine for level {level.name}")
    
    def has_engine(self, level: IndexLevel) -> bool:
        """エンジンが登録されているか
        
        Args:
            level: インデックスレベル
            
        Returns:
            登録されていればTrue
        """
        return level in self._engines
    
    def get_available_levels(self) -> List[IndexLevel]:
        """利用可能なレベルを取得
        
        Returns:
            利用可能なレベルのリスト
        """
        available = []
        for level, engine in self._engines.items():
            if engine.is_available():
                available.append(level)
        return sorted(available)
    
    def _get_max_level(self, budget: CostBudget) -> IndexLevel:
        """予算に応じた最大レベルを取得
        
        Args:
            budget: コスト予算
            
        Returns:
            許可される最大レベル
        """
        return self.budget_controller.get_max_level(budget)
    
    def _get_search_levels(self, budget: CostBudget) -> List[IndexLevel]:
        """検索に使用するレベルを取得
        
        Args:
            budget: コスト予算
            
        Returns:
            検索するレベルのリスト (昇順)
        """
        max_level = self._get_max_level(budget)
        available = self.get_available_levels()
        
        # 予算内で利用可能なレベルのみ
        return [
            level for level in available
            if level <= max_level
        ]
    
    async def search(
        self,
        query: str,
        budget: Optional[CostBudget] = None,
        context: Optional[ProgressiveSearchContext] = None,
    ) -> ProgressiveSearchResult:
        """Progressive 検索を実行
        
        Args:
            query: 検索クエリ
            budget: コスト予算 (Noneの場合はcontextまたはデフォルト)
            context: 検索コンテキスト
            
        Returns:
            Progressive 検索結果
            
        Raises:
            LevelNotBuiltError: 必要なレベルが構築されていない
            LevelNotAllowedError: レベルが予算で許可されていない
            SearchTimeoutError: タイムアウト
        """
        start_time = time.time()
        context = context or ProgressiveSearchContext()
        budget = budget or context.budget or self.config.default_budget
        
        # キャッシュチェック
        cache_key = f"{query}:{budget.value}"
        if self.config.enable_caching and cache_key in self._cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._cache[cache_key]
        
        # 検索レベル決定
        search_levels = self._get_search_levels(budget)
        
        if not search_levels:
            raise ProgressiveControllerError(
                f"No search engines available for budget {budget.value}"
            )
        
        logger.info(
            f"Searching with budget={budget.value}, "
            f"levels={[l.name for l in search_levels]}"
        )
        
        # 各レベルで検索
        level_results: Dict[IndexLevel, LevelSearchResult] = {}
        total_tokens = 0
        max_level_used = IndexLevel.RAW
        
        for level in search_levels:
            try:
                # レベル構築チェック
                if self.index_manager:
                    if not self.index_manager.state.is_level_built(level):
                        if context.auto_build and self.config.auto_build:
                            logger.info(f"Auto-building level {level.name}...")
                            await self.index_manager.build_level(level)
                        else:
                            logger.warning(
                                f"Level {level.name} not built, skipping"
                            )
                            continue
                
                # 検索実行
                result = await self._search_at_level(
                    query, level, context.max_results
                )
                level_results[level] = result
                total_tokens += result.tokens_used
                max_level_used = level
                
                # トークン使用量を記録
                if result.tokens_used > 0:
                    self.budget_controller.record_tokens(
                        prompt_tokens=result.tokens_used,
                        operation="query",
                        level=level,
                    )
                
            except Exception as e:
                logger.error(f"Error searching at level {level.name}: {e}")
                continue
        
        # 結果マージ
        merged_items = self._merge_results(
            level_results,
            strategy=self.config.merge_strategy,
            max_results=context.max_results,
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        result = ProgressiveSearchResult(
            query=query,
            budget=budget,
            max_level_used=max_level_used,
            levels_searched=list(level_results.keys()),
            level_results=level_results,
            merged_items=merged_items,
            total_tokens_used=total_tokens,
            processing_time_ms=processing_time,
            metadata={
                "available_levels": [l.name for l in self.get_available_levels()],
                "merge_strategy": self.config.merge_strategy,
            },
        )
        
        # キャッシュ保存
        if self.config.enable_caching:
            self._cache[cache_key] = result
        
        return result
    
    async def _search_at_level(
        self,
        query: str,
        level: IndexLevel,
        top_k: int = 10,
    ) -> LevelSearchResult:
        """指定レベルで検索
        
        Args:
            query: 検索クエリ
            level: インデックスレベル
            top_k: 取得件数
            
        Returns:
            レベル検索結果
        """
        if level not in self._engines:
            raise LevelNotBuiltError(f"No engine for level {level.name}")
        
        engine = self._engines[level]
        
        if not engine.is_available():
            raise LevelNotBuiltError(f"Engine for level {level.name} not available")
        
        try:
            # タイムアウト付きで実行
            result = await asyncio.wait_for(
                engine.search(query, top_k),
                timeout=self.config.timeout_seconds,
            )
            return result
        except asyncio.TimeoutError:
            raise SearchTimeoutError(
                f"Search at level {level.name} timed out"
            )
    
    def _merge_results(
        self,
        level_results: Dict[IndexLevel, LevelSearchResult],
        strategy: str = "score",
        max_results: int = 10,
    ) -> List[ProgressiveResultItem]:
        """レベル別結果をマージ
        
        Args:
            level_results: レベル別検索結果
            strategy: マージ戦略 (score, rrf, level_priority)
            max_results: 最大結果数
            
        Returns:
            マージされた結果
        """
        if not level_results:
            return []
        
        if strategy == "score":
            return self._merge_by_score(level_results, max_results)
        elif strategy == "rrf":
            return self._merge_by_rrf(level_results, max_results)
        elif strategy == "level_priority":
            return self._merge_by_level_priority(level_results, max_results)
        else:
            logger.warning(f"Unknown merge strategy: {strategy}, using score")
            return self._merge_by_score(level_results, max_results)
    
    def _merge_by_score(
        self,
        level_results: Dict[IndexLevel, LevelSearchResult],
        max_results: int,
    ) -> List[ProgressiveResultItem]:
        """スコア順でマージ
        
        すべてのレベルの結果をスコア順にソートして返す。
        """
        all_items: List[ProgressiveResultItem] = []
        
        for level, result in level_results.items():
            for item in result.items:
                # レベル情報を追加
                merged_item = ProgressiveResultItem(
                    content=item.content,
                    score=item.score,
                    source=item.source,
                    level=level,
                    metadata=item.metadata,
                )
                all_items.append(merged_item)
        
        # スコア順ソート
        all_items.sort(key=lambda x: x.score, reverse=True)
        
        # 重複除去 (content基準)
        seen_content = set()
        unique_items = []
        for item in all_items:
            content_key = item.content[:100]  # 最初の100文字で判定
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_items.append(item)
        
        return unique_items[:max_results]
    
    def _merge_by_rrf(
        self,
        level_results: Dict[IndexLevel, LevelSearchResult],
        max_results: int,
        k: int = 60,
    ) -> List[ProgressiveResultItem]:
        """Reciprocal Rank Fusion でマージ
        
        Args:
            level_results: レベル別検索結果
            max_results: 最大結果数
            k: RRFパラメータ (デフォルト60)
        """
        # コンテンツごとのRRFスコア
        rrf_scores: Dict[str, float] = {}
        content_items: Dict[str, ProgressiveResultItem] = {}
        
        for level, result in level_results.items():
            for rank, item in enumerate(result.items, 1):
                content_key = item.content[:100]
                rrf_score = 1.0 / (k + rank)
                
                if content_key not in rrf_scores:
                    rrf_scores[content_key] = 0.0
                    content_items[content_key] = ProgressiveResultItem(
                        content=item.content,
                        score=0.0,
                        source=item.source,
                        level=level,
                        metadata=item.metadata,
                    )
                
                rrf_scores[content_key] += rrf_score
        
        # スコア更新
        for content_key, item in content_items.items():
            item.score = rrf_scores[content_key]
        
        # ソート
        items = list(content_items.values())
        items.sort(key=lambda x: x.score, reverse=True)
        
        return items[:max_results]
    
    def _merge_by_level_priority(
        self,
        level_results: Dict[IndexLevel, LevelSearchResult],
        max_results: int,
    ) -> List[ProgressiveResultItem]:
        """レベル優先でマージ
        
        高いレベルの結果を優先する。
        """
        all_items: List[ProgressiveResultItem] = []
        seen_content = set()
        
        # 高いレベルから順に追加
        for level in sorted(level_results.keys(), reverse=True):
            result = level_results[level]
            for item in result.items:
                content_key = item.content[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    merged_item = ProgressiveResultItem(
                        content=item.content,
                        score=item.score,
                        source=item.source,
                        level=level,
                        metadata=item.metadata,
                    )
                    all_items.append(merged_item)
        
        return all_items[:max_results]
    
    async def search_at_budget(
        self,
        query: str,
        budget: CostBudget,
        max_results: int = 10,
    ) -> ProgressiveSearchResult:
        """指定予算で検索 (簡易メソッド)
        
        Args:
            query: 検索クエリ
            budget: コスト予算
            max_results: 最大結果数
            
        Returns:
            検索結果
        """
        context = ProgressiveSearchContext(
            budget=budget,
            max_results=max_results,
        )
        return await self.search(query, budget=budget, context=context)
    
    def get_status_summary(self) -> Dict[str, Any]:
        """ステータスサマリーを取得
        
        Returns:
            ステータス情報
        """
        available_levels = self.get_available_levels()
        
        engine_status = {}
        for level in IndexLevel:
            if level in self._engines:
                engine = self._engines[level]
                engine_status[level.name] = {
                    "registered": True,
                    "available": engine.is_available(),
                }
            else:
                engine_status[level.name] = {
                    "registered": False,
                    "available": False,
                }
        
        index_status = None
        if self.index_manager:
            index_status = {
                level.name: self.index_manager.state.is_level_built(level)
                for level in IndexLevel
            }
        
        return {
            "available_levels": [l.name for l in available_levels],
            "engine_status": engine_status,
            "index_status": index_status,
            "budget_summary": self.budget_controller.get_summary(),
            "config": self.config.to_dict(),
            "cache_size": len(self._cache),
        }
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        logger.debug("Cache cleared")
    
    def reset(self) -> None:
        """状態をリセット"""
        self._cache.clear()
        self.budget_controller.reset()
        logger.info("Controller reset")


def create_progressive_controller(
    index_manager: Optional[ProgressiveIndexManager] = None,
    budget_controller: Optional[BudgetController] = None,
    default_budget: CostBudget = CostBudget.STANDARD,
    auto_build: bool = False,
) -> ProgressiveController:
    """ProgressiveController を作成するファクトリ関数
    
    Args:
        index_manager: インデックスマネージャー
        budget_controller: 予算コントローラ
        default_budget: デフォルト予算
        auto_build: 未構築レベルを自動構築するか
        
    Returns:
        設定済みのProgressiveController
    """
    config = ProgressiveControllerConfig(
        default_budget=default_budget,
        auto_build=auto_build,
    )
    return ProgressiveController(
        index_manager=index_manager,
        budget_controller=budget_controller,
        config=config,
    )


__all__ = [
    "ProgressiveController",
    "create_progressive_controller",
]
