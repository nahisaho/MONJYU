"""HybridController implementation.

REQ-ARC-003: Hybrid GraphRAG Controller

複数の検索エンジンを並列実行し、結果をマージするコントローラ。
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Set

from .types import (
    AllEnginesFailedError,
    EngineResult,
    ExecutionMode,
    HybridControllerConfig,
    HybridControllerError,
    HybridResultItem,
    HybridSearchContext,
    HybridSearchEngineProtocol,
    HybridSearchResult,
    HybridSearchTimeoutError,
    MergeStrategy,
    NoEnginesRegisteredError,
)

logger = logging.getLogger(__name__)


class HybridController:
    """Hybrid GraphRAG コントローラ
    
    複数の検索エンジンを並列実行し、結果をマージする。
    RRF (Reciprocal Rank Fusion) をデフォルトのマージ戦略として使用。
    
    Example:
        >>> controller = HybridController()
        >>> 
        >>> # 検索エンジン登録
        >>> controller.register_engine("vector", vector_search_engine)
        >>> controller.register_engine("graph", graph_search_engine)
        >>> controller.register_engine("lazy", lazy_search_engine)
        >>> 
        >>> # 検索実行 (並列)
        >>> result = await controller.search("What is GraphRAG?")
        >>> print(f"Engines used: {result.engines_used}")
        >>> 
        >>> # 特定エンジンのみで検索
        >>> result = await controller.search(
        ...     query="GraphRAG overview",
        ...     engines=["vector", "graph"],
        ... )
    """
    
    def __init__(
        self,
        config: Optional[HybridControllerConfig] = None,
    ):
        """初期化
        
        Args:
            config: 設定
        """
        self.config = config or HybridControllerConfig()
        self._engines: Dict[str, HybridSearchEngineProtocol] = {}
        self._cache: Dict[str, HybridSearchResult] = {}
        self._cache_timestamps: Dict[str, float] = {}
    
    def register_engine(
        self,
        name: str,
        engine: HybridSearchEngineProtocol,
    ) -> None:
        """検索エンジンを登録
        
        Args:
            name: エンジン名
            engine: 検索エンジン
        """
        self._engines[name] = engine
        logger.debug(f"Registered engine: {name}")
    
    def unregister_engine(self, name: str) -> None:
        """検索エンジンを登録解除
        
        Args:
            name: エンジン名
        """
        if name in self._engines:
            del self._engines[name]
            logger.debug(f"Unregistered engine: {name}")
    
    def has_engine(self, name: str) -> bool:
        """エンジンが登録されているか
        
        Args:
            name: エンジン名
            
        Returns:
            登録されていればTrue
        """
        return name in self._engines
    
    def get_registered_engines(self) -> List[str]:
        """登録済みエンジン名を取得
        
        Returns:
            登録済みエンジン名のリスト
        """
        return list(self._engines.keys())
    
    def get_available_engines(self) -> List[str]:
        """利用可能なエンジン名を取得
        
        Returns:
            利用可能なエンジン名のリスト
        """
        return [
            name for name, engine in self._engines.items()
            if engine.is_available()
        ]
    
    def _get_cache_key(
        self,
        query: str,
        engines: Optional[List[str]],
        context: HybridSearchContext,
    ) -> str:
        """キャッシュキーを生成
        
        Args:
            query: 検索クエリ
            engines: 使用エンジン
            context: 検索コンテキスト
            
        Returns:
            キャッシュキー
        """
        key_data = f"{query}:{sorted(engines or [])}:{context.max_results}:{context.merge_strategy.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """キャッシュが有効か
        
        Args:
            cache_key: キャッシュキー
            
        Returns:
            有効であればTrue
        """
        if cache_key not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(cache_key, 0)
        return (time.time() - timestamp) < self.config.cache_ttl_seconds
    
    async def search(
        self,
        query: str,
        engines: Optional[List[str]] = None,
        context: Optional[HybridSearchContext] = None,
    ) -> HybridSearchResult:
        """検索を実行
        
        複数の検索エンジンを並列実行し、結果をマージする。
        
        Args:
            query: 検索クエリ
            engines: 使用するエンジン名リスト (Noneで全エンジン)
            context: 検索コンテキスト
            
        Returns:
            マージされた検索結果
            
        Raises:
            NoEnginesRegisteredError: エンジンが登録されていない
            AllEnginesFailedError: 全エンジンが失敗
            HybridSearchTimeoutError: 検索タイムアウト
        """
        start_time = time.time()
        context = context or HybridSearchContext()
        
        # 使用エンジンを決定
        target_engines = self._determine_target_engines(engines)
        if not target_engines:
            raise NoEnginesRegisteredError("No engines registered or available")
        
        # キャッシュチェック
        if self.config.enable_caching:
            cache_key = self._get_cache_key(query, engines, context)
            if self._is_cache_valid(cache_key):
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return self._cache[cache_key]
        
        # 実行モードに応じて検索
        execution_mode = context.execution_mode
        
        if execution_mode == ExecutionMode.PARALLEL:
            engine_results = await self._search_parallel(
                query, target_engines, context
            )
        elif execution_mode == ExecutionMode.SEQUENTIAL:
            engine_results = await self._search_sequential(
                query, target_engines, context
            )
        else:  # RACE
            engine_results = await self._search_race(
                query, target_engines, context
            )
        
        # エラーチェック
        successful = [r for r in engine_results.values() if r.success]
        if not successful:
            errors = {
                name: Exception(r.error or "Unknown error")
                for name, r in engine_results.items()
            }
            raise AllEnginesFailedError("All search engines failed", errors)
        
        if context.fail_on_partial and len(successful) < len(target_engines):
            failed = [name for name, r in engine_results.items() if not r.success]
            raise AllEnginesFailedError(
                f"Some engines failed: {failed}",
                {name: Exception(engine_results[name].error) for name in failed}
            )
        
        # 結果をマージ
        merge_strategy = context.merge_strategy
        merged_items = self._merge_results(
            engine_results,
            strategy=merge_strategy,
            max_results=context.max_results,
            engine_weights=context.engine_weights,
        )
        
        # 結果を構築
        total_time = (time.time() - start_time) * 1000
        result = HybridSearchResult(
            items=merged_items,
            engine_results=engine_results,
            engines_used=list(target_engines),
            merge_strategy=merge_strategy,
            total_processing_time_ms=total_time,
            metadata={
                "query": query,
                "successful_count": len(successful),
                "failed_count": len(target_engines) - len(successful),
            },
        )
        
        # キャッシュ保存
        if self.config.enable_caching:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
        
        logger.info(
            f"Hybrid search completed: {len(successful)}/{len(target_engines)} engines, "
            f"{len(merged_items)} results, {total_time:.1f}ms"
        )
        
        return result
    
    def _determine_target_engines(
        self,
        engines: Optional[List[str]],
    ) -> Set[str]:
        """使用するエンジンを決定
        
        Args:
            engines: 指定されたエンジン名リスト
            
        Returns:
            使用するエンジン名セット
        """
        if engines:
            # 指定されたエンジンのうち利用可能なもの
            available = set(self.get_available_engines())
            return set(engines) & available
        else:
            # 全ての利用可能なエンジン
            return set(self.get_available_engines())
    
    async def _search_parallel(
        self,
        query: str,
        engines: Set[str],
        context: HybridSearchContext,
    ) -> Dict[str, EngineResult]:
        """並列検索を実行
        
        Args:
            query: 検索クエリ
            engines: エンジン名セット
            context: 検索コンテキスト
            
        Returns:
            エンジン別結果
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def search_with_semaphore(name: str) -> tuple[str, EngineResult]:
            async with semaphore:
                return name, await self._search_single_engine(
                    name, query, context
                )
        
        tasks = [search_with_semaphore(name) for name in engines]
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=context.timeout_per_engine * 2,  # 全体タイムアウト
            )
        except asyncio.TimeoutError:
            raise HybridSearchTimeoutError(
                f"Parallel search timed out after {context.timeout_per_engine * 2}s"
            )
        
        # 結果を整理
        engine_results = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Engine search failed with exception: {result}")
                continue
            name, engine_result = result
            engine_results[name] = engine_result
        
        return engine_results
    
    async def _search_sequential(
        self,
        query: str,
        engines: Set[str],
        context: HybridSearchContext,
    ) -> Dict[str, EngineResult]:
        """順次検索を実行
        
        Args:
            query: 検索クエリ
            engines: エンジン名セット
            context: 検索コンテキスト
            
        Returns:
            エンジン別結果
        """
        engine_results = {}
        
        for name in engines:
            result = await self._search_single_engine(name, query, context)
            engine_results[name] = result
        
        return engine_results
    
    async def _search_race(
        self,
        query: str,
        engines: Set[str],
        context: HybridSearchContext,
    ) -> Dict[str, EngineResult]:
        """最初の成功結果を返す
        
        Args:
            query: 検索クエリ
            engines: エンジン名セット
            context: 検索コンテキスト
            
        Returns:
            エンジン別結果 (最初の成功のみ)
        """
        tasks = {
            name: asyncio.create_task(
                self._search_single_engine(name, query, context)
            )
            for name in engines
        }
        
        pending = set(tasks.values())
        engine_results = {}
        
        while pending:
            done, pending = await asyncio.wait(
                pending,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                # タスク名からエンジン名を取得
                for name, t in tasks.items():
                    if t == task:
                        try:
                            result = task.result()
                            engine_results[name] = result
                            
                            if result.success:
                                # 成功したら残りをキャンセル
                                for p in pending:
                                    p.cancel()
                                return engine_results
                        except Exception as e:
                            engine_results[name] = EngineResult(
                                engine_name=name,
                                success=False,
                                error=str(e),
                            )
                        break
        
        return engine_results
    
    async def _search_single_engine(
        self,
        name: str,
        query: str,
        context: HybridSearchContext,
    ) -> EngineResult:
        """単一エンジンで検索
        
        Args:
            name: エンジン名
            query: 検索クエリ
            context: 検索コンテキスト
            
        Returns:
            エンジン結果
        """
        engine = self._engines.get(name)
        if not engine:
            return EngineResult(
                engine_name=name,
                success=False,
                error=f"Engine not found: {name}",
            )
        
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                engine.search(
                    query,
                    max_results=context.max_results,
                    **context.custom_params,
                ),
                timeout=context.timeout_per_engine,
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # EngineResult が直接返された場合
            if isinstance(result, EngineResult):
                result.processing_time_ms = processing_time
                return result
            
            # HybridResultItem のリストが返された場合
            if isinstance(result, list):
                return EngineResult(
                    engine_name=name,
                    items=result,
                    total_count=len(result),
                    processing_time_ms=processing_time,
                    success=True,
                )
            
            # その他の場合
            return EngineResult(
                engine_name=name,
                success=False,
                error=f"Unexpected result type: {type(result)}",
            )
            
        except asyncio.TimeoutError:
            return EngineResult(
                engine_name=name,
                success=False,
                error=f"Timeout after {context.timeout_per_engine}s",
                processing_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            logger.warning(f"Engine {name} failed: {e}")
            return EngineResult(
                engine_name=name,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _merge_results(
        self,
        engine_results: Dict[str, EngineResult],
        strategy: MergeStrategy,
        max_results: int,
        engine_weights: Optional[Dict[str, float]] = None,
    ) -> List[HybridResultItem]:
        """結果をマージ
        
        Args:
            engine_results: エンジン別結果
            strategy: マージ戦略
            max_results: 最大結果数
            engine_weights: エンジン別重み
            
        Returns:
            マージされた結果
        """
        if strategy == MergeStrategy.RRF:
            return self._merge_by_rrf(engine_results, max_results)
        elif strategy == MergeStrategy.SCORE:
            return self._merge_by_score(engine_results, max_results)
        elif strategy == MergeStrategy.WEIGHTED:
            return self._merge_by_weighted(
                engine_results, max_results, engine_weights or {}
            )
        elif strategy == MergeStrategy.INTERLEAVE:
            return self._merge_by_interleave(engine_results, max_results)
        else:
            # デフォルトはRRF
            return self._merge_by_rrf(engine_results, max_results)
    
    def _merge_by_rrf(
        self,
        engine_results: Dict[str, EngineResult],
        max_results: int,
    ) -> List[HybridResultItem]:
        """Reciprocal Rank Fusion でマージ
        
        RRF(d) = Σ 1 / (k + rank(d))
        
        Args:
            engine_results: エンジン別結果
            max_results: 最大結果数
            
        Returns:
            マージされた結果
        """
        k = self.config.rrf_k
        
        # コンテンツをキーとしてスコアを集約
        content_scores: Dict[str, Dict[str, Any]] = {}
        
        for name, result in engine_results.items():
            if not result.success:
                continue
            
            for rank, item in enumerate(result.items, start=1):
                content_key = self._get_content_key(item.content)
                
                if content_key not in content_scores:
                    content_scores[content_key] = {
                        "content": item.content,
                        "source": item.source,
                        "engines": [],
                        "rrf_score": 0.0,
                        "original_scores": {},
                        "ranks": {},
                        "metadata": item.metadata.copy(),
                    }
                
                # RRFスコアを加算
                rrf_contribution = 1.0 / (k + rank)
                content_scores[content_key]["rrf_score"] += rrf_contribution
                content_scores[content_key]["engines"].append(name)
                content_scores[content_key]["original_scores"][name] = item.original_score or item.score
                content_scores[content_key]["ranks"][name] = rank
        
        # スコアでソートして上位を返す
        sorted_items = sorted(
            content_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )[:max_results]
        
        return [
            HybridResultItem(
                content=item["content"],
                score=item["rrf_score"],
                source=item["source"],
                engine=",".join(item["engines"]),  # 複数エンジンの場合はカンマ区切り
                original_score=max(item["original_scores"].values()) if item["original_scores"] else 0.0,
                rank=min(item["ranks"].values()) if item["ranks"] else 0,
                metadata={
                    **item["metadata"],
                    "contributing_engines": item["engines"],
                    "engine_scores": item["original_scores"],
                    "engine_ranks": item["ranks"],
                },
            )
            for item in sorted_items
        ]
    
    def _merge_by_score(
        self,
        engine_results: Dict[str, EngineResult],
        max_results: int,
    ) -> List[HybridResultItem]:
        """スコアベースでマージ
        
        Args:
            engine_results: エンジン別結果
            max_results: 最大結果数
            
        Returns:
            マージされた結果
        """
        all_items = []
        
        for name, result in engine_results.items():
            if not result.success:
                continue
            
            for rank, item in enumerate(result.items, start=1):
                item.engine = name
                item.rank = rank
                if item.original_score == 0.0:
                    item.original_score = item.score
                all_items.append(item)
        
        # スコアでソート
        all_items.sort(key=lambda x: x.score, reverse=True)
        
        # 重複を除去して上位を返す
        seen_contents = set()
        unique_items = []
        
        for item in all_items:
            content_key = self._get_content_key(item.content)
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_items.append(item)
                if len(unique_items) >= max_results:
                    break
        
        return unique_items
    
    def _merge_by_weighted(
        self,
        engine_results: Dict[str, EngineResult],
        max_results: int,
        weights: Dict[str, float],
    ) -> List[HybridResultItem]:
        """重み付けでマージ
        
        Args:
            engine_results: エンジン別結果
            max_results: 最大結果数
            weights: エンジン別重み
            
        Returns:
            マージされた結果
        """
        # デフォルト重みは1.0
        default_weight = 1.0
        
        content_scores: Dict[str, Dict[str, Any]] = {}
        
        for name, result in engine_results.items():
            if not result.success:
                continue
            
            weight = weights.get(name, default_weight)
            
            for rank, item in enumerate(result.items, start=1):
                content_key = self._get_content_key(item.content)
                
                if content_key not in content_scores:
                    content_scores[content_key] = {
                        "content": item.content,
                        "source": item.source,
                        "engines": [],
                        "weighted_score": 0.0,
                        "original_scores": {},
                        "metadata": item.metadata.copy(),
                    }
                
                # 重み付きスコアを加算
                weighted_contribution = (item.score or 1.0 / rank) * weight
                content_scores[content_key]["weighted_score"] += weighted_contribution
                content_scores[content_key]["engines"].append(name)
                content_scores[content_key]["original_scores"][name] = item.original_score or item.score
        
        # スコアでソートして上位を返す
        sorted_items = sorted(
            content_scores.values(),
            key=lambda x: x["weighted_score"],
            reverse=True,
        )[:max_results]
        
        return [
            HybridResultItem(
                content=item["content"],
                score=item["weighted_score"],
                source=item["source"],
                engine=",".join(item["engines"]),
                original_score=max(item["original_scores"].values()) if item["original_scores"] else 0.0,
                metadata={
                    **item["metadata"],
                    "contributing_engines": item["engines"],
                    "engine_scores": item["original_scores"],
                },
            )
            for item in sorted_items
        ]
    
    def _merge_by_interleave(
        self,
        engine_results: Dict[str, EngineResult],
        max_results: int,
    ) -> List[HybridResultItem]:
        """インターリーブでマージ
        
        各エンジンから順番に結果を取得してマージ。
        
        Args:
            engine_results: エンジン別結果
            max_results: 最大結果数
            
        Returns:
            マージされた結果
        """
        # 成功したエンジンの結果をリストに変換
        engine_items = {
            name: list(result.items)
            for name, result in engine_results.items()
            if result.success
        }
        
        if not engine_items:
            return []
        
        merged = []
        seen_contents = set()
        engine_names = list(engine_items.keys())
        current_idx = 0
        
        while len(merged) < max_results:
            # 全エンジンが空になったら終了
            if all(len(items) == 0 for items in engine_items.values()):
                break
            
            # 現在のエンジンから取得
            name = engine_names[current_idx % len(engine_names)]
            items = engine_items[name]
            
            if items:
                item = items.pop(0)
                content_key = self._get_content_key(item.content)
                
                if content_key not in seen_contents:
                    seen_contents.add(content_key)
                    item.engine = name
                    if item.original_score == 0.0:
                        item.original_score = item.score
                    merged.append(item)
            
            current_idx += 1
        
        return merged
    
    def _get_content_key(self, content: str) -> str:
        """コンテンツのユニークキーを生成
        
        Args:
            content: コンテンツ
            
        Returns:
            ユニークキー
        """
        # 正規化して比較
        normalized = content.strip().lower()[:200]  # 最初の200文字で比較
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_status(self) -> Dict[str, Any]:
        """ステータスを取得
        
        Returns:
            ステータス情報
        """
        return {
            "registered_engines": self.get_registered_engines(),
            "available_engines": self.get_available_engines(),
            "cache_enabled": self.config.enable_caching,
            "cache_size": len(self._cache),
            "config": self.config.to_dict(),
        }
    
    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Cache cleared")
    
    def reset(self) -> None:
        """コントローラをリセット"""
        self._engines.clear()
        self.clear_cache()
        logger.debug("Controller reset")


def create_hybrid_controller(
    config: Optional[HybridControllerConfig] = None,
    engines: Optional[Dict[str, HybridSearchEngineProtocol]] = None,
) -> HybridController:
    """HybridController を作成
    
    Args:
        config: 設定
        engines: 初期エンジン
        
    Returns:
        HybridController インスタンス
    """
    controller = HybridController(config=config)
    
    if engines:
        for name, engine in engines.items():
            controller.register_engine(name, engine)
    
    return controller
