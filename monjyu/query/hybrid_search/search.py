"""HybridSearch module - REQ-QRY-005.

複数の検索手法を組み合わせて最適な結果を返すハイブリッド検索。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from monjyu.query.hybrid_search.types import (
    FusionMethod,
    HybridSearchConfig,
    HybridSearchHit,
    HybridSearchResult,
    MethodSearchResult,
    SearchMethod,
)

logger = logging.getLogger(__name__)


# ========== Protocols ==========


@runtime_checkable
class VectorSearchProtocol(Protocol):
    """ベクトル検索プロトコル"""
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> Any:
        """検索を実行"""
        ...


@runtime_checkable
class LazySearchProtocol(Protocol):
    """Lazy検索プロトコル"""
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> Any:
        """検索を実行"""
        ...


@runtime_checkable
class GlobalSearchProtocol(Protocol):
    """グローバル検索プロトコル"""
    
    async def search(
        self,
        query: str,
        **kwargs,
    ) -> Any:
        """検索を実行"""
        ...


@runtime_checkable
class LocalSearchProtocol(Protocol):
    """ローカル検索プロトコル"""
    
    async def search(
        self,
        query: str,
        **kwargs,
    ) -> Any:
        """検索を実行"""
        ...


# ========== Result Merger ==========


class ResultMerger:
    """検索結果の融合"""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
    
    def merge(
        self,
        method_results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """
        複数メソッドの結果を融合
        
        Returns:
            融合されたヒットリストとソースマッピング
        """
        # 成功した結果のみ処理
        successful = [r for r in method_results if r.success]
        if not successful:
            return [], {}
        
        # 融合方式による分岐
        fusion_map = {
            FusionMethod.RRF: self._rrf_fusion,
            FusionMethod.WEIGHTED: self._weighted_fusion,
            FusionMethod.MAX: self._max_fusion,
            FusionMethod.COMBSUM: self._combsum_fusion,
            FusionMethod.COMBMNZ: self._combmnz_fusion,
        }
        
        fusion_func = fusion_map.get(self.config.fusion, self._rrf_fusion)
        return fusion_func(successful, top_k)
    
    def _rrf_fusion(
        self,
        results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """Reciprocal Rank Fusion"""
        k = self.config.rrf_k
        rrf_scores: Dict[str, float] = {}
        hit_map: Dict[str, HybridSearchHit] = {}
        sources: Dict[str, List[str]] = {}
        
        for method_result in results:
            method_name = method_result.method.value
            
            for rank, hit in enumerate(method_result.hits, start=1):
                chunk_id = hit.chunk_id
                
                if chunk_id not in rrf_scores:
                    rrf_scores[chunk_id] = 0.0
                    hit_map[chunk_id] = hit
                    sources[chunk_id] = []
                
                # RRFスコア: 1/(k + rank)
                rrf_scores[chunk_id] += 1.0 / (k + rank)
                sources[chunk_id].append(method_name)
        
        # スコア順でソート
        sorted_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True,
        )
        
        # 上位k件を返す
        merged = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            merged.append(HybridSearchHit(
                chunk_id=hit.chunk_id,
                score=rrf_scores[chunk_id],
                content=hit.content,
                metadata=hit.metadata,
                sources=sources[chunk_id],
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                section_type=hit.section_type,
            ))
        
        return merged, sources
    
    def _weighted_fusion(
        self,
        results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """重み付け融合"""
        weighted_scores: Dict[str, float] = {}
        hit_map: Dict[str, HybridSearchHit] = {}
        sources: Dict[str, List[str]] = {}
        
        for method_result in results:
            method = method_result.method
            method_name = method.value
            weight = self.config.method_weights.get(method, 0.5)
            
            for hit in method_result.hits:
                chunk_id = hit.chunk_id
                
                if chunk_id not in weighted_scores:
                    weighted_scores[chunk_id] = 0.0
                    hit_map[chunk_id] = hit
                    sources[chunk_id] = []
                
                weighted_scores[chunk_id] += hit.score * weight
                if method_name not in sources[chunk_id]:
                    sources[chunk_id].append(method_name)
        
        sorted_ids = sorted(
            weighted_scores.keys(),
            key=lambda x: weighted_scores[x],
            reverse=True,
        )
        
        merged = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            merged.append(HybridSearchHit(
                chunk_id=hit.chunk_id,
                score=weighted_scores[chunk_id],
                content=hit.content,
                metadata=hit.metadata,
                sources=sources[chunk_id],
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                section_type=hit.section_type,
            ))
        
        return merged, sources
    
    def _max_fusion(
        self,
        results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """最大スコア融合"""
        max_scores: Dict[str, float] = {}
        hit_map: Dict[str, HybridSearchHit] = {}
        sources: Dict[str, List[str]] = {}
        
        for method_result in results:
            method_name = method_result.method.value
            
            for hit in method_result.hits:
                chunk_id = hit.chunk_id
                
                if chunk_id not in max_scores:
                    max_scores[chunk_id] = 0.0
                    hit_map[chunk_id] = hit
                    sources[chunk_id] = []
                
                max_scores[chunk_id] = max(max_scores[chunk_id], hit.score)
                if method_name not in sources[chunk_id]:
                    sources[chunk_id].append(method_name)
        
        sorted_ids = sorted(
            max_scores.keys(),
            key=lambda x: max_scores[x],
            reverse=True,
        )
        
        merged = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            merged.append(HybridSearchHit(
                chunk_id=hit.chunk_id,
                score=max_scores[chunk_id],
                content=hit.content,
                metadata=hit.metadata,
                sources=sources[chunk_id],
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                section_type=hit.section_type,
            ))
        
        return merged, sources
    
    def _combsum_fusion(
        self,
        results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """CombSUM融合（スコア合計）"""
        sum_scores: Dict[str, float] = {}
        hit_map: Dict[str, HybridSearchHit] = {}
        sources: Dict[str, List[str]] = {}
        
        for method_result in results:
            method_name = method_result.method.value
            
            for hit in method_result.hits:
                chunk_id = hit.chunk_id
                
                if chunk_id not in sum_scores:
                    sum_scores[chunk_id] = 0.0
                    hit_map[chunk_id] = hit
                    sources[chunk_id] = []
                
                sum_scores[chunk_id] += hit.score
                if method_name not in sources[chunk_id]:
                    sources[chunk_id].append(method_name)
        
        sorted_ids = sorted(
            sum_scores.keys(),
            key=lambda x: sum_scores[x],
            reverse=True,
        )
        
        merged = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            merged.append(HybridSearchHit(
                chunk_id=hit.chunk_id,
                score=sum_scores[chunk_id],
                content=hit.content,
                metadata=hit.metadata,
                sources=sources[chunk_id],
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                section_type=hit.section_type,
            ))
        
        return merged, sources
    
    def _combmnz_fusion(
        self,
        results: List[MethodSearchResult],
        top_k: int,
    ) -> tuple[List[HybridSearchHit], Dict[str, List[str]]]:
        """CombMNZ融合（スコア合計 × 出現回数）"""
        sum_scores: Dict[str, float] = {}
        hit_count: Dict[str, int] = {}
        hit_map: Dict[str, HybridSearchHit] = {}
        sources: Dict[str, List[str]] = {}
        
        for method_result in results:
            method_name = method_result.method.value
            
            for hit in method_result.hits:
                chunk_id = hit.chunk_id
                
                if chunk_id not in sum_scores:
                    sum_scores[chunk_id] = 0.0
                    hit_count[chunk_id] = 0
                    hit_map[chunk_id] = hit
                    sources[chunk_id] = []
                
                sum_scores[chunk_id] += hit.score
                hit_count[chunk_id] += 1
                if method_name not in sources[chunk_id]:
                    sources[chunk_id].append(method_name)
        
        # CombMNZ: sum * count
        mnz_scores = {
            chunk_id: sum_scores[chunk_id] * hit_count[chunk_id]
            for chunk_id in sum_scores
        }
        
        sorted_ids = sorted(
            mnz_scores.keys(),
            key=lambda x: mnz_scores[x],
            reverse=True,
        )
        
        merged = []
        for chunk_id in sorted_ids[:top_k]:
            hit = hit_map[chunk_id]
            merged.append(HybridSearchHit(
                chunk_id=hit.chunk_id,
                score=mnz_scores[chunk_id],
                content=hit.content,
                metadata=hit.metadata,
                sources=sources[chunk_id],
                paper_id=hit.paper_id,
                paper_title=hit.paper_title,
                section_type=hit.section_type,
            ))
        
        return merged, sources


# ========== HybridSearch ==========


class HybridSearch:
    """ハイブリッド検索"""
    
    def __init__(
        self,
        config: Optional[HybridSearchConfig] = None,
        vector_search: Optional[VectorSearchProtocol] = None,
        lazy_search: Optional[LazySearchProtocol] = None,
        global_search: Optional[GlobalSearchProtocol] = None,
        local_search: Optional[LocalSearchProtocol] = None,
    ):
        self.config = config or HybridSearchConfig()
        self.vector_search = vector_search
        self.lazy_search = lazy_search
        self.global_search = global_search
        self.local_search = local_search
        self.merger = ResultMerger(self.config)
    
    async def search(
        self,
        query: str,
        methods: Optional[List[SearchMethod]] = None,
        top_k: Optional[int] = None,
    ) -> HybridSearchResult:
        """
        ハイブリッド検索を実行
        
        Args:
            query: 検索クエリ
            methods: 使用する検索メソッド（省略時はconfigのmethods）
            top_k: 返す結果数（省略時はconfigのtop_k）
        
        Returns:
            HybridSearchResult
        """
        start_time = time.time()
        methods = methods or self.config.methods
        top_k = top_k or self.config.top_k
        
        # 検索実行
        if self.config.parallel:
            method_results = await self._search_parallel(query, methods)
        else:
            method_results = await self._search_sequential(query, methods)
        
        # 結果融合
        merged_hits, _ = self.merger.merge(method_results, top_k)
        
        elapsed = (time.time() - start_time) * 1000
        
        logger.info(
            f"Hybrid search completed: {len(method_results)} methods, "
            f"{len(merged_hits)} results, {elapsed:.1f}ms"
        )
        
        return HybridSearchResult(
            query=query,
            hits=merged_hits,
            method_results=method_results,
            fusion_method=self.config.fusion,
            total_time_ms=elapsed,
        )
    
    def search_sync(
        self,
        query: str,
        methods: Optional[List[SearchMethod]] = None,
        top_k: Optional[int] = None,
    ) -> HybridSearchResult:
        """同期版検索"""
        return asyncio.get_event_loop().run_until_complete(
            self.search(query, methods, top_k)
        )
    
    async def _search_parallel(
        self,
        query: str,
        methods: List[SearchMethod],
    ) -> List[MethodSearchResult]:
        """並列検索"""
        tasks = []
        for method in methods:
            tasks.append(self._execute_method(query, method))
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Hybrid search timeout")
            results = [
                MethodSearchResult(
                    method=m,
                    hits=[],
                    success=False,
                    error="Timeout",
                )
                for m in methods
            ]
        
        # 例外を失敗結果に変換
        method_results = []
        for method, result in zip(methods, results):
            if isinstance(result, Exception):
                method_results.append(MethodSearchResult(
                    method=method,
                    hits=[],
                    success=False,
                    error=str(result),
                ))
            else:
                method_results.append(result)
        
        return method_results
    
    async def _search_sequential(
        self,
        query: str,
        methods: List[SearchMethod],
    ) -> List[MethodSearchResult]:
        """逐次検索"""
        results = []
        for method in methods:
            try:
                result = await asyncio.wait_for(
                    self._execute_method(query, method),
                    timeout=self.config.timeout_seconds / len(methods),
                )
                results.append(result)
            except asyncio.TimeoutError:
                results.append(MethodSearchResult(
                    method=method,
                    hits=[],
                    success=False,
                    error="Timeout",
                ))
            except Exception as e:
                results.append(MethodSearchResult(
                    method=method,
                    hits=[],
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    async def _execute_method(
        self,
        query: str,
        method: SearchMethod,
    ) -> MethodSearchResult:
        """個別メソッドを実行"""
        start_time = time.time()
        
        try:
            if method == SearchMethod.VECTOR:
                hits = await self._execute_vector(query)
            elif method == SearchMethod.LAZY:
                hits = await self._execute_lazy(query)
            elif method == SearchMethod.GLOBAL:
                hits = await self._execute_global(query)
            elif method == SearchMethod.LOCAL:
                hits = await self._execute_local(query)
            elif method == SearchMethod.KEYWORD:
                hits = await self._execute_keyword(query)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            elapsed = (time.time() - start_time) * 1000
            
            return MethodSearchResult(
                method=method,
                hits=hits,
                success=True,
                search_time_ms=elapsed,
            )
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.warning(f"Method {method.value} failed: {e}")
            
            return MethodSearchResult(
                method=method,
                hits=[],
                success=False,
                error=str(e),
                search_time_ms=elapsed,
            )
    
    async def _execute_vector(self, query: str) -> List[HybridSearchHit]:
        """ベクトル検索を実行"""
        if self.vector_search is None:
            raise ValueError("VectorSearch not configured")
        
        result = await self.vector_search.search(query, top_k=self.config.top_k)
        
        # 結果をHybridSearchHitに変換
        hits = []
        for hit in getattr(result, "hits", []):
            hits.append(HybridSearchHit(
                chunk_id=getattr(hit, "chunk_id", ""),
                score=getattr(hit, "score", 0.0),
                content=getattr(hit, "content", ""),
                metadata=getattr(hit, "metadata", {}),
                sources=["vector"],
                paper_id=getattr(hit, "paper_id", None),
                paper_title=getattr(hit, "paper_title", None),
                section_type=getattr(hit, "section_type", None),
            ))
        
        return hits
    
    async def _execute_lazy(self, query: str) -> List[HybridSearchHit]:
        """Lazy検索を実行"""
        if self.lazy_search is None:
            raise ValueError("LazySearch not configured")
        
        result = await self.lazy_search.search(query, top_k=self.config.top_k)
        
        # 結果をHybridSearchHitに変換
        hits = []
        chunks = getattr(result, "relevant_chunks", []) or getattr(result, "hits", [])
        for chunk in chunks:
            hits.append(HybridSearchHit(
                chunk_id=getattr(chunk, "chunk_id", getattr(chunk, "id", "")),
                score=getattr(chunk, "score", getattr(chunk, "relevance", 0.0)),
                content=getattr(chunk, "content", getattr(chunk, "text", "")),
                metadata=getattr(chunk, "metadata", {}),
                sources=["lazy"],
                paper_id=getattr(chunk, "paper_id", None),
                paper_title=getattr(chunk, "paper_title", None),
                section_type=getattr(chunk, "section_type", None),
            ))
        
        return hits
    
    async def _execute_global(self, query: str) -> List[HybridSearchHit]:
        """グローバル検索を実行"""
        if self.global_search is None:
            raise ValueError("GlobalSearch not configured")
        
        result = await self.global_search.search(query)
        
        # グローバル検索はコミュニティベースなので変換が必要
        hits = []
        for item in getattr(result, "map_results", []):
            hits.append(HybridSearchHit(
                chunk_id=getattr(item, "community_id", ""),
                score=getattr(item, "relevance_score", 0.0),
                content=getattr(item, "partial_answer", ""),
                metadata={"community_title": getattr(item, "community_title", "")},
                sources=["global"],
            ))
        
        return hits
    
    async def _execute_local(self, query: str) -> List[HybridSearchHit]:
        """ローカル検索を実行"""
        if self.local_search is None:
            raise ValueError("LocalSearch not configured")
        
        result = await self.local_search.search(query)
        
        hits = []
        for chunk in getattr(result, "relevant_chunks", []):
            hits.append(HybridSearchHit(
                chunk_id=getattr(chunk, "chunk_id", ""),
                score=getattr(chunk, "score", 0.0),
                content=getattr(chunk, "content", ""),
                metadata=getattr(chunk, "metadata", {}),
                sources=["local"],
            ))
        
        return hits
    
    async def _execute_keyword(self, query: str) -> List[HybridSearchHit]:
        """キーワード検索を実行（未実装）"""
        raise ValueError("Keyword search not implemented")


# ========== Factory ==========


def create_hybrid_search(
    methods: Optional[List[str]] = None,
    fusion: str = "rrf",
    **kwargs,
) -> HybridSearch:
    """
    HybridSearchインスタンスを作成
    
    Args:
        methods: 検索メソッド ["vector", "lazy", "global", "local"]
        fusion: 融合方式 "rrf", "weighted", "max", "combsum", "combmnz"
        **kwargs: 追加設定
    
    Returns:
        HybridSearch
    """
    # メソッドを変換
    search_methods = []
    if methods:
        for m in methods:
            try:
                search_methods.append(SearchMethod(m))
            except ValueError:
                logger.warning(f"Unknown method: {m}")
    else:
        search_methods = [SearchMethod.VECTOR, SearchMethod.LAZY]
    
    # 融合方式を変換
    try:
        fusion_method = FusionMethod(fusion)
    except ValueError:
        logger.warning(f"Unknown fusion: {fusion}, using RRF")
        fusion_method = FusionMethod.RRF
    
    config = HybridSearchConfig(
        methods=search_methods,
        fusion=fusion_method,
        rrf_k=kwargs.get("rrf_k", 60),
        top_k=kwargs.get("top_k", 10),
        min_score=kwargs.get("min_score", 0.0),
        parallel=kwargs.get("parallel", True),
        timeout_seconds=kwargs.get("timeout_seconds", 30.0),
    )
    
    return HybridSearch(
        config=config,
        vector_search=kwargs.get("vector_search"),
        lazy_search=kwargs.get("lazy_search"),
        global_search=kwargs.get("global_search"),
        local_search=kwargs.get("local_search"),
    )
