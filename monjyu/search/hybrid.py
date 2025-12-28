# Hybrid Search Engine
"""
Hybrid GraphRAG - 複数検索エンジン並列実行 & 結果マージ

REQ-ARC-003: Hybrid GraphRAG Controller
複数の検索方式(Vector, Lazy, Graph)を並列実行し、
RRFまたは重み付け融合で結果をマージする
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from monjyu.search.base import (
    Citation,
    SearchHit,
    SearchMode,
    SearchResults,
    SynthesizedAnswer,
)

if TYPE_CHECKING:
    from monjyu.lazy.engine import LazySearchEngine
    from monjyu.search.answer_synthesizer import LLMClientProtocol
    from monjyu.search.engine import VectorSearchEngine


# === Enums ===


class FusionMethod(Enum):
    """結果融合方式"""
    
    RRF = "rrf"                    # Reciprocal Rank Fusion
    WEIGHTED = "weighted"          # 重み付けスコア融合
    MAX = "max"                    # 最大スコア採用
    COMBSUM = "combsum"           # スコア合計
    COMBMNZ = "combmnz"           # スコア合計 × 出現回数


class SearchMethod(Enum):
    """検索方式"""
    
    VECTOR = "vector"
    LAZY = "lazy"
    KEYWORD = "keyword"


# === Data Classes ===


@dataclass
class HybridSearchConfig:
    """ハイブリッド検索設定"""
    
    # 使用する検索方式
    methods: list[SearchMethod] = field(
        default_factory=lambda: [SearchMethod.VECTOR, SearchMethod.LAZY]
    )
    
    # 融合設定
    fusion: FusionMethod = FusionMethod.RRF
    rrf_k: int = 60  # RRFの定数k
    
    # 方式ごとの重み（WEIGHTED融合時）
    method_weights: dict[SearchMethod, float] = field(default_factory=dict)
    
    # 検索パラメータ
    top_k: int = 10
    min_score: float = 0.0
    
    # 並列実行設定
    parallel: bool = True
    timeout_seconds: float = 30.0
    
    # 回答合成
    synthesize: bool = True
    
    def __post_init__(self):
        # デフォルト重み設定
        if not self.method_weights:
            self.method_weights = {
                SearchMethod.VECTOR: 0.5,
                SearchMethod.LAZY: 0.5,
                SearchMethod.KEYWORD: 0.3,
            }


@dataclass
class MethodResult:
    """各検索方式の結果"""
    
    method: SearchMethod
    hits: list[SearchHit]
    search_time_ms: float
    success: bool
    error: str | None = None
    
    # 追加メタデータ
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridSearchResult:
    """ハイブリッド検索結果"""
    
    query: str
    merged_hits: list[SearchHit]
    method_results: list[MethodResult]
    
    # 回答（合成時）
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    
    # メトリクス
    total_time_ms: float = 0.0
    fusion_method: FusionMethod = FusionMethod.RRF
    
    # 統計
    hit_sources: dict[str, list[str]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "query": self.query,
            "answer": self.answer,
            "merged_hits": [h.to_dict() for h in self.merged_hits],
            "citations": [c.to_dict() for c in self.citations],
            "method_results": [
                {
                    "method": r.method.value,
                    "hit_count": len(r.hits),
                    "search_time_ms": r.search_time_ms,
                    "success": r.success,
                }
                for r in self.method_results
            ],
            "total_time_ms": self.total_time_ms,
            "fusion_method": self.fusion_method.value,
        }


# === Result Merger ===


class ResultMerger:
    """検索結果マージャー"""
    
    def __init__(self, config: HybridSearchConfig):
        self.config = config
    
    def merge(
        self,
        method_results: list[MethodResult],
        top_k: int | None = None,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """
        複数の検索結果をマージ
        
        Args:
            method_results: 各検索方式の結果
            top_k: 返す結果数
        
        Returns:
            (マージされたヒット, ヒットID->ソース方式のマップ)
        """
        top_k = top_k or self.config.top_k
        
        # 融合方式に応じて処理
        if self.config.fusion == FusionMethod.RRF:
            return self._rrf_fusion(method_results, top_k)
        elif self.config.fusion == FusionMethod.WEIGHTED:
            return self._weighted_fusion(method_results, top_k)
        elif self.config.fusion == FusionMethod.MAX:
            return self._max_fusion(method_results, top_k)
        elif self.config.fusion == FusionMethod.COMBSUM:
            return self._combsum_fusion(method_results, top_k)
        elif self.config.fusion == FusionMethod.COMBMNZ:
            return self._combmnz_fusion(method_results, top_k)
        else:
            return self._rrf_fusion(method_results, top_k)
    
    def _rrf_fusion(
        self,
        method_results: list[MethodResult],
        top_k: int,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Score = Σ 1/(k + rank)
        """
        k = self.config.rrf_k
        scores: dict[str, float] = {}
        hit_map: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}
        
        for result in method_results:
            if not result.success:
                continue
            
            method_name = result.method.value
            
            for rank, hit in enumerate(result.hits, start=1):
                hit_id = hit.text_unit_id
                
                if hit_id not in scores:
                    scores[hit_id] = 0.0
                    hit_map[hit_id] = hit
                    sources[hit_id] = []
                
                # RRFスコア加算
                scores[hit_id] += 1.0 / (k + rank)
                
                # ソース記録
                if method_name not in sources[hit_id]:
                    sources[hit_id].append(method_name)
        
        # スコアでソート
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # 結果構築
        merged = []
        for hit_id in sorted_ids[:top_k]:
            hit = hit_map[hit_id]
            hit.score = scores[hit_id]
            merged.append(hit)
        
        return merged, sources
    
    def _weighted_fusion(
        self,
        method_results: list[MethodResult],
        top_k: int,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """重み付けスコア融合"""
        scores: dict[str, float] = {}
        hit_map: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}
        
        for result in method_results:
            if not result.success:
                continue
            
            method_name = result.method.value
            weight = self.config.method_weights.get(result.method, 1.0)
            
            # スコア正規化（最大スコアで割る）
            max_score = max((h.score for h in result.hits), default=1.0)
            if max_score == 0:
                max_score = 1.0
            
            for hit in result.hits:
                hit_id = hit.text_unit_id
                normalized_score = hit.score / max_score
                
                if hit_id not in scores:
                    scores[hit_id] = 0.0
                    hit_map[hit_id] = hit
                    sources[hit_id] = []
                
                scores[hit_id] += normalized_score * weight
                
                if method_name not in sources[hit_id]:
                    sources[hit_id].append(method_name)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged = []
        for hit_id in sorted_ids[:top_k]:
            hit = hit_map[hit_id]
            hit.score = scores[hit_id]
            merged.append(hit)
        
        return merged, sources
    
    def _max_fusion(
        self,
        method_results: list[MethodResult],
        top_k: int,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """最大スコア採用"""
        scores: dict[str, float] = {}
        hit_map: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}
        
        for result in method_results:
            if not result.success:
                continue
            
            method_name = result.method.value
            
            for hit in result.hits:
                hit_id = hit.text_unit_id
                
                if hit_id not in scores:
                    scores[hit_id] = hit.score
                    hit_map[hit_id] = hit
                    sources[hit_id] = [method_name]
                else:
                    if hit.score > scores[hit_id]:
                        scores[hit_id] = hit.score
                    if method_name not in sources[hit_id]:
                        sources[hit_id].append(method_name)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged = []
        for hit_id in sorted_ids[:top_k]:
            hit = hit_map[hit_id]
            hit.score = scores[hit_id]
            merged.append(hit)
        
        return merged, sources
    
    def _combsum_fusion(
        self,
        method_results: list[MethodResult],
        top_k: int,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """スコア合計"""
        scores: dict[str, float] = {}
        hit_map: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}
        
        for result in method_results:
            if not result.success:
                continue
            
            method_name = result.method.value
            
            for hit in result.hits:
                hit_id = hit.text_unit_id
                
                if hit_id not in scores:
                    scores[hit_id] = 0.0
                    hit_map[hit_id] = hit
                    sources[hit_id] = []
                
                scores[hit_id] += hit.score
                
                if method_name not in sources[hit_id]:
                    sources[hit_id].append(method_name)
        
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        merged = []
        for hit_id in sorted_ids[:top_k]:
            hit = hit_map[hit_id]
            hit.score = scores[hit_id]
            merged.append(hit)
        
        return merged, sources
    
    def _combmnz_fusion(
        self,
        method_results: list[MethodResult],
        top_k: int,
    ) -> tuple[list[SearchHit], dict[str, list[str]]]:
        """スコア合計 × 出現回数 (CombMNZ)"""
        scores: dict[str, float] = {}
        counts: dict[str, int] = {}
        hit_map: dict[str, SearchHit] = {}
        sources: dict[str, list[str]] = {}
        
        for result in method_results:
            if not result.success:
                continue
            
            method_name = result.method.value
            
            for hit in result.hits:
                hit_id = hit.text_unit_id
                
                if hit_id not in scores:
                    scores[hit_id] = 0.0
                    counts[hit_id] = 0
                    hit_map[hit_id] = hit
                    sources[hit_id] = []
                
                scores[hit_id] += hit.score
                counts[hit_id] += 1
                
                if method_name not in sources[hit_id]:
                    sources[hit_id].append(method_name)
        
        # CombMNZ: スコア × 出現回数
        final_scores = {
            hit_id: scores[hit_id] * counts[hit_id]
            for hit_id in scores
        }
        
        sorted_ids = sorted(final_scores.keys(), key=lambda x: final_scores[x], reverse=True)
        
        merged = []
        for hit_id in sorted_ids[:top_k]:
            hit = hit_map[hit_id]
            hit.score = final_scores[hit_id]
            merged.append(hit)
        
        return merged, sources


# === Hybrid Search Engine ===


class HybridSearchEngine:
    """
    Hybrid GraphRAG エンジン
    
    複数の検索方式を並列実行し、結果をマージして回答を生成
    """
    
    SYNTHESIS_PROMPT = """以下の複数の情報源から得られた情報に基づいて、質問に回答してください。

## 収集された情報
{context}

## 質問
{query}

## 指示
- 複数の情報源を統合して包括的に回答してください
- 矛盾する情報がある場合は、両方の見解を示してください
- 使用した情報源は [1], [2] のように引用してください
- 確実な情報と不確実な情報を区別してください
"""
    
    def __init__(
        self,
        vector_engine: "VectorSearchEngine | None" = None,
        lazy_engine: "LazySearchEngine | None" = None,
        llm_client: "LLMClientProtocol | None" = None,
        config: HybridSearchConfig | None = None,
    ):
        """
        Args:
            vector_engine: ベクトル検索エンジン
            lazy_engine: Lazy検索エンジン
            llm_client: LLMクライアント（回答合成用）
            config: ハイブリッド検索設定
        """
        self.vector_engine = vector_engine
        self.lazy_engine = lazy_engine
        self.llm_client = llm_client
        self.config = config or HybridSearchConfig()
        self.merger = ResultMerger(self.config)
    
    async def search(
        self,
        query: str,
        methods: list[SearchMethod] | None = None,
        top_k: int | None = None,
        synthesize: bool | None = None,
    ) -> HybridSearchResult:
        """
        ハイブリッド検索を実行
        
        Args:
            query: 検索クエリ
            methods: 使用する検索方式（Noneで設定値使用）
            top_k: 返す結果数
            synthesize: 回答を合成するか
        
        Returns:
            HybridSearchResult
        """
        start_time = time.time()
        
        methods = methods or self.config.methods
        top_k = top_k or self.config.top_k
        synthesize = synthesize if synthesize is not None else self.config.synthesize
        
        # 各検索方式を実行
        if self.config.parallel:
            method_results = await self._search_parallel(query, methods, top_k)
        else:
            method_results = await self._search_sequential(query, methods, top_k)
        
        # 結果をマージ
        merged_hits, hit_sources = self.merger.merge(method_results, top_k)
        
        # 回答合成
        answer = ""
        citations: list[Citation] = []
        
        if synthesize and self.llm_client and merged_hits:
            answer, citations = await self._synthesize_answer(query, merged_hits)
        
        total_time = (time.time() - start_time) * 1000
        
        return HybridSearchResult(
            query=query,
            merged_hits=merged_hits,
            method_results=method_results,
            answer=answer,
            citations=citations,
            total_time_ms=total_time,
            fusion_method=self.config.fusion,
            hit_sources=hit_sources,
        )
    
    def search_sync(
        self,
        query: str,
        methods: list[SearchMethod] | None = None,
        top_k: int | None = None,
        synthesize: bool | None = None,
    ) -> HybridSearchResult:
        """同期版ハイブリッド検索"""
        return asyncio.run(
            self.search(query, methods, top_k, synthesize)
        )
    
    async def _search_parallel(
        self,
        query: str,
        methods: list[SearchMethod],
        top_k: int,
    ) -> list[MethodResult]:
        """並列検索"""
        tasks = []
        
        for method in methods:
            task = self._execute_method(query, method, top_k)
            tasks.append(task)
        
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.timeout_seconds,
            )
        except asyncio.TimeoutError:
            results = [
                MethodResult(
                    method=m,
                    hits=[],
                    search_time_ms=0.0,
                    success=False,
                    error="Timeout",
                )
                for m in methods
            ]
        
        # 例外をMethodResultに変換
        method_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                method_results.append(MethodResult(
                    method=methods[i],
                    hits=[],
                    search_time_ms=0.0,
                    success=False,
                    error=str(result),
                ))
            else:
                method_results.append(result)
        
        return method_results
    
    async def _search_sequential(
        self,
        query: str,
        methods: list[SearchMethod],
        top_k: int,
    ) -> list[MethodResult]:
        """逐次検索"""
        results = []
        
        for method in methods:
            try:
                result = await self._execute_method(query, method, top_k)
                results.append(result)
            except Exception as e:
                results.append(MethodResult(
                    method=method,
                    hits=[],
                    search_time_ms=0.0,
                    success=False,
                    error=str(e),
                ))
        
        return results
    
    async def _execute_method(
        self,
        query: str,
        method: SearchMethod,
        top_k: int,
    ) -> MethodResult:
        """指定された検索方式を実行"""
        start_time = time.time()
        
        try:
            if method == SearchMethod.VECTOR:
                hits = await self._vector_search(query, top_k)
            elif method == SearchMethod.LAZY:
                hits = await self._lazy_search(query, top_k)
            elif method == SearchMethod.KEYWORD:
                hits = await self._keyword_search(query, top_k)
            else:
                raise ValueError(f"Unknown search method: {method}")
            
            search_time = (time.time() - start_time) * 1000
            
            return MethodResult(
                method=method,
                hits=hits,
                search_time_ms=search_time,
                success=True,
            )
        
        except Exception as e:
            search_time = (time.time() - start_time) * 1000
            return MethodResult(
                method=method,
                hits=[],
                search_time_ms=search_time,
                success=False,
                error=str(e),
            )
    
    async def _vector_search(self, query: str, top_k: int) -> list[SearchHit]:
        """ベクトル検索"""
        if not self.vector_engine:
            raise ValueError("Vector engine not configured")
        
        # VectorSearchEngine.search は同期メソッド
        import asyncio
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: self.vector_engine.search(query, top_k=top_k, synthesize=False)
        )
        
        return result.search_results.hits
    
    async def _lazy_search(self, query: str, top_k: int) -> list[SearchHit]:
        """Lazy検索"""
        if not self.lazy_engine:
            raise ValueError("Lazy engine not configured")
        
        result = await self.lazy_engine.search_async(query)
        
        # LazySearchResult からSearchHitに変換
        hits = []
        for claim in result.claims[:top_k]:
            hits.append(SearchHit(
                text_unit_id=claim.source_text_unit_id,
                document_id=claim.source_document_id,
                text=claim.text,
                score=claim.confidence,
            ))
        
        return hits
    
    async def _keyword_search(self, query: str, top_k: int) -> list[SearchHit]:
        """キーワード検索（BM25等）"""
        if not self.vector_engine:
            raise ValueError("Vector engine not configured")
        
        # ハイブリッド検索でキーワード重視
        import asyncio
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            lambda: self.vector_engine.search(
                query, 
                top_k=top_k, 
                mode=SearchMode.KEYWORD,
                synthesize=False
            )
        )
        
        return result.search_results.hits
    
    async def _synthesize_answer(
        self,
        query: str,
        hits: list[SearchHit],
    ) -> tuple[str, list[Citation]]:
        """回答を合成"""
        # コンテキスト構築
        context_parts = []
        citations = []
        
        for i, hit in enumerate(hits, start=1):
            context_parts.append(f"[{i}] {hit.text[:500]}")
            citations.append(Citation(
                text_unit_id=hit.text_unit_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                text_snippet=hit.text[:200],
                relevance_score=hit.score,
            ))
        
        context = "\n\n".join(context_parts)
        prompt = self.SYNTHESIS_PROMPT.format(context=context, query=query)
        
        # LLM呼び出し
        answer = await self._call_llm(prompt)
        
        return answer, citations
    
    async def _call_llm(self, prompt: str) -> str:
        """LLM呼び出し"""
        if hasattr(self.llm_client, 'generate_async'):
            return await self.llm_client.generate_async(prompt)
        elif hasattr(self.llm_client, 'generate'):
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.llm_client.generate,
                prompt
            )
        else:
            raise ValueError("LLM client must have generate or generate_async method")


# === Factory Functions ===


def create_hybrid_engine(
    vector_engine: "VectorSearchEngine | None" = None,
    lazy_engine: "LazySearchEngine | None" = None,
    llm_client: "LLMClientProtocol | None" = None,
    methods: list[str] | None = None,
    fusion: str = "rrf",
    **kwargs,
) -> HybridSearchEngine:
    """
    ハイブリッド検索エンジンを作成
    
    Args:
        vector_engine: ベクトル検索エンジン
        lazy_engine: Lazy検索エンジン
        llm_client: LLMクライアント
        methods: 検索方式リスト ["vector", "lazy", "keyword"]
        fusion: 融合方式 "rrf", "weighted", "max", "combsum", "combmnz"
        **kwargs: 追加設定
    
    Returns:
        HybridSearchEngine
    """
    # 検索方式を変換
    search_methods = []
    if methods:
        for m in methods:
            try:
                search_methods.append(SearchMethod(m))
            except ValueError:
                pass
    
    if not search_methods:
        search_methods = [SearchMethod.VECTOR, SearchMethod.LAZY]
    
    # 融合方式を変換
    try:
        fusion_method = FusionMethod(fusion)
    except ValueError:
        fusion_method = FusionMethod.RRF
    
    config = HybridSearchConfig(
        methods=search_methods,
        fusion=fusion_method,
        top_k=kwargs.get("top_k", 10),
        rrf_k=kwargs.get("rrf_k", 60),
        parallel=kwargs.get("parallel", True),
        synthesize=kwargs.get("synthesize", True),
    )
    
    return HybridSearchEngine(
        vector_engine=vector_engine,
        lazy_engine=lazy_engine,
        llm_client=llm_client,
        config=config,
    )
