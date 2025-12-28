"""Streaming Search Engine - REQ-QRY-006.

検索から回答合成まで全体をストリーミングするエンジン。
検索進捗・結果・回答をリアルタイム配信。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, List, Optional, Protocol

from monjyu.search.base import Citation, SearchHit, SearchResults, SynthesizedAnswer
from monjyu.search.streaming.types import (
    StreamCallbacks,
    StreamChunk,
    StreamChunkType,
    StreamConfig,
    StreamMetadata,
    StreamState,
)
from monjyu.search.streaming.synthesizer import (
    StreamingAnswerSynthesizer,
    StreamingLLMProtocol,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# === Protocols ===


class SearchEngineProtocol(Protocol):
    """検索エンジンプロトコル"""
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> SearchResults:
        """検索を実行"""
        ...


class ProgressCallback(Protocol):
    """進捗コールバックプロトコル"""
    
    async def on_progress(self, progress: float, message: str) -> None:
        """進捗更新"""
        ...


# === Data Classes ===


@dataclass
class StreamingSearchConfig:
    """ストリーミング検索設定"""
    
    # 検索設定
    top_k: int = 10
    min_score: float = 0.0
    
    # ストリーミング設定
    stream_config: StreamConfig = field(default_factory=StreamConfig)
    
    # 検索進捗通知
    notify_search_progress: bool = True
    
    # 検索結果ストリーミング
    stream_search_results: bool = True
    stream_result_batch_size: int = 3  # バッチサイズ
    
    # 合成設定
    synthesize: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "top_k": self.top_k,
            "min_score": self.min_score,
            "stream_config": self.stream_config.to_dict(),
            "notify_search_progress": self.notify_search_progress,
            "stream_search_results": self.stream_search_results,
            "stream_result_batch_size": self.stream_result_batch_size,
            "synthesize": self.synthesize,
        }


@dataclass
class StreamingSearchResult:
    """ストリーミング検索結果"""
    
    # 検索結果
    search_results: SearchResults
    
    # 合成回答
    answer: Optional[SynthesizedAnswer] = None
    
    # メタデータ
    metadata: Optional[StreamMetadata] = None
    
    # 全チャンク
    chunks: List[StreamChunk] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "search_results": self.search_results.to_dict(),
            "answer": self.answer.to_dict() if self.answer else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "chunk_count": len(self.chunks),
        }


# === Streaming Search Engine ===


class StreamingSearchEngine:
    """ストリーミング検索エンジン
    
    検索から回答合成まで全プロセスをストリーミング。
    各フェーズの進捗と結果をリアルタイムで配信。
    """
    
    def __init__(
        self,
        search_engine: SearchEngineProtocol,
        llm_client: StreamingLLMProtocol,
        config: Optional[StreamingSearchConfig] = None,
        system_prompt: str | None = None,
    ):
        """
        Args:
            search_engine: 検索エンジン
            llm_client: ストリーミングLLMクライアント
            config: ストリーミング検索設定
            system_prompt: 合成用システムプロンプト
        """
        self.search_engine = search_engine
        self.llm_client = llm_client
        self.config = config or StreamingSearchConfig()
        
        self._synthesizer = StreamingAnswerSynthesizer(
            llm_client=llm_client,
            config=self.config.stream_config,
            system_prompt=system_prompt,
        )
        
        self._metadata: Optional[StreamMetadata] = None
        self._cancelled = False
    
    async def search_stream(
        self,
        query: str,
        callbacks: Optional[StreamCallbacks] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        ストリーミング検索を実行
        
        Args:
            query: 検索クエリ
            callbacks: コールバック
            **kwargs: 追加パラメータ
            
        Yields:
            StreamChunk
        """
        self._cancelled = False
        self._metadata = StreamMetadata(
            query=query,
            state=StreamState.SEARCHING,
        )
        
        chunk_index = 0
        all_chunks: List[StreamChunk] = []
        
        # ===== 検索フェーズ =====
        
        # 検索開始通知
        search_start_chunk = StreamChunk(
            chunk_type=StreamChunkType.SEARCH_START,
            content="",
            index=chunk_index,
            data={"query": query, "top_k": self.config.top_k},
        )
        chunk_index += 1
        all_chunks.append(search_start_chunk)
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(search_start_chunk)
        yield search_start_chunk
        
        # 検索実行
        self._metadata.search_start_time = time.time()
        
        try:
            # 進捗通知（オプション）
            if self.config.notify_search_progress:
                progress_chunk = StreamChunk(
                    chunk_type=StreamChunkType.SEARCH_PROGRESS,
                    content="検索中...",
                    index=chunk_index,
                    data={"progress": 0.5},
                )
                chunk_index += 1
                all_chunks.append(progress_chunk)
                
                if callbacks and callbacks.on_chunk:
                    await callbacks.on_chunk(progress_chunk)
                yield progress_chunk
            
            # 検索実行
            top_k = kwargs.get("top_k", self.config.top_k)
            search_results = await self.search_engine.search(
                query=query,
                top_k=top_k,
                **kwargs,
            )
            
            self._metadata.search_end_time = time.time()
            self._metadata.total_search_hits = len(search_results.hits)
            
        except asyncio.CancelledError:
            self._metadata.state = StreamState.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Search error: {e}")
            self._metadata.state = StreamState.ERROR
            self._metadata.error_message = str(e)
            
            error_chunk = StreamChunk.error_chunk(str(e), chunk_index)
            chunk_index += 1
            
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield error_chunk
            return
        
        # 検索結果ストリーミング（オプション）
        if self.config.stream_search_results and search_results.hits:
            # バッチ単位で結果を配信
            batch_size = self.config.stream_result_batch_size
            for i in range(0, len(search_results.hits), batch_size):
                batch = search_results.hits[i:i + batch_size]
                
                result_chunk = StreamChunk(
                    chunk_type=StreamChunkType.SEARCH_RESULT,
                    content="",
                    index=chunk_index,
                    data={
                        "hits": [h.to_dict() for h in batch],
                        "batch_index": i // batch_size,
                        "total_hits": len(search_results.hits),
                    },
                )
                chunk_index += 1
                all_chunks.append(result_chunk)
                
                if callbacks and callbacks.on_chunk:
                    await callbacks.on_chunk(result_chunk)
                yield result_chunk
        
        # 検索完了通知
        search_complete_chunk = StreamChunk(
            chunk_type=StreamChunkType.SEARCH_COMPLETE,
            content="",
            index=chunk_index,
            data={
                "total_hits": len(search_results.hits),
                "search_time_ms": self._metadata.search_time_ms,
            },
        )
        chunk_index += 1
        all_chunks.append(search_complete_chunk)
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(search_complete_chunk)
        yield search_complete_chunk
        
        # キャンセルチェック
        if self._cancelled:
            self._metadata.state = StreamState.CANCELLED
            return
        
        # ===== 合成フェーズ =====
        
        if self.config.synthesize:
            self._metadata.state = StreamState.SYNTHESIZING
            
            # 合成器からストリーミング
            async for synthesis_chunk in self._synthesizer.synthesize_stream(
                query=query,
                context=search_results.hits,
                callbacks=callbacks,
            ):
                # インデックスを調整
                synthesis_chunk.index = chunk_index
                chunk_index += 1
                all_chunks.append(synthesis_chunk)
                
                # メタデータ更新
                if synthesis_chunk.chunk_type == StreamChunkType.TEXT:
                    self._metadata.tokens_generated += len(synthesis_chunk.content.split())
                    self._metadata.chunks_sent += 1
                
                yield synthesis_chunk
        
        # ===== 終了フェーズ =====
        
        self._metadata.state = StreamState.COMPLETED
        
        # 終了チャンク
        end_chunk = StreamChunk.end_chunk(self._metadata, chunk_index)
        all_chunks.append(end_chunk)
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(end_chunk)
        if callbacks and callbacks.on_complete:
            callbacks.on_complete(self._metadata)
        
        yield end_chunk
    
    async def search(
        self,
        query: str,
        **kwargs,
    ) -> StreamingSearchResult:
        """
        ストリーミング検索（結果収集版）
        
        全チャンクを収集して最終結果を返す。
        
        Args:
            query: 検索クエリ
            **kwargs: 追加パラメータ
            
        Returns:
            StreamingSearchResult
        """
        chunks: List[StreamChunk] = []
        search_results: Optional[SearchResults] = None
        answer_text = ""
        citations: List[Citation] = []
        
        async for chunk in self.search_stream(query, **kwargs):
            chunks.append(chunk)
            
            # 検索結果収集
            if chunk.chunk_type == StreamChunkType.SEARCH_RESULT:
                if chunk.data and "hits" in chunk.data:
                    if search_results is None:
                        from monjyu.search.base import SearchHit, SearchResults
                        search_results = SearchResults(
                            hits=[],
                            total_count=chunk.data.get("total_hits", 0),
                        )
                    
                    for hit_data in chunk.data["hits"]:
                        search_results.hits.append(SearchHit.from_dict(hit_data))
            
            # 回答テキスト収集
            elif chunk.chunk_type == StreamChunkType.TEXT:
                answer_text += chunk.content
            
            # 引用収集
            elif chunk.chunk_type == StreamChunkType.CITATION:
                if chunk.data:
                    citations.append(
                        Citation(
                            text_unit_id=chunk.data.get("citation_id", ""),
                            document_id="",
                            document_title=chunk.data.get("document_title", ""),
                            text_snippet=chunk.data.get("text_snippet", ""),
                            relevance_score=0.0,
                        )
                    )
        
        # デフォルト検索結果
        if search_results is None:
            search_results = SearchResults(hits=[], total_count=0)
        
        # 回答オブジェクト作成
        answer = None
        if answer_text:
            answer = SynthesizedAnswer(
                answer=answer_text,
                citations=citations,
                confidence=0.0,
                model=self.llm_client.model_name,
            )
        
        return StreamingSearchResult(
            search_results=search_results,
            answer=answer,
            metadata=self._metadata,
            chunks=chunks,
        )
    
    def cancel(self) -> None:
        """ストリーミングをキャンセル"""
        self._cancelled = True
        self._synthesizer.cancel()
    
    @property
    def metadata(self) -> Optional[StreamMetadata]:
        """現在のメタデータ"""
        return self._metadata


# === Multi-Engine Streaming ===


class MultiEngineStreamingSearch:
    """複数エンジンのストリーミング検索
    
    複数の検索エンジンを並列実行し、
    結果をストリーミングでマージ。
    """
    
    def __init__(
        self,
        engines: Dict[str, SearchEngineProtocol],
        llm_client: StreamingLLMProtocol,
        config: Optional[StreamingSearchConfig] = None,
        system_prompt: str | None = None,
    ):
        """
        Args:
            engines: 検索エンジン辞書 {名前: エンジン}
            llm_client: ストリーミングLLMクライアント
            config: ストリーミング検索設定
            system_prompt: 合成用システムプロンプト
        """
        self.engines = engines
        self.llm_client = llm_client
        self.config = config or StreamingSearchConfig()
        
        self._synthesizer = StreamingAnswerSynthesizer(
            llm_client=llm_client,
            config=self.config.stream_config,
            system_prompt=system_prompt,
        )
        
        self._metadata: Optional[StreamMetadata] = None
        self._cancelled = False
    
    async def search_stream(
        self,
        query: str,
        engine_names: Optional[List[str]] = None,
        callbacks: Optional[StreamCallbacks] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        複数エンジンでストリーミング検索
        
        Args:
            query: 検索クエリ
            engine_names: 使用するエンジン名（省略時は全エンジン）
            callbacks: コールバック
            **kwargs: 追加パラメータ
            
        Yields:
            StreamChunk
        """
        self._cancelled = False
        self._metadata = StreamMetadata(
            query=query,
            state=StreamState.SEARCHING,
        )
        
        chunk_index = 0
        
        # 使用エンジン決定
        engines_to_use = engine_names or list(self.engines.keys())
        
        # 検索開始通知
        start_chunk = StreamChunk(
            chunk_type=StreamChunkType.SEARCH_START,
            content="",
            index=chunk_index,
            data={
                "query": query,
                "engines": engines_to_use,
            },
        )
        chunk_index += 1
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(start_chunk)
        yield start_chunk
        
        # 並列検索実行
        self._metadata.search_start_time = time.time()
        
        async def search_with_engine(name: str) -> tuple[str, SearchResults]:
            engine = self.engines[name]
            top_k = kwargs.get("top_k", self.config.top_k)
            results = await engine.search(query=query, top_k=top_k, **kwargs)
            return name, results
        
        all_hits: List[SearchHit] = []
        
        try:
            # 並列実行
            tasks = [
                search_with_engine(name)
                for name in engines_to_use
                if name in self.engines
            ]
            
            completed = 0
            for coro in asyncio.as_completed(tasks):
                engine_name, results = await coro
                completed += 1
                
                # 進捗通知
                progress_chunk = StreamChunk(
                    chunk_type=StreamChunkType.SEARCH_PROGRESS,
                    content=f"{engine_name} 完了",
                    index=chunk_index,
                    data={
                        "engine": engine_name,
                        "hits": len(results.hits),
                        "progress": completed / len(tasks),
                    },
                )
                chunk_index += 1
                
                if callbacks and callbacks.on_chunk:
                    await callbacks.on_chunk(progress_chunk)
                yield progress_chunk
                
                # 結果追加
                all_hits.extend(results.hits)
                
                # 検索結果ストリーミング
                if self.config.stream_search_results:
                    result_chunk = StreamChunk(
                        chunk_type=StreamChunkType.SEARCH_RESULT,
                        content="",
                        index=chunk_index,
                        data={
                            "engine": engine_name,
                            "hits": [h.to_dict() for h in results.hits],
                        },
                    )
                    chunk_index += 1
                    
                    if callbacks and callbacks.on_chunk:
                        await callbacks.on_chunk(result_chunk)
                    yield result_chunk
            
            self._metadata.search_end_time = time.time()
            self._metadata.total_search_hits = len(all_hits)
            
        except asyncio.CancelledError:
            self._metadata.state = StreamState.CANCELLED
            raise
        except Exception as e:
            logger.error(f"Multi-engine search error: {e}")
            self._metadata.state = StreamState.ERROR
            self._metadata.error_message = str(e)
            
            error_chunk = StreamChunk.error_chunk(str(e), chunk_index)
            if callbacks and callbacks.on_error:
                callbacks.on_error(e)
            yield error_chunk
            return
        
        # 重複排除とスコアソート
        seen_ids = set()
        unique_hits = []
        for hit in sorted(all_hits, key=lambda h: h.score, reverse=True):
            if hit.text_unit_id not in seen_ids:
                seen_ids.add(hit.text_unit_id)
                unique_hits.append(hit)
        
        # Top-K制限
        top_k = kwargs.get("top_k", self.config.top_k)
        unique_hits = unique_hits[:top_k]
        
        # 検索完了通知
        complete_chunk = StreamChunk(
            chunk_type=StreamChunkType.SEARCH_COMPLETE,
            content="",
            index=chunk_index,
            data={
                "total_hits": len(unique_hits),
                "search_time_ms": self._metadata.search_time_ms,
            },
        )
        chunk_index += 1
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(complete_chunk)
        yield complete_chunk
        
        # 合成フェーズ
        if self.config.synthesize and unique_hits:
            self._metadata.state = StreamState.SYNTHESIZING
            
            async for synthesis_chunk in self._synthesizer.synthesize_stream(
                query=query,
                context=unique_hits,
                callbacks=callbacks,
            ):
                synthesis_chunk.index = chunk_index
                chunk_index += 1
                yield synthesis_chunk
        
        # 終了
        self._metadata.state = StreamState.COMPLETED
        
        end_chunk = StreamChunk.end_chunk(self._metadata, chunk_index)
        
        if callbacks and callbacks.on_chunk:
            await callbacks.on_chunk(end_chunk)
        if callbacks and callbacks.on_complete:
            callbacks.on_complete(self._metadata)
        
        yield end_chunk
    
    def cancel(self) -> None:
        """ストリーミングをキャンセル"""
        self._cancelled = True
        self._synthesizer.cancel()


# === Factory Functions ===


def create_streaming_engine(
    search_engine: SearchEngineProtocol,
    llm_client: StreamingLLMProtocol,
    config: Optional[StreamingSearchConfig] = None,
    system_prompt: str | None = None,
) -> StreamingSearchEngine:
    """ストリーミング検索エンジンを作成
    
    Args:
        search_engine: 検索エンジン
        llm_client: ストリーミングLLMクライアント
        config: ストリーミング検索設定
        system_prompt: 合成用システムプロンプト
        
    Returns:
        StreamingSearchEngine
    """
    return StreamingSearchEngine(
        search_engine=search_engine,
        llm_client=llm_client,
        config=config,
        system_prompt=system_prompt,
    )


def create_multi_engine_streaming(
    engines: Dict[str, SearchEngineProtocol],
    llm_client: StreamingLLMProtocol,
    config: Optional[StreamingSearchConfig] = None,
    system_prompt: str | None = None,
) -> MultiEngineStreamingSearch:
    """マルチエンジンストリーミング検索を作成
    
    Args:
        engines: 検索エンジン辞書
        llm_client: ストリーミングLLMクライアント
        config: ストリーミング検索設定
        system_prompt: 合成用システムプロンプト
        
    Returns:
        MultiEngineStreamingSearch
    """
    return MultiEngineStreamingSearch(
        engines=engines,
        llm_client=llm_client,
        config=config,
        system_prompt=system_prompt,
    )
