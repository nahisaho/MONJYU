# MONJYU Streaming Service
"""
monjyu.api.streaming - ストリーミングレスポンス

REQ-API-003: ストリーミング出力
WHERE streaming is enabled, the system SHALL stream response tokens as they are generated.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Generic, TypeVar

# Type variable for generic streaming content
T = TypeVar("T")


# ============================================================
# Enums
# ============================================================


class StreamingStatus(Enum):
    """ストリーミング状態"""
    
    PENDING = "pending"      # 開始待ち
    STREAMING = "streaming"  # ストリーミング中
    COMPLETED = "completed"  # 完了
    CANCELLED = "cancelled"  # キャンセル
    ERROR = "error"          # エラー


class ChunkType(Enum):
    """チャンクタイプ"""
    
    TEXT = "text"            # テキストトークン
    CITATION = "citation"    # 引用情報
    METADATA = "metadata"    # メタデータ
    PROGRESS = "progress"    # 進捗情報
    ERROR = "error"          # エラー
    DONE = "done"            # 完了マーカー


# ============================================================
# Exceptions
# ============================================================


class StreamingError(Exception):
    """ストリーミングエラー基底クラス"""
    pass


class StreamingCancelledError(StreamingError):
    """ストリーミングがキャンセルされた"""
    pass


class StreamingTimeoutError(StreamingError):
    """ストリーミングタイムアウト"""
    pass


# ============================================================
# Data Classes
# ============================================================


@dataclass
class StreamingChunk:
    """ストリーミングチャンク
    
    ストリーミングレスポンスの単位。
    テキスト、引用、メタデータなどを運ぶ。
    """
    
    # コンテンツ
    content: str
    chunk_type: ChunkType = ChunkType.TEXT
    
    # メタデータ
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    stream_id: str = ""
    sequence: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # 追加データ（引用情報など）
    data: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_text(self) -> bool:
        """テキストチャンクか"""
        return self.chunk_type == ChunkType.TEXT
    
    @property
    def is_done(self) -> bool:
        """完了チャンクか"""
        return self.chunk_type == ChunkType.DONE
    
    @property
    def is_error(self) -> bool:
        """エラーチャンクか"""
        return self.chunk_type == ChunkType.ERROR
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "chunk_id": self.chunk_id,
            "stream_id": self.stream_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "data": self.data,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamingChunk":
        """辞書から生成"""
        return cls(
            content=data["content"],
            chunk_type=ChunkType(data["chunk_type"]),
            chunk_id=data.get("chunk_id", str(uuid.uuid4())[:8]),
            stream_id=data.get("stream_id", ""),
            sequence=data.get("sequence", 0),
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data", {}),
        )
    
    @classmethod
    def text(cls, content: str, **kwargs: Any) -> "StreamingChunk":
        """テキストチャンクを生成"""
        return cls(content=content, chunk_type=ChunkType.TEXT, **kwargs)
    
    @classmethod
    def citation(cls, content: str, citation_data: dict[str, Any], **kwargs: Any) -> "StreamingChunk":
        """引用チャンクを生成"""
        return cls(
            content=content, 
            chunk_type=ChunkType.CITATION, 
            data=citation_data,
            **kwargs
        )
    
    @classmethod
    def progress(cls, message: str, percentage: float = 0.0, **kwargs: Any) -> "StreamingChunk":
        """進捗チャンクを生成"""
        return cls(
            content=message,
            chunk_type=ChunkType.PROGRESS,
            data={"percentage": percentage},
            **kwargs
        )
    
    @classmethod
    def done(cls, summary: str = "", **kwargs: Any) -> "StreamingChunk":
        """完了チャンクを生成"""
        return cls(content=summary, chunk_type=ChunkType.DONE, **kwargs)
    
    @classmethod
    def error(cls, message: str, error_code: str = "", **kwargs: Any) -> "StreamingChunk":
        """エラーチャンクを生成"""
        return cls(
            content=message,
            chunk_type=ChunkType.ERROR,
            data={"error_code": error_code},
            **kwargs
        )


@dataclass
class StreamingState:
    """ストリーミング状態
    
    ストリーミングセッションの状態を追跡。
    """
    
    stream_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: StreamingStatus = StreamingStatus.PENDING
    
    # 統計
    chunks_sent: int = 0
    bytes_sent: int = 0
    tokens_sent: int = 0
    
    # タイミング
    started_at: float | None = None
    completed_at: float | None = None
    
    # エラー
    error: str | None = None
    
    # キャンセル
    _cancel_requested: bool = field(default=False, repr=False)
    
    @property
    def is_active(self) -> bool:
        """アクティブか"""
        return self.status == StreamingStatus.STREAMING
    
    @property
    def is_completed(self) -> bool:
        """完了したか"""
        return self.status in (StreamingStatus.COMPLETED, StreamingStatus.CANCELLED, StreamingStatus.ERROR)
    
    @property
    def duration_ms(self) -> float:
        """処理時間(ms)"""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000
    
    @property
    def tokens_per_second(self) -> float:
        """トークン/秒"""
        duration_s = self.duration_ms / 1000
        if duration_s <= 0:
            return 0.0
        return self.tokens_sent / duration_s
    
    def start(self) -> None:
        """ストリーミング開始"""
        self.status = StreamingStatus.STREAMING
        self.started_at = time.time()
    
    def complete(self) -> None:
        """ストリーミング完了"""
        self.status = StreamingStatus.COMPLETED
        self.completed_at = time.time()
    
    def cancel(self) -> None:
        """ストリーミングキャンセル"""
        self._cancel_requested = True
        self.status = StreamingStatus.CANCELLED
        self.completed_at = time.time()
    
    def fail(self, error: str) -> None:
        """ストリーミング失敗"""
        self.status = StreamingStatus.ERROR
        self.error = error
        self.completed_at = time.time()
    
    def record_chunk(self, chunk: StreamingChunk) -> None:
        """チャンク送信を記録"""
        self.chunks_sent += 1
        self.bytes_sent += len(chunk.content.encode("utf-8"))
        # 簡易トークンカウント（空白区切り）
        if chunk.is_text:
            self.tokens_sent += len(chunk.content.split())
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "stream_id": self.stream_id,
            "status": self.status.value,
            "chunks_sent": self.chunks_sent,
            "bytes_sent": self.bytes_sent,
            "tokens_sent": self.tokens_sent,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "tokens_per_second": self.tokens_per_second,
            "error": self.error,
        }


@dataclass
class StreamingConfig:
    """ストリーミング設定"""
    
    # バッファリング
    buffer_size: int = 10  # チャンクをバッファリングする数
    flush_interval_ms: float = 50.0  # フラッシュ間隔(ms)
    
    # タイムアウト
    timeout_seconds: float = 300.0  # 全体タイムアウト
    idle_timeout_seconds: float = 30.0  # アイドルタイムアウト
    
    # コールバック
    on_chunk: Callable[[StreamingChunk], None] | None = None
    on_complete: Callable[[StreamingState], None] | None = None
    on_error: Callable[[Exception], None] | None = None
    
    # オプション
    include_citations: bool = True  # 引用情報を含めるか
    include_progress: bool = False  # 進捗情報を含めるか
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "buffer_size": self.buffer_size,
            "flush_interval_ms": self.flush_interval_ms,
            "timeout_seconds": self.timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "include_citations": self.include_citations,
            "include_progress": self.include_progress,
        }


@dataclass
class StreamingResult:
    """ストリーミング完了結果"""
    
    stream_id: str
    full_response: str
    chunks: list[StreamingChunk] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    state: StreamingState | None = None
    
    @property
    def chunk_count(self) -> int:
        """チャンク数"""
        return len(self.chunks)
    
    @property
    def citation_count(self) -> int:
        """引用数"""
        return len(self.citations)
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "stream_id": self.stream_id,
            "full_response": self.full_response,
            "chunk_count": self.chunk_count,
            "citation_count": self.citation_count,
            "state": self.state.to_dict() if self.state else None,
        }


# ============================================================
# Protocols
# ============================================================


class StreamingSourceProtocol(ABC):
    """ストリーミングソースプロトコル
    
    LLMや他のソースからストリーミングデータを取得するための
    インターフェース。
    """
    
    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """テキストをストリーミング"""
        ...
    
    @abstractmethod
    async def stream_with_citations(
        self,
        prompt: str,
        context: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        """引用付きでストリーミング"""
        ...


# ============================================================
# StreamingService
# ============================================================


class StreamingService:
    """ストリーミングサービス
    
    REQ-API-003: ストリーミング出力
    
    LLM応答をリアルタイムでストリーミングし、
    トークンごとにクライアントに送信する。
    
    Example:
        service = StreamingService()
        
        async for chunk in service.stream_search(query, context):
            print(chunk.content, end="", flush=True)
    """
    
    def __init__(
        self,
        config: StreamingConfig | None = None,
        source: StreamingSourceProtocol | None = None,
    ):
        """初期化
        
        Args:
            config: ストリーミング設定
            source: ストリーミングソース（LLMなど）
        """
        self.config = config or StreamingConfig()
        self._source = source
        self._active_streams: dict[str, StreamingState] = {}
    
    @property
    def active_stream_count(self) -> int:
        """アクティブなストリーム数"""
        return sum(1 for s in self._active_streams.values() if s.is_active)
    
    def get_stream_state(self, stream_id: str) -> StreamingState | None:
        """ストリーム状態を取得"""
        return self._active_streams.get(stream_id)
    
    def cancel_stream(self, stream_id: str) -> bool:
        """ストリームをキャンセル
        
        Args:
            stream_id: ストリームID
            
        Returns:
            キャンセルできたかどうか
        """
        state = self._active_streams.get(stream_id)
        if state and state.is_active:
            state.cancel()
            return True
        return False
    
    async def stream_text(
        self,
        text: str,
        delay_ms: float = 20.0,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """テキストをストリーミング（デモ/テスト用）
        
        Args:
            text: ストリーミングするテキスト
            delay_ms: チャンク間の遅延(ms)
            
        Yields:
            StreamingChunk
        """
        state = StreamingState()
        self._active_streams[state.stream_id] = state
        state.start()
        
        try:
            # 単語ごとにストリーミング
            words = text.split()
            for i, word in enumerate(words):
                if state._cancel_requested:
                    raise StreamingCancelledError("Stream cancelled")
                
                content = word + (" " if i < len(words) - 1 else "")
                chunk = StreamingChunk.text(
                    content=content,
                    stream_id=state.stream_id,
                    sequence=i,
                )
                
                state.record_chunk(chunk)
                yield chunk
                
                if self.config.on_chunk:
                    self.config.on_chunk(chunk)
                
                await asyncio.sleep(delay_ms / 1000)
            
            # 完了チャンク
            done_chunk = StreamingChunk.done(
                stream_id=state.stream_id,
                sequence=len(words),
            )
            yield done_chunk
            
            state.complete()
            
            if self.config.on_complete:
                self.config.on_complete(state)
                
        except StreamingCancelledError:
            state.cancel()
            raise
        except Exception as e:
            state.fail(str(e))
            if self.config.on_error:
                self.config.on_error(e)
            raise StreamingError(f"Streaming failed: {e}") from e
    
    async def stream_search(
        self,
        query: str,
        context: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamingChunk, None]:
        """検索結果をストリーミング
        
        Args:
            query: 検索クエリ
            context: コンテキスト（チャンク情報など）
            **kwargs: 追加パラメータ
            
        Yields:
            StreamingChunk
        """
        state = StreamingState()
        self._active_streams[state.stream_id] = state
        state.start()
        
        sequence = 0
        citations_sent: list[dict[str, Any]] = []
        
        try:
            # タイムアウト設定
            start_time = time.time()
            last_chunk_time = start_time
            
            # 進捗チャンク（オプション）
            if self.config.include_progress:
                progress_chunk = StreamingChunk.progress(
                    "Starting search...",
                    percentage=0.0,
                    stream_id=state.stream_id,
                    sequence=sequence,
                )
                sequence += 1
                yield progress_chunk
            
            # ソースがある場合はそれを使用
            if self._source and context:
                async for text, citations in self._source.stream_with_citations(
                    query, context, **kwargs
                ):
                    # タイムアウトチェック
                    current_time = time.time()
                    if current_time - start_time > self.config.timeout_seconds:
                        raise StreamingTimeoutError("Stream timeout exceeded")
                    if current_time - last_chunk_time > self.config.idle_timeout_seconds:
                        raise StreamingTimeoutError("Idle timeout exceeded")
                    last_chunk_time = current_time
                    
                    # キャンセルチェック
                    if state._cancel_requested:
                        raise StreamingCancelledError("Stream cancelled")
                    
                    # テキストチャンク
                    chunk = StreamingChunk.text(
                        content=text,
                        stream_id=state.stream_id,
                        sequence=sequence,
                    )
                    sequence += 1
                    state.record_chunk(chunk)
                    yield chunk
                    
                    if self.config.on_chunk:
                        self.config.on_chunk(chunk)
                    
                    # 引用チャンク（オプション）
                    if self.config.include_citations and citations:
                        for cit in citations:
                            if cit not in citations_sent:
                                citations_sent.append(cit)
                                cit_chunk = StreamingChunk.citation(
                                    content=f"[{len(citations_sent)}]",
                                    citation_data=cit,
                                    stream_id=state.stream_id,
                                    sequence=sequence,
                                )
                                sequence += 1
                                yield cit_chunk
            else:
                # ソースがない場合はデモレスポンス
                demo_response = f"This is a demo streaming response for query: '{query}'"
                async for chunk in self.stream_text(demo_response):
                    chunk.stream_id = state.stream_id
                    yield chunk
                    sequence += 1
            
            # 完了チャンク
            done_chunk = StreamingChunk.done(
                summary=f"Completed with {len(citations_sent)} citations",
                stream_id=state.stream_id,
                sequence=sequence,
                data={"citation_count": len(citations_sent)},
            )
            yield done_chunk
            
            state.complete()
            
            if self.config.on_complete:
                self.config.on_complete(state)
                
        except (StreamingCancelledError, StreamingTimeoutError):
            state.cancel()
            raise
        except Exception as e:
            state.fail(str(e))
            
            # エラーチャンク
            error_chunk = StreamingChunk.error(
                message=str(e),
                stream_id=state.stream_id,
                sequence=sequence,
            )
            yield error_chunk
            
            if self.config.on_error:
                self.config.on_error(e)
            raise StreamingError(f"Stream search failed: {e}") from e
    
    async def collect_stream(
        self,
        stream: AsyncGenerator[StreamingChunk, None],
    ) -> StreamingResult:
        """ストリームを収集して完全なレスポンスを取得
        
        Args:
            stream: ストリーミングジェネレータ
            
        Returns:
            StreamingResult
        """
        chunks: list[StreamingChunk] = []
        text_parts: list[str] = []
        citations: list[dict[str, Any]] = []
        stream_id = ""
        
        async for chunk in stream:
            chunks.append(chunk)
            stream_id = chunk.stream_id
            
            if chunk.is_text:
                text_parts.append(chunk.content)
            elif chunk.chunk_type == ChunkType.CITATION:
                citations.append(chunk.data)
        
        state = self._active_streams.get(stream_id)
        
        return StreamingResult(
            stream_id=stream_id,
            full_response="".join(text_parts),
            chunks=chunks,
            citations=citations,
            state=state,
        )
    
    def get_status(self) -> dict[str, Any]:
        """サービス状態を取得"""
        return {
            "active_streams": self.active_stream_count,
            "total_streams": len(self._active_streams),
            "config": self.config.to_dict(),
        }
    
    def clear_completed_streams(self) -> int:
        """完了したストリームをクリア
        
        Returns:
            クリアしたストリーム数
        """
        to_remove = [
            stream_id 
            for stream_id, state in self._active_streams.items() 
            if state.is_completed
        ]
        for stream_id in to_remove:
            del self._active_streams[stream_id]
        return len(to_remove)


# ============================================================
# Mock Source for Testing
# ============================================================


class MockStreamingSource(StreamingSourceProtocol):
    """モックストリーミングソース（テスト用）"""
    
    def __init__(
        self,
        response: str = "This is a mock response.",
        delay_ms: float = 20.0,
        fail_at: int | None = None,
    ):
        """初期化
        
        Args:
            response: 返すレスポンス
            delay_ms: チャンク間の遅延
            fail_at: 指定したチャンク番号で失敗する（テスト用）
        """
        self.response = response
        self.delay_ms = delay_ms
        self.fail_at = fail_at
    
    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """テキストをストリーミング"""
        words = self.response.split()
        for i, word in enumerate(words):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError(f"Simulated failure at chunk {i}")
            
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(self.delay_ms / 1000)
    
    async def stream_with_citations(
        self,
        prompt: str,
        context: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[tuple[str, list[dict[str, Any]]], None]:
        """引用付きでストリーミング"""
        words = self.response.split()
        
        # 引用を均等に配置
        citation_interval = max(1, len(words) // (len(context) + 1))
        citation_index = 0
        
        for i, word in enumerate(words):
            if self.fail_at is not None and i == self.fail_at:
                raise RuntimeError(f"Simulated failure at chunk {i}")
            
            text = word + (" " if i < len(words) - 1 else "")
            citations: list[dict[str, Any]] = []
            
            # 定期的に引用を追加
            if i > 0 and i % citation_interval == 0 and citation_index < len(context):
                citations = [context[citation_index]]
                citation_index += 1
            
            yield text, citations
            await asyncio.sleep(self.delay_ms / 1000)


# ============================================================
# Factory Function
# ============================================================


def create_streaming_service(
    config: StreamingConfig | dict[str, Any] | None = None,
    source: StreamingSourceProtocol | None = None,
) -> StreamingService:
    """StreamingServiceを作成
    
    Args:
        config: ストリーミング設定
        source: ストリーミングソース
        
    Returns:
        StreamingService
    """
    if isinstance(config, dict):
        config = StreamingConfig(**config)
    
    return StreamingService(config=config, source=source)
