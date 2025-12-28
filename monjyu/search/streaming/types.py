"""Streaming Types - REQ-QRY-006.

ストリーミングレスポンスの型定義。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable


class StreamChunkType(Enum):
    """ストリームチャンクの種類"""
    
    # 検索フェーズ
    SEARCH_START = "search_start"           # 検索開始
    SEARCH_PROGRESS = "search_progress"     # 検索進捗
    SEARCH_RESULT = "search_result"         # 検索結果（部分）
    SEARCH_COMPLETE = "search_complete"     # 検索完了
    
    # 合成フェーズ
    SYNTHESIS_START = "synthesis_start"     # 合成開始
    TEXT = "text"                           # テキストトークン
    CITATION = "citation"                   # 引用情報
    SYNTHESIS_COMPLETE = "synthesis_complete"  # 合成完了
    
    # メタデータ
    METADATA = "metadata"                   # メタデータ更新
    
    # 制御
    ERROR = "error"                         # エラー
    END = "end"                             # ストリーム終了


class StreamState(Enum):
    """ストリームの状態"""
    
    IDLE = "idle"               # 待機中
    SEARCHING = "searching"     # 検索中
    SYNTHESIZING = "synthesizing"  # 合成中
    STREAMING = "streaming"     # ストリーミング中
    COMPLETED = "completed"     # 完了
    ERROR = "error"             # エラー
    CANCELLED = "cancelled"     # キャンセル


@dataclass
class StreamConfig:
    """ストリーミング設定"""
    
    # チャンク設定
    chunk_size: int = 10                   # トークン単位（概算）
    include_citations: bool = True         # 引用を含める
    include_search_results: bool = True    # 検索結果を含める
    
    # タイミング
    search_update_interval_ms: int = 500   # 検索進捗更新間隔
    synthesis_flush_interval_ms: int = 100  # 合成フラッシュ間隔
    
    # バッファ
    buffer_size: int = 100                 # バッファサイズ（チャンク数）
    
    # タイムアウト
    search_timeout_seconds: float = 30.0
    synthesis_timeout_seconds: float = 60.0
    
    # 再試行
    max_retries: int = 3
    retry_delay_ms: int = 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "chunk_size": self.chunk_size,
            "include_citations": self.include_citations,
            "include_search_results": self.include_search_results,
            "search_update_interval_ms": self.search_update_interval_ms,
            "synthesis_flush_interval_ms": self.synthesis_flush_interval_ms,
            "buffer_size": self.buffer_size,
            "search_timeout_seconds": self.search_timeout_seconds,
            "synthesis_timeout_seconds": self.synthesis_timeout_seconds,
            "max_retries": self.max_retries,
            "retry_delay_ms": self.retry_delay_ms,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        """辞書から復元"""
        return cls(
            chunk_size=data.get("chunk_size", 10),
            include_citations=data.get("include_citations", True),
            include_search_results=data.get("include_search_results", True),
            search_update_interval_ms=data.get("search_update_interval_ms", 500),
            synthesis_flush_interval_ms=data.get("synthesis_flush_interval_ms", 100),
            buffer_size=data.get("buffer_size", 100),
            search_timeout_seconds=data.get("search_timeout_seconds", 30.0),
            synthesis_timeout_seconds=data.get("synthesis_timeout_seconds", 60.0),
            max_retries=data.get("max_retries", 3),
            retry_delay_ms=data.get("retry_delay_ms", 1000),
        )


@dataclass
class StreamMetadata:
    """ストリームメタデータ"""
    
    # クエリ情報
    query: str = ""
    
    # タイミング
    start_time: float = field(default_factory=time.time)
    search_start_time: Optional[float] = None
    search_end_time: Optional[float] = None
    synthesis_start_time: Optional[float] = None
    synthesis_end_time: Optional[float] = None
    
    # 統計
    total_search_hits: int = 0
    tokens_generated: int = 0
    chunks_sent: int = 0
    citations_found: int = 0
    
    # 状態
    state: StreamState = StreamState.IDLE
    error_message: Optional[str] = None
    
    @property
    def search_time_ms(self) -> float:
        """検索時間（ミリ秒）"""
        if self.search_start_time and self.search_end_time:
            return (self.search_end_time - self.search_start_time) * 1000
        return 0.0
    
    @property
    def synthesis_time_ms(self) -> float:
        """合成時間（ミリ秒）"""
        if self.synthesis_start_time and self.synthesis_end_time:
            return (self.synthesis_end_time - self.synthesis_start_time) * 1000
        return 0.0
    
    @property
    def total_time_ms(self) -> float:
        """合計時間（ミリ秒）"""
        return (time.time() - self.start_time) * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "query": self.query,
            "search_time_ms": self.search_time_ms,
            "synthesis_time_ms": self.synthesis_time_ms,
            "total_time_ms": self.total_time_ms,
            "total_search_hits": self.total_search_hits,
            "tokens_generated": self.tokens_generated,
            "chunks_sent": self.chunks_sent,
            "citations_found": self.citations_found,
            "state": self.state.value,
            "error_message": self.error_message,
        }


@dataclass
class StreamChunk:
    """ストリームチャンク"""
    
    chunk_type: StreamChunkType
    content: str = ""
    index: int = 0                      # チャンクインデックス
    timestamp: float = field(default_factory=time.time)
    
    # 追加データ
    data: Optional[Dict[str, Any]] = None
    
    # メタデータ参照
    metadata: Optional[StreamMetadata] = None
    
    @classmethod
    def text_chunk(cls, text: str, index: int) -> "StreamChunk":
        """テキストチャンクを作成"""
        return cls(
            chunk_type=StreamChunkType.TEXT,
            content=text,
            index=index,
        )
    
    @classmethod
    def citation_chunk(
        cls,
        citation_id: str,
        document_title: str,
        text_snippet: str,
        index: int,
    ) -> "StreamChunk":
        """引用チャンクを作成"""
        return cls(
            chunk_type=StreamChunkType.CITATION,
            content=f"[{citation_id}]",
            index=index,
            data={
                "citation_id": citation_id,
                "document_title": document_title,
                "text_snippet": text_snippet,
            },
        )
    
    @classmethod
    def error_chunk(cls, error_message: str, index: int) -> "StreamChunk":
        """エラーチャンクを作成"""
        return cls(
            chunk_type=StreamChunkType.ERROR,
            content=error_message,
            index=index,
            data={"error": error_message},
        )
    
    @classmethod
    def end_chunk(cls, metadata: StreamMetadata, index: int) -> "StreamChunk":
        """終了チャンクを作成"""
        return cls(
            chunk_type=StreamChunkType.END,
            content="",
            index=index,
            metadata=metadata,
            data=metadata.to_dict(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        result = {
            "chunk_type": self.chunk_type.value,
            "content": self.content,
            "index": self.index,
            "timestamp": self.timestamp,
        }
        if self.data:
            result["data"] = self.data
        if self.metadata:
            result["metadata"] = self.metadata.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamChunk":
        """辞書から復元"""
        return cls(
            chunk_type=StreamChunkType(data["chunk_type"]),
            content=data.get("content", ""),
            index=data.get("index", 0),
            timestamp=data.get("timestamp", time.time()),
            data=data.get("data"),
            metadata=None,  # メタデータは復元時に再構築
        )


@runtime_checkable
class StreamingCallbackProtocol(Protocol):
    """ストリーミングコールバックプロトコル"""
    
    async def on_chunk(self, chunk: StreamChunk) -> None:
        """チャンク受信時のコールバック"""
        ...
    
    async def on_error(self, error: Exception) -> None:
        """エラー発生時のコールバック"""
        ...
    
    async def on_complete(self, metadata: StreamMetadata) -> None:
        """完了時のコールバック"""
        ...


# 型エイリアス
ChunkCallback = Callable[[StreamChunk], None]
AsyncChunkCallback = Callable[[StreamChunk], Any]  # Coroutine
ErrorCallback = Callable[[Exception], None]
CompleteCallback = Callable[[StreamMetadata], None]


@dataclass
class StreamCallbacks:
    """ストリーミングコールバック群"""
    
    on_chunk: Optional[AsyncChunkCallback] = None
    on_error: Optional[ErrorCallback] = None
    on_complete: Optional[CompleteCallback] = None
    
    # 同期版コールバック
    on_chunk_sync: Optional[ChunkCallback] = None
    
    def has_callbacks(self) -> bool:
        """コールバックが設定されているか"""
        return any([
            self.on_chunk,
            self.on_error,
            self.on_complete,
            self.on_chunk_sync,
        ])
