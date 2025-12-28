# Search Base Types
"""
検索の基本型定義

FEAT-004: Vector Search
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


# === Enums ===


class SearchMode(Enum):
    """検索モード"""

    VECTOR = "vector"  # ベクトル検索のみ
    KEYWORD = "keyword"  # キーワード検索のみ (BM25)
    HYBRID = "hybrid"  # ハイブリッド検索


# === Data Classes ===


@dataclass
class SearchHit:
    """検索ヒット"""

    text_unit_id: str
    document_id: str
    text: str
    score: float  # 類似度スコア (0-1)

    # メタデータ
    chunk_index: int = 0
    document_title: str = ""

    # ハイブリッド検索用
    vector_score: float = 0.0
    keyword_score: float = 0.0

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "text_unit_id": self.text_unit_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "chunk_index": self.chunk_index,
            "document_title": self.document_title,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchHit":
        """辞書から復元"""
        return cls(
            text_unit_id=data["text_unit_id"],
            document_id=data["document_id"],
            text=data["text"],
            score=data["score"],
            chunk_index=data.get("chunk_index", 0),
            document_title=data.get("document_title", ""),
            vector_score=data.get("vector_score", 0.0),
            keyword_score=data.get("keyword_score", 0.0),
        )


@dataclass
class SearchResults:
    """検索結果"""

    hits: list[SearchHit]
    total_count: int
    query_vector: list[float] = field(default_factory=list)
    search_time_ms: float = 0.0

    @property
    def texts(self) -> list[str]:
        """テキストのリストを返す"""
        return [h.text for h in self.hits]

    @property
    def top_score(self) -> float:
        """最高スコアを返す"""
        return self.hits[0].score if self.hits else 0.0

    @property
    def text_unit_ids(self) -> list[str]:
        """TextUnit IDのリストを返す"""
        return [h.text_unit_id for h in self.hits]

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "hits": [h.to_dict() for h in self.hits],
            "total_count": self.total_count,
            "search_time_ms": self.search_time_ms,
        }


@dataclass
class Citation:
    """引用情報"""

    text_unit_id: str
    document_id: str
    document_title: str
    text_snippet: str  # 引用箇所
    relevance_score: float

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "text_unit_id": self.text_unit_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "text_snippet": self.text_snippet,
            "relevance_score": self.relevance_score,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Citation":
        """辞書から復元"""
        return cls(
            text_unit_id=data["text_unit_id"],
            document_id=data["document_id"],
            document_title=data["document_title"],
            text_snippet=data["text_snippet"],
            relevance_score=data["relevance_score"],
        )


@dataclass
class SynthesizedAnswer:
    """合成された回答"""

    answer: str
    citations: list[Citation]
    confidence: float = 0.0

    # メタデータ
    tokens_used: int = 0
    model: str = ""

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "model": self.model,
        }


@dataclass
class SearchResponse:
    """検索レスポンス"""

    query: str
    answer: SynthesizedAnswer
    search_results: SearchResults

    # パフォーマンス
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0

    # 検索パラメータ
    mode: SearchMode = SearchMode.VECTOR
    top_k: int = 10

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "query": self.query,
            "answer": self.answer.to_dict(),
            "search_results": self.search_results.to_dict(),
            "total_time_ms": self.total_time_ms,
            "search_time_ms": self.search_time_ms,
            "synthesis_time_ms": self.synthesis_time_ms,
            "mode": self.mode.value,
            "top_k": self.top_k,
        }


# === Protocols ===


@runtime_checkable
class QueryEncoderProtocol(Protocol):
    """クエリエンコーダープロトコル"""

    def encode(self, query: str) -> list[float]:
        """クエリを埋め込みベクトルに変換"""
        ...

    def encode_batch(self, queries: list[str]) -> list[list[float]]:
        """複数クエリを一括変換"""
        ...


@runtime_checkable
class VectorSearcherProtocol(Protocol):
    """ベクトル検索プロトコル"""

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> SearchResults:
        """ベクトル検索を実行"""
        ...

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> SearchResults:
        """ハイブリッド検索（ベクトル + キーワード）"""
        ...


@runtime_checkable
class AnswerSynthesizerProtocol(Protocol):
    """回答合成プロトコル"""

    def synthesize(
        self,
        query: str,
        context: list[SearchHit],
        system_prompt: str | None = None,
    ) -> SynthesizedAnswer:
        """コンテキストから回答を合成"""
        ...
