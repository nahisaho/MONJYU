# Lazy Search Base Types
"""
Lazy Search の基本型定義

FEAT-005: Lazy Search
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from monjyu.search.base import Citation


# === Enums ===


class SearchLevel(Enum):
    """検索レベル"""

    LEVEL_0 = 0  # Vector Search (Baseline)
    LEVEL_1 = 1  # NLP Graph + Community
    LEVEL_2 = 2  # LLM Entity Extraction (Future)
    LEVEL_3 = 3  # Full Graph (Future)
    LEVEL_4 = 4  # Graph + Community Summary (Future)


class RelevanceScore(Enum):
    """関連性スコア"""

    HIGH = 2  # 直接的に関連
    MEDIUM = 1  # 部分的に関連
    LOW = 0  # 関連なし


# === Data Classes ===


@dataclass
class Claim:
    """抽出されたクレーム"""

    text: str
    source_text_unit_id: str
    source_document_id: str
    confidence: float = 1.0

    # メタデータ
    extracted_at: str = ""
    relevance_score: RelevanceScore = RelevanceScore.HIGH

    def __post_init__(self):
        if not self.extracted_at:
            self.extracted_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "text": self.text,
            "source_text_unit_id": self.source_text_unit_id,
            "source_document_id": self.source_document_id,
            "confidence": self.confidence,
            "extracted_at": self.extracted_at,
            "relevance_score": self.relevance_score.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Claim":
        """辞書から復元"""
        return cls(
            text=data["text"],
            source_text_unit_id=data["source_text_unit_id"],
            source_document_id=data["source_document_id"],
            confidence=data.get("confidence", 1.0),
            extracted_at=data.get("extracted_at", ""),
            relevance_score=RelevanceScore[data.get("relevance_score", "HIGH")],
        )


@dataclass
class SearchCandidate:
    """検索候補"""

    id: str  # text_unit_id or community_id
    source: str  # "vector" | "community" | "graph"
    priority: float  # 優先度（高いほど優先）
    level: SearchLevel

    # メタデータ
    text: str = ""
    metadata: dict = field(default_factory=dict)

    def __lt__(self, other: "SearchCandidate") -> bool:
        """heapq用（優先度の高い順）"""
        return self.priority > other.priority

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SearchCandidate):
            return NotImplemented
        return self.id == other.id and self.source == other.source

    def __hash__(self) -> int:
        return hash((self.id, self.source))

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "id": self.id,
            "source": self.source,
            "priority": self.priority,
            "level": self.level.name,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class LazySearchState:
    """Lazy Search の状態"""

    query: str

    # 収集した情報
    context: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    # 訪問済み
    visited_text_units: set[str] = field(default_factory=set)
    visited_communities: set[str] = field(default_factory=set)

    # 優先度キュー
    priority_queue: list[SearchCandidate] = field(default_factory=list)

    # 統計
    llm_calls: int = 0
    tokens_used: int = 0
    current_level: SearchLevel = SearchLevel.LEVEL_0
    iterations: int = 0

    def add_candidate(self, candidate: SearchCandidate) -> None:
        """候補を追加"""
        heapq.heappush(self.priority_queue, candidate)

    def add_candidates(self, candidates: list[SearchCandidate]) -> None:
        """複数の候補を追加"""
        for candidate in candidates:
            self.add_candidate(candidate)

    def pop_candidate(self) -> SearchCandidate | None:
        """最優先の候補を取得"""
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)
        return None

    def peek_candidate(self) -> SearchCandidate | None:
        """最優先の候補を参照（取り出さない）"""
        if self.priority_queue:
            return self.priority_queue[0]
        return None

    def mark_visited(self, candidate: SearchCandidate) -> None:
        """訪問済みにマーク"""
        if candidate.source in ["vector", "graph"]:
            self.visited_text_units.add(candidate.id)
        elif candidate.source == "community":
            self.visited_communities.add(candidate.id)

    def is_visited(self, candidate: SearchCandidate) -> bool:
        """訪問済みか確認"""
        if candidate.source in ["vector", "graph"]:
            return candidate.id in self.visited_text_units
        elif candidate.source == "community":
            return candidate.id in self.visited_communities
        return False

    @property
    def queue_size(self) -> int:
        """キューサイズ"""
        return len(self.priority_queue)

    @property
    def visited_count(self) -> int:
        """訪問済み数"""
        return len(self.visited_text_units) + len(self.visited_communities)

    @property
    def claim_count(self) -> int:
        """クレーム数"""
        return len(self.claims)

    def to_dict(self) -> dict:
        """辞書に変換（デバッグ用）"""
        return {
            "query": self.query,
            "context_count": len(self.context),
            "claim_count": self.claim_count,
            "visited_text_units": len(self.visited_text_units),
            "visited_communities": len(self.visited_communities),
            "queue_size": self.queue_size,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "current_level": self.current_level.name,
            "iterations": self.iterations,
        }


@dataclass
class LazySearchResult:
    """Lazy Search 結果"""

    query: str
    answer: str
    claims: list[Claim]
    citations: list["Citation"]

    # メタデータ
    search_level_reached: SearchLevel
    llm_calls: int
    tokens_used: int
    total_time_ms: float

    # 内部状態（デバッグ用）
    final_state: LazySearchState | None = None

    def to_dict(self) -> dict:
        """辞書に変換"""
        result = {
            "query": self.query,
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "citations": [c.to_dict() for c in self.citations],
            "search_level_reached": self.search_level_reached.name,
            "llm_calls": self.llm_calls,
            "tokens_used": self.tokens_used,
            "total_time_ms": self.total_time_ms,
        }
        if self.final_state:
            result["final_state"] = self.final_state.to_dict()
        return result


@dataclass
class LazySearchConfig:
    """Lazy Search 設定"""

    # 初期検索設定
    initial_top_k: int = 20
    batch_size: int = 5

    # 深化設定
    max_iterations: int = 5
    max_llm_calls: int = 20

    # 関連性フィルタ
    min_relevance: RelevanceScore = RelevanceScore.MEDIUM

    # コミュニティ検索
    community_top_k: int = 3
    include_communities: bool = True

    # クレーム抽出
    max_claims_per_text: int = 5
    merge_duplicates: bool = True

    # 出力設定
    max_claims_in_answer: int = 20
    include_debug_state: bool = False

    # 並列処理
    parallel_relevance_test: bool = True
    max_workers: int = 5


# === Protocols ===


@runtime_checkable
class RelevanceTesterProtocol(Protocol):
    """関連性テストプロトコル"""

    def test(self, query: str, text: str) -> RelevanceScore:
        """単一テキストの関連性をテスト"""
        ...

    def test_batch(
        self, query: str, texts: list[str], parallel: bool = True
    ) -> list[RelevanceScore]:
        """バッチで関連性をテスト"""
        ...


@runtime_checkable
class ClaimExtractorProtocol(Protocol):
    """クレーム抽出プロトコル"""

    def extract(
        self,
        query: str,
        text: str,
        source_text_unit_id: str = "",
        source_document_id: str = "",
    ) -> list[Claim]:
        """テキストからクレームを抽出"""
        ...

    def extract_batch(
        self, query: str, candidates: list[SearchCandidate]
    ) -> list[Claim]:
        """バッチでクレーム抽出"""
        ...


@runtime_checkable
class IterativeDeepenerProtocol(Protocol):
    """動的深化プロトコル"""

    def should_deepen(self, state: LazySearchState) -> bool:
        """深化すべきか判定"""
        ...

    def get_next_candidates(
        self, state: LazySearchState, batch_size: int = 5
    ) -> list[SearchCandidate]:
        """次の候補を取得"""
        ...


@runtime_checkable
class CommunitySearcherProtocol(Protocol):
    """コミュニティ検索プロトコル"""

    def search(self, query: str, top_k: int = 5) -> list[SearchCandidate]:
        """クエリに関連するコミュニティを検索"""
        ...

    def get_text_units(self, community_id: str) -> list[tuple[str, str, str]]:
        """コミュニティに属するTextUnitを取得 (id, document_id, text)"""
        ...
