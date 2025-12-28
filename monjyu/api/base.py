# MONJYU API Base Types
"""
monjyu.api.base - Python API 基本型定義

FEAT-007: Python API (MONJYU Facade)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SearchMode(Enum):
    """検索モード"""

    VECTOR = "vector"  # Level 0: ベクトル検索のみ
    LAZY = "lazy"  # Level 0-1: LazyGraphRAG
    LOCAL = "local"  # Level 1+: ローカル検索（エンティティベース）
    GLOBAL = "global"  # Level 1+: グローバル検索（コミュニティベース）
    AUTO = "auto"  # 自動選択


class IndexLevel(Enum):
    """インデックスレベル"""

    LEVEL_0 = 0  # Baseline (Vector)
    LEVEL_1 = 1  # Lazy (NLP Graph)


class IndexStatus(Enum):
    """インデックス状態"""

    NOT_BUILT = "not_built"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


@dataclass
class MONJYUConfig:
    """MONJYU設定"""

    # 基本設定
    output_path: Path = field(default_factory=lambda: Path("./output"))

    # 環境設定
    environment: str = "local"  # "local" | "azure"

    # インデックス設定
    index_levels: list[IndexLevel] = field(
        default_factory=lambda: [IndexLevel.LEVEL_0, IndexLevel.LEVEL_1]
    )

    # 検索設定
    default_search_mode: SearchMode = SearchMode.LAZY
    default_top_k: int = 10

    # ドキュメント処理設定
    chunk_size: int = 1200
    chunk_overlap: int = 100

    # LLM設定
    llm_model: str = "llama3:8b-instruct-q4_K_M"
    embedding_model: str = "nomic-embed-text"

    # ローカル設定
    ollama_base_url: str = "http://192.168.224.1:11434"

    # Azure設定（オプション）
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None

    def __post_init__(self):
        """パス変換"""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)


@dataclass
class MONJYUStatus:
    """MONJYUステータス"""

    # インデックス状態
    index_status: IndexStatus = IndexStatus.NOT_BUILT
    index_levels_built: list[IndexLevel] = field(default_factory=list)

    # 統計
    document_count: int = 0
    text_unit_count: int = 0

    # Level 1 (NLP Graph)
    noun_phrase_count: int = 0
    community_count: int = 0

    # Citation Network
    citation_edge_count: int = 0

    # エラー
    last_error: str | None = None

    @property
    def is_ready(self) -> bool:
        """インデックスが準備完了か"""
        return self.index_status == IndexStatus.READY


@dataclass
class Citation:
    """引用情報"""

    doc_id: str
    title: str
    chunk_id: str | None = None
    text: str = ""
    relevance_score: float = 0.0


@dataclass
class SearchResult:
    """検索結果"""

    query: str
    answer: str
    citations: list[Citation] = field(default_factory=list)

    # メタデータ
    search_mode: SearchMode = SearchMode.VECTOR
    search_level: int = 0

    # パフォーマンス
    total_time_ms: float = 0.0
    llm_calls: int = 0

    # デバッグ情報（オプション）
    raw_results: list[dict[str, Any]] | None = None

    @property
    def citation_count(self) -> int:
        """引用数"""
        return len(self.citations)


@dataclass
class DocumentInfo:
    """ドキュメント情報"""

    id: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None

    # 統計
    chunk_count: int = 0

    # 引用メトリクス
    citation_count: int = 0
    reference_count: int = 0
    influence_score: float = 0.0


@dataclass
class IndexBuildResult:
    """インデックス構築結果"""

    success: bool
    level: IndexLevel
    duration_ms: float = 0.0
    error: str | None = None

    # 統計
    items_processed: int = 0
    items_indexed: int = 0
