# Query Router Types
"""
FEAT-014: QueryRouter データモデル定義

クエリルーティングの型定義
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


class SearchMode(Enum):
    """検索モード
    
    Attributes:
        AUTO: 自動選択（QueryRouterが決定）
        VECTOR: ベクトル検索（Baseline RAG）
        LAZY: LazyGraphRAG検索
        GRAPHRAG: Full GraphRAG検索
        HYBRID: ハイブリッド検索（複数モード併用）
    """
    AUTO = "auto"
    VECTOR = "vector"
    LAZY = "lazy"
    GRAPHRAG = "graphrag"
    HYBRID = "hybrid"
    
    @classmethod
    def from_string(cls, value: str) -> "SearchMode":
        """文字列からSearchModeを取得"""
        value_lower = value.lower()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        return cls.AUTO


class QueryType(Enum):
    """クエリタイプ
    
    学術論文検索におけるクエリの種類を分類。
    
    Attributes:
        SURVEY: サーベイ・傾向分析（「研究動向は？」「主要なアプローチは？」）
        EXPLORATION: 手法調査・探索（「○○を使った手法は？」「実装方法は？」）
        COMPARISON: 手法比較（「AとBの違いは？」「精度比較は？」）
        FACTOID: 具体的事実（「SOTAの精度は？」「データセットサイズは？」）
        CITATION: 先行研究・引用調査（「最初に提案したのは？」「元論文は？」）
        BENCHMARK: ベンチマーク調査（「性能一覧は？」「評価指標は？」）
        UNKNOWN: 分類不能
    """
    SURVEY = "survey"
    EXPLORATION = "exploration"
    COMPARISON = "comparison"
    FACTOID = "factoid"
    CITATION = "citation"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    """ルーティング決定
    
    QueryRouterによるルーティング決定の結果。
    
    Attributes:
        mode: 選択された検索モード
        query_type: 分類されたクエリタイプ
        confidence: 分類の確信度 (0.0-1.0)
        reasoning: 決定理由の説明
        params: モード固有のパラメータ
        fallback_mode: フォールバックモード
    
    Examples:
        >>> decision = RoutingDecision(
        ...     mode=SearchMode.LAZY,
        ...     query_type=QueryType.EXPLORATION,
        ...     confidence=0.85,
        ...     reasoning="Exploration query -> lazy search"
        ... )
    """
    mode: SearchMode
    query_type: QueryType
    confidence: float = 0.0
    reasoning: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    fallback_mode: Optional[SearchMode] = None
    
    @property
    def is_confident(self) -> bool:
        """確信度が十分か (0.7以上)"""
        return self.confidence >= 0.7
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "mode": self.mode.value,
            "query_type": self.query_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "params": self.params,
            "fallback_mode": self.fallback_mode.value if self.fallback_mode else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoutingDecision":
        """辞書から生成"""
        fallback = None
        if data.get("fallback_mode"):
            fallback = SearchMode.from_string(data["fallback_mode"])
        
        return cls(
            mode=SearchMode.from_string(data.get("mode", "auto")),
            query_type=QueryType(data.get("query_type", "unknown")),
            confidence=data.get("confidence", 0.0),
            reasoning=data.get("reasoning", ""),
            params=data.get("params", {}),
            fallback_mode=fallback,
        )


@dataclass
class RoutingContext:
    """ルーティングコンテキスト
    
    ルーティング判断に使用する追加情報。
    
    Attributes:
        index_level: 利用可能なインデックスレベル
        available_modes: 利用可能な検索モード
        query_history: 過去のクエリ履歴
        user_preference: ユーザー設定のモード
        budget: コスト予算
    """
    index_level: int = 1
    available_modes: List[SearchMode] = field(default_factory=lambda: [
        SearchMode.VECTOR, SearchMode.LAZY
    ])
    query_history: List[str] = field(default_factory=list)
    user_preference: Optional[SearchMode] = None
    budget: Optional[str] = None  # "minimal", "standard", "premium"
    
    def is_mode_available(self, mode: SearchMode) -> bool:
        """指定モードが利用可能か"""
        return mode in self.available_modes
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "index_level": self.index_level,
            "available_modes": [m.value for m in self.available_modes],
            "query_history": self.query_history,
            "user_preference": self.user_preference.value if self.user_preference else None,
            "budget": self.budget,
        }


# クエリタイプとモードのマッピング（デフォルト）
DEFAULT_MODE_MAPPING: Dict[QueryType, SearchMode] = {
    QueryType.SURVEY: SearchMode.GRAPHRAG,
    QueryType.EXPLORATION: SearchMode.LAZY,
    QueryType.COMPARISON: SearchMode.HYBRID,
    QueryType.FACTOID: SearchMode.VECTOR,
    QueryType.CITATION: SearchMode.LAZY,
    QueryType.BENCHMARK: SearchMode.HYBRID,
    QueryType.UNKNOWN: SearchMode.LAZY,
}

# インデックスレベルによるフォールバック
LEVEL_FALLBACK_MAPPING: Dict[int, Dict[SearchMode, SearchMode]] = {
    0: {  # Level 0: Vector only
        SearchMode.LAZY: SearchMode.VECTOR,
        SearchMode.GRAPHRAG: SearchMode.VECTOR,
        SearchMode.HYBRID: SearchMode.VECTOR,
    },
    1: {  # Level 1: Vector + Lazy
        SearchMode.GRAPHRAG: SearchMode.LAZY,
        SearchMode.HYBRID: SearchMode.LAZY,
    },
    2: {  # Level 2: Vector + Lazy + partial GraphRAG
        SearchMode.GRAPHRAG: SearchMode.LAZY,
    },
    3: {},  # Level 3: All modes available
    4: {},  # Level 4: All modes available
}
