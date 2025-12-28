"""HybridSearch module - REQ-QRY-005.

複数の検索手法（Vector, Lazy, Global, Local）を組み合わせて
RRF (Reciprocal Rank Fusion) などで結果を融合するハイブリッド検索。
"""

from monjyu.query.hybrid_search.types import (
    FusionMethod,
    HybridSearchConfig,
    HybridSearchHit,
    HybridSearchResult,
    MethodSearchResult,
    SearchMethod,
)
from monjyu.query.hybrid_search.search import (
    GlobalSearchProtocol,
    HybridSearch,
    LazySearchProtocol,
    LocalSearchProtocol,
    ResultMerger,
    VectorSearchProtocol,
    create_hybrid_search,
)

__all__ = [
    # Types
    "FusionMethod",
    "SearchMethod",
    "HybridSearchConfig",
    "HybridSearchHit",
    "HybridSearchResult",
    "MethodSearchResult",
    # Protocols
    "VectorSearchProtocol",
    "LazySearchProtocol",
    "GlobalSearchProtocol",
    "LocalSearchProtocol",
    # Search
    "HybridSearch",
    "ResultMerger",
    # Factory
    "create_hybrid_search",
]
