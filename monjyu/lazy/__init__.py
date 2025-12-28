# Lazy Search Module
"""
monjyu.lazy - LazyGraphRAG クエリエンジン

FEAT-005: Lazy Search (LazyGraphRAG Query)
"""

from monjyu.lazy.base import (
    SearchLevel,
    RelevanceScore,
    Claim,
    SearchCandidate,
    LazySearchState,
    LazySearchResult,
    LazySearchConfig,
    RelevanceTesterProtocol,
    ClaimExtractorProtocol,
    IterativeDeepenerProtocol,
    CommunitySearcherProtocol,
)
from monjyu.lazy.relevance_tester import RelevanceTester
from monjyu.lazy.claim_extractor import ClaimExtractor
from monjyu.lazy.iterative_deepener import IterativeDeepener
from monjyu.lazy.community_searcher import CommunitySearcher
from monjyu.lazy.engine import LazySearchEngine, create_local_lazy_engine

__all__ = [
    # Enums & Types
    "SearchLevel",
    "RelevanceScore",
    "Claim",
    "SearchCandidate",
    "LazySearchState",
    "LazySearchResult",
    "LazySearchConfig",
    # Protocols
    "RelevanceTesterProtocol",
    "ClaimExtractorProtocol",
    "IterativeDeepenerProtocol",
    "CommunitySearcherProtocol",
    # Implementations
    "RelevanceTester",
    "ClaimExtractor",
    "IterativeDeepener",
    "CommunitySearcher",
    "LazySearchEngine",
    # Factory
    "create_local_lazy_engine",
]
