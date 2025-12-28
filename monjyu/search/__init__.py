# Search Module
"""
monjyu.search - ベクトル検索 & ハイブリッド検索

FEAT-004: Vector Search (Baseline RAG)
REQ-ARC-003: Hybrid GraphRAG
"""

from monjyu.search.base import (
    SearchMode,
    SearchHit,
    SearchResults,
    Citation,
    SynthesizedAnswer,
    SearchResponse,
    QueryEncoderProtocol,
    VectorSearcherProtocol,
    AnswerSynthesizerProtocol,
)
from monjyu.search.query_encoder import QueryEncoder
from monjyu.search.vector_searcher import (
    LanceDBVectorSearcher,
)
from monjyu.search.answer_synthesizer import AnswerSynthesizer
from monjyu.search.engine import VectorSearchEngine, VectorSearchConfig
from monjyu.search.hybrid import (
    HybridSearchEngine,
    HybridSearchConfig,
    HybridSearchResult,
    FusionMethod,
    SearchMethod,
    MethodResult,
    ResultMerger,
    create_hybrid_engine,
)

__all__ = [
    # Enums & Types
    "SearchMode",
    "SearchHit",
    "SearchResults",
    "Citation",
    "SynthesizedAnswer",
    "SearchResponse",
    # Protocols
    "QueryEncoderProtocol",
    "VectorSearcherProtocol",
    "AnswerSynthesizerProtocol",
    # Implementations
    "QueryEncoder",
    "LanceDBVectorSearcher",
    "AnswerSynthesizer",
    "VectorSearchEngine",
    "VectorSearchConfig",
    # Hybrid Search
    "HybridSearchEngine",
    "HybridSearchConfig",
    "HybridSearchResult",
    "FusionMethod",
    "SearchMethod",
    "MethodResult",
    "ResultMerger",
    "create_hybrid_engine",
]
