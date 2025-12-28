# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
MONJYU - LazyGraphRAG Standalone Package.

MONJYU (文殊) represents wisdom - achieving great insights with minimal resources.
Like the proverb "三人寄れば文殊の知恵" (Three heads are better than one),
LazyGraphRAG achieves comparable quality at ~1/100th the cost.
"""

from lazy_search import (
    # Main search
    LazySearch,
    LazySearchData,
    LazySearchResult,
    # State management
    Claim,
    LazySearchState,
    RelevantSentence,
    # Components
    QueryExpander,
    QueryExpansionResult,
    RelevanceTester,
    RelevanceTestResult,
    ClaimExtractor,
    # Context
    LazyContextBuilder,
    LazySearchContext,
    merge_contexts,
    # Iterative deepening
    DeepenerConfig,
    DeepeningResult,
    IterativeDeepener,
    # Core interfaces
    ChatModel,
    ConversationHistory,
    ConversationRole,
    SearchResult,
    Tokenizer,
    TiktokenTokenizer,
    get_tokenizer,
    try_parse_json_object,
)
from config.lazy_search_config import LazySearchConfig

__all__ = [
    # Main
    "LazySearch",
    "LazySearchConfig",
    "LazySearchData",
    "LazySearchResult",
    # State
    "Claim",
    "LazySearchState",
    "RelevantSentence",
    # Components
    "QueryExpander",
    "QueryExpansionResult",
    "RelevanceTester",
    "RelevanceTestResult",
    "ClaimExtractor",
    # Context
    "LazyContextBuilder",
    "LazySearchContext",
    "merge_contexts",
    # Deepening
    "DeepenerConfig",
    "DeepeningResult",
    "IterativeDeepener",
    # Core
    "ChatModel",
    "ConversationHistory",
    "ConversationRole",
    "SearchResult",
    "Tokenizer",
    "TiktokenTokenizer",
    "get_tokenizer",
    "try_parse_json_object",
]

__version__ = "1.0.0"
