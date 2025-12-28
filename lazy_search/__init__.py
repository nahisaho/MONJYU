# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""MONJYU LazyGraphRAG search module.

Provides cost-efficient search implementation that achieves
comparable quality to full GraphRAG at ~1/100 of the cost
through iterative deepening and budget-controlled LLM calls.

This module includes standalone core interfaces that allow LazyGraphRAG
to be used independently from the full GraphRAG package.
"""

# Core interfaces for standalone usage
from lazy_search.core import (
    ChatModel,
    ConversationHistory,
    ConversationRole,
    ConversationTurn,
    QATurn,
    SearchResult,
    Tokenizer,
    TiktokenTokenizer,
    get_tokenizer,
    try_parse_json_object,
)

from lazy_search.claim_extractor import (
    ClaimExtractor,
)
from lazy_search.context import (
    LazyContextBuilder,
    LazySearchContext,
    merge_contexts,
)
from lazy_search.iterative_deepener import (
    DeepenerConfig,
    DeepeningResult,
    IterativeDeepener,
)
from lazy_search.query_expander import (
    QueryExpander,
    QueryExpansionResult,
)
from lazy_search.relevance_tester import (
    RelevanceTester,
    RelevanceTestResult,
)
from lazy_search.search import (
    LazySearch,
    LazySearchData,
    LazySearchResult,
)
from lazy_search.state import (
    Claim,
    LazySearchState,
    RelevantSentence,
)

__all__ = [
    # Main search
    "LazySearch",
    "LazySearchData",
    "LazySearchResult",
    # State management
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
    # Iterative deepening
    "DeepenerConfig",
    "DeepeningResult",
    "IterativeDeepener",
    # Core interfaces (for standalone usage)
    "ChatModel",
    "ConversationHistory",
    "ConversationRole",
    "ConversationTurn",
    "QATurn",
    "SearchResult",
    "Tokenizer",
    "TiktokenTokenizer",
    "get_tokenizer",
    "try_parse_json_object",
]
