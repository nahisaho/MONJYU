"""GlobalSearch module - Map-Reduce pattern for broad queries."""

from .prompts import (
    MAP_PROMPT_EN,
    MAP_PROMPT_JA,
    REDUCE_PROMPT_EN,
    REDUCE_PROMPT_JA,
    get_map_prompt,
    get_reduce_prompt,
)
from .search import (
    CommunityStoreProtocol,
    GlobalSearch,
    InMemoryCommunityStore,
    LLMClientProtocol,
    MockLLMClient,
)
from .types import (
    CommunityInfo,
    GlobalSearchConfig,
    GlobalSearchResult,
    MapResult,
)

__all__ = [
    # Types
    "GlobalSearchConfig",
    "GlobalSearchResult",
    "MapResult",
    "CommunityInfo",
    # Protocols
    "LLMClientProtocol",
    "CommunityStoreProtocol",
    # Search
    "GlobalSearch",
    # In-memory implementations
    "InMemoryCommunityStore",
    "MockLLMClient",
    # Prompts
    "MAP_PROMPT_EN",
    "MAP_PROMPT_JA",
    "REDUCE_PROMPT_EN",
    "REDUCE_PROMPT_JA",
    "get_map_prompt",
    "get_reduce_prompt",
]
