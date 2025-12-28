"""LocalSearch module - Entity-centric graph traversal for specific queries."""

from .prompts import (
    LOCAL_SEARCH_PROMPT_EN,
    LOCAL_SEARCH_PROMPT_JA,
    get_local_search_prompt,
)
from .search import (
    ChunkStoreProtocol,
    EntityStoreProtocol,
    InMemoryChunkStore,
    InMemoryEntityStore,
    InMemoryRelationshipStore,
    LLMClientProtocol,
    LocalSearch,
    MockLLMClient,
    RelationshipStoreProtocol,
)
from .types import (
    ChunkInfo,
    EntityInfo,
    EntityMatch,
    LocalSearchConfig,
    LocalSearchResult,
    RelationshipInfo,
)

__all__ = [
    # Types
    "LocalSearchConfig",
    "LocalSearchResult",
    "EntityInfo",
    "EntityMatch",
    "RelationshipInfo",
    "ChunkInfo",
    # Protocols
    "LLMClientProtocol",
    "EntityStoreProtocol",
    "RelationshipStoreProtocol",
    "ChunkStoreProtocol",
    # Search
    "LocalSearch",
    # In-memory implementations
    "InMemoryEntityStore",
    "InMemoryRelationshipStore",
    "InMemoryChunkStore",
    "MockLLMClient",
    # Prompts
    "LOCAL_SEARCH_PROMPT_EN",
    "LOCAL_SEARCH_PROMPT_JA",
    "get_local_search_prompt",
]
