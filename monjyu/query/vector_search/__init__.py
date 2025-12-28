"""VectorSearch module."""

from .types import (
    EmbedderProtocol,
    IndexedDocument,
    SearchHit,
    VectorSearchConfig,
    VectorSearchProtocol,
    VectorSearchResult,
)
from .in_memory import (
    InMemoryVectorSearch,
    cosine_similarity,
    create_in_memory_search,
    keyword_match_score,
)

__all__ = [
    # Types
    "SearchHit",
    "VectorSearchConfig",
    "VectorSearchResult",
    "IndexedDocument",
    "EmbedderProtocol",
    "VectorSearchProtocol",
    # In-Memory implementation
    "InMemoryVectorSearch",
    "create_in_memory_search",
    # Utilities
    "cosine_similarity",
    "keyword_match_score",
]
