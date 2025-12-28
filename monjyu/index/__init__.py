# Index Module
"""
Index components for MONJYU.

Provides abstraction for different vector index backends:
- LanceDB (local development)
- Azure AI Search (production)

And Progressive Index Manager for incremental index building.
"""

from monjyu.index.base import VectorIndexer, VectorIndexerProtocol, SearchResult
from monjyu.index.lancedb import LanceDBIndexer
from monjyu.index.manager import (
    IndexLevel,
    LevelStatus,
    ProgressiveIndexConfig,
    ProgressiveIndexManager,
    ProgressiveIndexState,
    create_progressive_index_manager,
)

__all__ = [
    "VectorIndexer",
    "VectorIndexerProtocol",
    "SearchResult",
    "LanceDBIndexer",
    # Progressive Index Manager
    "IndexLevel",
    "LevelStatus",
    "ProgressiveIndexConfig",
    "ProgressiveIndexManager",
    "ProgressiveIndexState",
    "create_progressive_index_manager",
]

# Azure AI Search is optional
HAS_AZURE_SEARCH = False
try:
    from monjyu.index.azure_search import AzureAISearchIndexer
    __all__.append("AzureAISearchIndexer")
    HAS_AZURE_SEARCH = True
except ImportError:
    AzureAISearchIndexer = None  # type: ignore

__all__.append("HAS_AZURE_SEARCH")
