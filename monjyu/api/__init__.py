# MONJYU API Module
"""
monjyu.api - Python API (MONJYU Facade)

FEAT-007: Python API
REQ-API-003: ストリーミング出力
"""

from monjyu.api.base import (
    SearchMode,
    IndexLevel,
    IndexStatus,
    MONJYUConfig,
    MONJYUStatus,
    Citation,
    SearchResult,
    DocumentInfo,
    IndexBuildResult,
)
from monjyu.api.config import ConfigManager, load_config
from monjyu.api.state import StateManager
from monjyu.api.factory import (
    ComponentFactory,
    EmbeddingClientProtocol,
    LLMClientProtocol,
    VectorSearcherProtocol,
    MockEmbeddingClient,
    MockLLMClient,
    MockVectorSearcher,
)
from monjyu.api.monjyu import MONJYU, create_monjyu
from monjyu.api.streaming import (
    ChunkType,
    StreamingStatus,
    StreamingChunk,
    StreamingState,
    StreamingConfig,
    StreamingResult,
    StreamingService,
    StreamingSourceProtocol,
    StreamingError,
    StreamingCancelledError,
    StreamingTimeoutError,
    MockStreamingSource,
    create_streaming_service,
)

__all__ = [
    # Enums
    "SearchMode",
    "IndexLevel",
    "IndexStatus",
    "ChunkType",
    "StreamingStatus",
    # Data Classes
    "MONJYUConfig",
    "MONJYUStatus",
    "Citation",
    "SearchResult",
    "DocumentInfo",
    "IndexBuildResult",
    "StreamingChunk",
    "StreamingState",
    "StreamingConfig",
    "StreamingResult",
    # Managers
    "ConfigManager",
    "StateManager",
    "ComponentFactory",
    # Protocols
    "EmbeddingClientProtocol",
    "LLMClientProtocol",
    "VectorSearcherProtocol",
    "StreamingSourceProtocol",
    # Mocks
    "MockEmbeddingClient",
    "MockLLMClient",
    "MockVectorSearcher",
    "MockStreamingSource",
    # Services
    "StreamingService",
    # Main API
    "MONJYU",
    # Errors
    "StreamingError",
    "StreamingCancelledError",
    "StreamingTimeoutError",
    # Factory Functions
    "create_monjyu",
    "create_streaming_service",
    "load_config",
]
