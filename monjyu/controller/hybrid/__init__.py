"""Hybrid Controller module.

REQ-ARC-003: Hybrid GraphRAG Controller

複数検索エンジンを並列実行し、結果をマージするコントローラ。
"""

from .types import (
    MergeStrategy,
    ExecutionMode,
    HybridControllerError,
    NoEnginesRegisteredError,
    AllEnginesFailedError,
    HybridSearchTimeoutError,
    HybridSearchEngineProtocol,
    HybridSearchContext,
    HybridResultItem,
    EngineResult,
    HybridSearchResult,
    HybridControllerConfig,
)

from .controller import (
    HybridController,
    create_hybrid_controller,
)

__all__ = [
    # Enums
    "MergeStrategy",
    "ExecutionMode",
    # Errors
    "HybridControllerError",
    "NoEnginesRegisteredError",
    "AllEnginesFailedError",
    "HybridSearchTimeoutError",
    # Protocols
    "HybridSearchEngineProtocol",
    # Types
    "HybridSearchContext",
    "HybridResultItem",
    "EngineResult",
    "HybridSearchResult",
    "HybridControllerConfig",
    # Controller
    "HybridController",
    "create_hybrid_controller",
]
