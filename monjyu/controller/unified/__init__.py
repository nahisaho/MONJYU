"""UnifiedController module."""

from .controller import UnifiedController, create_unified_controller
from .types import (
    EngineNotFoundError,
    EngineSearchResult,
    EngineUnavailableError,
    FALLBACK_ORDER,
    SearchContext,
    SearchEngineError,
    SearchEngineProtocol,
    SearchResultItem,
    SearchTimeoutError,
    UnifiedControllerConfig,
    UnifiedSearchResult,
)

__all__ = [
    # Controller
    "UnifiedController",
    "create_unified_controller",
    # Types
    "SearchContext",
    "SearchResultItem",
    "EngineSearchResult",
    "UnifiedSearchResult",
    "UnifiedControllerConfig",
    "SearchEngineProtocol",
    # Errors
    "SearchEngineError",
    "EngineNotFoundError",
    "EngineUnavailableError",
    "SearchTimeoutError",
    # Constants
    "FALLBACK_ORDER",
]
