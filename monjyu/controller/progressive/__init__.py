"""Progressive Controller module."""

from .controller import (
    ProgressiveController,
    create_progressive_controller,
)

from .types import (
    ProgressiveSearchContext,
    LevelSearchResult,
    ProgressiveResultItem,
    ProgressiveSearchResult,
    LevelSearchEngineProtocol,
    ProgressiveControllerConfig,
    ProgressiveControllerError,
    LevelNotBuiltError,
    LevelNotAllowedError,
    SearchTimeoutError,
)

__all__ = [
    # Controller
    "ProgressiveController",
    "create_progressive_controller",
    # Types
    "ProgressiveSearchContext",
    "LevelSearchResult",
    "ProgressiveResultItem",
    "ProgressiveSearchResult",
    "LevelSearchEngineProtocol",
    "ProgressiveControllerConfig",
    # Errors
    "ProgressiveControllerError",
    "LevelNotBuiltError",
    "LevelNotAllowedError",
    "SearchTimeoutError",
]
