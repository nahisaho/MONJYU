"""Controller module."""

from .unified import (
    UnifiedController,
    create_unified_controller,
    SearchContext,
    SearchResultItem,
    EngineSearchResult,
    UnifiedSearchResult,
    UnifiedControllerConfig,
    SearchEngineProtocol,
    SearchEngineError,
    EngineNotFoundError,
    EngineUnavailableError,
    SearchTimeoutError,
    FALLBACK_ORDER,
)

from .budget import (
    CostBudget,
    IndexLevel,
    TokenUsage,
    BudgetState,
    BudgetConfig,
    BudgetController,
    create_budget_controller,
    BUDGET_LEVEL_MAP,
    ESTIMATED_TOKENS_PER_LEVEL,
)

from .progressive import (
    ProgressiveController,
    create_progressive_controller,
    ProgressiveSearchContext,
    LevelSearchResult,
    ProgressiveResultItem,
    ProgressiveSearchResult,
    LevelSearchEngineProtocol,
    ProgressiveControllerConfig,
    ProgressiveControllerError,
    LevelNotBuiltError,
    LevelNotAllowedError,
    SearchTimeoutError as ProgressiveSearchTimeoutError,
)

from .hybrid import (
    HybridController,
    create_hybrid_controller,
    MergeStrategy,
    ExecutionMode,
    HybridSearchContext,
    HybridResultItem,
    EngineResult,
    HybridSearchResult,
    HybridSearchEngineProtocol,
    HybridControllerConfig,
    HybridControllerError,
    NoEnginesRegisteredError,
    AllEnginesFailedError,
    HybridSearchTimeoutError,
)

__all__ = [
    # Unified Controller
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
    # Budget Controller
    "CostBudget",
    "IndexLevel",
    "TokenUsage",
    "BudgetState",
    "BudgetConfig",
    "BudgetController",
    "create_budget_controller",
    "BUDGET_LEVEL_MAP",
    "ESTIMATED_TOKENS_PER_LEVEL",
    # Progressive Controller
    "ProgressiveController",
    "create_progressive_controller",
    "ProgressiveSearchContext",
    "LevelSearchResult",
    "ProgressiveResultItem",
    "ProgressiveSearchResult",
    "LevelSearchEngineProtocol",
    "ProgressiveControllerConfig",
    "ProgressiveControllerError",
    "LevelNotBuiltError",
    "LevelNotAllowedError",
    "ProgressiveSearchTimeoutError",
    # Hybrid Controller
    "HybridController",
    "create_hybrid_controller",
    "MergeStrategy",
    "ExecutionMode",
    "HybridSearchContext",
    "HybridResultItem",
    "EngineResult",
    "HybridSearchResult",
    "HybridSearchEngineProtocol",
    "HybridControllerConfig",
    "HybridControllerError",
    "NoEnginesRegisteredError",
    "AllEnginesFailedError",
    "HybridSearchTimeoutError",
]
