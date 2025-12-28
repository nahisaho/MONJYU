"""Budget Controller for Progressive GraphRAG.

This module provides cost budget management for Progressive GraphRAG,
mapping cost budgets to index levels and tracking token usage.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Optional


class CostBudget(str, Enum):
    """Cost budget levels for Progressive GraphRAG.
    
    Each budget level determines the maximum index level to use
    and affects the trade-off between quality and cost.
    
    Attributes:
        MINIMAL: Level 0-1 only (NLP-only, no LLM). Exploratory/one-off queries.
        STANDARD: Level 0-2 (limited LLM). General queries.
        PREMIUM: Level 0-3 (full LLM). High quality needed.
        UNLIMITED: Level 0-4 (all features). Highest quality.
    """
    
    MINIMAL = "minimal"
    STANDARD = "standard"
    PREMIUM = "premium"
    UNLIMITED = "unlimited"


class IndexLevel(IntEnum):
    """Index levels for Progressive GraphRAG.
    
    Higher levels provide better quality but higher cost.
    
    Level 0-1: NLP-only (no LLM cost for indexing)
    Level 2-4: LLM-based (requires LLM for indexing)
    """
    
    RAW = 0        # Chunks + embeddings
    LAZY = 1       # Noun phrase graph + communities
    PARTIAL = 2    # Entities + relationships (LLM)
    FULL = 3       # Community summaries (LLM)
    ENHANCED = 4   # Pre-extracted claims (LLM)


# Budget to max level mapping
BUDGET_LEVEL_MAP: dict[CostBudget, IndexLevel] = {
    CostBudget.MINIMAL: IndexLevel.LAZY,      # max Level 1
    CostBudget.STANDARD: IndexLevel.PARTIAL,  # max Level 2
    CostBudget.PREMIUM: IndexLevel.FULL,      # max Level 3
    CostBudget.UNLIMITED: IndexLevel.ENHANCED,  # max Level 4
}


# Estimated token costs per level (per 1000 chunks)
ESTIMATED_TOKENS_PER_LEVEL: dict[IndexLevel, dict[str, int]] = {
    IndexLevel.RAW: {
        "indexing": 0,       # No LLM cost
        "query": 5000,       # Embedding search + response
    },
    IndexLevel.LAZY: {
        "indexing": 0,       # NLP-only
        "query": 8000,       # Search + relevance testing
    },
    IndexLevel.PARTIAL: {
        "indexing": 50000,   # Entity/relationship extraction
        "query": 3000,       # Entity-based search
    },
    IndexLevel.FULL: {
        "indexing": 100000,  # Community summaries
        "query": 2000,       # Summary-based search
    },
    IndexLevel.ENHANCED: {
        "indexing": 150000,  # Claim extraction
        "query": 1500,       # Claim-based search
    },
}


@dataclass
class TokenUsage:
    """Token usage tracking for a single operation.
    
    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens used.
        operation: Type of operation (e.g., "indexing", "query").
        level: Index level of the operation.
        timestamp: When the operation occurred.
    """
    
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    operation: str = ""
    level: Optional[IndexLevel] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "operation": self.operation,
            "level": self.level.value if self.level is not None else None,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenUsage":
        """Create from dictionary."""
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            operation=data.get("operation", ""),
            level=IndexLevel(data["level"]) if data.get("level") is not None else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
        )


@dataclass
class BudgetState:
    """Accumulated budget state tracking.
    
    Attributes:
        total_tokens_used: Total tokens consumed.
        indexing_tokens: Tokens used for indexing operations.
        query_tokens: Tokens used for query operations.
        usage_history: List of individual token usages.
        operation_count: Number of operations performed.
    """
    
    total_tokens_used: int = 0
    indexing_tokens: int = 0
    query_tokens: int = 0
    usage_history: list[TokenUsage] = field(default_factory=list)
    operation_count: int = 0
    
    def record_usage(self, usage: TokenUsage) -> None:
        """Record a token usage.
        
        Args:
            usage: The token usage to record.
        """
        self.total_tokens_used += usage.total_tokens
        self.operation_count += 1
        
        if usage.operation == "indexing":
            self.indexing_tokens += usage.total_tokens
        elif usage.operation == "query":
            self.query_tokens += usage.total_tokens
        
        self.usage_history.append(usage)
    
    def get_usage_by_level(self, level: IndexLevel) -> int:
        """Get total tokens used at a specific level.
        
        Args:
            level: The index level.
            
        Returns:
            Total tokens used at that level.
        """
        return sum(
            u.total_tokens
            for u in self.usage_history
            if u.level == level
        )
    
    def get_usage_by_operation(self, operation: str) -> int:
        """Get total tokens used for a specific operation type.
        
        Args:
            operation: The operation type (e.g., "indexing", "query").
            
        Returns:
            Total tokens used for that operation.
        """
        return sum(
            u.total_tokens
            for u in self.usage_history
            if u.operation == operation
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_tokens_used": self.total_tokens_used,
            "indexing_tokens": self.indexing_tokens,
            "query_tokens": self.query_tokens,
            "operation_count": self.operation_count,
            "usage_history": [u.to_dict() for u in self.usage_history],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetState":
        """Create from dictionary."""
        state = cls(
            total_tokens_used=data.get("total_tokens_used", 0),
            indexing_tokens=data.get("indexing_tokens", 0),
            query_tokens=data.get("query_tokens", 0),
            operation_count=data.get("operation_count", 0),
        )
        state.usage_history = [
            TokenUsage.from_dict(u)
            for u in data.get("usage_history", [])
        ]
        return state
    
    def reset(self) -> None:
        """Reset all usage tracking."""
        self.total_tokens_used = 0
        self.indexing_tokens = 0
        self.query_tokens = 0
        self.operation_count = 0
        self.usage_history = []


@dataclass
class BudgetConfig:
    """Configuration for BudgetController.
    
    Attributes:
        default_budget: Default cost budget level.
        token_limit: Optional hard limit on total tokens (None = no limit).
        warn_threshold_ratio: Ratio of token_limit at which to warn (0.8 = 80%).
        track_history: Whether to track individual usage history.
        history_limit: Maximum number of usage records to keep.
    """
    
    default_budget: CostBudget = CostBudget.STANDARD
    token_limit: Optional[int] = None
    warn_threshold_ratio: float = 0.8
    track_history: bool = True
    history_limit: int = 1000


class BudgetController:
    """Budget controller for Progressive GraphRAG.
    
    Manages cost budgets and tracks token usage across operations.
    Maps cost budgets to appropriate index levels.
    
    Example:
        >>> controller = BudgetController()
        >>> max_level = controller.get_max_level(CostBudget.STANDARD)
        >>> print(max_level)  # IndexLevel.PARTIAL
        >>> controller.record_usage(TokenUsage(total_tokens=1000, operation="query"))
    """
    
    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
    ):
        """Initialize the budget controller.
        
        Args:
            config: Configuration for the controller.
        """
        self.config = config or BudgetConfig()
        self._state = BudgetState()
    
    @property
    def state(self) -> BudgetState:
        """Get the current budget state."""
        return self._state
    
    def get_max_level(self, budget: CostBudget) -> IndexLevel:
        """Get the maximum index level for a budget.
        
        Args:
            budget: The cost budget level.
            
        Returns:
            The maximum index level allowed for the budget.
        """
        return BUDGET_LEVEL_MAP[budget]
    
    def budget_to_level(self, budget: CostBudget) -> IndexLevel:
        """Alias for get_max_level for compatibility.
        
        Args:
            budget: The cost budget level.
            
        Returns:
            The maximum index level allowed for the budget.
        """
        return self.get_max_level(budget)
    
    def is_level_allowed(self, level: IndexLevel, budget: CostBudget) -> bool:
        """Check if a level is allowed under a budget.
        
        Args:
            level: The index level to check.
            budget: The cost budget level.
            
        Returns:
            True if the level is allowed under the budget.
        """
        max_level = self.get_max_level(budget)
        return level <= max_level
    
    def get_allowed_levels(self, budget: CostBudget) -> list[IndexLevel]:
        """Get all levels allowed under a budget.
        
        Args:
            budget: The cost budget level.
            
        Returns:
            List of allowed index levels.
        """
        max_level = self.get_max_level(budget)
        return [level for level in IndexLevel if level <= max_level]
    
    def estimate_cost(
        self,
        budget: CostBudget,
        chunk_count: int,
        operation: str = "both",
    ) -> dict[str, int]:
        """Estimate token cost for an operation.
        
        Args:
            budget: The cost budget level.
            chunk_count: Number of chunks to process.
            operation: Operation type ("indexing", "query", or "both").
            
        Returns:
            Estimated tokens for each level under the budget.
        """
        max_level = self.get_max_level(budget)
        estimates: dict[str, int] = {}
        scale = chunk_count / 1000  # Estimates are per 1000 chunks
        
        for level in IndexLevel:
            if level <= max_level:
                level_costs = ESTIMATED_TOKENS_PER_LEVEL[level]
                if operation == "indexing":
                    estimates[level.name] = int(level_costs["indexing"] * scale)
                elif operation == "query":
                    estimates[level.name] = int(level_costs["query"] * scale)
                else:  # both
                    estimates[level.name] = int(
                        (level_costs["indexing"] + level_costs["query"]) * scale
                    )
        
        estimates["total"] = sum(
            v for k, v in estimates.items() if k != "total"
        )
        return estimates
    
    def record_usage(self, usage: TokenUsage) -> None:
        """Record token usage.
        
        Args:
            usage: The token usage to record.
        """
        if self.config.track_history:
            # Limit history size
            if len(self._state.usage_history) >= self.config.history_limit:
                # Remove oldest entries
                self._state.usage_history = self._state.usage_history[
                    -(self.config.history_limit - 1):
                ]
        
        self._state.record_usage(usage)
    
    def record_tokens(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        operation: str = "",
        level: Optional[IndexLevel] = None,
    ) -> None:
        """Convenience method to record token usage.
        
        Args:
            prompt_tokens: Number of prompt tokens.
            completion_tokens: Number of completion tokens.
            operation: Operation type.
            level: Index level of the operation.
        """
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            operation=operation,
            level=level,
        )
        self.record_usage(usage)
    
    def is_over_limit(self) -> bool:
        """Check if token usage is over the configured limit.
        
        Returns:
            True if over the limit (only if token_limit is set).
        """
        if self.config.token_limit is None:
            return False
        return self._state.total_tokens_used >= self.config.token_limit
    
    def is_near_limit(self) -> bool:
        """Check if token usage is near the configured limit.
        
        Returns:
            True if near the warning threshold.
        """
        if self.config.token_limit is None:
            return False
        threshold = int(self.config.token_limit * self.config.warn_threshold_ratio)
        return self._state.total_tokens_used >= threshold
    
    def get_remaining_tokens(self) -> Optional[int]:
        """Get remaining tokens before limit.
        
        Returns:
            Remaining tokens, or None if no limit set.
        """
        if self.config.token_limit is None:
            return None
        return max(0, self.config.token_limit - self._state.total_tokens_used)
    
    def get_usage_ratio(self) -> Optional[float]:
        """Get the ratio of used tokens to limit.
        
        Returns:
            Usage ratio (0.0-1.0+), or None if no limit set.
        """
        if self.config.token_limit is None or self.config.token_limit == 0:
            return None
        return self._state.total_tokens_used / self.config.token_limit
    
    def get_summary(self) -> dict[str, Any]:
        """Get a summary of budget status.
        
        Returns:
            Dictionary with budget summary information.
        """
        summary: dict[str, Any] = {
            "total_tokens_used": self._state.total_tokens_used,
            "indexing_tokens": self._state.indexing_tokens,
            "query_tokens": self._state.query_tokens,
            "operation_count": self._state.operation_count,
            "default_budget": self.config.default_budget.value,
        }
        
        if self.config.token_limit is not None:
            summary["token_limit"] = self.config.token_limit
            summary["remaining_tokens"] = self.get_remaining_tokens()
            summary["usage_ratio"] = self.get_usage_ratio()
            summary["is_over_limit"] = self.is_over_limit()
            summary["is_near_limit"] = self.is_near_limit()
        
        # Usage by level
        summary["usage_by_level"] = {
            level.name: self._state.get_usage_by_level(level)
            for level in IndexLevel
        }
        
        return summary
    
    def reset(self) -> None:
        """Reset all usage tracking."""
        self._state.reset()
    
    def save_state(self) -> dict[str, Any]:
        """Save the current state for persistence.
        
        Returns:
            Dictionary representation of the state.
        """
        return {
            "state": self._state.to_dict(),
            "config": {
                "default_budget": self.config.default_budget.value,
                "token_limit": self.config.token_limit,
                "warn_threshold_ratio": self.config.warn_threshold_ratio,
                "track_history": self.config.track_history,
                "history_limit": self.config.history_limit,
            },
        }
    
    def load_state(self, data: dict[str, Any]) -> None:
        """Load state from persistence.
        
        Args:
            data: Dictionary representation of the state.
        """
        if "state" in data:
            self._state = BudgetState.from_dict(data["state"])
        
        if "config" in data:
            config_data = data["config"]
            self.config = BudgetConfig(
                default_budget=CostBudget(config_data.get("default_budget", "standard")),
                token_limit=config_data.get("token_limit"),
                warn_threshold_ratio=config_data.get("warn_threshold_ratio", 0.8),
                track_history=config_data.get("track_history", True),
                history_limit=config_data.get("history_limit", 1000),
            )


def create_budget_controller(
    default_budget: CostBudget = CostBudget.STANDARD,
    token_limit: Optional[int] = None,
) -> BudgetController:
    """Factory function to create a BudgetController.
    
    Args:
        default_budget: Default cost budget level.
        token_limit: Optional hard limit on total tokens.
        
    Returns:
        Configured BudgetController instance.
    """
    config = BudgetConfig(
        default_budget=default_budget,
        token_limit=token_limit,
    )
    return BudgetController(config=config)


__all__ = [
    "CostBudget",
    "IndexLevel",
    "TokenUsage",
    "BudgetState",
    "BudgetConfig",
    "BudgetController",
    "create_budget_controller",
    "BUDGET_LEVEL_MAP",
    "ESTIMATED_TOKENS_PER_LEVEL",
]
