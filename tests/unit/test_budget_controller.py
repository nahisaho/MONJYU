"""Tests for Budget Controller."""

import pytest
from datetime import datetime

from monjyu.controller.budget import (
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


class TestCostBudget:
    """Tests for CostBudget enum."""
    
    def test_budget_values(self):
        """Test budget enum values."""
        assert CostBudget.MINIMAL.value == "minimal"
        assert CostBudget.STANDARD.value == "standard"
        assert CostBudget.PREMIUM.value == "premium"
        assert CostBudget.UNLIMITED.value == "unlimited"
    
    def test_budget_from_string(self):
        """Test creating budget from string."""
        assert CostBudget("minimal") == CostBudget.MINIMAL
        assert CostBudget("standard") == CostBudget.STANDARD
        assert CostBudget("premium") == CostBudget.PREMIUM
        assert CostBudget("unlimited") == CostBudget.UNLIMITED


class TestIndexLevel:
    """Tests for IndexLevel enum."""
    
    def test_level_values(self):
        """Test index level values."""
        assert IndexLevel.RAW == 0
        assert IndexLevel.LAZY == 1
        assert IndexLevel.PARTIAL == 2
        assert IndexLevel.FULL == 3
        assert IndexLevel.ENHANCED == 4
    
    def test_level_ordering(self):
        """Test level ordering."""
        assert IndexLevel.RAW < IndexLevel.LAZY
        assert IndexLevel.LAZY < IndexLevel.PARTIAL
        assert IndexLevel.PARTIAL < IndexLevel.FULL
        assert IndexLevel.FULL < IndexLevel.ENHANCED
    
    def test_level_comparison(self):
        """Test level comparison."""
        assert IndexLevel.RAW <= IndexLevel.LAZY
        assert IndexLevel.PARTIAL >= IndexLevel.LAZY


class TestBudgetLevelMapping:
    """Tests for budget to level mapping."""
    
    def test_minimal_budget_level(self):
        """Test MINIMAL budget max level."""
        assert BUDGET_LEVEL_MAP[CostBudget.MINIMAL] == IndexLevel.LAZY
    
    def test_standard_budget_level(self):
        """Test STANDARD budget max level."""
        assert BUDGET_LEVEL_MAP[CostBudget.STANDARD] == IndexLevel.PARTIAL
    
    def test_premium_budget_level(self):
        """Test PREMIUM budget max level."""
        assert BUDGET_LEVEL_MAP[CostBudget.PREMIUM] == IndexLevel.FULL
    
    def test_unlimited_budget_level(self):
        """Test UNLIMITED budget max level."""
        assert BUDGET_LEVEL_MAP[CostBudget.UNLIMITED] == IndexLevel.ENHANCED


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""
    
    def test_default_values(self):
        """Test default token usage values."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.operation == ""
        assert usage.level is None
    
    def test_custom_values(self):
        """Test custom token usage values."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            operation="query",
            level=IndexLevel.PARTIAL,
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.operation == "query"
        assert usage.level == IndexLevel.PARTIAL
    
    def test_to_dict(self):
        """Test serialization to dict."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            operation="query",
            level=IndexLevel.PARTIAL,
        )
        data = usage.to_dict()
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["operation"] == "query"
        assert data["level"] == 2  # IndexLevel.PARTIAL.value
        assert "timestamp" in data
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "operation": "query",
            "level": 2,
            "timestamp": "2025-01-01T12:00:00",
        }
        usage = TokenUsage.from_dict(data)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.operation == "query"
        assert usage.level == IndexLevel.PARTIAL


class TestBudgetState:
    """Tests for BudgetState dataclass."""
    
    def test_default_values(self):
        """Test default budget state values."""
        state = BudgetState()
        assert state.total_tokens_used == 0
        assert state.indexing_tokens == 0
        assert state.query_tokens == 0
        assert state.operation_count == 0
        assert len(state.usage_history) == 0
    
    def test_record_usage(self):
        """Test recording token usage."""
        state = BudgetState()
        usage = TokenUsage(
            total_tokens=100,
            operation="query",
            level=IndexLevel.RAW,
        )
        state.record_usage(usage)
        
        assert state.total_tokens_used == 100
        assert state.query_tokens == 100
        assert state.indexing_tokens == 0
        assert state.operation_count == 1
        assert len(state.usage_history) == 1
    
    def test_record_indexing_usage(self):
        """Test recording indexing usage."""
        state = BudgetState()
        usage = TokenUsage(
            total_tokens=500,
            operation="indexing",
            level=IndexLevel.PARTIAL,
        )
        state.record_usage(usage)
        
        assert state.indexing_tokens == 500
        assert state.query_tokens == 0
    
    def test_get_usage_by_level(self):
        """Test getting usage by level."""
        state = BudgetState()
        state.record_usage(TokenUsage(total_tokens=100, level=IndexLevel.RAW))
        state.record_usage(TokenUsage(total_tokens=200, level=IndexLevel.PARTIAL))
        state.record_usage(TokenUsage(total_tokens=150, level=IndexLevel.RAW))
        
        assert state.get_usage_by_level(IndexLevel.RAW) == 250
        assert state.get_usage_by_level(IndexLevel.PARTIAL) == 200
        assert state.get_usage_by_level(IndexLevel.FULL) == 0
    
    def test_get_usage_by_operation(self):
        """Test getting usage by operation."""
        state = BudgetState()
        state.record_usage(TokenUsage(total_tokens=100, operation="query"))
        state.record_usage(TokenUsage(total_tokens=500, operation="indexing"))
        state.record_usage(TokenUsage(total_tokens=150, operation="query"))
        
        assert state.get_usage_by_operation("query") == 250
        assert state.get_usage_by_operation("indexing") == 500
    
    def test_reset(self):
        """Test resetting state."""
        state = BudgetState()
        state.record_usage(TokenUsage(total_tokens=100))
        state.reset()
        
        assert state.total_tokens_used == 0
        assert state.operation_count == 0
        assert len(state.usage_history) == 0
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        state = BudgetState()
        state.record_usage(TokenUsage(total_tokens=100, operation="query"))
        
        data = state.to_dict()
        restored = BudgetState.from_dict(data)
        
        assert restored.total_tokens_used == state.total_tokens_used
        assert restored.query_tokens == state.query_tokens
        assert restored.operation_count == state.operation_count


class TestBudgetController:
    """Tests for BudgetController."""
    
    def test_default_initialization(self):
        """Test default controller initialization."""
        controller = BudgetController()
        assert controller.config.default_budget == CostBudget.STANDARD
        assert controller.config.token_limit is None
    
    def test_custom_config(self):
        """Test controller with custom config."""
        config = BudgetConfig(
            default_budget=CostBudget.PREMIUM,
            token_limit=100000,
        )
        controller = BudgetController(config=config)
        assert controller.config.default_budget == CostBudget.PREMIUM
        assert controller.config.token_limit == 100000
    
    def test_get_max_level(self):
        """Test getting max level for budgets."""
        controller = BudgetController()
        
        assert controller.get_max_level(CostBudget.MINIMAL) == IndexLevel.LAZY
        assert controller.get_max_level(CostBudget.STANDARD) == IndexLevel.PARTIAL
        assert controller.get_max_level(CostBudget.PREMIUM) == IndexLevel.FULL
        assert controller.get_max_level(CostBudget.UNLIMITED) == IndexLevel.ENHANCED
    
    def test_budget_to_level_alias(self):
        """Test budget_to_level alias."""
        controller = BudgetController()
        assert controller.budget_to_level(CostBudget.STANDARD) == IndexLevel.PARTIAL
    
    def test_is_level_allowed(self):
        """Test level allowance check."""
        controller = BudgetController()
        
        # MINIMAL allows RAW and LAZY only
        assert controller.is_level_allowed(IndexLevel.RAW, CostBudget.MINIMAL)
        assert controller.is_level_allowed(IndexLevel.LAZY, CostBudget.MINIMAL)
        assert not controller.is_level_allowed(IndexLevel.PARTIAL, CostBudget.MINIMAL)
        
        # STANDARD allows up to PARTIAL
        assert controller.is_level_allowed(IndexLevel.PARTIAL, CostBudget.STANDARD)
        assert not controller.is_level_allowed(IndexLevel.FULL, CostBudget.STANDARD)
    
    def test_get_allowed_levels(self):
        """Test getting allowed levels."""
        controller = BudgetController()
        
        minimal_levels = controller.get_allowed_levels(CostBudget.MINIMAL)
        assert IndexLevel.RAW in minimal_levels
        assert IndexLevel.LAZY in minimal_levels
        assert IndexLevel.PARTIAL not in minimal_levels
        
        unlimited_levels = controller.get_allowed_levels(CostBudget.UNLIMITED)
        assert len(unlimited_levels) == 5  # All levels
    
    def test_record_usage(self):
        """Test recording token usage."""
        controller = BudgetController()
        usage = TokenUsage(total_tokens=100, operation="query")
        controller.record_usage(usage)
        
        assert controller.state.total_tokens_used == 100
        assert controller.state.operation_count == 1
    
    def test_record_tokens_convenience(self):
        """Test convenience method for recording tokens."""
        controller = BudgetController()
        controller.record_tokens(
            prompt_tokens=80,
            completion_tokens=20,
            operation="query",
            level=IndexLevel.RAW,
        )
        
        assert controller.state.total_tokens_used == 100
        assert controller.state.query_tokens == 100
    
    def test_token_limit_checks(self):
        """Test token limit checking."""
        config = BudgetConfig(token_limit=1000)
        controller = BudgetController(config=config)
        
        # Not over limit initially
        assert not controller.is_over_limit()
        assert not controller.is_near_limit()
        
        # Add 800 tokens (80% = at warning threshold)
        controller.record_tokens(prompt_tokens=800)
        assert not controller.is_over_limit()
        assert controller.is_near_limit()
        
        # Add more to exceed
        controller.record_tokens(prompt_tokens=250)
        assert controller.is_over_limit()
    
    def test_remaining_tokens(self):
        """Test remaining tokens calculation."""
        config = BudgetConfig(token_limit=1000)
        controller = BudgetController(config=config)
        
        assert controller.get_remaining_tokens() == 1000
        
        controller.record_tokens(prompt_tokens=300)
        assert controller.get_remaining_tokens() == 700
    
    def test_no_limit_remaining_tokens(self):
        """Test remaining tokens with no limit."""
        controller = BudgetController()
        assert controller.get_remaining_tokens() is None
    
    def test_usage_ratio(self):
        """Test usage ratio calculation."""
        config = BudgetConfig(token_limit=1000)
        controller = BudgetController(config=config)
        
        controller.record_tokens(prompt_tokens=250)
        assert controller.get_usage_ratio() == 0.25
    
    def test_estimate_cost(self):
        """Test cost estimation."""
        controller = BudgetController()
        
        # 1000 chunks, query only
        estimates = controller.estimate_cost(
            budget=CostBudget.STANDARD,
            chunk_count=1000,
            operation="query",
        )
        
        # Should have estimates for RAW, LAZY, PARTIAL
        assert "RAW" in estimates
        assert "LAZY" in estimates
        assert "PARTIAL" in estimates
        assert "FULL" not in estimates  # Not in STANDARD budget
        assert "total" in estimates
    
    def test_get_summary(self):
        """Test getting summary."""
        config = BudgetConfig(
            default_budget=CostBudget.PREMIUM,
            token_limit=10000,
        )
        controller = BudgetController(config=config)
        controller.record_tokens(prompt_tokens=500, operation="query")
        
        summary = controller.get_summary()
        
        assert summary["total_tokens_used"] == 500
        assert summary["default_budget"] == "premium"
        assert summary["token_limit"] == 10000
        assert summary["remaining_tokens"] == 9500
        assert "usage_by_level" in summary
    
    def test_reset(self):
        """Test reset."""
        controller = BudgetController()
        controller.record_tokens(prompt_tokens=1000)
        controller.reset()
        
        assert controller.state.total_tokens_used == 0
        assert controller.state.operation_count == 0
    
    def test_save_and_load_state(self):
        """Test state persistence."""
        controller = BudgetController(
            config=BudgetConfig(
                default_budget=CostBudget.PREMIUM,
                token_limit=50000,
            )
        )
        controller.record_tokens(prompt_tokens=500, operation="query")
        
        # Save
        data = controller.save_state()
        
        # Load into new controller
        new_controller = BudgetController()
        new_controller.load_state(data)
        
        assert new_controller.state.total_tokens_used == 500
        assert new_controller.config.default_budget == CostBudget.PREMIUM
        assert new_controller.config.token_limit == 50000
    
    def test_history_limit(self):
        """Test history size limit."""
        config = BudgetConfig(
            track_history=True,
            history_limit=5,
        )
        controller = BudgetController(config=config)
        
        # Record more than limit
        for i in range(10):
            controller.record_tokens(prompt_tokens=100)
        
        # History should be limited
        assert len(controller.state.usage_history) <= 5


class TestCreateBudgetController:
    """Tests for factory function."""
    
    def test_default_factory(self):
        """Test default factory creation."""
        controller = create_budget_controller()
        assert controller.config.default_budget == CostBudget.STANDARD
        assert controller.config.token_limit is None
    
    def test_custom_factory(self):
        """Test custom factory creation."""
        controller = create_budget_controller(
            default_budget=CostBudget.MINIMAL,
            token_limit=5000,
        )
        assert controller.config.default_budget == CostBudget.MINIMAL
        assert controller.config.token_limit == 5000


class TestEstimatedTokenCosts:
    """Tests for estimated token costs."""
    
    def test_all_levels_have_costs(self):
        """Test all levels have cost estimates."""
        for level in IndexLevel:
            assert level in ESTIMATED_TOKENS_PER_LEVEL
            assert "indexing" in ESTIMATED_TOKENS_PER_LEVEL[level]
            assert "query" in ESTIMATED_TOKENS_PER_LEVEL[level]
    
    def test_nlp_levels_no_indexing_cost(self):
        """Test NLP-only levels have no indexing cost."""
        assert ESTIMATED_TOKENS_PER_LEVEL[IndexLevel.RAW]["indexing"] == 0
        assert ESTIMATED_TOKENS_PER_LEVEL[IndexLevel.LAZY]["indexing"] == 0
    
    def test_llm_levels_have_indexing_cost(self):
        """Test LLM levels have indexing cost."""
        assert ESTIMATED_TOKENS_PER_LEVEL[IndexLevel.PARTIAL]["indexing"] > 0
        assert ESTIMATED_TOKENS_PER_LEVEL[IndexLevel.FULL]["indexing"] > 0
        assert ESTIMATED_TOKENS_PER_LEVEL[IndexLevel.ENHANCED]["indexing"] > 0
    
    def test_query_costs_decrease_with_level(self):
        """Test query costs generally decrease with higher levels."""
        # Higher levels have pre-computed data, so queries are cheaper
        costs = [
            ESTIMATED_TOKENS_PER_LEVEL[level]["query"]
            for level in [IndexLevel.RAW, IndexLevel.PARTIAL, IndexLevel.FULL, IndexLevel.ENHANCED]
        ]
        # LAZY is special (relevance testing), so we skip it
        assert costs[0] > costs[-1]  # RAW > ENHANCED for query
