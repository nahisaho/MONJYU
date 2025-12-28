# Iterative Deepener Coverage Tests
"""
Tests for monjyu.lazy.iterative_deepener to improve coverage from 59% to 75%+
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from monjyu.lazy.iterative_deepener import (
    IterativeDeepener,
    MockIterativeDeepener,
)
from monjyu.lazy.base import (
    LazySearchState,
    SearchCandidate,
    SearchLevel,
    Claim,
)


# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    mock = MagicMock()
    mock.generate.return_value = "INSUFFICIENT"
    return mock


@pytest.fixture
def mock_community_searcher():
    """Create mock community searcher."""
    mock = MagicMock()
    mock.get_text_units.return_value = [
        ("tu1", "doc1", "Text unit 1 content"),
        ("tu2", "doc1", "Text unit 2 content"),
    ]
    return mock


@pytest.fixture
def deepener(mock_llm_client):
    """Create IterativeDeepener with mock LLM."""
    return IterativeDeepener(
        llm_client=mock_llm_client,
        max_iterations=5,
        max_llm_calls=20,
        min_claims_for_check=3,
    )


@pytest.fixture
def search_state():
    """Create sample search state."""
    state = LazySearchState(query="test query")
    # Add some candidates to the queue
    # Use "vector" source so mark_visited works correctly
    for i in range(5):
        state.add_candidate(
            SearchCandidate(
                id=f"c{i}",
                source="vector",
                priority=0.9 - i * 0.1,
                level=SearchLevel.LEVEL_0,
                text=f"Text content {i}",
                metadata={},
            )
        )
    return state


# --------------------------------------------------------------------------- #
# IterativeDeepener Tests
# --------------------------------------------------------------------------- #


class TestIterativeDeepener:
    """Tests for IterativeDeepener class."""

    def test_init(self, mock_llm_client) -> None:
        """Test initialization."""
        deepener = IterativeDeepener(mock_llm_client)
        assert deepener.llm_client == mock_llm_client
        assert deepener.community_searcher is None
        assert deepener.max_iterations == 5
        assert deepener.max_llm_calls == 20
        assert deepener.min_claims_for_check == 3

    def test_init_with_community_searcher(
        self, mock_llm_client, mock_community_searcher
    ) -> None:
        """Test initialization with community searcher."""
        deepener = IterativeDeepener(
            mock_llm_client,
            community_searcher=mock_community_searcher,
        )
        assert deepener.community_searcher == mock_community_searcher

    def test_init_custom_params(self, mock_llm_client) -> None:
        """Test initialization with custom params."""
        deepener = IterativeDeepener(
            mock_llm_client,
            max_iterations=10,
            max_llm_calls=50,
            min_claims_for_check=5,
        )
        assert deepener.max_iterations == 10
        assert deepener.max_llm_calls == 50
        assert deepener.min_claims_for_check == 5

    def test_should_deepen_llm_calls_exceeded(
        self, deepener, search_state
    ) -> None:
        """Test should_deepen returns False when LLM calls exceeded."""
        search_state.llm_calls = 25
        assert deepener.should_deepen(search_state) is False

    def test_should_deepen_iterations_exceeded(
        self, deepener, search_state
    ) -> None:
        """Test should_deepen returns False when iterations exceeded."""
        search_state.iterations = 10
        assert deepener.should_deepen(search_state) is False

    def test_should_deepen_empty_queue(self, deepener) -> None:
        """Test should_deepen returns False when queue is empty."""
        state = LazySearchState(query="test")
        assert deepener.should_deepen(state) is False

    def test_should_deepen_few_claims(self, deepener, search_state) -> None:
        """Test should_deepen returns True when claims < min_claims_for_check."""
        # Add fewer claims than min_claims_for_check
        search_state.claims.append(Claim(text="Claim 1", source_text_unit_id="tu1", source_document_id="doc1"))
        search_state.claims.append(Claim(text="Claim 2", source_text_unit_id="tu2", source_document_id="doc2"))
        assert deepener.should_deepen(search_state) is True

    def test_should_deepen_sufficient(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen returns False when LLM says sufficient."""
        mock_llm_client.generate.return_value = "SUFFICIENT"
        # Add enough claims
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        assert deepener.should_deepen(search_state) is False

    def test_should_deepen_insufficient(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen returns True when LLM says insufficient."""
        mock_llm_client.generate.return_value = "INSUFFICIENT"
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        result = deepener.should_deepen(search_state)
        assert result is True

    def test_should_deepen_llm_exception(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen returns False on LLM exception."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        result = deepener.should_deepen(search_state)
        assert result is False

    def test_get_next_candidates_basic(self, deepener, search_state) -> None:
        """Test get_next_candidates returns candidates."""
        candidates = deepener.get_next_candidates(search_state, batch_size=3)
        assert len(candidates) == 3

    def test_get_next_candidates_marks_visited(
        self, deepener, search_state
    ) -> None:
        """Test get_next_candidates skips visited candidates."""
        # Get first candidate and mark it as visited
        first_candidate = search_state.peek_candidate()
        search_state.mark_visited(first_candidate)
        search_state.pop_candidate()  # Remove it from queue
        
        # Re-add it to test that visited candidates are skipped
        search_state.add_candidate(first_candidate)
        
        candidates = deepener.get_next_candidates(search_state, batch_size=5)
        # Should skip the visited one
        visited_ids = {first_candidate.id}
        assert all(c.id not in visited_ids for c in candidates)

    def test_get_next_candidates_empty_queue(self, deepener) -> None:
        """Test get_next_candidates with empty queue."""
        state = LazySearchState(query="test")
        candidates = deepener.get_next_candidates(state, batch_size=5)
        assert candidates == []

    def test_expand_from_community_no_searcher(self, deepener, search_state) -> None:
        """Test expand_from_community returns empty when no searcher."""
        candidates = deepener.expand_from_community("comm1", search_state)
        assert candidates == []

    def test_expand_from_community_with_searcher(
        self, mock_llm_client, mock_community_searcher, search_state
    ) -> None:
        """Test expand_from_community returns candidates."""
        deepener = IterativeDeepener(
            mock_llm_client,
            community_searcher=mock_community_searcher,
        )
        candidates = deepener.expand_from_community("comm1", search_state)
        assert len(candidates) == 2
        assert candidates[0].source == "community"

    def test_expand_from_community_skips_visited(
        self, mock_llm_client, mock_community_searcher, search_state
    ) -> None:
        """Test expand_from_community skips visited text units."""
        deepener = IterativeDeepener(
            mock_llm_client,
            community_searcher=mock_community_searcher,
        )
        search_state.visited_text_units.add("tu1")
        candidates = deepener.expand_from_community("comm1", search_state)
        assert len(candidates) == 1
        assert candidates[0].id == "tu2"


# --------------------------------------------------------------------------- #
# MockIterativeDeepener Tests
# --------------------------------------------------------------------------- #


class TestMockIterativeDeepener:
    """Tests for MockIterativeDeepener class."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        mock = MockIterativeDeepener()
        assert mock.should_deepen_result is True
        assert mock.max_deepening_count == 2
        assert mock.deepen_call_count == 0

    def test_init_custom(self) -> None:
        """Test initialization with custom values."""
        mock = MockIterativeDeepener(
            should_deepen_result=False,
            max_deepening_count=5,
        )
        assert mock.should_deepen_result is False
        assert mock.max_deepening_count == 5

    def test_should_deepen_max_iterations(self, search_state) -> None:
        """Test should_deepen respects max iterations."""
        mock = MockIterativeDeepener(max_deepening_count=2)
        search_state.iterations = 3
        assert mock.should_deepen(search_state) is False

    def test_should_deepen_empty_queue(self) -> None:
        """Test should_deepen returns False when queue empty."""
        mock = MockIterativeDeepener()
        state = LazySearchState(query="test")
        assert mock.should_deepen(state) is False

    def test_should_deepen_returns_result(self, search_state) -> None:
        """Test should_deepen returns configured result."""
        mock_true = MockIterativeDeepener(should_deepen_result=True)
        mock_false = MockIterativeDeepener(should_deepen_result=False)

        assert mock_true.should_deepen(search_state) is True
        assert mock_false.should_deepen(search_state) is False

    def test_should_deepen_increments_count(self, search_state) -> None:
        """Test should_deepen increments call count."""
        mock = MockIterativeDeepener()
        mock.should_deepen(search_state)
        mock.should_deepen(search_state)
        # Count only increments when returning should_deepen_result
        assert mock.deepen_call_count == 2

    def test_get_next_candidates(self, search_state) -> None:
        """Test get_next_candidates returns candidates from queue."""
        mock = MockIterativeDeepener()
        candidates = mock.get_next_candidates(search_state, batch_size=3)
        assert len(candidates) == 3

    def test_get_next_candidates_empty(self) -> None:
        """Test get_next_candidates with empty queue."""
        mock = MockIterativeDeepener()
        state = LazySearchState(query="test")
        candidates = mock.get_next_candidates(state, batch_size=5)
        assert candidates == []

    def test_expand_from_community(self, search_state) -> None:
        """Test expand_from_community returns empty list."""
        mock = MockIterativeDeepener()
        candidates = mock.expand_from_community("comm1", search_state)
        assert candidates == []


# --------------------------------------------------------------------------- #
# Edge Cases and Integration
# --------------------------------------------------------------------------- #


class TestIterativeDeepenerEdgeCases:
    """Edge case tests for IterativeDeepener."""

    def test_should_deepen_increments_llm_calls(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen increments LLM call count."""
        initial_calls = search_state.llm_calls
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        deepener.should_deepen(search_state)
        assert search_state.llm_calls == initial_calls + 1

    def test_should_deepen_with_whitespace_response(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen handles whitespace in response."""
        mock_llm_client.generate.return_value = "  INSUFFICIENT  \n"
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        result = deepener.should_deepen(search_state)
        assert result is True

    def test_should_deepen_with_lowercase_response(
        self, deepener, mock_llm_client, search_state
    ) -> None:
        """Test should_deepen handles lowercase response."""
        mock_llm_client.generate.return_value = "sufficient"
        for i in range(5):
            search_state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu{i}", source_document_id=f"doc{i}"))
        result = deepener.should_deepen(search_state)
        # lowercase "insufficient" not in uppercase "SUFFICIENT"
        assert result is False

    def test_get_next_candidates_respects_batch_size(
        self, deepener, search_state
    ) -> None:
        """Test get_next_candidates returns at most batch_size."""
        candidates = deepener.get_next_candidates(search_state, batch_size=2)
        assert len(candidates) == 2

    def test_sufficiency_prompt_format(self) -> None:
        """Test sufficiency prompt contains expected elements."""
        prompt = IterativeDeepener.SUFFICIENCY_PROMPT
        assert "{query}" in prompt
        assert "{claims}" in prompt
        assert "SUFFICIENT" in prompt
        assert "INSUFFICIENT" in prompt

    def test_expand_from_community_candidate_properties(
        self, mock_llm_client, mock_community_searcher, search_state
    ) -> None:
        """Test expanded candidates have correct properties."""
        deepener = IterativeDeepener(
            mock_llm_client,
            community_searcher=mock_community_searcher,
        )
        candidates = deepener.expand_from_community("comm1", search_state)
        
        for c in candidates:
            assert c.source == "community"
            assert c.priority == 0.5
            assert c.level == SearchLevel.LEVEL_1
            assert "document_id" in c.metadata
