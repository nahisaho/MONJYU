# Relevance Tester Coverage Tests
"""
Tests for monjyu.lazy.relevance_tester to improve coverage from 59% to 75%+
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from monjyu.lazy.relevance_tester import (
    RelevanceTester,
    MockRelevanceTester,
)
from monjyu.lazy.base import RelevanceScore, SearchCandidate, SearchLevel


# --------------------------------------------------------------------------- #
# Test Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    mock = MagicMock()
    mock.generate.return_value = "HIGH"
    return mock


@pytest.fixture
def relevance_tester(mock_llm_client):
    """Create RelevanceTester with mock LLM."""
    return RelevanceTester(
        llm_client=mock_llm_client,
        max_text_length=1000,
        max_workers=3,
    )


@pytest.fixture
def sample_candidates():
    """Create sample search candidates."""
    return [
        SearchCandidate(
            id="c1",
            source="test",
            priority=0.9,
            level=SearchLevel.LEVEL_0,
            text="This is a highly relevant document about machine learning.",
            metadata={},
        ),
        SearchCandidate(
            id="c2",
            source="test",
            priority=0.7,
            level=SearchLevel.LEVEL_0,
            text="This document has some relevance to the topic.",
            metadata={},
        ),
        SearchCandidate(
            id="c3",
            source="test",
            priority=0.3,
            level=SearchLevel.LEVEL_0,
            text="This is not really related to anything.",
            metadata={},
        ),
    ]


# --------------------------------------------------------------------------- #
# RelevanceTester Tests
# --------------------------------------------------------------------------- #


class TestRelevanceTester:
    """Tests for RelevanceTester class."""

    def test_init(self, mock_llm_client) -> None:
        """Test initialization."""
        tester = RelevanceTester(mock_llm_client)
        assert tester.llm_client == mock_llm_client
        assert tester.max_text_length == 1000
        assert tester.max_workers == 5

    def test_init_custom_params(self, mock_llm_client) -> None:
        """Test initialization with custom params."""
        tester = RelevanceTester(
            mock_llm_client,
            max_text_length=500,
            max_workers=10,
        )
        assert tester.max_text_length == 500
        assert tester.max_workers == 10

    def test_test_returns_high(self, relevance_tester) -> None:
        """Test test method returns HIGH score."""
        score = relevance_tester.test("query", "text")
        assert score == RelevanceScore.HIGH

    def test_test_returns_medium(self, relevance_tester, mock_llm_client) -> None:
        """Test test method returns MEDIUM score."""
        mock_llm_client.generate.return_value = "MEDIUM"
        score = relevance_tester.test("query", "text")
        assert score == RelevanceScore.MEDIUM

    def test_test_returns_low(self, relevance_tester, mock_llm_client) -> None:
        """Test test method returns LOW score."""
        mock_llm_client.generate.return_value = "LOW"
        score = relevance_tester.test("query", "text")
        assert score == RelevanceScore.LOW

    def test_test_truncates_long_text(self, relevance_tester, mock_llm_client) -> None:
        """Test test method truncates long text."""
        long_text = "x" * 2000
        relevance_tester.test("query", long_text)

        # Check that the prompt was called with truncated text
        call_args = mock_llm_client.generate.call_args[0][0]
        assert "..." in call_args
        # The full 2000 char text should not appear
        assert "x" * 2000 not in call_args

    def test_test_handles_exception(self, relevance_tester, mock_llm_client) -> None:
        """Test test method handles exceptions."""
        mock_llm_client.generate.side_effect = Exception("LLM error")
        score = relevance_tester.test("query", "text")
        assert score == RelevanceScore.LOW

    def test_test_batch_empty(self, relevance_tester) -> None:
        """Test test_batch with empty list."""
        result = relevance_tester.test_batch("query", [])
        assert result == []

    def test_test_batch_single(self, relevance_tester) -> None:
        """Test test_batch with single text (no parallel)."""
        result = relevance_tester.test_batch("query", ["text1"], parallel=False)
        assert len(result) == 1
        assert result[0] == RelevanceScore.HIGH

    def test_test_batch_parallel(self, relevance_tester, mock_llm_client) -> None:
        """Test test_batch with parallel processing."""
        mock_llm_client.generate.return_value = "HIGH"
        texts = ["text1", "text2", "text3"]
        result = relevance_tester.test_batch("query", texts, parallel=True)
        assert len(result) == 3
        assert all(s == RelevanceScore.HIGH for s in result)

    def test_test_batch_sequential(self, relevance_tester, mock_llm_client) -> None:
        """Test test_batch with sequential processing."""
        mock_llm_client.generate.return_value = "MEDIUM"
        texts = ["text1", "text2"]
        result = relevance_tester.test_batch("query", texts, parallel=False)
        assert len(result) == 2
        assert all(s == RelevanceScore.MEDIUM for s in result)

    def test_filter_relevant_empty(self, relevance_tester) -> None:
        """Test filter_relevant with empty candidates."""
        result = relevance_tester.filter_relevant("query", [])
        assert result == []

    def test_filter_relevant(
        self, relevance_tester, mock_llm_client, sample_candidates
    ) -> None:
        """Test filter_relevant filters by minimum relevance."""
        mock_llm_client.generate.side_effect = ["HIGH", "MEDIUM", "LOW"]
        result = relevance_tester.filter_relevant(
            "query",
            sample_candidates,
            min_relevance=RelevanceScore.MEDIUM,
            parallel=False,
        )
        # Should include HIGH and MEDIUM, exclude LOW
        assert len(result) == 2

    def test_filter_relevant_high_only(
        self, relevance_tester, mock_llm_client, sample_candidates
    ) -> None:
        """Test filter_relevant with HIGH minimum."""
        mock_llm_client.generate.side_effect = ["HIGH", "MEDIUM", "LOW"]
        result = relevance_tester.filter_relevant(
            "query",
            sample_candidates,
            min_relevance=RelevanceScore.HIGH,
            parallel=False,
        )
        assert len(result) == 1
        assert result[0][1] == RelevanceScore.HIGH

    def test_partition_by_relevance_empty(self, relevance_tester) -> None:
        """Test partition_by_relevance with empty candidates."""
        result = relevance_tester.partition_by_relevance("query", [])
        assert result[RelevanceScore.HIGH] == []
        assert result[RelevanceScore.MEDIUM] == []
        assert result[RelevanceScore.LOW] == []

    def test_partition_by_relevance(
        self, relevance_tester, mock_llm_client, sample_candidates
    ) -> None:
        """Test partition_by_relevance groups by score."""
        mock_llm_client.generate.side_effect = ["HIGH", "MEDIUM", "LOW"]
        result = relevance_tester.partition_by_relevance(
            "query", sample_candidates, parallel=False
        )
        assert len(result[RelevanceScore.HIGH]) == 1
        assert len(result[RelevanceScore.MEDIUM]) == 1
        assert len(result[RelevanceScore.LOW]) == 1


# --------------------------------------------------------------------------- #
# MockRelevanceTester Tests
# --------------------------------------------------------------------------- #


class TestMockRelevanceTester:
    """Tests for MockRelevanceTester class."""

    def test_init_default(self) -> None:
        """Test initialization with defaults."""
        tester = MockRelevanceTester()
        assert tester.default_score == RelevanceScore.MEDIUM
        assert tester.call_count == 0

    def test_init_custom_default(self) -> None:
        """Test initialization with custom default score."""
        tester = MockRelevanceTester(default_score=RelevanceScore.HIGH)
        assert tester.default_score == RelevanceScore.HIGH

    def test_set_keyword_score(self) -> None:
        """Test set_keyword_score method."""
        tester = MockRelevanceTester()
        tester.set_keyword_score("important", RelevanceScore.HIGH)
        assert tester._keyword_scores["important"] == RelevanceScore.HIGH

    def test_test_default_score(self) -> None:
        """Test test method returns default score."""
        tester = MockRelevanceTester(default_score=RelevanceScore.LOW)
        score = tester.test("query", "some text")
        assert score == RelevanceScore.LOW
        assert tester.call_count == 1

    def test_test_keyword_score(self) -> None:
        """Test test method returns keyword-based score."""
        tester = MockRelevanceTester(default_score=RelevanceScore.LOW)
        tester.set_keyword_score("important", RelevanceScore.HIGH)
        score = tester.test("query", "This is an important document")
        assert score == RelevanceScore.HIGH

    def test_test_keyword_case_insensitive(self) -> None:
        """Test keyword matching is case insensitive."""
        tester = MockRelevanceTester()
        tester.set_keyword_score("IMPORTANT", RelevanceScore.HIGH)
        score = tester.test("query", "This is important")
        assert score == RelevanceScore.HIGH

    def test_test_increments_call_count(self) -> None:
        """Test test method increments call count."""
        tester = MockRelevanceTester()
        tester.test("q1", "t1")
        tester.test("q2", "t2")
        tester.test("q3", "t3")
        assert tester.call_count == 3

    def test_test_batch(self) -> None:
        """Test test_batch method."""
        tester = MockRelevanceTester(default_score=RelevanceScore.MEDIUM)
        results = tester.test_batch("query", ["t1", "t2", "t3"])
        assert len(results) == 3
        assert all(s == RelevanceScore.MEDIUM for s in results)

    def test_filter_relevant_empty(self) -> None:
        """Test filter_relevant with empty candidates."""
        tester = MockRelevanceTester()
        result = tester.filter_relevant("query", [])
        assert result == []

    def test_filter_relevant(self) -> None:
        """Test filter_relevant with mock tester."""
        tester = MockRelevanceTester(default_score=RelevanceScore.MEDIUM)
        tester.set_keyword_score("important", RelevanceScore.HIGH)
        tester.set_keyword_score("irrelevant", RelevanceScore.LOW)

        candidates = [
            SearchCandidate(
                id="c1",
                source="test",
                priority=0.9,
                level=SearchLevel.LEVEL_0,
                text="This is important",
                metadata={},
            ),
            SearchCandidate(
                id="c2",
                source="test",
                priority=0.5,
                level=SearchLevel.LEVEL_0,
                text="Regular text",
                metadata={},
            ),
            SearchCandidate(
                id="c3",
                source="test",
                priority=0.1,
                level=SearchLevel.LEVEL_0,
                text="Irrelevant content",
                metadata={},
            ),
        ]

        result = tester.filter_relevant(
            "query", candidates, min_relevance=RelevanceScore.MEDIUM
        )
        # Should include HIGH and MEDIUM, not LOW
        assert len(result) == 2


# --------------------------------------------------------------------------- #
# Edge Cases and Integration
# --------------------------------------------------------------------------- #


class TestRelevanceTesterEdgeCases:
    """Edge case tests for RelevanceTester."""

    def test_test_with_whitespace_response(self, mock_llm_client) -> None:
        """Test handling of whitespace in LLM response."""
        mock_llm_client.generate.return_value = "  HIGH  \n"
        tester = RelevanceTester(mock_llm_client)
        score = tester.test("query", "text")
        assert score == RelevanceScore.HIGH

    def test_test_with_lowercase_response(self, mock_llm_client) -> None:
        """Test handling of lowercase LLM response."""
        mock_llm_client.generate.return_value = "medium"
        tester = RelevanceTester(mock_llm_client)
        score = tester.test("query", "text")
        assert score == RelevanceScore.MEDIUM

    def test_test_with_extra_text_in_response(self, mock_llm_client) -> None:
        """Test handling of extra text in LLM response."""
        mock_llm_client.generate.return_value = "I think this is HIGH relevance"
        tester = RelevanceTester(mock_llm_client)
        score = tester.test("query", "text")
        assert score == RelevanceScore.HIGH

    def test_test_with_unknown_response(self, mock_llm_client) -> None:
        """Test handling of unknown LLM response."""
        mock_llm_client.generate.return_value = "UNKNOWN"
        tester = RelevanceTester(mock_llm_client)
        score = tester.test("query", "text")
        assert score == RelevanceScore.LOW

    def test_relevance_prompt_format(self) -> None:
        """Test relevance prompt contains expected elements."""
        prompt = RelevanceTester.RELEVANCE_PROMPT
        assert "{query}" in prompt
        assert "{text}" in prompt
        assert "HIGH" in prompt
        assert "MEDIUM" in prompt
        assert "LOW" in prompt
