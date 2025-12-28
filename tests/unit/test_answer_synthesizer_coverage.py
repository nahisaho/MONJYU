# Answer Synthesizer Coverage Tests
"""
Tests for monjyu.search.answer_synthesizer to improve coverage from 53% to 75%+
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from monjyu.search.answer_synthesizer import (
    AnswerSynthesizer,
    OllamaLLMClient,
    OpenAILLMClient,
    AzureOpenAILLMClient,
    MockLLMClient,
)
from monjyu.search.base import SearchHit, Citation


# --------------------------------------------------------------------------- #
# Mock Data
# --------------------------------------------------------------------------- #
def create_search_hit(
    text_unit_id: str = "unit1",
    document_id: str = "doc1",
    document_title: str = "Test Document",
    text: str = "Sample text content",
    score: float = 0.9,
) -> SearchHit:
    """Create a SearchHit for testing."""
    return SearchHit(
        text_unit_id=text_unit_id,
        document_id=document_id,
        document_title=document_title,
        text=text,
        score=score,
    )


# --------------------------------------------------------------------------- #
# AnswerSynthesizer Tests
# --------------------------------------------------------------------------- #
class TestAnswerSynthesizer:
    """Tests for AnswerSynthesizer class."""

    def test_init_default_system_prompt(self) -> None:
        """Test initialization with default system prompt."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"

        synthesizer = AnswerSynthesizer(mock_client)
        assert synthesizer.llm_client is mock_client
        assert synthesizer.system_prompt == AnswerSynthesizer.DEFAULT_SYSTEM_PROMPT

    def test_init_custom_system_prompt(self) -> None:
        """Test initialization with custom system prompt."""
        mock_client = MagicMock()
        custom_prompt = "You are a helpful assistant."

        synthesizer = AnswerSynthesizer(mock_client, system_prompt=custom_prompt)
        assert synthesizer.system_prompt == custom_prompt

    def test_synthesize_empty_context(self) -> None:
        """Test synthesize with empty context."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"

        synthesizer = AnswerSynthesizer(mock_client)
        result = synthesizer.synthesize("test query", [])

        assert result.answer == "情報が見つかりませんでした。"
        assert result.citations == []
        assert result.confidence == 0.0
        assert result.model == "test-model"

    def test_synthesize_with_context(self) -> None:
        """Test synthesize with context."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.generate.return_value = "This is the answer. [1]"

        synthesizer = AnswerSynthesizer(mock_client)
        context = [create_search_hit()]
        result = synthesizer.synthesize("test query", context)

        assert result.answer == "This is the answer. [1]"
        assert len(result.citations) == 1
        mock_client.generate.assert_called_once()

    def test_synthesize_with_custom_system_prompt_override(self) -> None:
        """Test synthesize with system prompt override."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.generate.return_value = "Answer"

        synthesizer = AnswerSynthesizer(mock_client)
        context = [create_search_hit()]

        result = synthesizer.synthesize(
            "query", context, system_prompt="Custom override"
        )

        call_kwargs = mock_client.generate.call_args[1]
        assert call_kwargs["system_prompt"] == "Custom override"

    def test_synthesize_multiple_citations(self) -> None:
        """Test synthesize with multiple citations."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.generate.return_value = "Answer referencing [1] and [2]."

        synthesizer = AnswerSynthesizer(mock_client)
        context = [
            create_search_hit(text_unit_id="u1", document_id="d1", score=0.9),
            create_search_hit(text_unit_id="u2", document_id="d2", score=0.8),
        ]
        result = synthesizer.synthesize("query", context)

        assert len(result.citations) == 2

    def test_synthesize_invalid_citation_index(self) -> None:
        """Test synthesize ignores invalid citation indices."""
        mock_client = MagicMock()
        mock_client.model_name = "test-model"
        mock_client.generate.return_value = "Answer with [1] and [99]."  # 99 is invalid

        synthesizer = AnswerSynthesizer(mock_client)
        context = [create_search_hit()]
        result = synthesizer.synthesize("query", context)

        # Only [1] should be included
        assert len(result.citations) == 1

    def test_build_context(self) -> None:
        """Test _build_context method."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        hits = [
            create_search_hit(document_title="Doc 1", text="Content 1", score=0.9),
            create_search_hit(document_title="Doc 2", text="Content 2", score=0.8),
        ]
        context_text = synthesizer._build_context(hits)

        assert "[1] Doc 1" in context_text
        assert "[2] Doc 2" in context_text
        assert "Score: 0.900" in context_text
        assert "Content 1" in context_text

    def test_build_context_no_title(self) -> None:
        """Test _build_context with document without title."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        hits = [create_search_hit(document_title=None)]
        context_text = synthesizer._build_context(hits)

        assert "[1] Document" in context_text  # Fallback to "Document"

    def test_extract_citations_no_citations(self) -> None:
        """Test _extract_citations with no citation markers."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        context = [create_search_hit()]
        response, citations = synthesizer._extract_citations(
            "Answer without citations.", context
        )

        assert citations == []

    def test_extract_citations_long_text_truncated(self) -> None:
        """Test _extract_citations truncates long text."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        long_text = "x" * 300
        context = [create_search_hit(text=long_text)]
        _, citations = synthesizer._extract_citations("[1]", context)

        assert len(citations) == 1
        assert len(citations[0].text_snippet) <= 203  # 200 + "..."

    def test_estimate_confidence_no_citations(self) -> None:
        """Test _estimate_confidence with no citations."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        confidence = synthesizer._estimate_confidence([], [])
        assert confidence == 0.0

    def test_estimate_confidence_with_citations(self) -> None:
        """Test _estimate_confidence calculation."""
        mock_client = MagicMock()
        synthesizer = AnswerSynthesizer(mock_client)

        citations = [
            Citation(
                text_unit_id="u1",
                document_id="d1",
                document_title="Doc",
                text_snippet="Text",
                relevance_score=0.9,
            )
        ]
        context = [create_search_hit(score=0.9)]

        confidence = synthesizer._estimate_confidence(citations, context)
        assert 0.0 < confidence <= 1.0


# --------------------------------------------------------------------------- #
# OllamaLLMClient Tests
# --------------------------------------------------------------------------- #
class TestOllamaLLMClient:
    """Tests for OllamaLLMClient class."""

    def test_init(self) -> None:
        """Test initialization."""
        client = OllamaLLMClient()
        assert client._model == "llama3.1:8b"
        assert client.host == "http://localhost:11434"

    def test_init_custom_params(self) -> None:
        """Test initialization with custom params."""
        client = OllamaLLMClient(model="mistral", host="http://custom:11434")
        assert client._model == "mistral"
        assert client.host == "http://custom:11434"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        client = OllamaLLMClient(model="test-model")
        assert client.model_name == "test-model"

    def test_generate(self) -> None:
        """Test generate method."""
        client = OllamaLLMClient()
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Generated text"}}
        client._client = mock_ollama

        result = client.generate("prompt")
        assert result == "Generated text"

    def test_generate_with_system_prompt(self) -> None:
        """Test generate with system prompt."""
        client = OllamaLLMClient()
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Response"}}
        client._client = mock_ollama

        client.generate("user prompt", system_prompt="system prompt")

        call_args = mock_ollama.chat.call_args
        messages = call_args[1]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_generate_with_max_tokens(self) -> None:
        """Test generate with max_tokens."""
        client = OllamaLLMClient()
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Response"}}
        client._client = mock_ollama

        client.generate("prompt", max_tokens=100)

        call_args = mock_ollama.chat.call_args
        assert call_args[1]["options"]["num_predict"] == 100


# --------------------------------------------------------------------------- #
# OpenAILLMClient Tests
# --------------------------------------------------------------------------- #
class TestOpenAILLMClient:
    """Tests for OpenAILLMClient class."""

    def test_init(self) -> None:
        """Test initialization."""
        client = OpenAILLMClient()
        assert client._model == "gpt-4o-mini"

    def test_init_custom_params(self) -> None:
        """Test initialization with custom params."""
        client = OpenAILLMClient(model="gpt-4o", api_key="test-key")
        assert client._model == "gpt-4o"
        assert client._api_key == "test-key"

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        client = OpenAILLMClient(model="gpt-4")
        assert client.model_name == "gpt-4"

    def test_generate(self) -> None:
        """Test generate method."""
        client = OpenAILLMClient()
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Generated"))]
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        result = client.generate("prompt")
        assert result == "Generated"

    def test_generate_with_system_prompt(self) -> None:
        """Test generate with system prompt."""
        client = OpenAILLMClient()
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        client.generate("user prompt", system_prompt="system")

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 2

    def test_generate_with_max_tokens(self) -> None:
        """Test generate with max_tokens."""
        client = OpenAILLMClient()
        mock_openai = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_openai.chat.completions.create.return_value = mock_response
        client._client = mock_openai

        client.generate("prompt", max_tokens=500)

        call_kwargs = mock_openai.chat.completions.create.call_args[1]
        assert call_kwargs["max_tokens"] == 500


# --------------------------------------------------------------------------- #
# AzureOpenAILLMClient Tests
# --------------------------------------------------------------------------- #
class TestAzureOpenAILLMClient:
    """Tests for AzureOpenAILLMClient class."""

    def test_init(self) -> None:
        """Test initialization."""
        client = AzureOpenAILLMClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment_name="gpt-4",
        )
        assert client.endpoint == "https://test.openai.azure.com"
        assert client.deployment_name == "gpt-4"
        assert client.api_version == "2024-02-01"

    def test_model_name_property(self) -> None:
        """Test model_name returns deployment_name."""
        client = AzureOpenAILLMClient(
            endpoint="https://test.openai.azure.com",
            api_key="key",
            deployment_name="my-deployment",
        )
        assert client.model_name == "my-deployment"

    def test_generate(self) -> None:
        """Test generate method."""
        client = AzureOpenAILLMClient(
            endpoint="https://test.openai.azure.com",
            api_key="key",
            deployment_name="deploy",
        )
        mock_azure = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Azure response"))]
        mock_azure.chat.completions.create.return_value = mock_response
        client._client = mock_azure

        result = client.generate("prompt")
        assert result == "Azure response"


# --------------------------------------------------------------------------- #
# MockLLMClient Tests
# --------------------------------------------------------------------------- #
class TestMockLLMClient:
    """Tests for MockLLMClient class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        client = MockLLMClient()
        assert client._responses == {}
        assert client._default_response == "This is a mock response. [1]"

    def test_init_with_responses(self) -> None:
        """Test initialization with predefined responses."""
        responses = {"pattern1": "response1"}
        client = MockLLMClient(responses=responses)
        assert client._responses == responses

    def test_model_name(self) -> None:
        """Test model_name property."""
        client = MockLLMClient()
        assert client.model_name == "mock-llm"

    def test_set_response(self) -> None:
        """Test set_response method."""
        client = MockLLMClient()
        client.set_response("query", "custom response")
        assert client._responses["query"] == "custom response"

    def test_generate_default_response(self) -> None:
        """Test generate returns default response."""
        client = MockLLMClient()
        result = client.generate("unknown prompt")
        assert result == "This is a mock response. [1]"

    def test_generate_pattern_match(self) -> None:
        """Test generate returns matched pattern response."""
        client = MockLLMClient(responses={"keyword": "matched response"})
        result = client.generate("prompt with keyword in it")
        assert result == "matched response"

    def test_generate_ignores_params(self) -> None:
        """Test generate ignores system_prompt and max_tokens."""
        client = MockLLMClient()
        result = client.generate(
            "prompt", system_prompt="ignored", max_tokens=100
        )
        assert result == "This is a mock response. [1]"


# --------------------------------------------------------------------------- #
# Integration Tests
# --------------------------------------------------------------------------- #
class TestAnswerSynthesizerIntegration:
    """Integration tests for AnswerSynthesizer."""

    def test_full_synthesis_flow(self) -> None:
        """Test complete synthesis flow."""
        mock_client = MockLLMClient()
        mock_client.set_response(
            "コンテキスト", "Based on the context, the answer is X. [1] [2]"
        )

        synthesizer = AnswerSynthesizer(mock_client)
        context = [
            create_search_hit(document_title="Paper A", score=0.95),
            create_search_hit(document_title="Paper B", score=0.85),
        ]

        result = synthesizer.synthesize("What is X?", context)

        assert "answer is X" in result.answer
        assert len(result.citations) == 2
        assert result.model == "mock-llm"
        assert result.confidence > 0


# --------------------------------------------------------------------------- #
# Edge Case Tests
# --------------------------------------------------------------------------- #
class TestAnswerSynthesizerEdgeCases:
    """Edge case tests."""

    def test_synthesize_single_context_item(self) -> None:
        """Test with single context item."""
        mock_client = MagicMock()
        mock_client.model_name = "test"
        mock_client.generate.return_value = "Answer [1]"

        synthesizer = AnswerSynthesizer(mock_client)
        context = [create_search_hit()]

        result = synthesizer.synthesize("query", context)
        assert len(result.citations) == 1

    def test_synthesize_many_context_items(self) -> None:
        """Test with many context items."""
        mock_client = MagicMock()
        mock_client.model_name = "test"
        mock_client.generate.return_value = "Answer [1] [5] [10]"

        synthesizer = AnswerSynthesizer(mock_client)
        context = [create_search_hit(text_unit_id=f"u{i}") for i in range(15)]

        result = synthesizer.synthesize("query", context)
        # Should only include valid citations [1], [5], [10]
        assert len(result.citations) == 3

    def test_confidence_capped_at_one(self) -> None:
        """Test confidence is capped at 1.0."""
        mock_client = MagicMock()
        mock_client.model_name = "test"
        mock_client.generate.return_value = "[1]"

        synthesizer = AnswerSynthesizer(mock_client)
        # High score context
        context = [create_search_hit(score=1.0)]

        result = synthesizer.synthesize("query", context)
        assert result.confidence <= 1.0
