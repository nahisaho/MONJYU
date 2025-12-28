"""E2E test fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from tests.mock_provider import MockChatLLM


@pytest.fixture
def mock_chat_model():
    """Create a mock chat model for testing."""
    return MockChatLLM()


@pytest.fixture
def mock_embedding_client():
    """Create a mock embedding client for testing."""
    from tests.mock_provider import MockEmbedding
    return MockEmbedding()
