# Mock providers for testing
"""
Mock implementations for testing LazySearch components.
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import MagicMock


class MockChatLLM:
    """Mock ChatLLM for testing.
    
    Provides predictable responses for integration tests without
    requiring actual LLM API calls.
    """
    
    def __init__(
        self,
        response_text: str = '{"answer": "Test response"}',
        **kwargs: Any,
    ) -> None:
        """Initialize mock LLM.
        
        Args:
            response_text: Default response text to return
            **kwargs: Ignored (for compatibility)
        """
        self.response_text = response_text
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []
    
    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Mock chat completion.
        
        Args:
            messages: Input messages
            **kwargs: Ignored
            
        Returns:
            Configured response text
        """
        self.call_count += 1
        self.last_messages = messages
        return self.response_text
    
    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Mock streaming chat.
        
        Args:
            messages: Input messages
            **kwargs: Ignored
            
        Yields:
            Response text in chunks
        """
        self.call_count += 1
        self.last_messages = messages
        
        # Yield response in chunks
        words = self.response_text.split()
        for word in words:
            yield word + " "
    
    def set_response(self, response: str) -> None:
        """Set the response text for subsequent calls.
        
        Args:
            response: New response text
        """
        self.response_text = response
    
    def reset(self) -> None:
        """Reset call tracking."""
        self.call_count = 0
        self.last_messages = []


class MockEmbedding:
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 768) -> None:
        """Initialize mock embedding.
        
        Args:
            dimension: Embedding vector dimension
        """
        self.dimension = dimension
        self.call_count = 0
    
    async def embed(self, text: str) -> list[float]:
        """Generate mock embedding.
        
        Args:
            text: Input text
            
        Returns:
            Mock embedding vector
        """
        self.call_count += 1
        # Return deterministic embedding based on text hash
        import hashlib
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [(h >> (i * 4) & 0xF) / 15.0 for i in range(self.dimension)]
    
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch.
        
        Args:
            texts: Input texts
            
        Returns:
            List of mock embedding vectors
        """
        return [await self.embed(text) for text in texts]
