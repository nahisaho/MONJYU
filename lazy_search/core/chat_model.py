# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
ChatModel Protocol for LazyGraphRAG.

This is a minimal standalone version of the ChatModel protocol,
extracted from GraphRAG for independent use in LazyGraphRAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Generator


class ModelResponse:
    """Simple model response container."""

    def __init__(
        self,
        output: str | None = None,
        parsed_response: Any = None,
        history: list | None = None,
        usage: dict[str, int] | None = None,
    ):
        """Initialize ModelResponse."""
        self.output = output
        self.parsed_response = parsed_response
        self.history = history or []
        self.usage = usage or {}


class ChatModel(Protocol):
    """
    Protocol for a chat-based Language Model (LM).

    This protocol defines the methods required for a chat-based LM.
    Prompt is always required for the chat method, and any other keyword
    arguments are forwarded to the Model provider.
    """

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response for the given text asynchronously.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A ModelResponse containing the response.
        """
        ...

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """
        Generate a response for the given text using a streaming interface.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A generator that yields strings representing the response.
        """
        yield ""
        ...

    def chat(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> ModelResponse:
        """
        Generate a response for the given text.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A ModelResponse containing the response.
        """
        ...

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs: Any
    ) -> Generator[str, None]:
        """
        Generate a response for the given text using a streaming interface.

        Args:
            prompt: The text to generate a response for.
            history: The conversation history.
            **kwargs: Additional keyword arguments (e.g., model parameters).

        Returns
        -------
            A generator that yields strings representing the response.
        """
        ...
