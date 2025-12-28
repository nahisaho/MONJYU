# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
SearchResult dataclass for LazyGraphRAG.

This module contains the SearchResult dataclass, extracted from GraphRAG
for standalone use in LazyGraphRAG.
"""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class SearchResult:
    """A Structured Search Result."""

    response: str | dict[str, Any] | list[dict[str, Any]]
    """The response content from the search."""

    context_data: str | list[pd.DataFrame] | dict[str, pd.DataFrame]
    """Raw context data used to generate the response."""

    context_text: str | list[str] | dict[str, str]
    """Actual text strings that are in the context window, built from context_data."""

    completion_time: float
    """Time taken to complete the search in seconds."""

    llm_calls: int
    """Total number of LLM calls made."""

    prompt_tokens: int
    """Total number of prompt tokens used."""

    output_tokens: int
    """Total number of output tokens generated."""

    llm_calls_categories: dict[str, int] | None = field(default=None)
    """Breakdown of LLM calls by category."""

    prompt_tokens_categories: dict[str, int] | None = field(default=None)
    """Breakdown of prompt tokens by category."""

    output_tokens_categories: dict[str, int] | None = field(default=None)
    """Breakdown of output tokens by category."""
