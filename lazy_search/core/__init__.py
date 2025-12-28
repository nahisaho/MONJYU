# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
MONJYU Core - Standalone interfaces for LazyGraphRAG.

This module contains minimal interfaces extracted from GraphRAG to allow
LazyGraphRAG to operate independently. These are lightweight abstractions
that can be used without the full GraphRAG dependency.

Interfaces included:
- ChatModel: Protocol for chat-based LLM
- Tokenizer: Abstract base class for tokenization
- TiktokenTokenizer: Tiktoken-based tokenizer implementation
- SearchResult: Dataclass for search results
- ConversationHistory: Class for managing conversation history
- try_parse_json_object: JSON parsing utility
"""

from lazy_search.core.chat_model import ChatModel
from lazy_search.core.conversation import (
    ConversationHistory,
    ConversationRole,
    ConversationTurn,
    QATurn,
)
from lazy_search.core.search_result import SearchResult
from lazy_search.core.text_utils import (
    try_parse_json_object,
)
from lazy_search.core.tokenizer import (
    Tokenizer,
    TiktokenTokenizer,
    get_tokenizer,
)

__all__ = [
    "ChatModel",
    "ConversationHistory",
    "ConversationRole",
    "ConversationTurn",
    "QATurn",
    "SearchResult",
    "Tokenizer",
    "TiktokenTokenizer",
    "get_tokenizer",
    "try_parse_json_object",
]
