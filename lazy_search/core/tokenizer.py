# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Tokenizer interfaces for LazyGraphRAG.

This module contains the Tokenizer ABC and TiktokenTokenizer implementation,
extracted from GraphRAG for standalone use in LazyGraphRAG.
"""

from abc import ABC, abstractmethod

import tiktoken

# Default encoding model for tokenization
DEFAULT_ENCODING_MODEL = "cl100k_base"


class Tokenizer(ABC):
    """Tokenizer Abstract Base Class."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode the given text into a list of tokens.

        Args
        ----
            text (str): The input text to encode.

        Returns
        -------
            list[int]: A list of tokens representing the encoded text.
        """
        msg = "The encode method must be implemented by subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens back into a string.

        Args
        ----
            tokens (list[int]): A list of tokens to decode.

        Returns
        -------
            str: The decoded string from the list of tokens.
        """
        msg = "The decode method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in the given text.

        Args
        ----
            text (str): The input text to analyze.

        Returns
        -------
            int: The number of tokens in the input text.
        """
        return len(self.encode(text))


class TiktokenTokenizer(Tokenizer):
    """Tiktoken-based Tokenizer implementation."""

    def __init__(self, encoding_name: str = DEFAULT_ENCODING_MODEL) -> None:
        """Initialize the Tiktoken Tokenizer.

        Args
        ----
            encoding_name (str): The name of the Tiktoken encoding to use.
                               Defaults to cl100k_base.
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        """Encode the given text into a list of tokens.

        Args
        ----
            text (str): The input text to encode.

        Returns
        -------
            list[int]: A list of tokens representing the encoded text.
        """
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens back into a string.

        Args
        ----
            tokens (list[int]): A list of tokens to decode.

        Returns
        -------
            str: The decoded string from the list of tokens.
        """
        return self.encoding.decode(tokens)


def get_tokenizer(encoding_model: str = DEFAULT_ENCODING_MODEL) -> Tokenizer:
    """
    Get a tokenizer instance.

    Args
    ----
        encoding_model: The tiktoken encoding model to use.
                       Defaults to cl100k_base.

    Returns
    -------
        An instance of TiktokenTokenizer.
    """
    return TiktokenTokenizer(encoding_name=encoding_model)
