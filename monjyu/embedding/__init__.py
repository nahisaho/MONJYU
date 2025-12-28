# Embedding Module
"""
Embedding components for MONJYU Index Level 0.

Provides abstraction for different embedding backends:
- Ollama (local development)
- Azure OpenAI (production)
"""

from monjyu.embedding.base import EmbeddingClient, EmbeddingClientProtocol
from monjyu.embedding.ollama import OllamaEmbeddingClient

__all__ = [
    "EmbeddingClient",
    "EmbeddingClientProtocol",
    "OllamaEmbeddingClient",
]

# Azure client is optional
HAS_AZURE_OPENAI = False
try:
    from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
    __all__.append("AzureOpenAIEmbeddingClient")
    HAS_AZURE_OPENAI = True
except ImportError:
    AzureOpenAIEmbeddingClient = None  # type: ignore

__all__.append("HAS_AZURE_OPENAI")
