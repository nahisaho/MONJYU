# Index Level 1 Module
"""
Index Level 1 components for MONJYU.

Level 1 is the LazyGraphRAG foundation that provides:
- NLP-based keyword/noun phrase extraction
- Noun phrase co-occurrence graph
- Community detection (Leiden algorithm)
- All processing with ZERO LLM cost
"""

from monjyu.index.level1.builder import (
    Level1IndexBuilder,
    Level1IndexConfig,
    Level1Index,
)

__all__ = [
    "Level1IndexBuilder",
    "Level1IndexConfig",
    "Level1Index",
]
