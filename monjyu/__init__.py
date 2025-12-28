# MONJYU - Academic Paper RAG System
"""
MONJYU: Academic Paper RAG using Progressive GraphRAG

A specialized Retrieval-Augmented Generation system for AI for Science papers,
implementing LazyGraphRAG architecture with progressive index levels.
"""

__version__ = "0.1.0"
__author__ = "MONJYU Team"

# Main exports will be added as components are implemented
__all__ = [
    "__version__",
    # External API clients
    "SemanticScholarClient",
    "CrossRefClient",
    "UnifiedMetadataClient",
    "PaperMetadata",
    "Author",
    "Citation",
]

# Lazy imports for external API clients
def __getattr__(name):
    """Lazy import for external API clients."""
    if name in ("SemanticScholarClient", "create_semantic_scholar_client"):
        from monjyu.external.semantic_scholar import SemanticScholarClient, create_semantic_scholar_client
        return SemanticScholarClient if name == "SemanticScholarClient" else create_semantic_scholar_client
    if name in ("CrossRefClient", "create_crossref_client"):
        from monjyu.external.crossref import CrossRefClient, create_crossref_client
        return CrossRefClient if name == "CrossRefClient" else create_crossref_client
    if name in ("UnifiedMetadataClient", "create_unified_client"):
        from monjyu.external.unified import UnifiedMetadataClient, create_unified_client
        return UnifiedMetadataClient if name == "UnifiedMetadataClient" else create_unified_client
    if name in ("PaperMetadata", "Author", "Citation"):
        from monjyu.external.base import PaperMetadata, Author, Citation
        return {"PaperMetadata": PaperMetadata, "Author": Author, "Citation": Citation}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
