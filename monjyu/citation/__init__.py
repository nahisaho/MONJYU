# Citation Network Module
"""
monjyu.citation - 引用ネットワーク構築・分析

FEAT-006: Citation Network
"""

from monjyu.citation.base import (
    ReferenceMatchStatus,
    ResolvedReference,
    CitationEdge,
    CitationGraph,
    DocumentMetrics,
    CitationPath,
    RelatedPaper,
    CitationNetworkConfig,
)
from monjyu.citation.resolver import ReferenceResolver
from monjyu.citation.builder import CitationGraphBuilder
from monjyu.citation.metrics import MetricsCalculator
from monjyu.citation.analyzer import CitationAnalyzer
from monjyu.citation.manager import CitationNetworkManager, CitationNetworkBuildResult

__all__ = [
    # Enums & Types
    "ReferenceMatchStatus",
    "ResolvedReference",
    "CitationEdge",
    "CitationGraph",
    "DocumentMetrics",
    "CitationPath",
    "RelatedPaper",
    "CitationNetworkConfig",
    # Components
    "ReferenceResolver",
    "CitationGraphBuilder",
    "MetricsCalculator",
    "CitationAnalyzer",
    # Manager
    "CitationNetworkManager",
    "CitationNetworkBuildResult",
]
