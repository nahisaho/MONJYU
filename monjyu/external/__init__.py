# External API Clients - NFR-INT-001
"""
MONJYU External API Module

学術論文メタデータ取得のための外部API統合

Supported APIs:
- Semantic Scholar: 論文検索・引用ネットワーク
- CrossRef: DOI解決・メタデータ取得
"""

from monjyu.external.base import (
    ExternalAPIClient,
    ExternalAPIConfig,
    ExternalAPIError,
    RateLimitError,
    APIResponseError,
    PaperMetadata,
    Author,
    Citation,
)
from monjyu.external.semantic_scholar import (
    SemanticScholarClient,
    SemanticScholarConfig,
    create_semantic_scholar_client,
)
from monjyu.external.crossref import (
    CrossRefClient,
    CrossRefConfig,
    create_crossref_client,
)
from monjyu.external.unified import (
    UnifiedMetadataClient,
    UnifiedMetadataConfig,
    create_unified_client,
)

__all__ = [
    # Base
    "ExternalAPIClient",
    "ExternalAPIConfig",
    "ExternalAPIError",
    "RateLimitError",
    "APIResponseError",
    "PaperMetadata",
    "Author",
    "Citation",
    # Semantic Scholar
    "SemanticScholarClient",
    "SemanticScholarConfig",
    "create_semantic_scholar_client",
    # CrossRef
    "CrossRefClient",
    "CrossRefConfig",
    "create_crossref_client",
    # Unified
    "UnifiedMetadataClient",
    "UnifiedMetadataConfig",
    "create_unified_client",
]
