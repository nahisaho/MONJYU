# Query Router - FEAT-014
"""
MONJYU Query Router Module

クエリを分類し最適な検索モードにルーティング
"""

from monjyu.query.router.types import (
    SearchMode,
    QueryType,
    RoutingDecision,
    RoutingContext,
    DEFAULT_MODE_MAPPING,
    LEVEL_FALLBACK_MAPPING,
)
from monjyu.query.router.router import (
    QueryRouter,
    QueryRouterConfig,
    QueryRouterProtocol,
    ChatModelProtocol,
    create_router,
)

__all__ = [
    # Types
    "SearchMode",
    "QueryType",
    "RoutingDecision",
    "RoutingContext",
    "DEFAULT_MODE_MAPPING",
    "LEVEL_FALLBACK_MAPPING",
    # Router
    "QueryRouter",
    "QueryRouterConfig",
    "QueryRouterProtocol",
    "ChatModelProtocol",
    "create_router",
]
