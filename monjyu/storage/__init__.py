# Storage Module
"""
Storage components for MONJYU.

Provides abstraction for different storage backends:
- Parquet (local file storage)
- Cache (in-memory/Redis caching with LRU eviction)
"""

from monjyu.storage.cache import (
    CacheBackend,
    CacheEntry,
    CacheError,
    CacheManager,
    CacheManagerConfig,
    CacheProtocol,
    CacheStats,
    MemoryCache,
    RedisCache,
    create_cache_manager,
)
from monjyu.storage.parquet import ParquetStorage

__all__ = [
    # Parquet
    "ParquetStorage",
    # Cache
    "CacheBackend",
    "CacheEntry",
    "CacheError",
    "CacheManager",
    "CacheManagerConfig",
    "CacheProtocol",
    "CacheStats",
    "MemoryCache",
    "RedisCache",
    "create_cache_manager",
]
