"""CacheManager module.

REQ-STG-003: Progressive インデックスキャッシュ

検索結果およびインデックスレベルのキャッシュ管理。
LRU eviction、TTL、レベル別キャッシュをサポート。
"""

import asyncio
import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheBackend(Enum):
    """キャッシュバックエンド"""
    
    MEMORY = "memory"  # ローカルメモリ (開発用)
    REDIS = "redis"  # Redis (本番用)
    LANCEDB = "lancedb"  # LanceDB (永続化)


class CacheError(Exception):
    """キャッシュエラー基底クラス"""
    pass


class CacheConnectionError(CacheError):
    """接続エラー"""
    pass


class CacheSerializationError(CacheError):
    """シリアライズエラー"""
    pass


@dataclass
class CacheEntry(Generic[T]):
    """キャッシュエントリ
    
    Attributes:
        key: キャッシュキー
        value: キャッシュ値
        created_at: 作成時刻
        expires_at: 有効期限 (None = 無期限)
        access_count: アクセス回数
        last_accessed_at: 最終アクセス時刻
        level: インデックスレベル (Optional)
        metadata: メタデータ
    """
    
    key: str
    value: T
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed_at: float = field(default_factory=time.time)
    level: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """有効期限切れか"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    @property
    def ttl_remaining(self) -> Optional[float]:
        """残りTTL (秒)"""
        if self.expires_at is None:
            return None
        remaining = self.expires_at - time.time()
        return max(0, remaining)
    
    def touch(self) -> None:
        """アクセス記録を更新"""
        self.access_count += 1
        self.last_accessed_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed_at": self.last_accessed_at,
            "level": self.level,
            "metadata": self.metadata,
        }


@dataclass
class CacheStats:
    """キャッシュ統計
    
    Attributes:
        hits: ヒット数
        misses: ミス数
        evictions: 追い出し数
        size: 現在のサイズ
        max_size: 最大サイズ
    """
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    max_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """ヒット率"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate,
        }


class CacheProtocol(ABC):
    """キャッシュプロトコル"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得
        
        Args:
            key: キャッシュキー
            
        Returns:
            キャッシュ値 (存在しない場合はNone)
        """
        ...
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """キャッシュ設定
        
        Args:
            key: キャッシュキー
            value: キャッシュ値
            ttl: TTL (秒)
            level: インデックスレベル
        """
        ...
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """キャッシュ削除
        
        Args:
            key: キャッシュキー
            
        Returns:
            削除成功したらTrue
        """
        ...
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """キャッシュ存在確認
        
        Args:
            key: キャッシュキー
            
        Returns:
            存在すればTrue
        """
        ...
    
    @abstractmethod
    async def clear(self) -> int:
        """全キャッシュをクリア
        
        Returns:
            クリアしたエントリ数
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """統計を取得
        
        Returns:
            キャッシュ統計
        """
        ...


class MemoryCache(CacheProtocol):
    """ローカルメモリキャッシュ (LRU)
    
    開発・テスト用のインメモリキャッシュ。
    OrderedDictを使用したLRU evictionをサポート。
    
    Example:
        >>> cache = MemoryCache(max_size=100)
        >>> await cache.set("key1", "value1", ttl=60)
        >>> value = await cache.get("key1")
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None,
    ):
        """初期化
        
        Args:
            max_size: 最大エントリ数
            default_ttl: デフォルトTTL (秒)
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._stats = CacheStats(max_size=max_size)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats.misses += 1
                return None
            
            # 有効期限チェック
            if entry.is_expired:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.size = len(self._cache)
                return None
            
            # LRU: アクセスしたエントリを末尾に移動
            self._cache.move_to_end(key)
            entry.touch()
            self._stats.hits += 1
            
            return entry.value
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """キャッシュ設定"""
        async with self._lock:
            # TTL計算
            actual_ttl = ttl or self._default_ttl
            expires_at = None
            if actual_ttl:
                expires_at = time.time() + actual_ttl
            
            # 既存エントリの更新
            if key in self._cache:
                entry = self._cache[key]
                entry.value = value
                entry.expires_at = expires_at
                entry.level = level
                self._cache.move_to_end(key)
                return
            
            # LRU eviction
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._stats.evictions += 1
            
            # 新規エントリ追加
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                level=level,
            )
            self._stats.size = len(self._cache)
    
    async def delete(self, key: str) -> bool:
        """キャッシュ削除"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """キャッシュ存在確認"""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if entry.is_expired:
                del self._cache[key]
                self._stats.size = len(self._cache)
                return False
            return True
    
    async def clear(self) -> int:
        """全キャッシュをクリア"""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._stats.size = 0
            return count
    
    async def clear_expired(self) -> int:
        """期限切れエントリをクリア"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(expired_keys)
    
    async def clear_by_level(self, level: int) -> int:
        """指定レベルのエントリをクリア"""
        async with self._lock:
            level_keys = [
                key for key, entry in self._cache.items()
                if entry.level == level
            ]
            for key in level_keys:
                del self._cache[key]
            self._stats.size = len(self._cache)
            return len(level_keys)
    
    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """キーリストを取得
        
        Args:
            pattern: 前方一致パターン (Optional)
            
        Returns:
            キーリスト
        """
        async with self._lock:
            if pattern:
                return [k for k in self._cache.keys() if k.startswith(pattern)]
            return list(self._cache.keys())
    
    async def get_entry(self, key: str) -> Optional[CacheEntry]:
        """エントリを取得 (メタデータ含む)"""
        async with self._lock:
            return self._cache.get(key)
    
    def get_stats(self) -> CacheStats:
        """統計を取得"""
        return self._stats
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        self._stats = CacheStats(
            max_size=self._max_size,
            size=len(self._cache),
        )


class RedisCache(CacheProtocol):
    """Redis キャッシュ (本番環境)
    
    Redis をバックエンドとしたキャッシュ。
    TTL、分散キャッシュをサポート。
    
    Example:
        >>> cache = RedisCache(host="localhost", port=6379)
        >>> await cache.connect()
        >>> await cache.set("key1", "value1", ttl=60)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        key_prefix: str = "monjyu:",
        default_ttl: Optional[int] = None,
    ):
        """初期化
        
        Args:
            host: Redis ホスト
            port: Redis ポート
            password: パスワード
            db: データベース番号
            key_prefix: キープレフィックス
            default_ttl: デフォルトTTL (秒)
        """
        self._host = host
        self._port = port
        self._password = password
        self._db = db
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._client = None
        self._stats = CacheStats()
        self._connected = False
    
    def _full_key(self, key: str) -> str:
        """フルキーを生成"""
        return f"{self._key_prefix}{key}"
    
    async def connect(self) -> None:
        """Redis に接続"""
        try:
            import redis.asyncio as redis
            
            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                password=self._password,
                db=self._db,
                decode_responses=True,
            )
            # 接続テスト
            await self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self._host}:{self._port}")
        except ImportError:
            raise CacheConnectionError(
                "redis package not installed. Install with: pip install redis"
            )
        except Exception as e:
            raise CacheConnectionError(f"Failed to connect to Redis: {e}")
    
    async def disconnect(self) -> None:
        """Redis から切断"""
        if self._client:
            await self._client.close()
            self._connected = False
            logger.info("Disconnected from Redis")
    
    @property
    def is_connected(self) -> bool:
        """接続中か"""
        return self._connected
    
    async def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        try:
            full_key = self._full_key(key)
            value = await self._client.get(full_key)
            
            if value is None:
                self._stats.misses += 1
                return None
            
            self._stats.hits += 1
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise CacheSerializationError(f"Failed to deserialize cache value: {e}")
        except Exception as e:
            logger.warning(f"Redis get error: {e}")
            self._stats.misses += 1
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """キャッシュ設定"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        try:
            full_key = self._full_key(key)
            
            # メタデータ付きで保存
            data = {
                "value": value,
                "level": level,
                "created_at": time.time(),
            }
            serialized = json.dumps(data)
            
            actual_ttl = ttl or self._default_ttl
            if actual_ttl:
                await self._client.setex(full_key, actual_ttl, serialized)
            else:
                await self._client.set(full_key, serialized)
        except (TypeError, ValueError) as e:
            raise CacheSerializationError(f"Failed to serialize cache value: {e}")
        except Exception as e:
            logger.warning(f"Redis set error: {e}")
    
    async def delete(self, key: str) -> bool:
        """キャッシュ削除"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        full_key = self._full_key(key)
        result = await self._client.delete(full_key)
        return result > 0
    
    async def exists(self, key: str) -> bool:
        """キャッシュ存在確認"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        full_key = self._full_key(key)
        return await self._client.exists(full_key) > 0
    
    async def clear(self) -> int:
        """全キャッシュをクリア (プレフィックス付きのみ)"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        pattern = f"{self._key_prefix}*"
        keys = []
        async for key in self._client.scan_iter(match=pattern):
            keys.append(key)
        
        if keys:
            return await self._client.delete(*keys)
        return 0
    
    async def clear_by_level(self, level: int) -> int:
        """指定レベルのエントリをクリア"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        # レベルでフィルタリングして削除
        pattern = f"{self._key_prefix}*"
        deleted = 0
        
        async for key in self._client.scan_iter(match=pattern):
            value = await self._client.get(key)
            if value:
                try:
                    data = json.loads(value)
                    if data.get("level") == level:
                        await self._client.delete(key)
                        deleted += 1
                except json.JSONDecodeError:
                    pass
        
        return deleted
    
    async def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """キーリストを取得"""
        if not self._client:
            raise CacheConnectionError("Not connected to Redis")
        
        if pattern:
            full_pattern = f"{self._key_prefix}{pattern}*"
        else:
            full_pattern = f"{self._key_prefix}*"
        
        keys = []
        prefix_len = len(self._key_prefix)
        async for key in self._client.scan_iter(match=full_pattern):
            keys.append(key[prefix_len:])  # プレフィックスを除去
        
        return keys
    
    def get_stats(self) -> CacheStats:
        """統計を取得"""
        return self._stats
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        self._stats = CacheStats()


@dataclass
class CacheManagerConfig:
    """CacheManager 設定
    
    Attributes:
        backend: キャッシュバックエンド
        max_size: 最大エントリ数 (Memory)
        default_ttl: デフォルトTTL (秒)
        redis_host: Redis ホスト
        redis_port: Redis ポート
        redis_password: Redis パスワード
        redis_db: Redis データベース番号
        key_prefix: キープレフィックス
        level_ttls: レベル別TTL
    """
    
    backend: CacheBackend = CacheBackend.MEMORY
    max_size: int = 1000
    default_ttl: Optional[int] = 3600  # 1 hour
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    key_prefix: str = "monjyu:"
    level_ttls: Dict[int, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "backend": self.backend.value,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_db": self.redis_db,
            "key_prefix": self.key_prefix,
            "level_ttls": self.level_ttls,
        }


class CacheManager:
    """キャッシュマネージャ
    
    検索結果およびインデックスレベルのキャッシュを管理。
    複数のバックエンド (Memory, Redis) をサポート。
    
    Example:
        >>> manager = CacheManager()
        >>> 
        >>> # 検索結果キャッシュ
        >>> cache_key = manager.generate_key("search", query="test")
        >>> await manager.set(cache_key, result, ttl=60)
        >>> 
        >>> # レベル別キャッシュ
        >>> await manager.set_by_level(0, "index_0", index_data)
        >>> 
        >>> # キャッシュ取得
        >>> cached = await manager.get(cache_key)
    """
    
    def __init__(
        self,
        config: Optional[CacheManagerConfig] = None,
    ):
        """初期化
        
        Args:
            config: 設定
        """
        self.config = config or CacheManagerConfig()
        self._cache: Optional[CacheProtocol] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """キャッシュを初期化"""
        if self._initialized:
            return
        
        if self.config.backend == CacheBackend.MEMORY:
            self._cache = MemoryCache(
                max_size=self.config.max_size,
                default_ttl=self.config.default_ttl,
            )
        elif self.config.backend == CacheBackend.REDIS:
            self._cache = RedisCache(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                key_prefix=self.config.key_prefix,
                default_ttl=self.config.default_ttl,
            )
            await self._cache.connect()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        self._initialized = True
        logger.info(f"CacheManager initialized with {self.config.backend.value} backend")
    
    async def close(self) -> None:
        """キャッシュを閉じる"""
        if self._cache and isinstance(self._cache, RedisCache):
            await self._cache.disconnect()
        self._initialized = False
    
    @staticmethod
    def generate_key(
        namespace: str,
        **kwargs: Any,
    ) -> str:
        """キャッシュキーを生成
        
        Args:
            namespace: 名前空間 (例: "search", "index")
            **kwargs: キー要素
            
        Returns:
            キャッシュキー
        """
        key_parts = [namespace]
        for k, v in sorted(kwargs.items()):
            if v is not None:
                key_parts.append(f"{k}={v}")
        
        key_str = ":".join(key_parts)
        # 長いキーはハッシュ化
        if len(key_str) > 200:
            hash_part = hashlib.md5(key_str.encode()).hexdigest()[:16]
            return f"{namespace}:{hash_part}"
        return key_str
    
    async def get(self, key: str) -> Optional[Any]:
        """キャッシュ取得
        
        Args:
            key: キャッシュキー
            
        Returns:
            キャッシュ値
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._cache.get(key)
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> None:
        """キャッシュ設定
        
        Args:
            key: キャッシュキー
            value: キャッシュ値
            ttl: TTL (秒)
            level: インデックスレベル
        """
        if not self._initialized:
            await self.initialize()
        
        # レベル別TTL
        if level is not None and ttl is None:
            ttl = self.config.level_ttls.get(level, self.config.default_ttl)
        
        await self._cache.set(key, value, ttl=ttl, level=level)
    
    async def set_by_level(
        self,
        level: int,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """レベル別キャッシュ設定
        
        Args:
            level: インデックスレベル
            key: キャッシュキー
            value: キャッシュ値
            ttl: TTL (秒)
        """
        full_key = f"level{level}:{key}"
        await self.set(full_key, value, ttl=ttl, level=level)
    
    async def get_by_level(
        self,
        level: int,
        key: str,
    ) -> Optional[Any]:
        """レベル別キャッシュ取得
        
        Args:
            level: インデックスレベル
            key: キャッシュキー
            
        Returns:
            キャッシュ値
        """
        full_key = f"level{level}:{key}"
        return await self.get(full_key)
    
    async def delete(self, key: str) -> bool:
        """キャッシュ削除
        
        Args:
            key: キャッシュキー
            
        Returns:
            削除成功したらTrue
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._cache.delete(key)
    
    async def exists(self, key: str) -> bool:
        """キャッシュ存在確認
        
        Args:
            key: キャッシュキー
            
        Returns:
            存在すればTrue
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._cache.exists(key)
    
    async def clear(self) -> int:
        """全キャッシュをクリア
        
        Returns:
            クリアしたエントリ数
        """
        if not self._initialized:
            await self.initialize()
        
        count = await self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
    
    async def clear_by_level(self, level: int) -> int:
        """指定レベルのキャッシュをクリア
        
        Args:
            level: インデックスレベル
            
        Returns:
            クリアしたエントリ数
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(self._cache, (MemoryCache, RedisCache)):
            count = await self._cache.clear_by_level(level)
            logger.info(f"Cleared {count} cache entries for level {level}")
            return count
        return 0
    
    async def clear_expired(self) -> int:
        """期限切れキャッシュをクリア
        
        Returns:
            クリアしたエントリ数
        """
        if not self._initialized:
            await self.initialize()
        
        if isinstance(self._cache, MemoryCache):
            count = await self._cache.clear_expired()
            logger.info(f"Cleared {count} expired cache entries")
            return count
        return 0
    
    def get_stats(self) -> CacheStats:
        """統計を取得
        
        Returns:
            キャッシュ統計
        """
        if not self._cache:
            return CacheStats()
        return self._cache.get_stats()
    
    def reset_stats(self) -> None:
        """統計をリセット"""
        if self._cache:
            self._cache.reset_stats()
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
        level: Optional[int] = None,
    ) -> Any:
        """キャッシュ取得、なければ生成して設定
        
        Args:
            key: キャッシュキー
            factory: 値生成関数
            ttl: TTL (秒)
            level: インデックスレベル
            
        Returns:
            キャッシュ値または生成値
        """
        cached = await self.get(key)
        if cached is not None:
            return cached
        
        # 生成
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()
        
        await self.set(key, value, ttl=ttl, level=level)
        return value
    
    def get_status(self) -> Dict[str, Any]:
        """ステータスを取得
        
        Returns:
            ステータス情報
        """
        stats = self.get_stats()
        return {
            "initialized": self._initialized,
            "backend": self.config.backend.value,
            "stats": stats.to_dict(),
            "config": self.config.to_dict(),
        }


def create_cache_manager(
    backend: Union[str, CacheBackend] = CacheBackend.MEMORY,
    **kwargs: Any,
) -> CacheManager:
    """CacheManager を作成
    
    Args:
        backend: キャッシュバックエンド
        **kwargs: 追加設定
        
    Returns:
        CacheManager インスタンス
    """
    if isinstance(backend, str):
        backend = CacheBackend(backend)
    
    config = CacheManagerConfig(backend=backend, **kwargs)
    return CacheManager(config=config)
