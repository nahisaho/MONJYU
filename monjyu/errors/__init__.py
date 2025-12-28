"""MONJYU Error Handling Framework.

統一的なエラー管理、リトライ、サーキットブレーカーを提供。

Example:
    >>> from monjyu.errors import (
    ...     MONJYUError, ConfigurationError, IndexError, QueryError,
    ...     retry, circuit_breaker, ErrorHandler
    ... )
    >>>
    >>> # カスタム例外
    >>> raise QueryError("Search failed", query="test", details={"cause": "timeout"})
    >>>
    >>> # リトライデコレータ
    >>> @retry(max_attempts=3, delay=1.0, backoff=2.0)
    ... async def fetch_data():
    ...     ...
    >>>
    >>> # サーキットブレーカー
    >>> @circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
    ... async def call_external_service():
    ...     ...
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

__all__ = [
    # Base exceptions
    "MONJYUError",
    "ConfigurationError",
    "IndexError",
    "QueryError",
    "LLMError",
    "StorageError",
    "PDFProcessError",
    "EmbeddingError",
    "GraphError",
    "CacheError",
    "ValidationError",
    "TimeoutError",
    "RateLimitError",
    "AuthenticationError",
    "NetworkError",
    # Error context
    "ErrorContext",
    "ErrorSeverity",
    # Retry
    "RetryConfig",
    "retry",
    # Circuit breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "circuit_breaker",
    # Error handler
    "ErrorHandler",
    "create_error_handler",
]


# ============================================================
# Error Severity
# ============================================================


class ErrorSeverity(str, Enum):
    """エラー重要度"""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def to_logging_level(self) -> int:
        """ロギングレベルに変換"""
        mapping = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


# ============================================================
# Error Context
# ============================================================


@dataclass
class ErrorContext:
    """エラーコンテキスト情報
    
    エラー発生時の詳細な情報を保持。
    
    Attributes:
        error_id: ユニークなエラーID
        timestamp: エラー発生時刻
        component: エラー発生コンポーネント
        operation: 実行中の操作
        details: 追加の詳細情報
        stack_trace: スタックトレース
        correlation_id: リクエスト相関ID
    """
    
    error_id: str = field(default_factory=lambda: f"err_{int(time.time() * 1000)}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    component: str | None = None
    operation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    stack_trace: str | None = None
    correlation_id: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "details": self.details,
            "stack_trace": self.stack_trace,
            "correlation_id": self.correlation_id,
        }


# ============================================================
# Base Exception Classes
# ============================================================


class MONJYUError(Exception):
    """MONJYU基底例外クラス
    
    すべてのMONJYU例外の基底クラス。
    構造化されたエラー情報を提供。
    
    Attributes:
        message: エラーメッセージ
        code: エラーコード
        severity: エラー重要度
        context: エラーコンテキスト
        cause: 原因となった例外
    
    Example:
        >>> raise MONJYUError(
        ...     "Operation failed",
        ...     code="ERR001",
        ...     severity=ErrorSeverity.ERROR,
        ...     details={"key": "value"}
        ... )
    """
    
    default_code: str = "MONJYU_ERROR"
    default_severity: ErrorSeverity = ErrorSeverity.ERROR
    
    def __init__(
        self,
        message: str,
        code: str | None = None,
        severity: ErrorSeverity | None = None,
        cause: Exception | None = None,
        component: str | None = None,
        operation: str | None = None,
        correlation_id: str | None = None,
        **details: Any,
    ):
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.severity = severity or self.default_severity
        self.cause = cause
        
        # コンテキストを構築
        self.context = ErrorContext(
            component=component,
            operation=operation,
            details=details,
            stack_trace=traceback.format_exc() if cause else None,
            correlation_id=correlation_id,
        )
    
    def __str__(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.context.component:
            parts.append(f"(component: {self.context.component})")
        if self.context.operation:
            parts.append(f"(operation: {self.context.operation})")
        if self.cause:
            parts.append(f"caused by: {self.cause}")
        return " ".join(parts)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"severity={self.severity.value!r})"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "severity": self.severity.value,
            "context": self.context.to_dict(),
            "cause": str(self.cause) if self.cause else None,
        }
    
    def with_context(self, **kwargs: Any) -> "MONJYUError":
        """追加のコンテキストを設定"""
        self.context.details.update(kwargs)
        return self
    
    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        message: str | None = None,
        **kwargs: Any,
    ) -> "MONJYUError":
        """既存の例外からMONJYUErrorを作成"""
        return cls(
            message=message or str(exc),
            cause=exc,
            **kwargs,
        )


# ============================================================
# Specific Exception Classes
# ============================================================


class ConfigurationError(MONJYUError):
    """設定エラー
    
    設定ファイルの読み込みや検証に失敗した場合。
    """
    
    default_code = "CONFIG_ERROR"
    default_severity = ErrorSeverity.ERROR


class IndexError(MONJYUError):
    """インデックスエラー
    
    インデックスの構築、読み込み、更新に失敗した場合。
    """
    
    default_code = "INDEX_ERROR"
    default_severity = ErrorSeverity.ERROR


class QueryError(MONJYUError):
    """クエリエラー
    
    検索クエリの実行に失敗した場合。
    """
    
    default_code = "QUERY_ERROR"
    default_severity = ErrorSeverity.ERROR
    
    def __init__(
        self,
        message: str,
        query: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if query:
            self.context.details["query"] = query


class LLMError(MONJYUError):
    """LLMエラー
    
    LLM APIの呼び出しに失敗した場合。
    """
    
    default_code = "LLM_ERROR"
    default_severity = ErrorSeverity.ERROR
    
    def __init__(
        self,
        message: str,
        model: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if model:
            self.context.details["model"] = model
        if provider:
            self.context.details["provider"] = provider


class StorageError(MONJYUError):
    """ストレージエラー
    
    ファイルやデータベースの読み書きに失敗した場合。
    """
    
    default_code = "STORAGE_ERROR"
    default_severity = ErrorSeverity.ERROR
    
    def __init__(
        self,
        message: str,
        path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if path:
            self.context.details["path"] = path


class PDFProcessError(MONJYUError):
    """PDF処理エラー
    
    PDFの解析や変換に失敗した場合。
    """
    
    default_code = "PDF_ERROR"
    default_severity = ErrorSeverity.ERROR
    
    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        page: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if file_path:
            self.context.details["file_path"] = file_path
        if page is not None:
            self.context.details["page"] = page


class EmbeddingError(MONJYUError):
    """埋め込みエラー
    
    ベクトル埋め込みの生成に失敗した場合。
    """
    
    default_code = "EMBEDDING_ERROR"
    default_severity = ErrorSeverity.ERROR


class GraphError(MONJYUError):
    """グラフエラー
    
    グラフ構築や操作に失敗した場合。
    """
    
    default_code = "GRAPH_ERROR"
    default_severity = ErrorSeverity.ERROR


class CacheError(MONJYUError):
    """キャッシュエラー
    
    キャッシュの読み書きに失敗した場合。
    """
    
    default_code = "CACHE_ERROR"
    default_severity = ErrorSeverity.WARNING


class ValidationError(MONJYUError):
    """バリデーションエラー
    
    入力データの検証に失敗した場合。
    """
    
    default_code = "VALIDATION_ERROR"
    default_severity = ErrorSeverity.WARNING
    
    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if field:
            self.context.details["field"] = field
        if value is not None:
            self.context.details["value"] = repr(value)


class TimeoutError(MONJYUError):
    """タイムアウトエラー
    
    操作がタイムアウトした場合。
    """
    
    default_code = "TIMEOUT_ERROR"
    default_severity = ErrorSeverity.WARNING
    
    def __init__(
        self,
        message: str,
        timeout_seconds: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if timeout_seconds is not None:
            self.context.details["timeout_seconds"] = timeout_seconds


class RateLimitError(MONJYUError):
    """レート制限エラー
    
    APIレート制限に達した場合。
    """
    
    default_code = "RATE_LIMIT_ERROR"
    default_severity = ErrorSeverity.WARNING
    
    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if retry_after is not None:
            self.context.details["retry_after"] = retry_after


class AuthenticationError(MONJYUError):
    """認証エラー
    
    認証に失敗した場合。
    """
    
    default_code = "AUTH_ERROR"
    default_severity = ErrorSeverity.ERROR


class NetworkError(MONJYUError):
    """ネットワークエラー
    
    ネットワーク接続に失敗した場合。
    """
    
    default_code = "NETWORK_ERROR"
    default_severity = ErrorSeverity.ERROR


# ============================================================
# Retry Configuration
# ============================================================


@dataclass
class RetryConfig:
    """リトライ設定
    
    Attributes:
        max_attempts: 最大試行回数
        delay: 初期遅延（秒）
        backoff: バックオフ係数
        max_delay: 最大遅延（秒）
        exceptions: リトライ対象の例外タイプ
        on_retry: リトライ時のコールバック
    """
    
    max_attempts: int = 3
    delay: float = 1.0
    backoff: float = 2.0
    max_delay: float = 60.0
    exceptions: tuple[type[Exception], ...] = (Exception,)
    on_retry: Callable[[int, Exception, float], None] | None = None
    
    def calculate_delay(self, attempt: int) -> float:
        """遅延時間を計算"""
        delay = self.delay * (self.backoff ** (attempt - 1))
        return min(delay, self.max_delay)


# Type variable for return type
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Callable[[F], F]:
    """リトライデコレータ
    
    指定された例外が発生した場合、自動的にリトライ。
    
    Args:
        max_attempts: 最大試行回数
        delay: 初期遅延（秒）
        backoff: バックオフ係数
        max_delay: 最大遅延（秒）
        exceptions: リトライ対象の例外タイプ
        on_retry: リトライ時のコールバック
    
    Example:
        >>> @retry(max_attempts=3, delay=1.0, backoff=2.0)
        ... async def fetch_data():
        ...     response = await client.get("/api/data")
        ...     return response.json()
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        delay=delay,
        backoff=backoff,
        max_delay=max_delay,
        exceptions=exceptions,
        on_retry=on_retry,
    )
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        raise
                    
                    wait_time = config.calculate_delay(attempt)
                    
                    if config.on_retry:
                        config.on_retry(attempt, e, wait_time)
                    
                    await asyncio.sleep(wait_time)
            
            # Should not reach here
            if last_exception:
                raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Exception | None = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        raise
                    
                    wait_time = config.calculate_delay(attempt)
                    
                    if config.on_retry:
                        config.on_retry(attempt, e, wait_time)
                    
                    time.sleep(wait_time)
            
            # Should not reach here
            if last_exception:
                raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# ============================================================
# Circuit Breaker
# ============================================================


class CircuitState(str, Enum):
    """サーキットブレーカー状態"""
    
    CLOSED = "closed"      # 正常動作中
    OPEN = "open"          # 遮断中
    HALF_OPEN = "half_open"  # 復旧テスト中


@dataclass
class CircuitBreakerConfig:
    """サーキットブレーカー設定
    
    Attributes:
        failure_threshold: 失敗しきい値
        recovery_timeout: 復旧待機時間（秒）
        half_open_max_calls: ハーフオープン時の最大呼び出し数
        exceptions: 対象の例外タイプ
    """
    
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 1
    exceptions: tuple[type[Exception], ...] = (Exception,)


class CircuitBreaker:
    """サーキットブレーカー
    
    連続した失敗を検出し、サービスを保護。
    
    Example:
        >>> breaker = CircuitBreaker(failure_threshold=5)
        >>> 
        >>> async def call_service():
        ...     async with breaker:
        ...         return await external_api.call()
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        exceptions: tuple[type[Exception], ...] = (Exception,),
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        self.config = CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_max_calls=half_open_max_calls,
            exceptions=exceptions,
        )
        self.on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """現在の状態を取得"""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """失敗回数を取得"""
        return self._failure_count
    
    def _set_state(self, new_state: CircuitState) -> None:
        """状態を設定"""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            if self.on_state_change:
                self.on_state_change(old_state, new_state)
    
    def _should_allow_request(self) -> bool:
        """リクエストを許可するか判定"""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            # タイムアウト経過で HALF_OPEN に遷移
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.config.recovery_timeout
            ):
                self._set_state(CircuitState.HALF_OPEN)
                self._half_open_calls = 0
                return True
            return False
        
        # HALF_OPEN
        if self._half_open_calls < self.config.half_open_max_calls:
            return True
        return False
    
    def _on_success(self) -> None:
        """成功時の処理"""
        if self._state == CircuitState.HALF_OPEN:
            self._set_state(CircuitState.CLOSED)
        self._failure_count = 0
        self._half_open_calls = 0
    
    def _on_failure(self) -> None:
        """失敗時の処理"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            self._set_state(CircuitState.OPEN)
        elif self._failure_count >= self.config.failure_threshold:
            self._set_state(CircuitState.OPEN)
    
    async def __aenter__(self) -> "CircuitBreaker":
        """非同期コンテキストマネージャ開始"""
        async with self._lock:
            if not self._should_allow_request():
                raise CircuitOpenError(
                    f"Circuit breaker is {self._state.value}",
                    state=self._state,
                    failure_count=self._failure_count,
                )
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1
        return self
    
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """非同期コンテキストマネージャ終了"""
        async with self._lock:
            if exc_type is None:
                self._on_success()
            elif issubclass(exc_type, self.config.exceptions):
                self._on_failure()
        return False
    
    def __enter__(self) -> "CircuitBreaker":
        """同期コンテキストマネージャ開始"""
        if not self._should_allow_request():
            raise CircuitOpenError(
                f"Circuit breaker is {self._state.value}",
                state=self._state,
                failure_count=self._failure_count,
            )
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
        return self
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        """同期コンテキストマネージャ終了"""
        if exc_type is None:
            self._on_success()
        elif issubclass(exc_type, self.config.exceptions):
            self._on_failure()
        return False
    
    def reset(self) -> None:
        """サーキットブレーカーをリセット"""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
    
    def get_status(self) -> dict[str, Any]:
        """ステータスを取得"""
        return {
            "state": self._state.value,
            "failure_count": self._failure_count,
            "last_failure_time": self._last_failure_time,
        }


class CircuitOpenError(MONJYUError):
    """サーキットオープンエラー
    
    サーキットブレーカーが開いている場合。
    """
    
    default_code = "CIRCUIT_OPEN"
    default_severity = ErrorSeverity.WARNING
    
    def __init__(
        self,
        message: str,
        state: CircuitState | None = None,
        failure_count: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(message, **kwargs)
        if state:
            self.context.details["state"] = state.value
        if failure_count is not None:
            self.context.details["failure_count"] = failure_count


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    half_open_max_calls: int = 1,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
) -> Callable[[F], F]:
    """サーキットブレーカーデコレータ
    
    関数にサーキットブレーカーパターンを適用。
    
    Args:
        failure_threshold: 失敗しきい値
        recovery_timeout: 復旧待機時間（秒）
        half_open_max_calls: ハーフオープン時の最大呼び出し数
        exceptions: 対象の例外タイプ
        on_state_change: 状態変更時のコールバック
    
    Example:
        >>> @circuit_breaker(failure_threshold=5, recovery_timeout=30.0)
        ... async def call_external_api():
        ...     return await api.call()
    """
    breaker = CircuitBreaker(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        half_open_max_calls=half_open_max_calls,
        exceptions=exceptions,
        on_state_change=on_state_change,
    )
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            async with breaker:
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with breaker:
                return func(*args, **kwargs)
        
        # Attach breaker for inspection
        if asyncio.iscoroutinefunction(func):
            async_wrapper.circuit_breaker = breaker  # type: ignore
            return async_wrapper  # type: ignore
        sync_wrapper.circuit_breaker = breaker  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# ============================================================
# Error Handler
# ============================================================


@dataclass
class ErrorHandlerConfig:
    """エラーハンドラ設定"""
    
    log_errors: bool = True
    log_level: ErrorSeverity = ErrorSeverity.ERROR
    include_stack_trace: bool = True
    max_context_size: int = 1000


class ErrorHandler:
    """統合エラーハンドラ
    
    エラーのログ記録、変換、集約を管理。
    
    Example:
        >>> handler = ErrorHandler()
        >>> 
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     handler.handle(e, component="search", operation="query")
    """
    
    def __init__(
        self,
        config: ErrorHandlerConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config or ErrorHandlerConfig()
        self.logger = logger or logging.getLogger("monjyu.errors")
        
        self._error_counts: dict[str, int] = {}
        self._recent_errors: list[MONJYUError] = []
        self._max_recent_errors = 100
    
    def handle(
        self,
        error: Exception,
        component: str | None = None,
        operation: str | None = None,
        reraise: bool = True,
        **context: Any,
    ) -> MONJYUError:
        """エラーを処理
        
        Args:
            error: 処理するエラー
            component: コンポーネント名
            operation: 操作名
            reraise: エラーを再送出するか
            **context: 追加のコンテキスト
        
        Returns:
            変換されたMONJYUError
        
        Raises:
            MONJYUError: reraise=Trueの場合
        """
        # MONJYUErrorに変換
        if isinstance(error, MONJYUError):
            monjyu_error = error
            if component:
                monjyu_error.context.component = component
            if operation:
                monjyu_error.context.operation = operation
            monjyu_error.context.details.update(context)
        else:
            monjyu_error = MONJYUError.from_exception(
                error,
                component=component,
                operation=operation,
                **context,
            )
        
        # ログ記録
        if self.config.log_errors:
            self._log_error(monjyu_error)
        
        # 統計更新
        self._update_stats(monjyu_error)
        
        if reraise:
            raise monjyu_error
        
        return monjyu_error
    
    def _log_error(self, error: MONJYUError) -> None:
        """エラーをログ記録"""
        level = error.severity.to_logging_level()
        
        message = str(error)
        if self.config.include_stack_trace and error.context.stack_trace:
            message += f"\n{error.context.stack_trace}"
        
        self.logger.log(level, message, extra={"error": error.to_dict()})
    
    def _update_stats(self, error: MONJYUError) -> None:
        """統計を更新"""
        error_type = error.__class__.__name__
        self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1
        
        self._recent_errors.append(error)
        if len(self._recent_errors) > self._max_recent_errors:
            self._recent_errors.pop(0)
    
    def get_stats(self) -> dict[str, Any]:
        """エラー統計を取得"""
        return {
            "error_counts": dict(self._error_counts),
            "total_errors": sum(self._error_counts.values()),
            "recent_error_count": len(self._recent_errors),
        }
    
    def get_recent_errors(self, limit: int = 10) -> list[dict[str, Any]]:
        """最近のエラーを取得"""
        return [e.to_dict() for e in self._recent_errors[-limit:]]
    
    def clear_stats(self) -> None:
        """統計をクリア"""
        self._error_counts.clear()
        self._recent_errors.clear()


# Global error handler
_default_handler: ErrorHandler | None = None


def create_error_handler(
    config: ErrorHandlerConfig | None = None,
    logger: logging.Logger | None = None,
) -> ErrorHandler:
    """エラーハンドラを作成"""
    global _default_handler
    _default_handler = ErrorHandler(config=config, logger=logger)
    return _default_handler


def get_error_handler() -> ErrorHandler:
    """グローバルエラーハンドラを取得"""
    global _default_handler
    if _default_handler is None:
        _default_handler = ErrorHandler()
    return _default_handler
