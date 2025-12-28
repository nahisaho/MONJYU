# MONJYU Observability Module
"""
monjyu.observability - テレメトリ、ロギング、メトリクス

Azure Monitor / OpenTelemetry 互換のObservabilityフレームワーク。
ローカル開発ではコンソール出力、本番ではAzure Application Insightsに送信。
"""

from __future__ import annotations

import asyncio
import functools
import logging
import sys
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Generator, TypeVar

# Type variable for generic decorators
F = TypeVar("F", bound=Callable[..., Any])


# ============================================================
# Enums
# ============================================================


class LogLevel(Enum):
    """ログレベル"""
    
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    
    def to_logging_level(self) -> int:
        """Python logging レベルに変換"""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return mapping[self]


class MetricType(Enum):
    """メトリクスタイプ"""
    
    COUNTER = "counter"         # 累積カウンタ
    GAUGE = "gauge"             # 瞬間値
    HISTOGRAM = "histogram"     # 分布
    TIMER = "timer"             # 時間計測


class SpanStatus(Enum):
    """スパンステータス"""
    
    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


# ============================================================
# Exceptions
# ============================================================


class ObservabilityError(Exception):
    """Observabilityエラー基底クラス"""
    pass


# ============================================================
# Data Classes
# ============================================================


@dataclass
class SpanContext:
    """スパンコンテキスト（トレーシング用）"""
    
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:16])
    parent_span_id: str | None = None
    
    def child(self) -> "SpanContext":
        """子スパンコンテキストを生成"""
        return SpanContext(
            trace_id=self.trace_id,
            parent_span_id=self.span_id,
        )


@dataclass
class Span:
    """トレーシングスパン"""
    
    name: str
    context: SpanContext
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    status: SpanStatus = SpanStatus.UNSET
    
    # 属性
    attributes: dict[str, Any] = field(default_factory=dict)
    
    # イベント（ログ的な情報）
    events: list[dict[str, Any]] = field(default_factory=list)
    
    # エラー情報
    error: str | None = None
    
    @property
    def duration_ms(self) -> float:
        """スパンの持続時間(ms)"""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000
    
    @property
    def is_finished(self) -> bool:
        """スパンが終了したか"""
        return self.end_time is not None
    
    def set_attribute(self, key: str, value: Any) -> None:
        """属性を設定"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """イベントを追加"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
    
    def set_status(self, status: SpanStatus, error: str | None = None) -> None:
        """ステータスを設定"""
        self.status = status
        if error:
            self.error = error
    
    def finish(self, status: SpanStatus | None = None) -> None:
        """スパンを終了"""
        self.end_time = time.time()
        if status:
            self.status = status
        elif self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "name": self.name,
            "trace_id": self.context.trace_id,
            "span_id": self.context.span_id,
            "parent_span_id": self.context.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


@dataclass
class MetricValue:
    """メトリクス値"""
    
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)
    unit: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "unit": self.unit,
        }


@dataclass
class LogEntry:
    """ログエントリ"""
    
    level: LogLevel
    message: str
    timestamp: float = field(default_factory=time.time)
    logger_name: str = "monjyu"
    
    # コンテキスト情報
    trace_id: str | None = None
    span_id: str | None = None
    
    # 追加データ
    extra: dict[str, Any] = field(default_factory=dict)
    
    # 例外情報
    exception: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "level": self.level.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "logger_name": self.logger_name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "extra": self.extra,
            "exception": self.exception,
        }
    
    def format(self, include_trace: bool = True) -> str:
        """フォーマット済み文字列を生成"""
        dt = datetime.fromtimestamp(self.timestamp)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        parts = [
            f"[{time_str}]",
            f"[{self.level.value.upper():8s}]",
            f"[{self.logger_name}]",
        ]
        
        if include_trace and self.trace_id:
            parts.append(f"[trace:{self.trace_id[:8]}]")
        
        parts.append(self.message)
        
        if self.extra:
            extras = " ".join(f"{k}={v}" for k, v in self.extra.items())
            parts.append(f"| {extras}")
        
        return " ".join(parts)


@dataclass
class ObservabilityConfig:
    """Observability設定"""
    
    # ログ設定
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_to_console: bool = True
    log_to_file: str | None = None
    
    # メトリクス設定
    metrics_enabled: bool = True
    metrics_prefix: str = "monjyu"
    
    # トレーシング設定
    tracing_enabled: bool = True
    sample_rate: float = 1.0  # 1.0 = 100%サンプリング
    
    # Azure Application Insights
    app_insights_connection_string: str | None = None
    
    # エクスポート設定
    export_interval_seconds: float = 60.0
    batch_size: int = 100
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "log_level": self.log_level.value,
            "log_to_console": self.log_to_console,
            "log_to_file": self.log_to_file,
            "metrics_enabled": self.metrics_enabled,
            "metrics_prefix": self.metrics_prefix,
            "tracing_enabled": self.tracing_enabled,
            "sample_rate": self.sample_rate,
            "app_insights_enabled": self.app_insights_connection_string is not None,
        }


# ============================================================
# Exporter Protocol
# ============================================================


class TelemetryExporterProtocol(ABC):
    """テレメトリエクスポーターのプロトコル"""
    
    @abstractmethod
    def export_spans(self, spans: list[Span]) -> None:
        """スパンをエクスポート"""
        ...
    
    @abstractmethod
    def export_metrics(self, metrics: list[MetricValue]) -> None:
        """メトリクスをエクスポート"""
        ...
    
    @abstractmethod
    def export_logs(self, logs: list[LogEntry]) -> None:
        """ログをエクスポート"""
        ...
    
    @abstractmethod
    def flush(self) -> None:
        """バッファをフラッシュ"""
        ...


class ConsoleExporter(TelemetryExporterProtocol):
    """コンソールエクスポーター（開発用）"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def export_spans(self, spans: list[Span]) -> None:
        """スパンをコンソールに出力"""
        if not self.verbose:
            return
        for span in spans:
            status_icon = "✅" if span.status == SpanStatus.OK else "❌"
            print(f"[SPAN] {status_icon} {span.name} ({span.duration_ms:.2f}ms)")
    
    def export_metrics(self, metrics: list[MetricValue]) -> None:
        """メトリクスをコンソールに出力"""
        if not self.verbose:
            return
        for metric in metrics:
            tags = " ".join(f"{k}={v}" for k, v in metric.tags.items())
            print(f"[METRIC] {metric.name}={metric.value} {tags}")
    
    def export_logs(self, logs: list[LogEntry]) -> None:
        """ログをコンソールに出力"""
        for log in logs:
            print(log.format())
    
    def flush(self) -> None:
        """フラッシュ（コンソールは即時出力なので何もしない）"""
        pass


class InMemoryExporter(TelemetryExporterProtocol):
    """インメモリエクスポーター（テスト用）"""
    
    def __init__(self):
        self.spans: list[Span] = []
        self.metrics: list[MetricValue] = []
        self.logs: list[LogEntry] = []
    
    def export_spans(self, spans: list[Span]) -> None:
        """スパンをメモリに保存"""
        self.spans.extend(spans)
    
    def export_metrics(self, metrics: list[MetricValue]) -> None:
        """メトリクスをメモリに保存"""
        self.metrics.extend(metrics)
    
    def export_logs(self, logs: list[LogEntry]) -> None:
        """ログをメモリに保存"""
        self.logs.extend(logs)
    
    def flush(self) -> None:
        """フラッシュ（何もしない）"""
        pass
    
    def clear(self) -> None:
        """データをクリア"""
        self.spans.clear()
        self.metrics.clear()
        self.logs.clear()


# ============================================================
# Tracer
# ============================================================


class Tracer:
    """トレーサー
    
    分散トレーシングのためのスパン管理。
    
    Example:
        tracer = Tracer()
        
        with tracer.start_span("search") as span:
            span.set_attribute("query", query)
            # do search...
    """
    
    def __init__(
        self,
        service_name: str = "monjyu",
        exporter: TelemetryExporterProtocol | None = None,
        config: ObservabilityConfig | None = None,
    ):
        self.service_name = service_name
        self.exporter = exporter or ConsoleExporter()
        self.config = config or ObservabilityConfig()
        
        self._current_span: Span | None = None
        self._span_stack: list[Span] = []
        self._pending_spans: list[Span] = []
    
    @property
    def current_span(self) -> Span | None:
        """現在のスパン"""
        return self._current_span
    
    @property
    def current_context(self) -> SpanContext | None:
        """現在のスパンコンテキスト"""
        return self._current_span.context if self._current_span else None
    
    @contextmanager
    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """スパンを開始（コンテキストマネージャ）
        
        Args:
            name: スパン名
            attributes: 初期属性
            
        Yields:
            Span
        """
        if not self.config.tracing_enabled:
            # トレーシング無効時はダミースパンを返す
            dummy = Span(name=name, context=SpanContext())
            yield dummy
            return
        
        # コンテキスト生成
        if self._current_span:
            context = self._current_span.context.child()
        else:
            context = SpanContext()
        
        # スパン生成
        span = Span(name=name, context=context)
        span.set_attribute("service.name", self.service_name)
        
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        
        # スタックにプッシュ
        if self._current_span:
            self._span_stack.append(self._current_span)
        self._current_span = span
        
        try:
            yield span
            span.finish(SpanStatus.OK)
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            span.finish()
            raise
        finally:
            # スタックからポップ
            self._pending_spans.append(span)
            self._current_span = self._span_stack.pop() if self._span_stack else None
            
            # バッチエクスポート
            if len(self._pending_spans) >= self.config.batch_size:
                self._export_spans()
    
    def _export_spans(self) -> None:
        """保留中のスパンをエクスポート"""
        if self._pending_spans:
            self.exporter.export_spans(self._pending_spans)
            self._pending_spans.clear()
    
    def flush(self) -> None:
        """すべての保留中スパンをエクスポート"""
        self._export_spans()
        self.exporter.flush()


# ============================================================
# Metrics
# ============================================================


class MetricsCollector:
    """メトリクスコレクター
    
    カウンタ、ゲージ、ヒストグラムなどのメトリクスを収集。
    
    Example:
        metrics = MetricsCollector()
        
        metrics.increment("search.count")
        metrics.gauge("index.documents", 1000)
        metrics.timer("search.latency_ms", 150.5)
    """
    
    def __init__(
        self,
        exporter: TelemetryExporterProtocol | None = None,
        config: ObservabilityConfig | None = None,
    ):
        self.exporter = exporter or ConsoleExporter()
        self.config = config or ObservabilityConfig()
        
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[float]] = {}
        self._pending_metrics: list[MetricValue] = []
    
    def _metric_name(self, name: str) -> str:
        """プレフィックス付きメトリクス名"""
        return f"{self.config.metrics_prefix}.{name}"
    
    def increment(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """カウンタをインクリメント"""
        if not self.config.metrics_enabled:
            return
        
        full_name = self._metric_name(name)
        self._counters[full_name] = self._counters.get(full_name, 0) + value
        
        metric = MetricValue(
            name=full_name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags or {},
        )
        self._pending_metrics.append(metric)
    
    def decrement(
        self,
        name: str,
        value: float = 1.0,
        tags: dict[str, str] | None = None,
    ) -> None:
        """カウンタをデクリメント"""
        self.increment(name, -value, tags)
    
    def gauge(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """ゲージ値を設定"""
        if not self.config.metrics_enabled:
            return
        
        full_name = self._metric_name(name)
        self._gauges[full_name] = value
        
        metric = MetricValue(
            name=full_name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags or {},
        )
        self._pending_metrics.append(metric)
    
    def histogram(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """ヒストグラムに値を追加"""
        if not self.config.metrics_enabled:
            return
        
        full_name = self._metric_name(name)
        if full_name not in self._histograms:
            self._histograms[full_name] = []
        self._histograms[full_name].append(value)
        
        metric = MetricValue(
            name=full_name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags or {},
        )
        self._pending_metrics.append(metric)
    
    def timer(
        self,
        name: str,
        value_ms: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """タイマー値を記録"""
        if not self.config.metrics_enabled:
            return
        
        full_name = self._metric_name(name)
        
        metric = MetricValue(
            name=full_name,
            value=value_ms,
            metric_type=MetricType.TIMER,
            tags=tags or {},
            unit="ms",
        )
        self._pending_metrics.append(metric)
    
    @contextmanager
    def measure_time(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> Generator[None, None, None]:
        """時間計測コンテキストマネージャ"""
        start = time.time()
        try:
            yield
        finally:
            elapsed_ms = (time.time() - start) * 1000
            self.timer(name, elapsed_ms, tags)
    
    def get_counter(self, name: str) -> float:
        """カウンタ値を取得"""
        return self._counters.get(self._metric_name(name), 0.0)
    
    def get_gauge(self, name: str) -> float | None:
        """ゲージ値を取得"""
        return self._gauges.get(self._metric_name(name))
    
    def get_histogram_stats(self, name: str) -> dict[str, float] | None:
        """ヒストグラム統計を取得"""
        full_name = self._metric_name(name)
        values = self._histograms.get(full_name)
        
        if not values:
            return None
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }
    
    def flush(self) -> None:
        """保留中のメトリクスをエクスポート"""
        if self._pending_metrics:
            self.exporter.export_metrics(self._pending_metrics)
            self._pending_metrics.clear()
        self.exporter.flush()
    
    def get_status(self) -> dict[str, Any]:
        """現在の状態を取得"""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histogram_count": {k: len(v) for k, v in self._histograms.items()},
        }


# ============================================================
# Logger
# ============================================================


class Logger:
    """構造化ロガー
    
    トレースID付きの構造化ログを出力。
    
    Example:
        logger = Logger("monjyu.search")
        
        logger.info("Search started", query=query, mode=mode)
        logger.error("Search failed", error=str(e))
    """
    
    def __init__(
        self,
        name: str = "monjyu",
        exporter: TelemetryExporterProtocol | None = None,
        config: ObservabilityConfig | None = None,
        tracer: Tracer | None = None,
    ):
        self.name = name
        self.exporter = exporter or ConsoleExporter()
        self.config = config or ObservabilityConfig()
        self.tracer = tracer
        
        # 標準のPythonロガーも設定
        self._setup_python_logger()
    
    def _setup_python_logger(self) -> None:
        """Python標準ロガーをセットアップ"""
        self._python_logger = logging.getLogger(self.name)
        self._python_logger.setLevel(self.config.log_level.to_logging_level())
        
        # ハンドラが未設定の場合のみ追加
        if not self._python_logger.handlers:
            if self.config.log_to_console:
                handler = logging.StreamHandler(sys.stdout)
                handler.setFormatter(logging.Formatter(self.config.log_format))
                self._python_logger.addHandler(handler)
            
            if self.config.log_to_file:
                file_handler = logging.FileHandler(self.config.log_to_file)
                file_handler.setFormatter(logging.Formatter(self.config.log_format))
                self._python_logger.addHandler(file_handler)
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        exception: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """ログを記録"""
        # トレースコンテキストを取得
        trace_id = None
        span_id = None
        if self.tracer and self.tracer.current_context:
            trace_id = self.tracer.current_context.trace_id
            span_id = self.tracer.current_context.span_id
        
        entry = LogEntry(
            level=level,
            message=message,
            logger_name=self.name,
            trace_id=trace_id,
            span_id=span_id,
            extra=kwargs,
            exception=str(exception) if exception else None,
        )
        
        self.exporter.export_logs([entry])
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """DEBUGログ"""
        if self.config.log_level.to_logging_level() <= logging.DEBUG:
            self._log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """INFOログ"""
        if self.config.log_level.to_logging_level() <= logging.INFO:
            self._log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """WARNINGログ"""
        if self.config.log_level.to_logging_level() <= logging.WARNING:
            self._log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, exception: Exception | None = None, **kwargs: Any) -> None:
        """ERRORログ"""
        if self.config.log_level.to_logging_level() <= logging.ERROR:
            self._log(LogLevel.ERROR, message, exception=exception, **kwargs)
    
    def critical(self, message: str, exception: Exception | None = None, **kwargs: Any) -> None:
        """CRITICALログ"""
        self._log(LogLevel.CRITICAL, message, exception=exception, **kwargs)
    
    def child(self, name: str) -> "Logger":
        """子ロガーを生成"""
        return Logger(
            name=f"{self.name}.{name}",
            exporter=self.exporter,
            config=self.config,
            tracer=self.tracer,
        )


# ============================================================
# Observability Manager (統合クラス)
# ============================================================


class Observability:
    """Observability統合マネージャー
    
    トレーサー、メトリクス、ロガーを統合管理。
    
    Example:
        obs = Observability.create()
        
        # ロギング
        obs.logger.info("Starting search")
        
        # トレーシング
        with obs.tracer.start_span("search") as span:
            span.set_attribute("query", query)
            
            # メトリクス
            obs.metrics.increment("search.requests")
            
            # 処理...
            
        obs.metrics.timer("search.latency_ms", elapsed_ms)
    """
    
    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        exporter: TelemetryExporterProtocol | None = None,
    ):
        self.config = config or ObservabilityConfig()
        self.exporter = exporter or ConsoleExporter()
        
        self.tracer = Tracer(exporter=self.exporter, config=self.config)
        self.metrics = MetricsCollector(exporter=self.exporter, config=self.config)
        self.logger = Logger(exporter=self.exporter, config=self.config, tracer=self.tracer)
    
    @classmethod
    def create(
        cls,
        config: ObservabilityConfig | dict[str, Any] | None = None,
        exporter: TelemetryExporterProtocol | None = None,
    ) -> "Observability":
        """Observabilityインスタンスを作成"""
        if isinstance(config, dict):
            # LogLevelの変換
            if "log_level" in config and isinstance(config["log_level"], str):
                config["log_level"] = LogLevel(config["log_level"])
            config = ObservabilityConfig(**config)
        
        return cls(config=config, exporter=exporter)
    
    @classmethod
    def create_for_testing(cls) -> tuple["Observability", InMemoryExporter]:
        """テスト用Observabilityを作成"""
        exporter = InMemoryExporter()
        obs = cls(exporter=exporter)
        return obs, exporter
    
    def flush(self) -> None:
        """すべてのテレメトリをフラッシュ"""
        self.tracer.flush()
        self.metrics.flush()
    
    def get_status(self) -> dict[str, Any]:
        """ステータスを取得"""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.get_status(),
        }


# ============================================================
# Decorators
# ============================================================


def traced(
    name: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Callable[[F], F]:
    """関数をトレースするデコレータ
    
    Example:
        @traced("search.execute")
        def execute_search(query: str) -> SearchResult:
            ...
    """
    def decorator(func: F) -> F:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = _get_default_observability()
            with obs.tracer.start_span(span_name, attributes) as span:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = _get_default_observability()
            with obs.tracer.start_span(span_name, attributes) as span:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.set_status(SpanStatus.ERROR, str(e))
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


def timed(
    name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """関数の実行時間を計測するデコレータ
    
    Example:
        @timed("search.latency")
        def execute_search(query: str) -> SearchResult:
            ...
    """
    def decorator(func: F) -> F:
        metric_name = name or f"{func.__module__}.{func.__qualname__}.duration"
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = _get_default_observability()
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                obs.metrics.timer(metric_name, elapsed_ms, tags)
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            obs = _get_default_observability()
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.time() - start) * 1000
                obs.metrics.timer(metric_name, elapsed_ms, tags)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# ============================================================
# Global Instance
# ============================================================

_default_observability: Observability | None = None


def _get_default_observability() -> Observability:
    """デフォルトのObservabilityインスタンスを取得"""
    global _default_observability
    if _default_observability is None:
        _default_observability = Observability.create()
    return _default_observability


def configure_observability(
    config: ObservabilityConfig | dict[str, Any] | None = None,
    exporter: TelemetryExporterProtocol | None = None,
) -> Observability:
    """グローバルObservabilityを設定"""
    global _default_observability
    _default_observability = Observability.create(config, exporter)
    return _default_observability


def get_tracer() -> Tracer:
    """グローバルトレーサーを取得"""
    return _get_default_observability().tracer


def get_metrics() -> MetricsCollector:
    """グローバルメトリクスコレクターを取得"""
    return _get_default_observability().metrics


def get_logger(name: str = "monjyu") -> Logger:
    """ロガーを取得"""
    obs = _get_default_observability()
    if name == "monjyu":
        return obs.logger
    return obs.logger.child(name)
