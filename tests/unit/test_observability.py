"""Tests for Observability module.

Telemetry, Logging, Metrics, Tracing tests.
"""

import asyncio
import time

import pytest

from monjyu.observability import (
    ConsoleExporter,
    InMemoryExporter,
    LogEntry,
    Logger,
    LogLevel,
    MetricsCollector,
    MetricType,
    MetricValue,
    Observability,
    ObservabilityConfig,
    Span,
    SpanContext,
    SpanStatus,
    TelemetryExporterProtocol,
    Tracer,
    configure_observability,
    get_logger,
    get_metrics,
    get_tracer,
    timed,
    traced,
)


# ============================================================
# SpanContext Tests
# ============================================================


class TestSpanContext:
    """SpanContext tests."""
    
    def test_context_creation(self):
        """Test context creation."""
        ctx = SpanContext()
        
        assert ctx.trace_id is not None
        assert ctx.span_id is not None
        assert ctx.parent_span_id is None
    
    def test_child_context(self):
        """Test child context creation."""
        parent = SpanContext()
        child = parent.child()
        
        assert child.trace_id == parent.trace_id  # Same trace
        assert child.parent_span_id == parent.span_id  # Links to parent
        assert child.span_id != parent.span_id  # Different span


# ============================================================
# Span Tests
# ============================================================


class TestSpan:
    """Span tests."""
    
    def test_span_creation(self):
        """Test span creation."""
        span = Span(name="test", context=SpanContext())
        
        assert span.name == "test"
        assert span.status == SpanStatus.UNSET
        assert not span.is_finished
    
    def test_span_attributes(self):
        """Test span attributes."""
        span = Span(name="test", context=SpanContext())
        
        span.set_attribute("key", "value")
        span.set_attribute("count", 42)
        
        assert span.attributes["key"] == "value"
        assert span.attributes["count"] == 42
    
    def test_span_events(self):
        """Test span events."""
        span = Span(name="test", context=SpanContext())
        
        span.add_event("started", {"step": 1})
        span.add_event("completed")
        
        assert len(span.events) == 2
        assert span.events[0]["name"] == "started"
    
    def test_span_finish(self):
        """Test span finish."""
        span = Span(name="test", context=SpanContext())
        time.sleep(0.01)
        span.finish()
        
        assert span.is_finished
        assert span.status == SpanStatus.OK
        assert span.duration_ms > 0
    
    def test_span_error(self):
        """Test span with error."""
        span = Span(name="test", context=SpanContext())
        span.set_status(SpanStatus.ERROR, "Something failed")
        span.finish()
        
        assert span.status == SpanStatus.ERROR
        assert span.error == "Something failed"
    
    def test_span_to_dict(self):
        """Test span serialization."""
        span = Span(name="test", context=SpanContext())
        span.set_attribute("key", "value")
        span.finish()
        
        data = span.to_dict()
        
        assert data["name"] == "test"
        assert data["status"] == "ok"
        assert data["attributes"]["key"] == "value"


# ============================================================
# MetricValue Tests
# ============================================================


class TestMetricValue:
    """MetricValue tests."""
    
    def test_metric_creation(self):
        """Test metric creation."""
        metric = MetricValue(
            name="test.counter",
            value=1.0,
            metric_type=MetricType.COUNTER,
        )
        
        assert metric.name == "test.counter"
        assert metric.value == 1.0
        assert metric.metric_type == MetricType.COUNTER
    
    def test_metric_with_tags(self):
        """Test metric with tags."""
        metric = MetricValue(
            name="test.gauge",
            value=100.0,
            metric_type=MetricType.GAUGE,
            tags={"env": "test", "host": "localhost"},
        )
        
        assert metric.tags["env"] == "test"
    
    def test_metric_to_dict(self):
        """Test metric serialization."""
        metric = MetricValue(
            name="test.timer",
            value=150.5,
            metric_type=MetricType.TIMER,
            unit="ms",
        )
        
        data = metric.to_dict()
        
        assert data["name"] == "test.timer"
        assert data["type"] == "timer"
        assert data["unit"] == "ms"


# ============================================================
# LogEntry Tests
# ============================================================


class TestLogEntry:
    """LogEntry tests."""
    
    def test_entry_creation(self):
        """Test entry creation."""
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
        )
        
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
    
    def test_entry_with_trace(self):
        """Test entry with trace context."""
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test",
            trace_id="trace123",
            span_id="span456",
        )
        
        assert entry.trace_id == "trace123"
        assert entry.span_id == "span456"
    
    def test_entry_format(self):
        """Test entry formatting."""
        entry = LogEntry(
            level=LogLevel.INFO,
            message="Test message",
            logger_name="test",
        )
        
        formatted = entry.format()
        
        assert "INFO" in formatted
        assert "Test message" in formatted
        assert "test" in formatted
    
    def test_entry_to_dict(self):
        """Test entry serialization."""
        entry = LogEntry(
            level=LogLevel.ERROR,
            message="Error occurred",
            exception="ValueError: invalid",
        )
        
        data = entry.to_dict()
        
        assert data["level"] == "error"
        assert data["exception"] == "ValueError: invalid"


# ============================================================
# ObservabilityConfig Tests
# ============================================================


class TestObservabilityConfig:
    """ObservabilityConfig tests."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = ObservabilityConfig()
        
        assert config.log_level == LogLevel.INFO
        assert config.metrics_enabled is True
        assert config.tracing_enabled is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ObservabilityConfig(
            log_level=LogLevel.DEBUG,
            metrics_prefix="myapp",
            sample_rate=0.5,
        )
        
        assert config.log_level == LogLevel.DEBUG
        assert config.metrics_prefix == "myapp"
        assert config.sample_rate == 0.5
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = ObservabilityConfig()
        
        data = config.to_dict()
        
        assert data["log_level"] == "info"
        assert "metrics_enabled" in data


# ============================================================
# InMemoryExporter Tests
# ============================================================


class TestInMemoryExporter:
    """InMemoryExporter tests."""
    
    def test_export_spans(self):
        """Test span export."""
        exporter = InMemoryExporter()
        span = Span(name="test", context=SpanContext())
        span.finish()
        
        exporter.export_spans([span])
        
        assert len(exporter.spans) == 1
        assert exporter.spans[0].name == "test"
    
    def test_export_metrics(self):
        """Test metric export."""
        exporter = InMemoryExporter()
        metric = MetricValue(name="test", value=1.0, metric_type=MetricType.COUNTER)
        
        exporter.export_metrics([metric])
        
        assert len(exporter.metrics) == 1
    
    def test_export_logs(self):
        """Test log export."""
        exporter = InMemoryExporter()
        entry = LogEntry(level=LogLevel.INFO, message="test")
        
        exporter.export_logs([entry])
        
        assert len(exporter.logs) == 1
    
    def test_clear(self):
        """Test clearing data."""
        exporter = InMemoryExporter()
        exporter.export_logs([LogEntry(level=LogLevel.INFO, message="test")])
        
        exporter.clear()
        
        assert len(exporter.logs) == 0


# ============================================================
# Tracer Tests
# ============================================================


class TestTracer:
    """Tracer tests."""
    
    def test_tracer_creation(self):
        """Test tracer creation."""
        tracer = Tracer()
        
        assert tracer.current_span is None
    
    def test_start_span(self):
        """Test starting a span."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        
        with tracer.start_span("test") as span:
            assert span.name == "test"
            assert tracer.current_span == span
        
        assert tracer.current_span is None
    
    def test_nested_spans(self):
        """Test nested spans."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        
        with tracer.start_span("parent") as parent:
            parent_trace_id = parent.context.trace_id
            
            with tracer.start_span("child") as child:
                assert child.context.trace_id == parent_trace_id
                assert child.context.parent_span_id == parent.context.span_id
    
    def test_span_with_attributes(self):
        """Test span with attributes."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        
        with tracer.start_span("test", {"key": "value"}) as span:
            span.set_attribute("extra", 123)
        
        tracer.flush()
        
        assert exporter.spans[0].attributes["key"] == "value"
        assert exporter.spans[0].attributes["extra"] == 123
    
    def test_span_error_handling(self):
        """Test span error handling."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        
        with pytest.raises(ValueError):
            with tracer.start_span("test") as span:
                raise ValueError("Test error")
        
        tracer.flush()
        
        assert exporter.spans[0].status == SpanStatus.ERROR
        assert "Test error" in exporter.spans[0].error
    
    def test_tracing_disabled(self):
        """Test with tracing disabled."""
        config = ObservabilityConfig(tracing_enabled=False)
        tracer = Tracer(config=config)
        
        with tracer.start_span("test") as span:
            span.set_attribute("key", "value")
        
        # Should not raise, span is a dummy


# ============================================================
# MetricsCollector Tests
# ============================================================


class TestMetricsCollector:
    """MetricsCollector tests."""
    
    def test_increment(self):
        """Test counter increment."""
        exporter = InMemoryExporter()
        metrics = MetricsCollector(exporter=exporter)
        
        metrics.increment("requests")
        metrics.increment("requests")
        metrics.increment("requests", 3)
        
        assert metrics.get_counter("requests") == 5
    
    def test_decrement(self):
        """Test counter decrement."""
        metrics = MetricsCollector()
        
        metrics.increment("balance", 100)
        metrics.decrement("balance", 30)
        
        assert metrics.get_counter("balance") == 70
    
    def test_gauge(self):
        """Test gauge."""
        metrics = MetricsCollector()
        
        metrics.gauge("temperature", 23.5)
        
        assert metrics.get_gauge("temperature") == 23.5
    
    def test_gauge_update(self):
        """Test gauge update."""
        metrics = MetricsCollector()
        
        metrics.gauge("connections", 10)
        metrics.gauge("connections", 15)
        
        assert metrics.get_gauge("connections") == 15
    
    def test_histogram(self):
        """Test histogram."""
        metrics = MetricsCollector()
        
        for v in [10, 20, 30, 40, 50]:
            metrics.histogram("latency", v)
        
        stats = metrics.get_histogram_stats("latency")
        
        assert stats["count"] == 5
        assert stats["min"] == 10
        assert stats["max"] == 50
        assert stats["mean"] == 30
    
    def test_timer(self):
        """Test timer."""
        exporter = InMemoryExporter()
        metrics = MetricsCollector(exporter=exporter)
        
        metrics.timer("search.latency", 150.5)
        metrics.flush()
        
        assert len(exporter.metrics) == 1
        assert exporter.metrics[0].metric_type == MetricType.TIMER
    
    def test_measure_time(self):
        """Test time measurement context manager."""
        exporter = InMemoryExporter()
        metrics = MetricsCollector(exporter=exporter)
        
        with metrics.measure_time("operation"):
            time.sleep(0.01)
        
        metrics.flush()
        
        assert len(exporter.metrics) == 1
        assert exporter.metrics[0].value > 0
    
    def test_metrics_with_tags(self):
        """Test metrics with tags."""
        exporter = InMemoryExporter()
        metrics = MetricsCollector(exporter=exporter)
        
        metrics.increment("requests", tags={"endpoint": "/search", "method": "GET"})
        metrics.flush()
        
        assert exporter.metrics[0].tags["endpoint"] == "/search"
    
    def test_metrics_disabled(self):
        """Test with metrics disabled."""
        config = ObservabilityConfig(metrics_enabled=False)
        metrics = MetricsCollector(config=config)
        
        metrics.increment("requests")
        
        # Should not raise, no metrics recorded
        assert metrics.get_counter("requests") == 0


# ============================================================
# Logger Tests
# ============================================================


class TestLogger:
    """Logger tests."""
    
    def test_logger_creation(self):
        """Test logger creation."""
        logger = Logger("test")
        
        assert logger.name == "test"
    
    def test_log_levels(self):
        """Test all log levels."""
        exporter = InMemoryExporter()
        config = ObservabilityConfig(log_level=LogLevel.DEBUG)
        logger = Logger("test", exporter=exporter, config=config)
        
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        assert len(exporter.logs) == 5
        assert exporter.logs[0].level == LogLevel.DEBUG
        assert exporter.logs[4].level == LogLevel.CRITICAL
    
    def test_log_with_extra(self):
        """Test log with extra data."""
        exporter = InMemoryExporter()
        logger = Logger("test", exporter=exporter)
        
        logger.info("Search completed", query="test", results=10)
        
        assert exporter.logs[0].extra["query"] == "test"
        assert exporter.logs[0].extra["results"] == 10
    
    def test_log_with_exception(self):
        """Test log with exception."""
        exporter = InMemoryExporter()
        logger = Logger("test", exporter=exporter)
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            logger.error("Operation failed", exception=e)
        
        # The exception is stored as str(e), which is just "Test error"
        assert exporter.logs[0].exception is not None
        assert "Test error" in exporter.logs[0].exception
    
    def test_log_with_trace_context(self):
        """Test log with trace context."""
        exporter = InMemoryExporter()
        tracer = Tracer(exporter=exporter)
        logger = Logger("test", exporter=exporter, tracer=tracer)
        
        with tracer.start_span("operation") as span:
            logger.info("Inside span")
        
        assert exporter.logs[0].trace_id == span.context.trace_id
        assert exporter.logs[0].span_id == span.context.span_id
    
    def test_child_logger(self):
        """Test child logger creation."""
        logger = Logger("monjyu")
        child = logger.child("search")
        
        assert child.name == "monjyu.search"
    
    def test_log_level_filtering(self):
        """Test log level filtering."""
        exporter = InMemoryExporter()
        config = ObservabilityConfig(log_level=LogLevel.WARNING)
        logger = Logger("test", exporter=exporter, config=config)
        
        logger.debug("Debug")
        logger.info("Info")
        logger.warning("Warning")
        logger.error("Error")
        
        # Only WARNING and ERROR should be recorded
        assert len(exporter.logs) == 2


# ============================================================
# Observability Manager Tests
# ============================================================


class TestObservability:
    """Observability manager tests."""
    
    def test_create(self):
        """Test creation."""
        obs = Observability.create()
        
        assert obs.tracer is not None
        assert obs.metrics is not None
        assert obs.logger is not None
    
    def test_create_with_dict_config(self):
        """Test creation with dict config."""
        obs = Observability.create(config={"log_level": "debug"})
        
        assert obs.config.log_level == LogLevel.DEBUG
    
    def test_create_for_testing(self):
        """Test creation for testing."""
        obs, exporter = Observability.create_for_testing()
        
        obs.logger.info("Test")
        
        assert len(exporter.logs) == 1
    
    def test_get_status(self):
        """Test getting status."""
        obs = Observability.create()
        
        status = obs.get_status()
        
        assert "config" in status
        assert "metrics" in status
    
    def test_flush(self):
        """Test flushing all telemetry."""
        obs, exporter = Observability.create_for_testing()
        
        with obs.tracer.start_span("test"):
            pass
        obs.metrics.increment("counter")
        
        obs.flush()
        
        # Spans and metrics should be exported
        assert len(exporter.spans) > 0


# ============================================================
# Decorator Tests
# ============================================================


class TestDecorators:
    """Decorator tests."""
    
    def test_traced_decorator(self):
        """Test @traced decorator."""
        exporter = InMemoryExporter()
        obs = configure_observability(exporter=exporter)
        
        @traced("test.function")
        def my_function():
            return 42
        
        result = my_function()
        
        assert result == 42
        obs.flush()
        assert any(s.name == "test.function" for s in exporter.spans)
    
    def test_traced_with_error(self):
        """Test @traced decorator with error."""
        exporter = InMemoryExporter()
        obs = configure_observability(exporter=exporter)
        
        @traced("error.function")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        obs.flush()
        span = next(s for s in exporter.spans if s.name == "error.function")
        assert span.status == SpanStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_traced_async(self):
        """Test @traced decorator with async function."""
        exporter = InMemoryExporter()
        obs = configure_observability(exporter=exporter)
        
        @traced("async.function")
        async def async_function():
            await asyncio.sleep(0.01)
            return "done"
        
        result = await async_function()
        
        assert result == "done"
        obs.flush()
        assert any(s.name == "async.function" for s in exporter.spans)
    
    def test_timed_decorator(self):
        """Test @timed decorator."""
        exporter = InMemoryExporter()
        obs = configure_observability(exporter=exporter)
        
        @timed("test.duration")
        def slow_function():
            time.sleep(0.01)
            return "done"
        
        result = slow_function()
        
        assert result == "done"
        obs.flush()
        
        timer_metric = next(
            (m for m in exporter.metrics if "test.duration" in m.name),
            None
        )
        assert timer_metric is not None
        assert timer_metric.value > 0


# ============================================================
# Global Functions Tests
# ============================================================


class TestGlobalFunctions:
    """Global function tests."""
    
    def test_configure_observability(self):
        """Test global configuration."""
        obs = configure_observability(config={"log_level": "warning"})
        
        assert obs.config.log_level == LogLevel.WARNING
    
    def test_get_tracer(self):
        """Test getting global tracer."""
        tracer = get_tracer()
        
        assert isinstance(tracer, Tracer)
    
    def test_get_metrics(self):
        """Test getting global metrics."""
        metrics = get_metrics()
        
        assert isinstance(metrics, MetricsCollector)
    
    def test_get_logger(self):
        """Test getting logger."""
        logger = get_logger("test.module")
        
        assert "test.module" in logger.name


# ============================================================
# Integration Tests
# ============================================================


class TestObservabilityIntegration:
    """Integration tests."""
    
    def test_full_workflow(self):
        """Test complete observability workflow."""
        obs, exporter = Observability.create_for_testing()
        
        # Start a traced operation
        with obs.tracer.start_span("search", {"query": "test query"}) as span:
            obs.logger.info("Search started", query="test query")
            
            # Record metrics
            obs.metrics.increment("search.requests")
            
            # Simulate some work
            with obs.metrics.measure_time("search.latency"):
                time.sleep(0.01)
            
            # Nested span
            with obs.tracer.start_span("retrieve_documents") as child:
                child.set_attribute("doc_count", 10)
                obs.logger.debug("Retrieved documents", count=10)
            
            obs.logger.info("Search completed")
        
        obs.flush()
        
        # Verify spans
        assert len(exporter.spans) == 2
        search_span = next(s for s in exporter.spans if s.name == "search")
        assert search_span.attributes["query"] == "test query"
        
        # Verify logs have trace context
        assert all(log.trace_id == search_span.context.trace_id for log in exporter.logs)
        
        # Verify metrics
        assert any("search.requests" in m.name for m in exporter.metrics)
        assert any("search.latency" in m.name for m in exporter.metrics)
    
    def test_error_propagation(self):
        """Test error propagation through observability."""
        obs, exporter = Observability.create_for_testing()
        
        with pytest.raises(RuntimeError):
            with obs.tracer.start_span("failing_operation") as span:
                obs.logger.info("Starting operation")
                raise RuntimeError("Intentional failure")
        
        obs.flush()
        
        # Span should have error status
        assert exporter.spans[0].status == SpanStatus.ERROR
        assert "Intentional failure" in exporter.spans[0].error
