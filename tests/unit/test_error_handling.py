"""Tests for Error Handling Framework.

Exception classes, Retry, Circuit Breaker tests.
"""

import asyncio
import time

import pytest

from monjyu.errors import (
    AuthenticationError,
    CacheError,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ConfigurationError,
    EmbeddingError,
    ErrorContext,
    ErrorHandler,
    ErrorHandlerConfig,
    ErrorSeverity,
    GraphError,
    IndexError,
    LLMError,
    MONJYUError,
    NetworkError,
    PDFProcessError,
    QueryError,
    RateLimitError,
    RetryConfig,
    StorageError,
    TimeoutError,
    ValidationError,
    circuit_breaker,
    create_error_handler,
    retry,
)


# ============================================================
# ErrorSeverity Tests
# ============================================================


class TestErrorSeverity:
    """ErrorSeverity tests."""
    
    def test_severity_values(self):
        """Test severity values."""
        assert ErrorSeverity.DEBUG.value == "debug"
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_to_logging_level(self):
        """Test conversion to logging level."""
        import logging
        
        assert ErrorSeverity.DEBUG.to_logging_level() == logging.DEBUG
        assert ErrorSeverity.ERROR.to_logging_level() == logging.ERROR


# ============================================================
# ErrorContext Tests
# ============================================================


class TestErrorContext:
    """ErrorContext tests."""
    
    def test_context_creation(self):
        """Test context creation."""
        ctx = ErrorContext()
        
        assert ctx.error_id.startswith("err_")
        assert ctx.timestamp is not None
    
    def test_context_with_details(self):
        """Test context with details."""
        ctx = ErrorContext(
            component="search",
            operation="query",
            details={"key": "value"},
            correlation_id="req-123",
        )
        
        assert ctx.component == "search"
        assert ctx.operation == "query"
        assert ctx.details["key"] == "value"
        assert ctx.correlation_id == "req-123"
    
    def test_to_dict(self):
        """Test context serialization."""
        ctx = ErrorContext(component="test")
        
        data = ctx.to_dict()
        
        assert "error_id" in data
        assert data["component"] == "test"


# ============================================================
# MONJYUError Tests
# ============================================================


class TestMONJYUError:
    """MONJYUError tests."""
    
    def test_basic_error(self):
        """Test basic error creation."""
        error = MONJYUError("Something went wrong")
        
        assert error.message == "Something went wrong"
        assert error.code == "MONJYU_ERROR"
        assert error.severity == ErrorSeverity.ERROR
    
    def test_error_with_code(self):
        """Test error with custom code."""
        error = MONJYUError("Error", code="CUSTOM_001")
        
        assert error.code == "CUSTOM_001"
    
    def test_error_with_context(self):
        """Test error with context."""
        error = MONJYUError(
            "Error",
            component="index",
            operation="build",
            key="value",
        )
        
        assert error.context.component == "index"
        assert error.context.operation == "build"
        assert error.context.details["key"] == "value"
    
    def test_error_with_cause(self):
        """Test error with cause."""
        cause = ValueError("Original error")
        error = MONJYUError("Wrapped error", cause=cause)
        
        assert error.cause is cause
        assert "Original error" in str(error)
    
    def test_error_str(self):
        """Test error string representation."""
        error = MONJYUError(
            "Error message",
            code="ERR001",
            component="search",
        )
        
        error_str = str(error)
        
        assert "[ERR001]" in error_str
        assert "Error message" in error_str
        assert "search" in error_str
    
    def test_error_repr(self):
        """Test error repr."""
        error = MONJYUError("Test")
        
        repr_str = repr(error)
        
        assert "MONJYUError" in repr_str
        assert "Test" in repr_str
    
    def test_to_dict(self):
        """Test error serialization."""
        error = MONJYUError("Test error", code="TEST001")
        
        data = error.to_dict()
        
        assert data["error_type"] == "MONJYUError"
        assert data["message"] == "Test error"
        assert data["code"] == "TEST001"
        assert "context" in data
    
    def test_with_context(self):
        """Test adding context."""
        error = MONJYUError("Error")
        error.with_context(extra="data", count=42)
        
        assert error.context.details["extra"] == "data"
        assert error.context.details["count"] == 42
    
    def test_from_exception(self):
        """Test creating from exception."""
        original = ValueError("Original")
        error = MONJYUError.from_exception(original)
        
        assert error.cause is original
        assert "Original" in error.message


# ============================================================
# Specific Exception Tests
# ============================================================


class TestSpecificExceptions:
    """Specific exception tests."""
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        
        assert error.code == "CONFIG_ERROR"
    
    def test_index_error(self):
        """Test IndexError."""
        error = IndexError("Index failed")
        
        assert error.code == "INDEX_ERROR"
    
    def test_query_error(self):
        """Test QueryError with query."""
        error = QueryError("Search failed", query="test query")
        
        assert error.code == "QUERY_ERROR"
        assert error.context.details["query"] == "test query"
    
    def test_llm_error(self):
        """Test LLMError with model info."""
        error = LLMError(
            "API failed",
            model="gpt-4",
            provider="azure",
        )
        
        assert error.code == "LLM_ERROR"
        assert error.context.details["model"] == "gpt-4"
        assert error.context.details["provider"] == "azure"
    
    def test_storage_error(self):
        """Test StorageError with path."""
        error = StorageError("Write failed", path="/data/file.txt")
        
        assert error.code == "STORAGE_ERROR"
        assert error.context.details["path"] == "/data/file.txt"
    
    def test_pdf_process_error(self):
        """Test PDFProcessError."""
        error = PDFProcessError(
            "Parse failed",
            file_path="/docs/paper.pdf",
            page=10,
        )
        
        assert error.code == "PDF_ERROR"
        assert error.context.details["file_path"] == "/docs/paper.pdf"
        assert error.context.details["page"] == 10
    
    def test_embedding_error(self):
        """Test EmbeddingError."""
        error = EmbeddingError("Embedding failed")
        
        assert error.code == "EMBEDDING_ERROR"
    
    def test_graph_error(self):
        """Test GraphError."""
        error = GraphError("Graph construction failed")
        
        assert error.code == "GRAPH_ERROR"
    
    def test_cache_error(self):
        """Test CacheError."""
        error = CacheError("Cache miss")
        
        assert error.code == "CACHE_ERROR"
        assert error.severity == ErrorSeverity.WARNING
    
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid value",
            field="email",
            value="not-an-email",
        )
        
        assert error.code == "VALIDATION_ERROR"
        assert error.context.details["field"] == "email"
    
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Operation timed out", timeout_seconds=30.0)
        
        assert error.code == "TIMEOUT_ERROR"
        assert error.context.details["timeout_seconds"] == 30.0
    
    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limited", retry_after=60.0)
        
        assert error.code == "RATE_LIMIT_ERROR"
        assert error.context.details["retry_after"] == 60.0
    
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Invalid credentials")
        
        assert error.code == "AUTH_ERROR"
    
    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Connection refused")
        
        assert error.code == "NETWORK_ERROR"


# ============================================================
# RetryConfig Tests
# ============================================================


class TestRetryConfig:
    """RetryConfig tests."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.delay == 1.0
        assert config.backoff == 2.0
    
    def test_calculate_delay(self):
        """Test delay calculation."""
        config = RetryConfig(delay=1.0, backoff=2.0, max_delay=60.0)
        
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
    
    def test_max_delay_cap(self):
        """Test max delay cap."""
        config = RetryConfig(delay=10.0, backoff=2.0, max_delay=30.0)
        
        # delay * 2^4 = 160, but capped at 30
        assert config.calculate_delay(5) == 30.0


# ============================================================
# Retry Decorator Tests
# ============================================================


class TestRetryDecorator:
    """Retry decorator tests."""
    
    def test_retry_success(self):
        """Test successful execution."""
        call_count = 0
        
        @retry(max_attempts=3)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeed()
        
        assert result == "success"
        assert call_count == 1
    
    def test_retry_on_failure(self):
        """Test retry on failure."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def fail_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = fail_twice()
        
        assert result == "success"
        assert call_count == 3
    
    def test_retry_exhausted(self):
        """Test retry exhaustion."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError):
            always_fail()
        
        assert call_count == 3
    
    def test_retry_specific_exceptions(self):
        """Test retry for specific exceptions."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01, exceptions=(ValueError,))
        def fail_with_type():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable")
            raise TypeError("Not retryable")
        
        with pytest.raises(TypeError):
            fail_with_type()
        
        assert call_count == 2
    
    def test_retry_callback(self):
        """Test retry callback."""
        retry_info = []
        
        def on_retry(attempt, exc, wait_time):
            retry_info.append((attempt, str(exc), wait_time))
        
        @retry(max_attempts=3, delay=0.01, on_retry=on_retry)
        def fail_once():
            if len(retry_info) == 0:
                raise ValueError("First failure")
            return "success"
        
        fail_once()
        
        assert len(retry_info) == 1
        assert retry_info[0][0] == 1
    
    @pytest.mark.asyncio
    async def test_retry_async(self):
        """Test async retry."""
        call_count = 0
        
        @retry(max_attempts=3, delay=0.01)
        async def async_fail_once():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temp")
            return "done"
        
        result = await async_fail_once()
        
        assert result == "done"
        assert call_count == 2


# ============================================================
# Circuit Breaker Tests
# ============================================================


class TestCircuitBreaker:
    """CircuitBreaker tests."""
    
    def test_initial_state(self):
        """Test initial state."""
        breaker = CircuitBreaker()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    def test_success_keeps_closed(self):
        """Test success keeps circuit closed."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        with breaker:
            pass  # Success
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    def test_failure_increments_count(self):
        """Test failure increments count."""
        breaker = CircuitBreaker(failure_threshold=5)
        
        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("Failure")
        
        assert breaker.failure_count == 1
        assert breaker.state == CircuitState.CLOSED
    
    def test_threshold_opens_circuit(self):
        """Test threshold opens circuit."""
        breaker = CircuitBreaker(failure_threshold=3)
        
        for _ in range(3):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3
    
    def test_open_rejects_requests(self):
        """Test open circuit rejects requests."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=10.0)
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        # Next request should be rejected
        with pytest.raises(CircuitOpenError):
            with breaker:
                pass
    
    def test_recovery_to_half_open(self):
        """Test recovery to half-open state."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.01,
        )
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        assert breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.02)
        
        # This should transition to HALF_OPEN and allow
        with breaker:
            pass
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_half_open_failure_reopens(self):
        """Test half-open failure reopens circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.01,
        )
        
        # Open the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        time.sleep(0.02)
        
        # Fail in half-open
        with pytest.raises(ValueError):
            with breaker:
                raise ValueError("Still failing")
        
        assert breaker.state == CircuitState.OPEN
    
    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        breaker.reset()
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
    
    def test_state_change_callback(self):
        """Test state change callback."""
        changes = []
        
        def on_change(old, new):
            changes.append((old.value, new.value))
        
        breaker = CircuitBreaker(
            failure_threshold=2,
            on_state_change=on_change,
        )
        
        for _ in range(2):
            with pytest.raises(ValueError):
                with breaker:
                    raise ValueError("Failure")
        
        assert len(changes) == 1
        assert changes[0] == ("closed", "open")
    
    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self):
        """Test async circuit breaker."""
        breaker = CircuitBreaker(failure_threshold=2)
        
        async with breaker:
            await asyncio.sleep(0.01)
        
        assert breaker.state == CircuitState.CLOSED
    
    def test_get_status(self):
        """Test get status."""
        breaker = CircuitBreaker()
        
        status = breaker.get_status()
        
        assert status["state"] == "closed"
        assert status["failure_count"] == 0


# ============================================================
# Circuit Breaker Decorator Tests
# ============================================================


class TestCircuitBreakerDecorator:
    """Circuit breaker decorator tests."""
    
    def test_decorator_success(self):
        """Test decorator on success."""
        @circuit_breaker(failure_threshold=3)
        def succeed():
            return "ok"
        
        result = succeed()
        
        assert result == "ok"
        assert succeed.circuit_breaker.state == CircuitState.CLOSED
    
    def test_decorator_failure(self):
        """Test decorator on failure."""
        @circuit_breaker(failure_threshold=2)
        def fail():
            raise ValueError("Fail")
        
        for _ in range(2):
            with pytest.raises(ValueError):
                fail()
        
        assert fail.circuit_breaker.state == CircuitState.OPEN
    
    @pytest.mark.asyncio
    async def test_decorator_async(self):
        """Test async decorator."""
        @circuit_breaker(failure_threshold=2)
        async def async_succeed():
            await asyncio.sleep(0.01)
            return "async ok"
        
        result = await async_succeed()
        
        assert result == "async ok"


# ============================================================
# Error Handler Tests
# ============================================================


class TestErrorHandler:
    """ErrorHandler tests."""
    
    def test_handler_creation(self):
        """Test handler creation."""
        handler = ErrorHandler()
        
        assert handler.config.log_errors is True
    
    def test_handle_exception(self):
        """Test handling exception."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Original")
        except Exception as e:
            with pytest.raises(MONJYUError):
                handler.handle(e, component="test")
    
    def test_handle_without_reraise(self):
        """Test handling without reraise."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Original")
        except Exception as e:
            error = handler.handle(e, reraise=False)
        
        assert isinstance(error, MONJYUError)
        assert "Original" in error.message
    
    def test_handle_monjyu_error(self):
        """Test handling MONJYUError."""
        handler = ErrorHandler()
        
        original = QueryError("Query failed", query="test")
        
        with pytest.raises(QueryError):
            handler.handle(original, component="search")
    
    def test_get_stats(self):
        """Test getting stats."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Test")
        except Exception as e:
            handler.handle(e, reraise=False)
        
        stats = handler.get_stats()
        
        assert stats["total_errors"] == 1
        assert "MONJYUError" in stats["error_counts"]
    
    def test_get_recent_errors(self):
        """Test getting recent errors."""
        handler = ErrorHandler()
        
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except Exception as e:
                handler.handle(e, reraise=False)
        
        recent = handler.get_recent_errors(limit=2)
        
        assert len(recent) == 2
    
    def test_clear_stats(self):
        """Test clearing stats."""
        handler = ErrorHandler()
        
        try:
            raise ValueError("Test")
        except Exception as e:
            handler.handle(e, reraise=False)
        
        handler.clear_stats()
        
        assert handler.get_stats()["total_errors"] == 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ErrorHandlerConfig(
            log_errors=False,
            include_stack_trace=False,
        )
        handler = ErrorHandler(config=config)
        
        assert handler.config.log_errors is False


# ============================================================
# Global Functions Tests
# ============================================================


class TestGlobalFunctions:
    """Global function tests."""
    
    def test_create_error_handler(self):
        """Test creating global handler."""
        handler = create_error_handler()
        
        assert isinstance(handler, ErrorHandler)


# ============================================================
# Integration Tests
# ============================================================


class TestErrorHandlingIntegration:
    """Integration tests."""
    
    def test_full_error_workflow(self):
        """Test complete error workflow."""
        handler = ErrorHandler()
        
        # Create a specific error
        error = LLMError(
            "API call failed",
            model="gpt-4",
            provider="azure",
            component="query",
            operation="generate_answer",
        )
        
        # Handle it
        with pytest.raises(LLMError) as exc_info:
            handler.handle(error)
        
        caught = exc_info.value
        
        assert caught.code == "LLM_ERROR"
        assert caught.context.details["model"] == "gpt-4"
        assert caught.context.component == "query"
    
    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry with circuit breaker."""
        call_count = 0
        
        @circuit_breaker(failure_threshold=5)
        @retry(max_attempts=3, delay=0.01)
        async def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = await flaky_operation()
        
        assert result == "success"
        assert call_count == 3
    
    def test_exception_chaining(self):
        """Test exception chaining."""
        def inner():
            raise ValueError("Inner error")
        
        def middle():
            try:
                inner()
            except ValueError as e:
                raise LLMError("LLM failed", cause=e, model="gpt-4")
        
        def outer():
            try:
                middle()
            except LLMError as e:
                raise QueryError("Query failed", cause=e, query="test")
        
        with pytest.raises(QueryError) as exc_info:
            outer()
        
        error = exc_info.value
        assert error.cause is not None
        assert isinstance(error.cause, LLMError)
