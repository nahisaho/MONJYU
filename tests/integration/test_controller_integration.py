"""Integration tests for Controller components.

Tests the interaction between Controllers, CacheManager, StreamingService,
Observability, and Error Handling.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pytest

# Controllers
from monjyu.controller.unified import (
    UnifiedController,
    UnifiedControllerConfig,
)
from monjyu.controller.budget import (
    BudgetController,
    BudgetConfig,
    CostBudget,
    TokenUsage,
)
from monjyu.controller.progressive import (
    ProgressiveController,
    ProgressiveControllerConfig,
)
from monjyu.controller.hybrid import (
    HybridController,
    HybridControllerConfig,
    MergeStrategy,
)

# Storage - Use MemoryCache directly for simpler testing
from monjyu.storage.cache import (
    MemoryCache,
    CacheStats,
)

# Observability
from monjyu.observability import (
    InMemoryExporter,
    Observability,
    ObservabilityConfig,
    SpanStatus,
    traced,
)

# Error Handling
from monjyu.errors import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ErrorHandler,
    LLMError,
    QueryError,
    retry,
)


# ============================================================
# Test-only Data Classes
# ============================================================


class SearchMode(str, Enum):
    """Search mode for tests."""
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"


class ControllerStatus(str, Enum):
    """Controller status for tests."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


@dataclass
class ControllerResult:
    """Controller result for tests."""
    answer: str
    sources: list[str]
    mode: SearchMode
    status: ControllerStatus
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "mode": self.mode.value,
            "status": self.status.value,
        }


@dataclass
class MockSearchResult:
    """Mock search result."""
    query: str
    answer: str
    sources: list[str]
    score: float = 0.85


# ============================================================
# Mock Components
# ============================================================


class MockSearchEngine:
    """Mock search engine for testing."""
    
    def __init__(self, delay: float = 0.01, fail_count: int = 0):
        self.delay = delay
        self.fail_count = fail_count
        self.call_count = 0
        self.queries: list[str] = []
    
    async def search(
        self,
        query: str,
        mode: str = "local",
    ) -> MockSearchResult:
        """Execute mock search."""
        self.call_count += 1
        self.queries.append(query)
        
        if self.fail_count > 0 and self.call_count <= self.fail_count:
            raise LLMError(f"Search failed (attempt {self.call_count})")
        
        await asyncio.sleep(self.delay)
        
        return MockSearchResult(
            query=query,
            answer=f"Answer for: {query}",
            sources=["source1.pdf", "source2.pdf"],
            score=0.85 + (self.call_count * 0.01),
        )


class MockStreamingSource:
    """Mock streaming source for testing."""
    
    def __init__(self, chunks: list[str] | None = None):
        self.chunks = chunks or ["Hello ", "World ", "!"]
    
    async def stream(self):
        """Stream mock chunks."""
        for chunk in self.chunks:
            yield chunk
            await asyncio.sleep(0.01)


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def memory_cache():
    """Create a MemoryCache for testing (sync API)."""
    return MemoryCache(max_size=100, default_ttl=60.0)


@pytest.fixture
def observability():
    """Create Observability with InMemoryExporter."""
    exporter = InMemoryExporter()
    obs = Observability(exporter=exporter)
    return obs, exporter


@pytest.fixture
def circuit_breaker():
    """Create a CircuitBreaker for testing."""
    return CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=1.0,
        half_open_max_calls=1,
    )


@pytest.fixture
def error_handler():
    """Create an ErrorHandler for testing."""
    return ErrorHandler()


@pytest.fixture
def budget_controller():
    """Create a BudgetController for testing."""
    config = BudgetConfig(
        default_budget=CostBudget.STANDARD,
        track_history=True,
    )
    return BudgetController(config=config)


# ============================================================
# Controller + CacheManager Integration Tests
# ============================================================


class TestControllerCacheIntegration:
    """Controller + CacheManager integration tests."""
    
    @pytest.mark.asyncio
    async def test_cache_search_results(self, memory_cache):
        """Test caching search results."""
        # First search - cache miss
        query = "What is GraphRAG?"
        cache_key = f"search:{query}:local"
        
        cached = await memory_cache.get(cache_key)
        assert cached is None
        
        # Simulate search and cache result
        result = ControllerResult(
            answer="GraphRAG is...",
            sources=["paper1.pdf"],
            mode=SearchMode.LOCAL,
            status=ControllerStatus.SUCCESS,
        )
        await memory_cache.set(cache_key, result.to_dict())
        
        # Second lookup - cache hit
        cached = await memory_cache.get(cache_key)
        assert cached is not None
        assert cached["answer"] == "GraphRAG is..."
        assert cached["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_cache_with_ttl(self, memory_cache):
        """Test cache with TTL expiration."""
        key = "ttl_test"
        value = {"data": "test"}
        
        # Set with short TTL
        await memory_cache.set(key, value, ttl=0.1)
        
        # Should be present immediately
        cached = await memory_cache.get(key)
        assert cached is not None
        
        # Wait for expiration
        await asyncio.sleep(0.15)
        
        # Should be expired
        cached = await memory_cache.get(key)
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        small_cache = MemoryCache(max_size=3, default_ttl=60.0)
        
        # Fill cache
        await small_cache.set("key1", "value1")
        await small_cache.set("key2", "value2")
        await small_cache.set("key3", "value3")
        
        # Access key1 to make it recently used
        await small_cache.get("key1")
        
        # Add key4 - should evict key2 (least recently used)
        await small_cache.set("key4", "value4")
        
        # key2 should be evicted
        assert await small_cache.get("key2") is None
        # key1, key3, key4 should still exist
        assert await small_cache.get("key1") == "value1"
        assert await small_cache.get("key3") == "value3"
        assert await small_cache.get("key4") == "value4"
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, memory_cache):
        """Test cache statistics tracking."""
        # Generate some operations
        await memory_cache.set("key1", "value1")
        await memory_cache.get("key1")  # hit
        await memory_cache.get("nonexistent")  # miss
        
        stats = memory_cache.get_stats()
        assert stats.hits >= 1
        assert stats.misses >= 1


# ============================================================
# Controller + StreamingService Integration Tests
# ============================================================


class TestControllerStreamingIntegration:
    """Controller + StreamingService integration tests."""
    
    @pytest.mark.asyncio
    async def test_streaming_search_response(self):
        """Test streaming search response."""
        chunks_received: list[str] = []
        
        async def callback(chunk: str):
            chunks_received.append(chunk)
        
        source = MockStreamingSource(["Result: ", "GraphRAG ", "is great!"])
        
        async for chunk in source.stream():
            await callback(chunk)
        
        assert len(chunks_received) == 3
        assert "".join(chunks_received) == "Result: GraphRAG is great!"
    
    @pytest.mark.asyncio
    async def test_streaming_with_error_recovery(self, error_handler):
        """Test streaming with error recovery."""
        error_count = 0
        chunks_received: list[str] = []
        
        async def error_prone_stream():
            nonlocal error_count
            for i, chunk in enumerate(["A", "B", "C"]):
                if i == 1 and error_count == 0:
                    error_count += 1
                    raise LLMError("Temporary error")
                yield chunk
                await asyncio.sleep(0.01)
        
        # First attempt - will fail
        with pytest.raises(LLMError):
            async for chunk in error_prone_stream():
                chunks_received.append(chunk)
        
        # Second attempt - should succeed
        chunks_received.clear()
        async for chunk in error_prone_stream():
            chunks_received.append(chunk)
        
        assert chunks_received == ["A", "B", "C"]
    
    @pytest.mark.asyncio
    async def test_streaming_timeout(self):
        """Test streaming with timeout."""
        async def slow_stream():
            yield "start"
            await asyncio.sleep(10)  # Very slow
            yield "end"
        
        chunks_received = []
        
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(0.1):
                async for chunk in slow_stream():
                    chunks_received.append(chunk)
        
        assert len(chunks_received) == 1
        assert chunks_received[0] == "start"


# ============================================================
# Controller + Observability Integration Tests
# ============================================================


class TestControllerObservabilityIntegration:
    """Controller + Observability integration tests."""
    
    @pytest.mark.asyncio
    async def test_traced_search_operation(self, observability):
        """Test tracing of search operation."""
        obs, exporter = observability
        
        with obs.tracer.start_span("test.search") as span:
            span.set_attribute("query", "test query")
            span.set_attribute("mode", "local")
            
            # Simulate search
            await asyncio.sleep(0.01)
            
            span.set_status(SpanStatus.OK)
        
        obs.tracer.flush()
        
        # Verify span was recorded
        assert len(exporter.spans) >= 1
        search_span = next(
            (s for s in exporter.spans if s.name == "test.search"),
            None
        )
        assert search_span is not None
        assert search_span.attributes["query"] == "test query"
        assert search_span.status == SpanStatus.OK
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, observability):
        """Test metrics collection during search."""
        obs, exporter = observability
        
        # Record some metrics
        obs.metrics.increment("search.requests")
        obs.metrics.increment("search.requests")
        obs.metrics.gauge("search.active_queries", 5)
        obs.metrics.timer("search.latency_ms", 150.5)
        
        obs.metrics.flush()
        
        # Verify metrics
        assert obs.metrics.get_counter("search.requests") == 2
        assert obs.metrics.get_gauge("search.active_queries") == 5
    
    @pytest.mark.asyncio
    async def test_error_span_recording(self, observability):
        """Test error recording in spans."""
        obs, exporter = observability
        
        with pytest.raises(QueryError):
            with obs.tracer.start_span("test.error") as span:
                span.set_attribute("query", "bad query")
                raise QueryError("Invalid query")
        
        obs.tracer.flush()
        
        # Verify error was recorded
        error_span = next(
            (s for s in exporter.spans if s.name == "test.error"),
            None
        )
        assert error_span is not None
        assert error_span.status == SpanStatus.ERROR
        assert "Invalid query" in (error_span.error or "")
    
    @pytest.mark.asyncio
    async def test_nested_spans(self, observability):
        """Test nested span hierarchy."""
        obs, exporter = observability
        
        with obs.tracer.start_span("parent") as parent:
            parent.set_attribute("level", "1")
            
            with obs.tracer.start_span("child") as child:
                child.set_attribute("level", "2")
                
                with obs.tracer.start_span("grandchild") as grandchild:
                    grandchild.set_attribute("level", "3")
        
        obs.tracer.flush()
        
        # Verify hierarchy
        assert len(exporter.spans) == 3
        
        parent_span = next(s for s in exporter.spans if s.name == "parent")
        child_span = next(s for s in exporter.spans if s.name == "child")
        grandchild_span = next(s for s in exporter.spans if s.name == "grandchild")
        
        assert child_span.context.parent_span_id == parent_span.context.span_id
        assert grandchild_span.context.parent_span_id == child_span.context.span_id


# ============================================================
# Controller + ErrorHandling Integration Tests
# ============================================================


class TestControllerErrorHandlingIntegration:
    """Controller + Error Handling integration tests."""
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """Test retry decorator on transient errors."""
        engine = MockSearchEngine(fail_count=2)
        
        @retry(
            max_attempts=5,
            exceptions=(LLMError,),
            delay=0.01,
            backoff=1.0,
        )
        async def search_with_retry(query: str):
            return await engine.search(query)
        
        result = await search_with_retry("test query")
        
        # Should succeed on third attempt
        assert result.answer == "Answer for: test query"
        assert engine.call_count == 3
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self, circuit_breaker):
        """Test circuit breaker opens after failures."""
        failure_count = 0
        
        async def failing_operation():
            nonlocal failure_count
            failure_count += 1
            raise LLMError("Always fails")
        
        # Trigger failures up to threshold
        for _ in range(3):
            with pytest.raises(LLMError):
                async with circuit_breaker:
                    await failing_operation()
        
        # Circuit should be open
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Further calls should fail fast
        with pytest.raises(CircuitOpenError):
            async with circuit_breaker:
                await failing_operation()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery through half-open state."""
        call_count = 0
        should_succeed = False
        
        async def conditional_operation():
            nonlocal call_count
            call_count += 1
            if not should_succeed:
                raise LLMError("Failing")
            return "success"
        
        # Open the circuit
        for _ in range(3):
            with pytest.raises(LLMError):
                async with circuit_breaker:
                    await conditional_operation()
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Now make the operation succeed
        should_succeed = True
        
        async with circuit_breaker:
            result = await conditional_operation()
        
        assert result == "success"
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_error_handler_with_context(self, error_handler):
        """Test error handler with context capture."""
        context = {
            "query": "test query",
            "mode": "local",
            "user_id": "test_user",
        }
        
        try:
            raise QueryError("Query failed", context=context)
        except QueryError as e:
            # context is now an ErrorContext object
            # The original dict is stored in context.details['context']
            assert e.context is not None
            assert e.context.details.get("context") == context


# ============================================================
# Full Integration Tests
# ============================================================


class TestFullIntegration:
    """Full integration tests combining all components."""
    
    @pytest.mark.asyncio
    async def test_search_with_all_components(
        self,
        memory_cache,
        observability,
        circuit_breaker,
    ):
        """Test search flow with cache, observability, and circuit breaker."""
        obs, exporter = observability
        engine = MockSearchEngine()
        
        async def search_with_observability(query: str) -> MockSearchResult:
            with obs.tracer.start_span("search") as span:
                span.set_attribute("query", query)
                
                # Check cache first
                cache_key = f"search:{query}"
                cached = await memory_cache.get(cache_key)
                
                if cached:
                    span.set_attribute("cache_hit", True)
                    obs.metrics.increment("cache.hits")
                    return MockSearchResult(**cached)
                
                span.set_attribute("cache_hit", False)
                obs.metrics.increment("cache.misses")
                
                # Execute search through circuit breaker
                async with circuit_breaker:
                    result = await engine.search(query)
                
                # Cache result
                await memory_cache.set(cache_key, {
                    "query": result.query,
                    "answer": result.answer,
                    "sources": result.sources,
                    "score": result.score,
                })
                
                obs.metrics.increment("search.executed")
                return result
        
        # First search - cache miss
        result1 = await search_with_observability("What is RAG?")
        assert result1.answer == "Answer for: What is RAG?"
        
        # Second search - cache hit
        result2 = await search_with_observability("What is RAG?")
        assert result2.answer == "Answer for: What is RAG?"
        
        # Verify observability
        obs.tracer.flush()
        obs.metrics.flush()
        
        assert len(exporter.spans) == 2
        assert obs.metrics.get_counter("cache.hits") == 1
        assert obs.metrics.get_counter("cache.misses") == 1
        assert obs.metrics.get_counter("search.executed") == 1
    
    @pytest.mark.asyncio
    async def test_streaming_with_observability_and_errors(self, observability):
        """Test streaming with observability and error handling."""
        obs, exporter = observability
        chunks_collected: list[str] = []
        
        async def stream_with_tracing():
            with obs.tracer.start_span("stream") as span:
                span.set_attribute("type", "search_result")
                
                source = MockStreamingSource(["Part 1", " Part 2", " Part 3"])
                
                async for chunk in source.stream():
                    chunks_collected.append(chunk)
                    obs.metrics.increment("stream.chunks")
                
                obs.metrics.gauge("stream.total_chunks", len(chunks_collected))
        
        await stream_with_tracing()
        
        obs.tracer.flush()
        
        assert len(chunks_collected) == 3
        assert obs.metrics.get_counter("stream.chunks") == 3
        assert obs.metrics.get_gauge("stream.total_chunks") == 3
    
    @pytest.mark.asyncio
    async def test_budget_with_cache_and_observability(
        self,
        budget_controller,
        memory_cache,
        observability,
    ):
        """Test budget tracking with caching and observability."""
        obs, exporter = observability
        
        with obs.tracer.start_span("budget_search") as span:
            # Simulate a query with budget tracking
            span.set_attribute("budget", budget_controller.config.default_budget.value)
            
            # Check cache to save budget
            cache_key = "budget:expensive_query"
            cached = await memory_cache.get(cache_key)
            
            if not cached:
                # Record token usage
                usage = TokenUsage(
                    prompt_tokens=500,
                    completion_tokens=200,
                    total_tokens=700,
                    operation="query",
                )
                budget_controller.record_usage(usage)
                
                # Cache the result
                await memory_cache.set(cache_key, {
                    "answer": "Expensive result",
                    "tokens_used": 700,
                })
                
                obs.metrics.increment("budget.queries_executed")
            else:
                obs.metrics.increment("budget.cache_hits")
        
        # Second query should hit cache
        with obs.tracer.start_span("budget_search") as span:
            cached = await memory_cache.get(cache_key)
            assert cached is not None
            obs.metrics.increment("budget.cache_hits")
        
        obs.metrics.flush()
        
        assert obs.metrics.get_counter("budget.queries_executed") == 1
        assert obs.metrics.get_counter("budget.cache_hits") == 1
    
    @pytest.mark.asyncio
    async def test_progressive_levels_with_circuit_breaker(
        self,
        circuit_breaker,
        observability,
    ):
        """Test progressive levels with circuit breaker protection."""
        obs, exporter = observability
        
        level_results: list[dict] = []
        
        async def search_level(level: int) -> dict:
            with obs.tracer.start_span(f"level_{level}") as span:
                span.set_attribute("level", level)
                
                # Simulate level-specific search
                await asyncio.sleep(0.01 * level)
                
                return {
                    "level": level,
                    "result": f"Level {level} result",
                    "score": 0.5 + (level * 0.1),
                }
        
        # Search through levels progressively
        for level in range(3):
            async with circuit_breaker:
                result = await search_level(level)
            level_results.append(result)
            obs.metrics.increment("progressive.levels_searched")
        
        obs.tracer.flush()
        
        assert len(level_results) == 3
        assert level_results[2]["score"] > level_results[0]["score"]
        assert obs.metrics.get_counter("progressive.levels_searched") == 3
    
    @pytest.mark.asyncio
    async def test_hybrid_merge_with_all_components(
        self,
        memory_cache,
        observability,
    ):
        """Test hybrid merge with caching and observability."""
        obs, exporter = observability
        
        async def run_hybrid_search(query: str) -> list[dict]:
            with obs.tracer.start_span("hybrid_search") as span:
                span.set_attribute("query", query)
                
                # Check cache
                cache_key = f"hybrid:{query}"
                cached = await memory_cache.get(cache_key)
                
                if cached:
                    obs.metrics.increment("hybrid.cache_hits")
                    return cached
                
                results = []
                
                # Simulate multiple engine searches
                for engine_name in ["local", "global", "entity"]:
                    with obs.tracer.start_span(f"engine.{engine_name}"):
                        await asyncio.sleep(0.01)
                        results.append({
                            "engine": engine_name,
                            "answer": f"{engine_name} answer",
                            "score": 0.7 + len(engine_name) * 0.01,
                        })
                        obs.metrics.increment(f"hybrid.{engine_name}_searched")
                
                # Merge results
                merged = sorted(results, key=lambda x: x["score"], reverse=True)
                
                # Cache
                await memory_cache.set(cache_key, merged)
                
                return merged
        
        # First search
        result1 = await run_hybrid_search("test query")
        assert len(result1) == 3
        
        # Second search - should hit cache
        result2 = await run_hybrid_search("test query")
        assert result2 == result1
        
        obs.metrics.flush()
        
        assert obs.metrics.get_counter("hybrid.cache_hits") == 1


# ============================================================
# Edge Case Tests
# ============================================================


class TestEdgeCases:
    """Edge case and error boundary tests."""
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, memory_cache, observability):
        """Test handling of empty queries."""
        obs, exporter = observability
        
        async def handle_query(query: str):
            with obs.tracer.start_span("query") as span:
                span.set_attribute("query", query)
                
                if not query.strip():
                    span.set_status(SpanStatus.ERROR, "Empty query")
                    raise QueryError("Query cannot be empty")
                
                return {"result": f"Result for {query}"}
        
        with pytest.raises(QueryError):
            await handle_query("")
        
        obs.tracer.flush()
        
        error_span = exporter.spans[-1]
        assert error_span.status == SpanStatus.ERROR
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_access(self, memory_cache):
        """Test concurrent cache access."""
        async def write_cache(key: str, value: str):
            await memory_cache.set(key, value)
            await asyncio.sleep(0.01)
            return await memory_cache.get(key)
        
        # Run concurrent writes
        tasks = [
            write_cache(f"key_{i}", f"value_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # All writes should succeed
        for i, result in enumerate(results):
            assert result == f"value_{i}"
    
    @pytest.mark.asyncio
    async def test_observability_under_high_load(self, observability):
        """Test observability under high load."""
        obs, exporter = observability
        
        async def generate_spans(count: int):
            for i in range(count):
                with obs.tracer.start_span(f"span_{i}") as span:
                    span.set_attribute("index", i)
                    obs.metrics.increment("load.operations")
        
        # Generate many spans
        await generate_spans(100)
        
        obs.tracer.flush()
        obs.metrics.flush()
        
        assert obs.metrics.get_counter("load.operations") == 100
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_reset_after_success(self, circuit_breaker):
        """Test circuit breaker failure count resets after success."""
        call_count = 0
        
        async def intermittent_operation():
            nonlocal call_count
            call_count += 1
            
            # Fail twice, then succeed, then fail twice, etc.
            if call_count % 3 != 0:
                raise LLMError("Intermittent failure")
            return "success"
        
        # First two calls fail
        for _ in range(2):
            with pytest.raises(LLMError):
                async with circuit_breaker:
                    await intermittent_operation()
        
        # Third call succeeds - should reset failure count
        async with circuit_breaker:
            result = await intermittent_operation()
        assert result == "success"
        
        # Circuit should still be closed
        assert circuit_breaker.state == CircuitState.CLOSED
        
        # Two more failures should not open circuit
        # (failure count was reset)
        for _ in range(2):
            with pytest.raises(LLMError):
                async with circuit_breaker:
                    await intermittent_operation()
        
        # Sixth call succeeds
        async with circuit_breaker:
            result = await intermittent_operation()
        assert result == "success"
