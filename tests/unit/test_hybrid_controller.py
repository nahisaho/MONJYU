"""Tests for HybridController.

REQ-ARC-003: Hybrid GraphRAG Controller
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from monjyu.controller.hybrid import (
    AllEnginesFailedError,
    EngineResult,
    ExecutionMode,
    HybridController,
    HybridControllerConfig,
    HybridResultItem,
    HybridSearchContext,
    HybridSearchEngineProtocol,
    HybridSearchResult,
    MergeStrategy,
    NoEnginesRegisteredError,
    create_hybrid_controller,
)


# ============================================================
# Mock Classes
# ============================================================


class MockSearchEngine:
    """Mock search engine for testing."""
    
    def __init__(
        self,
        name: str,
        results: Optional[List[HybridResultItem]] = None,
        available: bool = True,
        delay: float = 0.0,
        should_fail: bool = False,
        fail_error: Optional[str] = None,
    ):
        self._name = name
        self._results = results or []
        self._available = available
        self._delay = delay
        self._should_fail = should_fail
        self._fail_error = fail_error or "Mock engine failure"
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> EngineResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        
        if self._should_fail:
            raise RuntimeError(self._fail_error)
        
        return EngineResult(
            engine_name=self._name,
            items=self._results[:max_results],
            total_count=len(self._results),
            success=True,
        )
    
    def is_available(self) -> bool:
        return self._available
    
    @property
    def engine_name(self) -> str:
        return self._name


def create_mock_results(
    prefix: str,
    count: int = 5,
    base_score: float = 1.0,
) -> List[HybridResultItem]:
    """Create mock search results."""
    return [
        HybridResultItem(
            content=f"{prefix} result {i}",
            score=base_score - (i * 0.1),
            source=f"source_{prefix}_{i}.txt",
            metadata={"index": i},
        )
        for i in range(count)
    ]


# ============================================================
# Test Classes
# ============================================================


class TestHybridControllerBasics:
    """Basic HybridController tests."""
    
    def test_create_controller(self):
        """Test controller creation."""
        controller = HybridController()
        assert controller is not None
        assert controller.config is not None
    
    def test_create_with_config(self):
        """Test controller creation with config."""
        config = HybridControllerConfig(
            default_merge_strategy=MergeStrategy.SCORE,
            rrf_k=30,
        )
        controller = HybridController(config=config)
        assert controller.config.default_merge_strategy == MergeStrategy.SCORE
        assert controller.config.rrf_k == 30
    
    def test_register_engine(self):
        """Test engine registration."""
        controller = HybridController()
        engine = MockSearchEngine("test")
        
        controller.register_engine("test", engine)
        
        assert controller.has_engine("test")
        assert "test" in controller.get_registered_engines()
    
    def test_unregister_engine(self):
        """Test engine unregistration."""
        controller = HybridController()
        engine = MockSearchEngine("test")
        
        controller.register_engine("test", engine)
        controller.unregister_engine("test")
        
        assert not controller.has_engine("test")
    
    def test_get_available_engines(self):
        """Test getting available engines."""
        controller = HybridController()
        
        available = MockSearchEngine("available", available=True)
        unavailable = MockSearchEngine("unavailable", available=False)
        
        controller.register_engine("available", available)
        controller.register_engine("unavailable", unavailable)
        
        engines = controller.get_available_engines()
        assert "available" in engines
        assert "unavailable" not in engines


class TestHybridControllerSearch:
    """HybridController search tests."""
    
    @pytest.mark.asyncio
    async def test_search_basic(self):
        """Test basic search."""
        controller = HybridController()
        
        engine = MockSearchEngine(
            "test",
            results=create_mock_results("test", 3),
        )
        controller.register_engine("test", engine)
        
        result = await controller.search("test query")
        
        assert isinstance(result, HybridSearchResult)
        assert len(result.items) > 0
        assert "test" in result.engines_used
    
    @pytest.mark.asyncio
    async def test_search_multiple_engines(self):
        """Test search with multiple engines."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("vector", results=create_mock_results("vector", 3))
        engine2 = MockSearchEngine("graph", results=create_mock_results("graph", 3))
        engine3 = MockSearchEngine("lazy", results=create_mock_results("lazy", 3))
        
        controller.register_engine("vector", engine1)
        controller.register_engine("graph", engine2)
        controller.register_engine("lazy", engine3)
        
        result = await controller.search("test query")
        
        assert len(result.engines_used) == 3
        assert result.merge_strategy == MergeStrategy.RRF
    
    @pytest.mark.asyncio
    async def test_search_specific_engines(self):
        """Test search with specific engines."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("vector", results=create_mock_results("vector", 3))
        engine2 = MockSearchEngine("graph", results=create_mock_results("graph", 3))
        
        controller.register_engine("vector", engine1)
        controller.register_engine("graph", engine2)
        
        result = await controller.search("test query", engines=["vector"])
        
        assert result.engines_used == ["vector"]
    
    @pytest.mark.asyncio
    async def test_search_no_engines_error(self):
        """Test search with no engines raises error."""
        controller = HybridController()
        
        with pytest.raises(NoEnginesRegisteredError):
            await controller.search("test query")
    
    @pytest.mark.asyncio
    async def test_search_all_engines_failed(self):
        """Test search when all engines fail."""
        controller = HybridController()
        
        engine = MockSearchEngine("failing", should_fail=True)
        controller.register_engine("failing", engine)
        
        with pytest.raises(AllEnginesFailedError):
            await controller.search("test query")
    
    @pytest.mark.asyncio
    async def test_search_partial_failure(self):
        """Test search with partial engine failure."""
        controller = HybridController()
        
        good_engine = MockSearchEngine("good", results=create_mock_results("good", 3))
        bad_engine = MockSearchEngine("bad", should_fail=True)
        
        controller.register_engine("good", good_engine)
        controller.register_engine("bad", bad_engine)
        
        result = await controller.search("test query")
        
        assert len(result.successful_engines) == 1
        assert len(result.failed_engines) == 1
        assert "good" in result.successful_engines
        assert "bad" in result.failed_engines


class TestHybridControllerMergeStrategies:
    """Merge strategy tests."""
    
    @pytest.mark.asyncio
    async def test_merge_by_rrf(self):
        """Test RRF merge strategy."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3, 0.9))
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3, 0.8))
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(merge_strategy=MergeStrategy.RRF)
        result = await controller.search("query", context=context)
        
        assert result.merge_strategy == MergeStrategy.RRF
        assert len(result.items) > 0
    
    @pytest.mark.asyncio
    async def test_merge_by_score(self):
        """Test score-based merge strategy."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3, 0.9))
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3, 0.5))
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(merge_strategy=MergeStrategy.SCORE)
        result = await controller.search("query", context=context)
        
        assert result.merge_strategy == MergeStrategy.SCORE
        # 最高スコアの結果が最初に来るはず
        if result.items:
            assert result.items[0].score >= result.items[-1].score
    
    @pytest.mark.asyncio
    async def test_merge_by_weighted(self):
        """Test weighted merge strategy."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3))
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3))
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(
            merge_strategy=MergeStrategy.WEIGHTED,
            engine_weights={"e1": 2.0, "e2": 1.0},
        )
        result = await controller.search("query", context=context)
        
        assert result.merge_strategy == MergeStrategy.WEIGHTED
    
    @pytest.mark.asyncio
    async def test_merge_by_interleave(self):
        """Test interleave merge strategy."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3))
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3))
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(merge_strategy=MergeStrategy.INTERLEAVE)
        result = await controller.search("query", context=context)
        
        assert result.merge_strategy == MergeStrategy.INTERLEAVE


class TestHybridControllerExecutionModes:
    """Execution mode tests."""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Test parallel execution mode."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3), delay=0.1)
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3), delay=0.1)
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(execution_mode=ExecutionMode.PARALLEL)
        
        import time
        start = time.time()
        result = await controller.search("query", context=context)
        elapsed = time.time() - start
        
        # 並列なので0.2秒未満で完了するはず
        assert elapsed < 0.25
        assert len(result.engines_used) == 2
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self):
        """Test sequential execution mode."""
        controller = HybridController()
        
        engine1 = MockSearchEngine("e1", results=create_mock_results("e1", 3), delay=0.05)
        engine2 = MockSearchEngine("e2", results=create_mock_results("e2", 3), delay=0.05)
        
        controller.register_engine("e1", engine1)
        controller.register_engine("e2", engine2)
        
        context = HybridSearchContext(execution_mode=ExecutionMode.SEQUENTIAL)
        
        import time
        start = time.time()
        result = await controller.search("query", context=context)
        elapsed = time.time() - start
        
        # 順次なので0.1秒以上かかるはず
        assert elapsed >= 0.1
        assert len(result.engines_used) == 2
    
    @pytest.mark.asyncio
    async def test_race_execution(self):
        """Test race execution mode."""
        controller = HybridController()
        
        fast_engine = MockSearchEngine("fast", results=create_mock_results("fast", 3), delay=0.01)
        slow_engine = MockSearchEngine("slow", results=create_mock_results("slow", 3), delay=0.5)
        
        controller.register_engine("fast", fast_engine)
        controller.register_engine("slow", slow_engine)
        
        context = HybridSearchContext(execution_mode=ExecutionMode.RACE)
        
        import time
        start = time.time()
        result = await controller.search("query", context=context)
        elapsed = time.time() - start
        
        # 最初の成功で終了するので速いはず
        assert elapsed < 0.3
        assert "fast" in result.successful_engines


class TestHybridSearchResult:
    """HybridSearchResult tests."""
    
    def test_result_creation(self):
        """Test result creation."""
        items = create_mock_results("test", 3)
        result = HybridSearchResult(
            items=items,
            engines_used=["engine1"],
            merge_strategy=MergeStrategy.RRF,
        )
        
        assert result.total_count == 3
        assert result.engines_used == ["engine1"]
    
    def test_result_to_dict(self):
        """Test result serialization."""
        items = create_mock_results("test", 2)
        result = HybridSearchResult(
            items=items,
            engines_used=["engine1"],
            merge_strategy=MergeStrategy.RRF,
        )
        
        data = result.to_dict()
        
        assert "items" in data
        assert "engines_used" in data
        assert "merge_strategy" in data
        assert data["merge_strategy"] == "rrf"
    
    def test_result_from_dict(self):
        """Test result deserialization."""
        data = {
            "items": [
                {"content": "test", "score": 0.9, "source": "test.txt"},
            ],
            "engine_results": {},
            "engines_used": ["engine1"],
            "merge_strategy": "score",
            "total_processing_time_ms": 100.0,
        }
        
        result = HybridSearchResult.from_dict(data)
        
        assert len(result.items) == 1
        assert result.merge_strategy == MergeStrategy.SCORE
    
    def test_successful_failed_engines(self):
        """Test successful/failed engines properties."""
        result = HybridSearchResult(
            engine_results={
                "good": EngineResult(engine_name="good", success=True),
                "bad": EngineResult(engine_name="bad", success=False, error="error"),
            },
        )
        
        assert result.successful_engines == ["good"]
        assert result.failed_engines == ["bad"]


class TestHybridSearchContext:
    """HybridSearchContext tests."""
    
    def test_context_defaults(self):
        """Test context default values."""
        context = HybridSearchContext()
        
        assert context.max_results == 10
        assert context.merge_strategy == MergeStrategy.RRF
        assert context.execution_mode == ExecutionMode.PARALLEL
        assert context.timeout_per_engine == 30.0
    
    def test_context_to_dict(self):
        """Test context serialization."""
        context = HybridSearchContext(
            max_results=5,
            merge_strategy=MergeStrategy.SCORE,
        )
        
        data = context.to_dict()
        
        assert data["max_results"] == 5
        assert data["merge_strategy"] == "score"


class TestHybridResultItem:
    """HybridResultItem tests."""
    
    def test_item_creation(self):
        """Test item creation."""
        item = HybridResultItem(
            content="test content",
            score=0.9,
            source="test.txt",
            engine="vector",
        )
        
        assert item.content == "test content"
        assert item.score == 0.9
    
    def test_item_to_dict(self):
        """Test item serialization."""
        item = HybridResultItem(
            content="test",
            score=0.9,
            engine="vector",
            original_score=0.85,
            rank=1,
        )
        
        data = item.to_dict()
        
        assert data["content"] == "test"
        assert data["score"] == 0.9
        assert data["engine"] == "vector"
        assert data["original_score"] == 0.85
        assert data["rank"] == 1


class TestHybridControllerConfig:
    """HybridControllerConfig tests."""
    
    def test_config_defaults(self):
        """Test config default values."""
        config = HybridControllerConfig()
        
        assert config.default_merge_strategy == MergeStrategy.RRF
        assert config.default_execution_mode == ExecutionMode.PARALLEL
        assert config.rrf_k == 60
        assert config.max_concurrent == 5
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = HybridControllerConfig(
            rrf_k=30,
            enable_caching=True,
        )
        
        data = config.to_dict()
        
        assert data["rrf_k"] == 30
        assert data["enable_caching"] is True


class TestCreateHybridController:
    """create_hybrid_controller factory tests."""
    
    def test_create_default(self):
        """Test default creation."""
        controller = create_hybrid_controller()
        
        assert controller is not None
        assert controller.config is not None
    
    def test_create_with_config(self):
        """Test creation with config."""
        config = HybridControllerConfig(rrf_k=30)
        controller = create_hybrid_controller(config=config)
        
        assert controller.config.rrf_k == 30
    
    def test_create_with_engines(self):
        """Test creation with engines."""
        engines = {
            "e1": MockSearchEngine("e1"),
            "e2": MockSearchEngine("e2"),
        }
        controller = create_hybrid_controller(engines=engines)
        
        assert controller.has_engine("e1")
        assert controller.has_engine("e2")


class TestHybridControllerStatusAndCache:
    """Status and cache tests."""
    
    def test_get_status(self):
        """Test getting status."""
        controller = HybridController()
        engine = MockSearchEngine("test")
        controller.register_engine("test", engine)
        
        status = controller.get_status()
        
        assert "registered_engines" in status
        assert "available_engines" in status
        assert "cache_enabled" in status
        assert "test" in status["registered_engines"]
    
    @pytest.mark.asyncio
    async def test_caching(self):
        """Test result caching."""
        config = HybridControllerConfig(
            enable_caching=True,
            cache_ttl_seconds=60,
        )
        controller = HybridController(config=config)
        
        engine = MockSearchEngine("test", results=create_mock_results("test", 3))
        controller.register_engine("test", engine)
        
        # First search
        result1 = await controller.search("query")
        
        # Second search should use cache
        result2 = await controller.search("query")
        
        assert controller.get_status()["cache_size"] == 1
    
    def test_clear_cache(self):
        """Test cache clearing."""
        config = HybridControllerConfig(enable_caching=True)
        controller = HybridController(config=config)
        controller._cache["test"] = HybridSearchResult()
        
        controller.clear_cache()
        
        assert len(controller._cache) == 0
    
    def test_reset(self):
        """Test controller reset."""
        controller = HybridController()
        engine = MockSearchEngine("test")
        controller.register_engine("test", engine)
        controller._cache["test"] = HybridSearchResult()
        
        controller.reset()
        
        assert not controller.has_engine("test")
        assert len(controller._cache) == 0


class TestEngineResult:
    """EngineResult tests."""
    
    def test_engine_result_success(self):
        """Test successful engine result."""
        result = EngineResult(
            engine_name="test",
            items=create_mock_results("test", 3),
            success=True,
        )
        
        assert result.success
        assert result.engine_name == "test"
        assert len(result.items) == 3
    
    def test_engine_result_failure(self):
        """Test failed engine result."""
        result = EngineResult(
            engine_name="test",
            success=False,
            error="Connection failed",
        )
        
        assert not result.success
        assert result.error == "Connection failed"
    
    def test_engine_result_to_dict(self):
        """Test engine result serialization."""
        result = EngineResult(
            engine_name="test",
            items=create_mock_results("test", 2),
            success=True,
            processing_time_ms=50.0,
        )
        
        data = result.to_dict()
        
        assert data["engine_name"] == "test"
        assert data["success"] is True
        assert data["processing_time_ms"] == 50.0
        assert len(data["items"]) == 2
