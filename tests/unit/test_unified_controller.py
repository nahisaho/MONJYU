"""UnifiedController unit tests."""

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.query.router.types import QueryType, SearchMode


# ==== モックエンジン ====


class MockSearchEngine:
    """テスト用モック検索エンジン"""
    
    def __init__(
        self,
        name: str = "mock_engine",
        available: bool = True,
        results: Optional[list] = None,
        raise_error: Optional[Exception] = None,
    ):
        self._name = name
        self._available = available
        self._results = results or []
        self._raise_error = raise_error
    
    async def search(self, query: str, context=None):
        from monjyu.controller.unified import EngineSearchResult, SearchResultItem
        
        if self._raise_error:
            raise self._raise_error
        
        return EngineSearchResult(
            items=[
                SearchResultItem(content=r, score=1.0 - i * 0.1)
                for i, r in enumerate(self._results)
            ],
            total_count=len(self._results),
            metadata={"engine": self._name},
        )
    
    def is_available(self) -> bool:
        return self._available
    
    @property
    def name(self) -> str:
        return self._name


# ==== 設定テスト ====


class TestUnifiedControllerConfig:
    """UnifiedControllerConfig のテスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        from monjyu.controller.unified import UnifiedControllerConfig
        
        config = UnifiedControllerConfig()
        
        assert config.default_mode == SearchMode.AUTO
        assert config.enable_auto_routing is True
        assert config.fallback_enabled is True
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 2
        assert config.retry_delay_seconds == 1.0
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        from monjyu.controller.unified import UnifiedControllerConfig
        
        config = UnifiedControllerConfig(
            default_mode=SearchMode.VECTOR,
            enable_auto_routing=False,
            fallback_enabled=False,
            timeout_seconds=60.0,
            max_retries=5,
            retry_delay_seconds=2.0,
        )
        
        assert config.default_mode == SearchMode.VECTOR
        assert config.enable_auto_routing is False
        assert config.fallback_enabled is False
        assert config.timeout_seconds == 60.0
        assert config.max_retries == 5
        assert config.retry_delay_seconds == 2.0
    
    def test_config_to_dict(self):
        """設定辞書変換テスト"""
        from monjyu.controller.unified import UnifiedControllerConfig
        
        config = UnifiedControllerConfig()
        data = config.to_dict()
        
        assert data["default_mode"] == "auto"
        assert data["enable_auto_routing"] is True


# ==== SearchContext テスト ====


class TestSearchContext:
    """SearchContext のテスト"""
    
    def test_default_context(self):
        """デフォルトコンテキストテスト"""
        from monjyu.controller.unified import SearchContext
        
        ctx = SearchContext()
        
        assert ctx.query_type is None
        assert ctx.max_results == 10
        assert ctx.include_metadata is True
        assert ctx.language == "auto"
    
    def test_context_to_dict(self):
        """コンテキスト辞書変換テスト"""
        from monjyu.controller.unified import SearchContext
        
        ctx = SearchContext(
            query_type=QueryType.SURVEY,
            max_results=20,
        )
        
        data = ctx.to_dict()
        assert data["query_type"] == "survey"
        assert data["max_results"] == 20


# ==== SearchResultItem テスト ====


class TestSearchResultItem:
    """SearchResultItem のテスト"""
    
    def test_item_creation(self):
        """アイテム作成テスト"""
        from monjyu.controller.unified import SearchResultItem
        
        item = SearchResultItem(
            content="Test content",
            score=0.95,
            source="paper.pdf",
        )
        
        assert item.content == "Test content"
        assert item.score == 0.95
        assert item.source == "paper.pdf"


# ==== UnifiedSearchResult テスト ====


class TestUnifiedSearchResult:
    """UnifiedSearchResult のテスト"""
    
    def test_result_creation(self):
        """結果作成テスト"""
        from monjyu.controller.unified import UnifiedSearchResult, SearchResultItem
        
        result = UnifiedSearchResult(
            mode_used=SearchMode.VECTOR,
            query_type=QueryType.FACTOID,
            items=[SearchResultItem(content="Test", score=0.9)],
            total_count=1,
            processing_time_ms=123.45,
        )
        
        assert result.mode_used == SearchMode.VECTOR
        assert result.query_type == QueryType.FACTOID
        assert len(result.items) == 1
        assert result.processing_time_ms == 123.45
    
    def test_result_to_dict(self):
        """結果辞書変換テスト"""
        from monjyu.controller.unified import UnifiedSearchResult, SearchResultItem
        
        result = UnifiedSearchResult(
            mode_used=SearchMode.LAZY,
            query_type=QueryType.EXPLORATION,
            items=[SearchResultItem(content="Content", score=0.8)],
            total_count=1,
        )
        
        data = result.to_dict()
        
        assert data["mode_used"] == "lazy"
        assert data["query_type"] == "exploration"
        assert len(data["items"]) == 1
    
    def test_result_from_dict(self):
        """辞書から結果作成テスト"""
        from monjyu.controller.unified import UnifiedSearchResult
        
        data = {
            "mode_used": "vector",
            "query_type": "factoid",
            "items": [{"content": "Test", "score": 0.9}],
            "total_count": 1,
            "processing_time_ms": 100.0,
        }
        
        result = UnifiedSearchResult.from_dict(data)
        
        assert result.mode_used == SearchMode.VECTOR
        assert result.query_type == QueryType.FACTOID


# ==== エンジン管理テスト ====


class TestEngineManagement:
    """エンジン管理のテスト"""
    
    @pytest.fixture
    def controller(self):
        """コントローラ作成"""
        from monjyu.controller.unified import UnifiedController
        return UnifiedController()
    
    def test_register_engine(self, controller):
        """エンジン登録テスト"""
        engine = MockSearchEngine(name="vector")
        controller.register_engine(SearchMode.VECTOR, engine)
        
        assert controller.has_engine(SearchMode.VECTOR)
        assert SearchMode.VECTOR in controller.get_available_modes()
    
    def test_unregister_engine(self, controller):
        """エンジン登録解除テスト"""
        engine = MockSearchEngine(name="vector")
        controller.register_engine(SearchMode.VECTOR, engine)
        controller.unregister_engine(SearchMode.VECTOR)
        
        assert not controller.has_engine(SearchMode.VECTOR)
    
    def test_get_available_modes(self, controller):
        """利用可能モード取得テスト"""
        controller.register_engine(
            SearchMode.VECTOR,
            MockSearchEngine(name="vector", available=True),
        )
        controller.register_engine(
            SearchMode.LAZY,
            MockSearchEngine(name="lazy", available=False),
        )
        
        available = controller.get_available_modes()
        
        assert SearchMode.VECTOR in available
        assert SearchMode.LAZY not in available
    
    def test_has_engine(self, controller):
        """エンジン存在チェックテスト"""
        engine = MockSearchEngine(name="vector")
        controller.register_engine(SearchMode.VECTOR, engine)
        
        assert controller.has_engine(SearchMode.VECTOR)
        assert not controller.has_engine(SearchMode.LAZY)


# ==== 検索テスト ====


class TestUnifiedControllerSearch:
    """検索のテスト"""
    
    @pytest.fixture
    def controller_with_engines(self):
        """エンジン付きコントローラ作成"""
        from monjyu.controller.unified import UnifiedController, UnifiedControllerConfig
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,  # テスト用に無効化
        )
        controller = UnifiedController(config=config)
        
        controller.register_engine(
            SearchMode.VECTOR,
            MockSearchEngine(name="vector", results=["Vector result 1", "Vector result 2"]),
        )
        controller.register_engine(
            SearchMode.LAZY,
            MockSearchEngine(name="lazy", results=["Lazy result 1"]),
        )
        
        return controller
    
    @pytest.mark.asyncio
    async def test_search_with_specified_mode(self, controller_with_engines):
        """指定モード検索テスト"""
        result = await controller_with_engines.search(
            "test query",
            mode=SearchMode.VECTOR,
        )
        
        assert result.mode_used == SearchMode.VECTOR
        assert len(result.items) == 2
        assert result.items[0].content == "Vector result 1"
    
    @pytest.mark.asyncio
    async def test_search_with_default_mode(self, controller_with_engines):
        """デフォルトモード検索テスト"""
        from monjyu.controller.unified import UnifiedControllerConfig
        
        controller_with_engines.config = UnifiedControllerConfig(
            default_mode=SearchMode.LAZY,
            enable_auto_routing=False,
        )
        
        result = await controller_with_engines.search("test query")
        
        assert result.mode_used == SearchMode.LAZY
    
    @pytest.mark.asyncio
    async def test_search_engine_not_found(self):
        """エンジン未登録エラーテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            EngineNotFoundError,
        )
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
        )
        controller = UnifiedController(config=config)
        
        with pytest.raises(EngineNotFoundError):
            await controller.search("test", mode=SearchMode.VECTOR)
    
    @pytest.mark.asyncio
    async def test_search_engine_unavailable(self):
        """エンジン利用不可エラーテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            EngineUnavailableError,
        )
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
        )
        controller = UnifiedController(config=config)
        controller.register_engine(
            SearchMode.VECTOR,
            MockSearchEngine(name="vector", available=False),
        )
        
        with pytest.raises(EngineUnavailableError):
            await controller.search("test", mode=SearchMode.VECTOR)
    
    @pytest.mark.asyncio
    async def test_search_processing_time(self, controller_with_engines):
        """処理時間計測テスト"""
        result = await controller_with_engines.search(
            "test query",
            mode=SearchMode.VECTOR,
        )
        
        assert result.processing_time_ms > 0


# ==== 自動ルーティングテスト ====


class TestAutoRouting:
    """自動ルーティングのテスト"""
    
    @pytest.fixture
    def controller_with_routing(self):
        """ルーティング有効コントローラ作成"""
        from monjyu.controller.unified import UnifiedController, UnifiedControllerConfig
        from monjyu.query.router import QueryRouter, QueryRouterConfig
        
        config = UnifiedControllerConfig(
            enable_auto_routing=True,
        )
        # フォールバックを無効にしてテスト
        router_config = QueryRouterConfig(enable_fallback=False)
        router = QueryRouter(config=router_config)
        controller = UnifiedController(router=router, config=config)
        
        # 全モードのエンジンを登録
        for mode in [SearchMode.VECTOR, SearchMode.LAZY, SearchMode.GRAPHRAG, SearchMode.HYBRID]:
            controller.register_engine(
                mode,
                MockSearchEngine(name=mode.value, results=[f"{mode.value} result"]),
            )
        
        return controller
    
    @pytest.mark.asyncio
    async def test_auto_route_survey_to_graphrag(self, controller_with_routing):
        """サーベイクエリ→GraphRAGルーティングテスト"""
        result = await controller_with_routing.search(
            "深層学習の研究動向は？",
            mode=SearchMode.AUTO,
        )
        
        assert result.mode_used == SearchMode.GRAPHRAG
        assert result.query_type == QueryType.SURVEY
    
    @pytest.mark.asyncio
    async def test_auto_route_exploration_to_lazy(self, controller_with_routing):
        """探索クエリ→LAZYルーティングテスト"""
        result = await controller_with_routing.search(
            "BERTの実装方法を教えて",
            mode=SearchMode.AUTO,
        )
        
        assert result.mode_used == SearchMode.LAZY
        assert result.query_type == QueryType.EXPLORATION
    
    @pytest.mark.asyncio
    async def test_auto_route_factoid_to_vector(self, controller_with_routing):
        """事実クエリ→VECTORルーティングテスト"""
        result = await controller_with_routing.search(
            "パラメータ数はいくつ？",
            mode=SearchMode.AUTO,
        )
        
        assert result.mode_used == SearchMode.VECTOR
        assert result.query_type == QueryType.FACTOID
    
    @pytest.mark.asyncio
    async def test_routing_confidence_recorded(self, controller_with_routing):
        """ルーティング確信度記録テスト"""
        result = await controller_with_routing.search(
            "深層学習の研究動向は？",
            mode=SearchMode.AUTO,
        )
        
        assert result.routing_confidence > 0


# ==== フォールバックテスト ====


class TestFallback:
    """フォールバックのテスト"""
    
    @pytest.fixture
    def controller_with_fallback(self):
        """フォールバック有効コントローラ作成"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchEngineError,
        )
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=True,
        )
        controller = UnifiedController(config=config)
        
        # GRAPHRAGは失敗するエンジン
        controller.register_engine(
            SearchMode.GRAPHRAG,
            MockSearchEngine(
                name="graphrag",
                raise_error=SearchEngineError("GraphRAG failed"),
            ),
        )
        # LAZYは成功するエンジン
        controller.register_engine(
            SearchMode.LAZY,
            MockSearchEngine(name="lazy", results=["Fallback result"]),
        )
        
        return controller
    
    @pytest.mark.asyncio
    async def test_fallback_on_engine_failure(self, controller_with_fallback):
        """エンジン失敗時フォールバックテスト"""
        result = await controller_with_fallback.search(
            "test query",
            mode=SearchMode.GRAPHRAG,
        )
        
        assert result.mode_used == SearchMode.LAZY
        assert result.fallback_used is True
        assert result.fallback_mode == SearchMode.LAZY
    
    @pytest.mark.asyncio
    async def test_fallback_disabled_raises_error(self):
        """フォールバック無効時エラーテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchEngineError,
        )
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
        )
        controller = UnifiedController(config=config)
        controller.register_engine(
            SearchMode.GRAPHRAG,
            MockSearchEngine(
                name="graphrag",
                raise_error=SearchEngineError("GraphRAG failed"),
            ),
        )
        
        with pytest.raises(SearchEngineError):
            await controller.search("test", mode=SearchMode.GRAPHRAG)


# ==== リトライテスト ====


class TestRetry:
    """リトライのテスト"""
    
    @pytest.mark.asyncio
    async def test_retry_on_transient_error(self):
        """一時エラー時リトライテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchEngineError,
        )
        
        call_count = 0
        
        class TransientErrorEngine:
            def __init__(self):
                self.call_count = 0
            
            async def search(self, query, context=None):
                from monjyu.controller.unified import EngineSearchResult, SearchResultItem
                self.call_count += 1
                if self.call_count < 2:
                    raise SearchEngineError("Transient error")
                return EngineSearchResult(
                    items=[SearchResultItem(content="Success")],
                    total_count=1,
                )
            
            def is_available(self):
                return True
            
            @property
            def name(self):
                return "transient"
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
            max_retries=3,
            retry_delay_seconds=0.01,
        )
        controller = UnifiedController(config=config)
        engine = TransientErrorEngine()
        controller.register_engine(SearchMode.VECTOR, engine)
        
        result = await controller.search_with_retry("test", mode=SearchMode.VECTOR)
        
        assert engine.call_count == 2
        assert result.items[0].content == "Success"
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """最大リトライ超過テスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchEngineError,
        )
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
            max_retries=2,
            retry_delay_seconds=0.01,
        )
        controller = UnifiedController(config=config)
        controller.register_engine(
            SearchMode.VECTOR,
            MockSearchEngine(
                name="always_fail",
                raise_error=SearchEngineError("Always fails"),
            ),
        )
        
        with pytest.raises(SearchEngineError):
            await controller.search_with_retry("test", mode=SearchMode.VECTOR)


# ==== タイムアウトテスト ====


class TestTimeout:
    """タイムアウトのテスト"""
    
    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """検索タイムアウトテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchTimeoutError,
        )
        
        class SlowEngine:
            async def search(self, query, context=None):
                await asyncio.sleep(10)  # 長いスリープ
            
            def is_available(self):
                return True
            
            @property
            def name(self):
                return "slow"
        
        config = UnifiedControllerConfig(
            enable_auto_routing=False,
            fallback_enabled=False,
            timeout_seconds=0.1,
        )
        controller = UnifiedController(config=config)
        controller.register_engine(SearchMode.VECTOR, SlowEngine())
        
        with pytest.raises(SearchTimeoutError):
            await controller.search("test", mode=SearchMode.VECTOR)


# ==== ファクトリ関数テスト ====


class TestCreateUnifiedController:
    """create_unified_controller のテスト"""
    
    def test_create_default(self):
        """デフォルト作成テスト"""
        from monjyu.controller.unified import create_unified_controller
        
        controller = create_unified_controller()
        
        assert controller is not None
        assert controller.config.default_mode == SearchMode.AUTO
    
    def test_create_with_config(self):
        """設定付き作成テスト"""
        from monjyu.controller.unified import (
            create_unified_controller,
            UnifiedControllerConfig,
        )
        
        config = UnifiedControllerConfig(
            default_mode=SearchMode.LAZY,
            timeout_seconds=60.0,
        )
        controller = create_unified_controller(config=config)
        
        assert controller.config.default_mode == SearchMode.LAZY
        assert controller.config.timeout_seconds == 60.0
    
    def test_create_with_router(self):
        """ルーター付き作成テスト"""
        from monjyu.controller.unified import create_unified_controller
        from monjyu.query.router import QueryRouter
        
        router = QueryRouter()
        controller = create_unified_controller(router=router)
        
        assert controller.router is router


# ==== 統合テスト ====


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_search_workflow(self):
        """完全な検索ワークフローテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchContext,
        )
        from monjyu.query.router import QueryRouter, QueryRouterConfig
        
        # セットアップ
        config = UnifiedControllerConfig(
            enable_auto_routing=True,
            fallback_enabled=True,
        )
        # ルーターのフォールバックを無効に
        router_config = QueryRouterConfig(enable_fallback=False)
        router = QueryRouter(config=router_config)
        controller = UnifiedController(router=router, config=config)
        
        # エンジン登録
        controller.register_engine(
            SearchMode.VECTOR,
            MockSearchEngine(name="vector", results=["Vector result"]),
        )
        controller.register_engine(
            SearchMode.LAZY,
            MockSearchEngine(name="lazy", results=["Lazy result"]),
        )
        controller.register_engine(
            SearchMode.GRAPHRAG,
            MockSearchEngine(name="graphrag", results=["GraphRAG result"]),
        )
        controller.register_engine(
            SearchMode.HYBRID,
            MockSearchEngine(name="hybrid", results=["Hybrid result"]),
        )
        
        # 検索実行
        context = SearchContext(max_results=5)
        result = await controller.search(
            "深層学習の研究動向について教えて",
            mode=SearchMode.AUTO,
            context=context,
        )
        
        # 検証
        assert result.mode_used == SearchMode.GRAPHRAG
        assert result.query_type == QueryType.SURVEY
        assert result.processing_time_ms > 0
        assert len(result.items) > 0
    
    @pytest.mark.asyncio
    async def test_context_with_user_preference(self):
        """ユーザー優先設定付きコンテキストテスト"""
        from monjyu.controller.unified import (
            UnifiedController,
            UnifiedControllerConfig,
            SearchContext,
        )
        from monjyu.query.router import QueryRouter
        
        config = UnifiedControllerConfig(enable_auto_routing=True)
        router = QueryRouter()
        controller = UnifiedController(router=router, config=config)
        
        # 全モードのエンジン登録
        for mode in [SearchMode.VECTOR, SearchMode.LAZY, SearchMode.GRAPHRAG, SearchMode.HYBRID]:
            controller.register_engine(
                mode,
                MockSearchEngine(name=mode.value, results=[f"{mode.value} result"]),
            )
        
        # ユーザーがVECTORを優先
        context = SearchContext(user_preference=SearchMode.VECTOR)
        result = await controller.search(
            "深層学習の研究動向は？",  # 通常はGRAPHRAG
            mode=SearchMode.AUTO,
            context=context,
        )
        
        # ユーザー優先設定が反映される
        assert result.mode_used == SearchMode.VECTOR
