"""Tests for ProgressiveController."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from monjyu.controller.budget import (
    BudgetController,
    BudgetConfig,
    CostBudget,
    IndexLevel,
)
from monjyu.controller.progressive import (
    ProgressiveController,
    ProgressiveControllerConfig,
    ProgressiveSearchContext,
    ProgressiveSearchResult,
    LevelSearchResult,
    ProgressiveResultItem,
    LevelNotBuiltError,
    ProgressiveControllerError,
    create_progressive_controller,
)


class MockLevelEngine:
    """モックレベル検索エンジン"""
    
    def __init__(self, level: IndexLevel, available: bool = True):
        self._level = level
        self._available = available
        self._results: list[ProgressiveResultItem] = []
    
    @property
    def level(self) -> IndexLevel:
        return self._level
    
    def set_results(self, results: list[ProgressiveResultItem]) -> None:
        self._results = results
    
    async def search(self, query: str, top_k: int = 10) -> LevelSearchResult:
        return LevelSearchResult(
            level=self._level,
            items=self._results[:top_k],
            score=0.8,
            tokens_used=100,
            processing_time_ms=50.0,
        )
    
    def is_available(self) -> bool:
        return self._available


class TestProgressiveControllerBasics:
    """ProgressiveController 基本テスト"""
    
    def test_initialization_default(self):
        """デフォルト初期化"""
        controller = ProgressiveController()
        assert controller.config.default_budget == CostBudget.STANDARD
        assert controller.budget_controller is not None
    
    def test_initialization_with_config(self):
        """カスタム設定で初期化"""
        config = ProgressiveControllerConfig(
            default_budget=CostBudget.PREMIUM,
            auto_build=True,
        )
        controller = ProgressiveController(config=config)
        assert controller.config.default_budget == CostBudget.PREMIUM
        assert controller.config.auto_build is True
    
    def test_register_engine(self):
        """エンジン登録"""
        controller = ProgressiveController()
        engine = MockLevelEngine(IndexLevel.RAW)
        
        controller.register_engine(IndexLevel.RAW, engine)
        
        assert controller.has_engine(IndexLevel.RAW)
        assert not controller.has_engine(IndexLevel.LAZY)
    
    def test_unregister_engine(self):
        """エンジン登録解除"""
        controller = ProgressiveController()
        engine = MockLevelEngine(IndexLevel.RAW)
        
        controller.register_engine(IndexLevel.RAW, engine)
        controller.unregister_engine(IndexLevel.RAW)
        
        assert not controller.has_engine(IndexLevel.RAW)
    
    def test_get_available_levels(self):
        """利用可能レベル取得"""
        controller = ProgressiveController()
        
        engine_raw = MockLevelEngine(IndexLevel.RAW, available=True)
        engine_lazy = MockLevelEngine(IndexLevel.LAZY, available=True)
        engine_partial = MockLevelEngine(IndexLevel.PARTIAL, available=False)
        
        controller.register_engine(IndexLevel.RAW, engine_raw)
        controller.register_engine(IndexLevel.LAZY, engine_lazy)
        controller.register_engine(IndexLevel.PARTIAL, engine_partial)
        
        available = controller.get_available_levels()
        
        assert IndexLevel.RAW in available
        assert IndexLevel.LAZY in available
        assert IndexLevel.PARTIAL not in available


class TestProgressiveControllerSearch:
    """ProgressiveController 検索テスト"""
    
    @pytest.fixture
    def controller_with_engines(self):
        """エンジン付きコントローラ"""
        controller = ProgressiveController()
        
        # RAW エンジン
        engine_raw = MockLevelEngine(IndexLevel.RAW)
        engine_raw.set_results([
            ProgressiveResultItem(
                content="RAW result 1",
                score=0.9,
                source="doc1",
            ),
            ProgressiveResultItem(
                content="RAW result 2",
                score=0.8,
                source="doc2",
            ),
        ])
        controller.register_engine(IndexLevel.RAW, engine_raw)
        
        # LAZY エンジン
        engine_lazy = MockLevelEngine(IndexLevel.LAZY)
        engine_lazy.set_results([
            ProgressiveResultItem(
                content="LAZY result 1",
                score=0.85,
                source="doc3",
            ),
        ])
        controller.register_engine(IndexLevel.LAZY, engine_lazy)
        
        # PARTIAL エンジン
        engine_partial = MockLevelEngine(IndexLevel.PARTIAL)
        engine_partial.set_results([
            ProgressiveResultItem(
                content="PARTIAL result 1",
                score=0.95,
                source="doc4",
            ),
        ])
        controller.register_engine(IndexLevel.PARTIAL, engine_partial)
        
        return controller
    
    @pytest.mark.asyncio
    async def test_search_minimal_budget(self, controller_with_engines):
        """MINIMAL予算での検索"""
        controller = controller_with_engines
        
        result = await controller.search(
            query="test query",
            budget=CostBudget.MINIMAL,
        )
        
        # MINIMAL は Level 0-1 のみ
        assert result.budget == CostBudget.MINIMAL
        assert IndexLevel.RAW in result.levels_searched
        assert IndexLevel.LAZY in result.levels_searched
        assert IndexLevel.PARTIAL not in result.levels_searched
    
    @pytest.mark.asyncio
    async def test_search_standard_budget(self, controller_with_engines):
        """STANDARD予算での検索"""
        controller = controller_with_engines
        
        result = await controller.search(
            query="test query",
            budget=CostBudget.STANDARD,
        )
        
        # STANDARD は Level 0-2
        assert result.budget == CostBudget.STANDARD
        assert IndexLevel.RAW in result.levels_searched
        assert IndexLevel.PARTIAL in result.levels_searched
    
    @pytest.mark.asyncio
    async def test_search_with_context(self, controller_with_engines):
        """コンテキスト付き検索"""
        controller = controller_with_engines
        
        context = ProgressiveSearchContext(
            budget=CostBudget.MINIMAL,
            max_results=5,
        )
        
        result = await controller.search(
            query="test query",
            context=context,
        )
        
        assert result.budget == CostBudget.MINIMAL
        assert len(result.merged_items) <= 5
    
    @pytest.mark.asyncio
    async def test_search_records_tokens(self, controller_with_engines):
        """トークン使用量の記録"""
        controller = controller_with_engines
        
        initial_tokens = controller.budget_controller.state.total_tokens_used
        
        await controller.search(
            query="test query",
            budget=CostBudget.MINIMAL,
        )
        
        # トークンが記録されている
        assert controller.budget_controller.state.total_tokens_used > initial_tokens
    
    @pytest.mark.asyncio
    async def test_search_caching(self, controller_with_engines):
        """検索結果キャッシュ"""
        controller = controller_with_engines
        controller.config.enable_caching = True
        
        # 1回目
        result1 = await controller.search(
            query="cached query",
            budget=CostBudget.MINIMAL,
        )
        
        # 2回目 (キャッシュヒット)
        result2 = await controller.search(
            query="cached query",
            budget=CostBudget.MINIMAL,
        )
        
        assert result1.query == result2.query
        assert len(controller._cache) == 1
    
    @pytest.mark.asyncio
    async def test_search_no_engines_error(self):
        """エンジンなしでエラー"""
        controller = ProgressiveController()
        
        with pytest.raises(ProgressiveControllerError):
            await controller.search("test query")


class TestProgressiveControllerMergeStrategies:
    """マージ戦略テスト"""
    
    @pytest.fixture
    def level_results(self):
        """テスト用レベル結果"""
        return {
            IndexLevel.RAW: LevelSearchResult(
                level=IndexLevel.RAW,
                items=[
                    ProgressiveResultItem(content="A", score=0.9),
                    ProgressiveResultItem(content="B", score=0.7),
                ],
            ),
            IndexLevel.LAZY: LevelSearchResult(
                level=IndexLevel.LAZY,
                items=[
                    ProgressiveResultItem(content="C", score=0.95),
                    ProgressiveResultItem(content="A", score=0.85),  # 重複
                ],
            ),
        }
    
    def test_merge_by_score(self, level_results):
        """スコア順マージ"""
        controller = ProgressiveController()
        
        merged = controller._merge_by_score(level_results, max_results=10)
        
        # スコア順になっている
        assert merged[0].content == "C"  # 0.95
        assert merged[0].score == 0.95
        # 重複は除去される
        assert len([m for m in merged if m.content == "A"]) == 1
    
    def test_merge_by_rrf(self, level_results):
        """RRFマージ"""
        controller = ProgressiveController()
        
        merged = controller._merge_by_rrf(level_results, max_results=10)
        
        # RRFスコアが計算されている
        assert all(m.score > 0 for m in merged)
        # 複数リストに出現するアイテムはスコアが高くなる
        a_item = next((m for m in merged if m.content == "A"), None)
        c_item = next((m for m in merged if m.content == "C"), None)
        assert a_item is not None
        assert a_item.score > c_item.score  # Aは2つのリストに出現
    
    def test_merge_by_level_priority(self, level_results):
        """レベル優先マージ"""
        controller = ProgressiveController()
        
        merged = controller._merge_by_level_priority(level_results, max_results=10)
        
        # 高いレベルが先に来る
        assert merged[0].level == IndexLevel.LAZY
        # 重複は除去される
        assert len([m for m in merged if m.content == "A"]) == 1


class TestProgressiveSearchResult:
    """ProgressiveSearchResult テスト"""
    
    def test_to_dict(self):
        """辞書変換"""
        result = ProgressiveSearchResult(
            query="test",
            budget=CostBudget.STANDARD,
            max_level_used=IndexLevel.PARTIAL,
            levels_searched=[IndexLevel.RAW, IndexLevel.PARTIAL],
            merged_items=[
                ProgressiveResultItem(
                    content="test content",
                    score=0.9,
                    level=IndexLevel.RAW,
                )
            ],
            total_tokens_used=500,
            processing_time_ms=100.0,
        )
        
        data = result.to_dict()
        
        assert data["query"] == "test"
        assert data["budget"] == "standard"
        assert data["max_level_used"] == 2
        assert len(data["merged_items"]) == 1
    
    def test_from_dict(self):
        """辞書から作成"""
        data = {
            "query": "test",
            "budget": "standard",
            "max_level_used": 2,
            "levels_searched": [0, 2],
            "level_results": {},
            "merged_items": [
                {
                    "content": "test content",
                    "score": 0.9,
                    "source": None,
                    "level": 0,
                    "metadata": {},
                }
            ],
            "total_tokens_used": 500,
            "processing_time_ms": 100.0,
            "metadata": {},
        }
        
        result = ProgressiveSearchResult.from_dict(data)
        
        assert result.query == "test"
        assert result.budget == CostBudget.STANDARD
        assert result.max_level_used == IndexLevel.PARTIAL
        assert len(result.merged_items) == 1


class TestProgressiveSearchContext:
    """ProgressiveSearchContext テスト"""
    
    def test_default_values(self):
        """デフォルト値"""
        context = ProgressiveSearchContext()
        assert context.budget == CostBudget.STANDARD
        assert context.max_results == 10
        assert context.auto_build is True
    
    def test_to_dict(self):
        """辞書変換"""
        context = ProgressiveSearchContext(
            budget=CostBudget.PREMIUM,
            max_results=20,
        )
        
        data = context.to_dict()
        
        assert data["budget"] == "premium"
        assert data["max_results"] == 20


class TestProgressiveControllerConfig:
    """ProgressiveControllerConfig テスト"""
    
    def test_default_values(self):
        """デフォルト値"""
        config = ProgressiveControllerConfig()
        assert config.default_budget == CostBudget.STANDARD
        assert config.auto_build is False
        assert config.merge_strategy == "score"
    
    def test_to_dict(self):
        """辞書変換"""
        config = ProgressiveControllerConfig(
            default_budget=CostBudget.UNLIMITED,
            auto_build=True,
            merge_strategy="rrf",
        )
        
        data = config.to_dict()
        
        assert data["default_budget"] == "unlimited"
        assert data["auto_build"] is True
        assert data["merge_strategy"] == "rrf"


class TestCreateProgressiveController:
    """ファクトリ関数テスト"""
    
    def test_default_factory(self):
        """デフォルトファクトリ"""
        controller = create_progressive_controller()
        
        assert controller.config.default_budget == CostBudget.STANDARD
        assert controller.config.auto_build is False
    
    def test_custom_factory(self):
        """カスタムファクトリ"""
        controller = create_progressive_controller(
            default_budget=CostBudget.PREMIUM,
            auto_build=True,
        )
        
        assert controller.config.default_budget == CostBudget.PREMIUM
        assert controller.config.auto_build is True


class TestProgressiveControllerStatusAndReset:
    """ステータスとリセットテスト"""
    
    def test_get_status_summary(self):
        """ステータスサマリー取得"""
        controller = ProgressiveController()
        engine = MockLevelEngine(IndexLevel.RAW)
        controller.register_engine(IndexLevel.RAW, engine)
        
        summary = controller.get_status_summary()
        
        assert "available_levels" in summary
        assert "engine_status" in summary
        assert "budget_summary" in summary
        assert "config" in summary
    
    def test_clear_cache(self):
        """キャッシュクリア"""
        controller = ProgressiveController()
        controller._cache["test"] = MagicMock()
        
        controller.clear_cache()
        
        assert len(controller._cache) == 0
    
    def test_reset(self):
        """リセット"""
        controller = ProgressiveController()
        controller._cache["test"] = MagicMock()
        controller.budget_controller.record_tokens(prompt_tokens=1000)
        
        controller.reset()
        
        assert len(controller._cache) == 0
        assert controller.budget_controller.state.total_tokens_used == 0


class TestLevelSearchResult:
    """LevelSearchResult テスト"""
    
    def test_to_dict(self):
        """辞書変換"""
        result = LevelSearchResult(
            level=IndexLevel.LAZY,
            items=[
                ProgressiveResultItem(content="test", score=0.9),
            ],
            score=0.85,
            tokens_used=100,
            processing_time_ms=50.0,
        )
        
        data = result.to_dict()
        
        assert data["level"] == 1
        assert data["level_name"] == "LAZY"
        assert data["score"] == 0.85
        assert len(data["items"]) == 1


class TestProgressiveResultItem:
    """ProgressiveResultItem テスト"""
    
    def test_to_dict(self):
        """辞書変換"""
        item = ProgressiveResultItem(
            content="test content",
            score=0.95,
            source="doc1",
            level=IndexLevel.RAW,
            metadata={"key": "value"},
        )
        
        data = item.to_dict()
        
        assert data["content"] == "test content"
        assert data["score"] == 0.95
        assert data["source"] == "doc1"
        assert data["level"] == 0
        assert data["level_name"] == "RAW"
        assert data["metadata"] == {"key": "value"}
    
    def test_to_dict_without_level(self):
        """レベルなしの辞書変換"""
        item = ProgressiveResultItem(
            content="test content",
            score=0.95,
        )
        
        data = item.to_dict()
        
        assert data["content"] == "test content"
        assert data["level"] is None
        assert data["level_name"] is None
