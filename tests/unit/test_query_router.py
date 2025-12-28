# Test Query Router - FEAT-014
"""
QueryRouter の単体テスト
"""

import pytest
from unittest.mock import AsyncMock
import json


class TestSearchMode:
    """SearchMode のテスト"""
    
    def test_search_mode_values(self):
        """SearchMode値テスト"""
        from monjyu.query.router.types import SearchMode
        
        assert SearchMode.AUTO.value == "auto"
        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.LAZY.value == "lazy"
        assert SearchMode.GRAPHRAG.value == "graphrag"
        assert SearchMode.HYBRID.value == "hybrid"
    
    def test_from_string(self):
        """from_stringテスト"""
        from monjyu.query.router.types import SearchMode
        
        assert SearchMode.from_string("vector") == SearchMode.VECTOR
        assert SearchMode.from_string("LAZY") == SearchMode.LAZY
        assert SearchMode.from_string("unknown") == SearchMode.AUTO


class TestQueryType:
    """QueryType のテスト"""
    
    def test_query_type_values(self):
        """QueryType値テスト"""
        from monjyu.query.router.types import QueryType
        
        assert QueryType.SURVEY.value == "survey"
        assert QueryType.EXPLORATION.value == "exploration"
        assert QueryType.COMPARISON.value == "comparison"
        assert QueryType.FACTOID.value == "factoid"
        assert QueryType.CITATION.value == "citation"
        assert QueryType.BENCHMARK.value == "benchmark"


class TestRoutingDecision:
    """RoutingDecision のテスト"""
    
    def test_decision_creation(self):
        """Decision作成テスト"""
        from monjyu.query.router.types import (
            RoutingDecision,
            SearchMode,
            QueryType,
        )
        
        decision = RoutingDecision(
            mode=SearchMode.LAZY,
            query_type=QueryType.EXPLORATION,
            confidence=0.85,
            reasoning="Test reasoning",
        )
        
        assert decision.mode == SearchMode.LAZY
        assert decision.query_type == QueryType.EXPLORATION
        assert decision.confidence == 0.85
    
    def test_is_confident(self):
        """is_confidentプロパティテスト"""
        from monjyu.query.router.types import (
            RoutingDecision,
            SearchMode,
            QueryType,
        )
        
        confident = RoutingDecision(
            mode=SearchMode.LAZY,
            query_type=QueryType.EXPLORATION,
            confidence=0.8,
        )
        assert confident.is_confident is True
        
        not_confident = RoutingDecision(
            mode=SearchMode.LAZY,
            query_type=QueryType.UNKNOWN,
            confidence=0.5,
        )
        assert not_confident.is_confident is False
    
    def test_to_dict(self):
        """to_dictテスト"""
        from monjyu.query.router.types import (
            RoutingDecision,
            SearchMode,
            QueryType,
        )
        
        decision = RoutingDecision(
            mode=SearchMode.GRAPHRAG,
            query_type=QueryType.SURVEY,
            confidence=0.9,
            params={"community_level": 2},
        )
        
        data = decision.to_dict()
        
        assert data["mode"] == "graphrag"
        assert data["query_type"] == "survey"
        assert data["confidence"] == 0.9
        assert data["params"]["community_level"] == 2
    
    def test_from_dict(self):
        """from_dictテスト"""
        from monjyu.query.router.types import (
            RoutingDecision,
            SearchMode,
            QueryType,
        )
        
        data = {
            "mode": "hybrid",
            "query_type": "comparison",
            "confidence": 0.75,
        }
        
        decision = RoutingDecision.from_dict(data)
        
        assert decision.mode == SearchMode.HYBRID
        assert decision.query_type == QueryType.COMPARISON


class TestRoutingContext:
    """RoutingContext のテスト"""
    
    def test_context_creation(self):
        """Context作成テスト"""
        from monjyu.query.router.types import RoutingContext, SearchMode
        
        context = RoutingContext(
            index_level=2,
            available_modes=[SearchMode.VECTOR, SearchMode.LAZY, SearchMode.HYBRID],
        )
        
        assert context.index_level == 2
        assert len(context.available_modes) == 3
    
    def test_is_mode_available(self):
        """is_mode_availableテスト"""
        from monjyu.query.router.types import RoutingContext, SearchMode
        
        context = RoutingContext(
            available_modes=[SearchMode.VECTOR, SearchMode.LAZY],
        )
        
        assert context.is_mode_available(SearchMode.VECTOR) is True
        assert context.is_mode_available(SearchMode.LAZY) is True
        assert context.is_mode_available(SearchMode.GRAPHRAG) is False


class TestQueryRouterConfig:
    """QueryRouterConfig のテスト"""
    
    def test_config_defaults(self):
        """デフォルト設定テスト"""
        from monjyu.query.router.router import QueryRouterConfig
        
        config = QueryRouterConfig()
        
        assert config.use_llm_classification is False
        assert config.llm_confidence_threshold == 0.7
        assert config.enable_fallback is True


class TestQueryRouterRuleBased:
    """ルールベース分類のテスト"""
    
    @pytest.fixture
    def router(self):
        """ルーター作成"""
        from monjyu.query.router import QueryRouter
        return QueryRouter()
    
    # ==== サーベイ分類テスト ====
    
    @pytest.mark.asyncio
    async def test_classify_survey_ja(self, router):
        """日本語サーベイクエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "この分野の研究動向は？",
            "深層学習のトレンドを教えて",
            "NLPの概要を説明して",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.SURVEY, f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_classify_survey_en(self, router):
        """英語サーベイクエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "What is the state-of-the-art in NLP?",
            "Give me an overview of transformers",
            "Survey of deep learning methods",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.SURVEY, f"Failed for: {query}"
    
    # ==== 探索分類テスト ====
    
    @pytest.mark.asyncio
    async def test_classify_exploration_ja(self, router):
        """日本語探索クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "Transformerを使った手法は？",
            "BERTの実装方法を教えて",
            "アテンション機構のアプローチについて",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.EXPLORATION, f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_classify_exploration_en(self, router):
        """英語探索クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "How to implement attention mechanism?",
            "What is BERT?",
            "Methods for text classification",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.EXPLORATION, f"Failed for: {query}"
    
    # ==== 比較分類テスト ====
    
    @pytest.mark.asyncio
    async def test_classify_comparison_ja(self, router):
        """日本語比較クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "BERTとGPTの違いは？",
            "CNNとRNNの比較",
            "どちらが優れている？",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.COMPARISON, f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_classify_comparison_en(self, router):
        """英語比較クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "Compare BERT vs GPT",
            "What is the difference between CNN and RNN?",
            "Which model is better for NER?",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.COMPARISON, f"Failed for: {query}"
    
    # ==== 事実分類テスト ====
    
    @pytest.mark.asyncio
    async def test_classify_factoid_ja(self, router):
        """日本語事実クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "パラメータ数はいくつ？",
            "データセットのサイズは？",
            "精度は何%？",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.FACTOID, f"Failed for: {query}"
    
    @pytest.mark.asyncio
    async def test_classify_factoid_en(self, router):
        """英語事実クエリ分類"""
        from monjyu.query.router import QueryType
        
        queries = [
            "How many parameters does GPT-3 have?",
            "What is the accuracy on ImageNet?",
        ]
        
        for query in queries:
            query_type, confidence = router.classify_sync(query)
            assert query_type == QueryType.FACTOID, f"Failed for: {query}"


class TestQueryRouterRouting:
    """ルーティングロジックのテスト"""
    
    @pytest.fixture
    def router(self):
        """ルーター作成"""
        from monjyu.query.router import QueryRouter
        return QueryRouter()
    
    @pytest.mark.asyncio
    async def test_route_survey_to_graphrag(self, router):
        """サーベイ→GRAPHRAG"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            index_level=3,
            available_modes=[
                SearchMode.VECTOR, SearchMode.LAZY,
                SearchMode.GRAPHRAG, SearchMode.HYBRID
            ],
        )
        
        decision = await router.route("研究動向は？", context)
        
        assert decision.mode == SearchMode.GRAPHRAG
    
    @pytest.mark.asyncio
    async def test_route_exploration_to_lazy(self, router):
        """探索→LAZY"""
        from monjyu.query.router import SearchMode
        
        decision = await router.route("Transformerを使った手法は？")
        
        assert decision.mode == SearchMode.LAZY
    
    @pytest.mark.asyncio
    async def test_route_comparison_to_hybrid(self, router):
        """比較→HYBRID"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            index_level=2,
            available_modes=[
                SearchMode.VECTOR, SearchMode.LAZY, SearchMode.HYBRID
            ],
        )
        
        decision = await router.route("BERTとGPTの違いは？", context)
        
        assert decision.mode == SearchMode.HYBRID
    
    @pytest.mark.asyncio
    async def test_route_factoid_to_vector(self, router):
        """事実→VECTOR"""
        from monjyu.query.router import SearchMode
        
        decision = await router.route("精度は何%？")
        
        assert decision.mode == SearchMode.VECTOR


class TestQueryRouterFallback:
    """フォールバックのテスト"""
    
    @pytest.fixture
    def router(self):
        """ルーター作成"""
        from monjyu.query.router import QueryRouter
        return QueryRouter()
    
    @pytest.mark.asyncio
    async def test_fallback_graphrag_to_lazy(self, router):
        """GRAPHRAG→LAZYフォールバック（Level 1）"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            index_level=1,
            available_modes=[SearchMode.VECTOR, SearchMode.LAZY],
        )
        
        decision = await router.route("研究動向は？", context)
        
        # GRAPHRAGは利用不可なのでLAZYにフォールバック
        assert decision.mode == SearchMode.LAZY
        assert decision.fallback_mode == SearchMode.GRAPHRAG
    
    @pytest.mark.asyncio
    async def test_fallback_to_vector(self, router):
        """VECTORフォールバック（Level 0）"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            index_level=0,
            available_modes=[SearchMode.VECTOR],
        )
        
        decision = await router.route("手法について教えて", context)
        
        # LAZYは利用不可なのでVECTORにフォールバック
        assert decision.mode == SearchMode.VECTOR
    
    @pytest.mark.asyncio
    async def test_user_preference_override(self, router):
        """ユーザー指定モード優先"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            user_preference=SearchMode.VECTOR,
            available_modes=[SearchMode.VECTOR, SearchMode.LAZY],
        )
        
        decision = await router.route("研究動向は？", context)
        
        # ユーザー指定が優先
        assert decision.mode == SearchMode.VECTOR
        assert decision.confidence == 1.0


class TestQueryRouterLLMClassification:
    """LLM分類のテスト"""
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLM"""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=json.dumps({
            "type": "SURVEY",
            "confidence": 0.9,
            "reasoning": "Research trend question"
        }))
        return mock
    
    @pytest.mark.asyncio
    async def test_llm_classification(self, mock_llm):
        """LLM分類テスト"""
        from monjyu.query.router import (
            QueryRouter,
            QueryRouterConfig,
            QueryType,
        )
        
        config = QueryRouterConfig(
            use_llm_classification=True,
            llm_confidence_threshold=0.9,  # 高い閾値で必ずLLMが呼ばれるように
        )
        router = QueryRouter(mock_llm, config)
        
        # ルールベースで確信度が0.9未満になるクエリ
        decision = await router.route("これについて教えて")
        
        # LLMが呼ばれたことを確認
        assert mock_llm.generate.called


class TestQueryRouterParams:
    """パラメータ決定のテスト"""
    
    @pytest.fixture
    def router(self):
        """ルーター作成"""
        from monjyu.query.router import QueryRouter
        return QueryRouter()
    
    @pytest.mark.asyncio
    async def test_vector_params(self, router):
        """VECTORモードパラメータ"""
        decision = await router.route("精度は何%？")
        
        assert "top_k" in decision.params
    
    @pytest.mark.asyncio
    async def test_lazy_params(self, router):
        """LAZYモードパラメータ"""
        decision = await router.route("Transformerを使った手法は？")
        
        assert "max_iterations" in decision.params
    
    @pytest.mark.asyncio
    async def test_hybrid_params(self, router):
        """HYBRIDモードパラメータ"""
        from monjyu.query.router import SearchMode, RoutingContext
        
        context = RoutingContext(
            index_level=2,
            available_modes=[
                SearchMode.VECTOR, SearchMode.LAZY, SearchMode.HYBRID
            ],
        )
        
        decision = await router.route("比較して", context)
        
        assert "fusion_method" in decision.params
        assert "engines" in decision.params


class TestCreateRouter:
    """create_routerファクトリ関数のテスト"""
    
    def test_create_router_default(self):
        """デフォルト作成"""
        from monjyu.query.router import create_router
        
        router = create_router()
        
        assert router is not None
        assert router.config.use_llm_classification is False
    
    def test_create_router_with_llm(self):
        """LLM付き作成"""
        from monjyu.query.router import create_router, SearchMode
        
        mock_llm = AsyncMock()
        
        router = create_router(
            llm_client=mock_llm,
            use_llm=True,
            default_mode=SearchMode.VECTOR,
        )
        
        assert router.llm_client is mock_llm
        assert router.config.use_llm_classification is True


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_routing_workflow(self):
        """完全ワークフローテスト"""
        from monjyu.query.router import (
            QueryRouter,
            RoutingContext,
            SearchMode,
            QueryType,
        )
        
        router = QueryRouter()
        
        # テストクエリセット
        test_cases = [
            ("研究動向を教えて", QueryType.SURVEY),
            ("BERTの実装方法は？", QueryType.EXPLORATION),
            ("CNNとRNNの違いは？", QueryType.COMPARISON),
            ("パラメータ数はいくつ？", QueryType.FACTOID),
        ]
        
        for query, expected_type in test_cases:
            decision = await router.route(query)
            
            assert decision.query_type == expected_type, f"Failed for: {query}"
            assert decision.confidence > 0.5
            assert decision.reasoning != ""
    
    @pytest.mark.asyncio
    async def test_context_aware_routing(self):
        """コンテキスト考慮ルーティング"""
        from monjyu.query.router import (
            QueryRouter,
            RoutingContext,
            SearchMode,
        )
        
        router = QueryRouter()
        
        # Level 3では全モード利用可能
        context_full = RoutingContext(
            index_level=3,
            available_modes=[
                SearchMode.VECTOR, SearchMode.LAZY,
                SearchMode.GRAPHRAG, SearchMode.HYBRID,
            ],
        )
        
        decision = await router.route("研究動向は？", context_full)
        assert decision.mode == SearchMode.GRAPHRAG
        
        # Level 1ではフォールバック
        context_limited = RoutingContext(
            index_level=1,
            available_modes=[SearchMode.VECTOR, SearchMode.LAZY],
        )
        
        decision = await router.route("研究動向は？", context_limited)
        assert decision.mode == SearchMode.LAZY
        assert decision.fallback_mode == SearchMode.GRAPHRAG
