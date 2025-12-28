# Hybrid Search Engine Unit Tests
"""
REQ-ARC-003: Hybrid GraphRAG - 単体テスト
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from monjyu.search.hybrid import (
    FusionMethod,
    SearchMethod,
    HybridSearchConfig,
    MethodResult,
    HybridSearchResult,
    ResultMerger,
    HybridSearchEngine,
    create_hybrid_engine,
)
from monjyu.search.base import SearchHit, Citation


# ========== Fixtures ==========


@pytest.fixture
def sample_hits_vector():
    """ベクトル検索結果のサンプル"""
    return [
        SearchHit(
            text_unit_id="tu_001",
            document_id="doc_001",
            text="Vector result 1",
            score=0.95,
        ),
        SearchHit(
            text_unit_id="tu_002",
            document_id="doc_001",
            text="Vector result 2",
            score=0.85,
        ),
        SearchHit(
            text_unit_id="tu_003",
            document_id="doc_002",
            text="Vector result 3",
            score=0.75,
        ),
    ]


@pytest.fixture
def sample_hits_lazy():
    """Lazy検索結果のサンプル"""
    return [
        SearchHit(
            text_unit_id="tu_002",
            document_id="doc_001",
            text="Lazy result 1 (same as vector tu_002)",
            score=0.90,
        ),
        SearchHit(
            text_unit_id="tu_004",
            document_id="doc_002",
            text="Lazy result 2",
            score=0.80,
        ),
        SearchHit(
            text_unit_id="tu_005",
            document_id="doc_003",
            text="Lazy result 3",
            score=0.70,
        ),
    ]


@pytest.fixture
def method_result_vector(sample_hits_vector):
    """ベクトル検索のMethodResult"""
    return MethodResult(
        method=SearchMethod.VECTOR,
        hits=sample_hits_vector,
        search_time_ms=50.0,
        success=True,
    )


@pytest.fixture
def method_result_lazy(sample_hits_lazy):
    """Lazy検索のMethodResult"""
    return MethodResult(
        method=SearchMethod.LAZY,
        hits=sample_hits_lazy,
        search_time_ms=100.0,
        success=True,
    )


@pytest.fixture
def method_result_failed():
    """失敗したMethodResult"""
    return MethodResult(
        method=SearchMethod.KEYWORD,
        hits=[],
        search_time_ms=0.0,
        success=False,
        error="Not configured",
    )


# ========== Test: FusionMethod Enum ==========


class TestFusionMethodEnum:
    """FusionMethod列挙型のテスト"""
    
    def test_all_methods_defined(self):
        """全ての融合方式が定義されている"""
        assert FusionMethod.RRF.value == "rrf"
        assert FusionMethod.WEIGHTED.value == "weighted"
        assert FusionMethod.MAX.value == "max"
        assert FusionMethod.COMBSUM.value == "combsum"
        assert FusionMethod.COMBMNZ.value == "combmnz"
    
    def test_from_string(self):
        """文字列から変換できる"""
        assert FusionMethod("rrf") == FusionMethod.RRF
        assert FusionMethod("weighted") == FusionMethod.WEIGHTED


class TestSearchMethodEnum:
    """SearchMethod列挙型のテスト"""
    
    def test_all_methods_defined(self):
        """全ての検索方式が定義されている"""
        assert SearchMethod.VECTOR.value == "vector"
        assert SearchMethod.LAZY.value == "lazy"
        assert SearchMethod.KEYWORD.value == "keyword"


# ========== Test: HybridSearchConfig ==========


class TestHybridSearchConfig:
    """HybridSearchConfig設定のテスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        config = HybridSearchConfig()
        
        assert SearchMethod.VECTOR in config.methods
        assert SearchMethod.LAZY in config.methods
        assert config.fusion == FusionMethod.RRF
        assert config.rrf_k == 60
        assert config.top_k == 10
        assert config.parallel is True
        assert config.synthesize is True
    
    def test_custom_config(self):
        """カスタム設定"""
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            fusion=FusionMethod.WEIGHTED,
            rrf_k=100,
            top_k=20,
            parallel=False,
        )
        
        assert config.methods == [SearchMethod.VECTOR]
        assert config.fusion == FusionMethod.WEIGHTED
        assert config.rrf_k == 100
        assert config.top_k == 20
        assert config.parallel is False
    
    def test_default_weights(self):
        """デフォルトの重み設定"""
        config = HybridSearchConfig()
        
        assert SearchMethod.VECTOR in config.method_weights
        assert SearchMethod.LAZY in config.method_weights
        assert config.method_weights[SearchMethod.VECTOR] == 0.5
        assert config.method_weights[SearchMethod.LAZY] == 0.5


# ========== Test: MethodResult ==========


class TestMethodResult:
    """MethodResultのテスト"""
    
    def test_success_result(self, sample_hits_vector):
        """成功結果"""
        result = MethodResult(
            method=SearchMethod.VECTOR,
            hits=sample_hits_vector,
            search_time_ms=50.0,
            success=True,
        )
        
        assert result.method == SearchMethod.VECTOR
        assert len(result.hits) == 3
        assert result.success is True
        assert result.error is None
    
    def test_failed_result(self):
        """失敗結果"""
        result = MethodResult(
            method=SearchMethod.LAZY,
            hits=[],
            search_time_ms=0.0,
            success=False,
            error="Connection timeout",
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"
        assert len(result.hits) == 0


# ========== Test: HybridSearchResult ==========


class TestHybridSearchResult:
    """HybridSearchResultのテスト"""
    
    def test_to_dict(self, sample_hits_vector, method_result_vector, method_result_lazy):
        """辞書変換"""
        result = HybridSearchResult(
            query="test query",
            merged_hits=sample_hits_vector,
            method_results=[method_result_vector, method_result_lazy],
            answer="Test answer",
            citations=[],
            total_time_ms=150.0,
            fusion_method=FusionMethod.RRF,
        )
        
        data = result.to_dict()
        
        assert data["query"] == "test query"
        assert data["answer"] == "Test answer"
        assert len(data["merged_hits"]) == 3
        assert len(data["method_results"]) == 2
        assert data["fusion_method"] == "rrf"
        assert data["total_time_ms"] == 150.0


# ========== Test: ResultMerger ==========


class TestResultMerger:
    """ResultMergerのテスト"""
    
    def test_rrf_fusion(self, method_result_vector, method_result_lazy):
        """RRF融合"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF, rrf_k=60)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        # tu_002は両方に出現するので上位に
        assert len(merged) == 5
        assert "tu_002" in [h.text_unit_id for h in merged[:3]]
        
        # ソース確認
        assert "vector" in sources["tu_002"]
        assert "lazy" in sources["tu_002"]
        assert "vector" in sources["tu_001"]
        assert "lazy" not in sources["tu_001"]
    
    def test_rrf_score_calculation(self, method_result_vector, method_result_lazy):
        """RRFスコア計算の検証"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF, rrf_k=60)
        merger = ResultMerger(config)
        
        merged, _ = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=10,
        )
        
        # tu_002のRRFスコア: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
        tu_002_hit = next(h for h in merged if h.text_unit_id == "tu_002")
        expected_score = 1.0 / (60 + 2) + 1.0 / (60 + 1)
        assert abs(tu_002_hit.score - expected_score) < 0.0001
    
    def test_weighted_fusion(self, method_result_vector, method_result_lazy):
        """重み付け融合"""
        config = HybridSearchConfig(
            fusion=FusionMethod.WEIGHTED,
            method_weights={
                SearchMethod.VECTOR: 0.7,
                SearchMethod.LAZY: 0.3,
            },
        )
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        assert len(merged) == 5
        # スコアが設定されている
        for hit in merged:
            assert hit.score > 0
    
    def test_max_fusion(self, method_result_vector, method_result_lazy):
        """最大スコア融合"""
        config = HybridSearchConfig(fusion=FusionMethod.MAX)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        assert len(merged) == 5
        # tu_001はvectorのみで0.95
        tu_001_hit = next(h for h in merged if h.text_unit_id == "tu_001")
        assert tu_001_hit.score == 0.95
    
    def test_combsum_fusion(self, method_result_vector, method_result_lazy):
        """スコア合計融合"""
        config = HybridSearchConfig(fusion=FusionMethod.COMBSUM)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        assert len(merged) == 5
        # tu_002: 0.85 + 0.90 = 1.75
        tu_002_hit = next(h for h in merged if h.text_unit_id == "tu_002")
        assert abs(tu_002_hit.score - 1.75) < 0.01
    
    def test_combmnz_fusion(self, method_result_vector, method_result_lazy):
        """CombMNZ融合"""
        config = HybridSearchConfig(fusion=FusionMethod.COMBMNZ)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        assert len(merged) == 5
        # tu_002: (0.85 + 0.90) * 2 = 3.5
        tu_002_hit = next(h for h in merged if h.text_unit_id == "tu_002")
        assert abs(tu_002_hit.score - 3.5) < 0.01
    
    def test_handles_failed_results(self, method_result_vector, method_result_failed):
        """失敗した結果をスキップ"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge(
            [method_result_vector, method_result_failed],
            top_k=5,
        )
        
        # vectorの結果のみ
        assert len(merged) == 3
        for hit_id, src_list in sources.items():
            assert "keyword" not in src_list
    
    def test_empty_results(self):
        """空の結果"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF)
        merger = ResultMerger(config)
        
        merged, sources = merger.merge([], top_k=10)
        
        assert merged == []
        assert sources == {}
    
    def test_top_k_limit(self, method_result_vector, method_result_lazy):
        """top_k制限"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF)
        merger = ResultMerger(config)
        
        merged, _ = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=2,
        )
        
        assert len(merged) == 2


# ========== Test: HybridSearchEngine ==========


class TestHybridSearchEngine:
    """HybridSearchEngineのテスト"""
    
    def test_init_default(self):
        """デフォルト初期化"""
        engine = HybridSearchEngine()
        
        assert engine.vector_engine is None
        assert engine.lazy_engine is None
        assert engine.llm_client is None
        assert engine.config is not None
    
    def test_init_with_config(self):
        """設定付き初期化"""
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            fusion=FusionMethod.WEIGHTED,
        )
        engine = HybridSearchEngine(config=config)
        
        assert engine.config.methods == [SearchMethod.VECTOR]
        assert engine.config.fusion == FusionMethod.WEIGHTED
    
    @pytest.mark.asyncio
    async def test_search_no_engines(self):
        """エンジンなしで検索（エラー）"""
        engine = HybridSearchEngine()
        
        result = await engine.search("test query")
        
        # 全て失敗
        assert all(not r.success for r in result.method_results)
        assert len(result.merged_hits) == 0
    
    @pytest.mark.asyncio
    async def test_search_with_vector_engine(self, sample_hits_vector):
        """ベクトルエンジンのみで検索"""
        # モックエンジン
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            config=config,
        )
        
        result = await engine.search("test query")
        
        assert len(result.method_results) == 1
        assert result.method_results[0].method == SearchMethod.VECTOR
        assert result.method_results[0].success is True
        assert len(result.merged_hits) == 3
    
    @pytest.mark.asyncio
    async def test_search_parallel(self, sample_hits_vector, sample_hits_lazy):
        """並列検索"""
        # モックエンジン
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        mock_lazy = AsyncMock()
        mock_lazy.search_async.return_value = Mock(
            claims=[
                Mock(
                    source_text_unit_id=h.text_unit_id,
                    source_document_id=h.document_id,
                    text=h.text,
                    confidence=h.score,
                )
                for h in sample_hits_lazy
            ]
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
            parallel=True,
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            lazy_engine=mock_lazy,
            config=config,
        )
        
        result = await engine.search("test query")
        
        assert len(result.method_results) == 2
        # tu_002は両方に出現
        hit_ids = [h.text_unit_id for h in result.merged_hits]
        assert "tu_002" in hit_ids
    
    @pytest.mark.asyncio
    async def test_search_sequential(self, sample_hits_vector):
        """逐次検索"""
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            parallel=False,
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            config=config,
        )
        
        result = await engine.search("test query")
        
        assert len(result.method_results) == 1
        assert result.method_results[0].success is True
    
    @pytest.mark.asyncio
    async def test_search_with_synthesis(self, sample_hits_vector):
        """回答合成付き検索"""
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        mock_llm = AsyncMock()
        mock_llm.generate_async.return_value = "Synthesized answer"
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            synthesize=True,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            llm_client=mock_llm,
            config=config,
        )
        
        result = await engine.search("test query")
        
        assert result.answer == "Synthesized answer"
        assert len(result.citations) == 3
    
    def test_search_sync(self, sample_hits_vector):
        """同期検索"""
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            config=config,
        )
        
        result = engine.search_sync("test query")
        
        assert len(result.merged_hits) == 3
    
    @pytest.mark.asyncio
    async def test_search_timeout(self):
        """タイムアウト処理"""
        # 遅いモックエンジン
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(10)
            return Mock(search_results=Mock(hits=[]))
        
        mock_vector = Mock()
        mock_vector.search = slow_search
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            timeout_seconds=0.1,
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            config=config,
        )
        
        result = await engine.search("test query")
        
        # タイムアウトでエラー
        assert not result.method_results[0].success or "Timeout" in str(result.method_results[0].error)


# ========== Test: Factory Function ==========


class TestCreateHybridEngine:
    """create_hybrid_engine関数のテスト"""
    
    def test_default_creation(self):
        """デフォルト作成"""
        engine = create_hybrid_engine()
        
        assert isinstance(engine, HybridSearchEngine)
        assert SearchMethod.VECTOR in engine.config.methods
        assert SearchMethod.LAZY in engine.config.methods
    
    def test_custom_methods(self):
        """カスタムメソッド"""
        engine = create_hybrid_engine(
            methods=["vector", "keyword"],
        )
        
        assert SearchMethod.VECTOR in engine.config.methods
        assert SearchMethod.KEYWORD in engine.config.methods
        assert SearchMethod.LAZY not in engine.config.methods
    
    def test_custom_fusion(self):
        """カスタム融合方式"""
        engine = create_hybrid_engine(fusion="weighted")
        
        assert engine.config.fusion == FusionMethod.WEIGHTED
    
    def test_invalid_method_ignored(self):
        """無効なメソッドは無視"""
        engine = create_hybrid_engine(methods=["vector", "invalid"])
        
        assert SearchMethod.VECTOR in engine.config.methods
        assert len(engine.config.methods) == 1
    
    def test_invalid_fusion_fallback(self):
        """無効な融合方式はRRFにフォールバック"""
        engine = create_hybrid_engine(fusion="invalid")
        
        assert engine.config.fusion == FusionMethod.RRF
    
    def test_kwargs_passed(self):
        """追加引数が渡される"""
        engine = create_hybrid_engine(
            top_k=20,
            rrf_k=100,
            parallel=False,
        )
        
        assert engine.config.top_k == 20
        assert engine.config.rrf_k == 100
        assert engine.config.parallel is False


# ========== Test: Module Exports ==========


class TestModuleExports:
    """モジュールエクスポートのテスト"""
    
    def test_hybrid_exports(self):
        """hybridモジュールのエクスポート"""
        from monjyu.search.hybrid import (
            FusionMethod,
            SearchMethod,
            HybridSearchConfig,
            MethodResult,
            HybridSearchResult,
            ResultMerger,
            HybridSearchEngine,
            create_hybrid_engine,
        )
        
        assert FusionMethod is not None
        assert SearchMethod is not None
        assert HybridSearchConfig is not None
        assert MethodResult is not None
        assert HybridSearchResult is not None
        assert ResultMerger is not None
        assert HybridSearchEngine is not None
        assert create_hybrid_engine is not None
    
    def test_search_package_exports(self):
        """searchパッケージのエクスポート"""
        from monjyu.search import (
            HybridSearchEngine,
            HybridSearchConfig,
            HybridSearchResult,
            FusionMethod,
            SearchMethod,
            MethodResult,
            ResultMerger,
            create_hybrid_engine,
        )
        
        assert HybridSearchEngine is not None
        assert HybridSearchConfig is not None
        assert create_hybrid_engine is not None


# ========== Test: Integration Scenarios ==========


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
    def test_fusion_order_consistency(self, method_result_vector, method_result_lazy):
        """融合結果の順序一貫性"""
        config = HybridSearchConfig(fusion=FusionMethod.RRF)
        merger = ResultMerger(config)
        
        # 同じ入力で複数回実行
        results = []
        for _ in range(5):
            merged, _ = merger.merge(
                [method_result_vector, method_result_lazy],
                top_k=5,
            )
            results.append([h.text_unit_id for h in merged])
        
        # 全て同じ順序
        for i in range(1, len(results)):
            assert results[i] == results[0]
    
    def test_all_fusion_methods_produce_results(
        self,
        method_result_vector,
        method_result_lazy,
    ):
        """全ての融合方式が結果を生成"""
        for fusion in FusionMethod:
            config = HybridSearchConfig(fusion=fusion)
            merger = ResultMerger(config)
            
            merged, sources = merger.merge(
                [method_result_vector, method_result_lazy],
                top_k=5,
            )
            
            assert len(merged) > 0, f"{fusion} produced no results"
            assert len(sources) > 0, f"{fusion} produced no sources"
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, sample_hits_vector):
        """一部エンジン失敗時のグレースフルデグラデーション"""
        mock_vector = Mock()
        mock_vector.search.return_value = Mock(
            search_results=Mock(hits=sample_hits_vector)
        )
        
        # Lazyエンジンは設定なし（失敗する）
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
            synthesize=False,
        )
        engine = HybridSearchEngine(
            vector_engine=mock_vector,
            lazy_engine=None,  # 未設定
            config=config,
        )
        
        result = await engine.search("test query")
        
        # Vectorは成功、Lazyは失敗
        vector_result = next(r for r in result.method_results if r.method == SearchMethod.VECTOR)
        lazy_result = next(r for r in result.method_results if r.method == SearchMethod.LAZY)
        
        assert vector_result.success is True
        assert lazy_result.success is False
        
        # 結果はVectorのみ
        assert len(result.merged_hits) == 3
