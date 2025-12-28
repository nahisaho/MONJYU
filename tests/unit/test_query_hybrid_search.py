# HybridSearch (Query Layer) Unit Tests
"""
REQ-QRY-005: HybridSearch - Query層単体テスト
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from dataclasses import dataclass

from monjyu.query.hybrid_search import (
    FusionMethod,
    SearchMethod,
    HybridSearchConfig,
    HybridSearchHit,
    HybridSearchResult,
    MethodSearchResult,
    HybridSearch,
    ResultMerger,
    create_hybrid_search,
)


# ========== Fixtures ==========


@pytest.fixture
def sample_vector_hits():
    """ベクトル検索結果のサンプル"""
    return [
        HybridSearchHit(
            chunk_id="chunk_001",
            score=0.95,
            content="Vector result 1",
            sources=["vector"],
        ),
        HybridSearchHit(
            chunk_id="chunk_002",
            score=0.85,
            content="Vector result 2",
            sources=["vector"],
        ),
        HybridSearchHit(
            chunk_id="chunk_003",
            score=0.75,
            content="Vector result 3",
            sources=["vector"],
        ),
    ]


@pytest.fixture
def sample_lazy_hits():
    """Lazy検索結果のサンプル"""
    return [
        HybridSearchHit(
            chunk_id="chunk_002",  # Vectorと重複
            score=0.90,
            content="Lazy result 1 (same as vector chunk_002)",
            sources=["lazy"],
        ),
        HybridSearchHit(
            chunk_id="chunk_004",
            score=0.80,
            content="Lazy result 2",
            sources=["lazy"],
        ),
        HybridSearchHit(
            chunk_id="chunk_005",
            score=0.70,
            content="Lazy result 3",
            sources=["lazy"],
        ),
    ]


@pytest.fixture
def method_result_vector(sample_vector_hits):
    """ベクトル検索のMethodSearchResult"""
    return MethodSearchResult(
        method=SearchMethod.VECTOR,
        hits=sample_vector_hits,
        success=True,
        search_time_ms=50.0,
    )


@pytest.fixture
def method_result_lazy(sample_lazy_hits):
    """Lazy検索のMethodSearchResult"""
    return MethodSearchResult(
        method=SearchMethod.LAZY,
        hits=sample_lazy_hits,
        success=True,
        search_time_ms=100.0,
    )


@pytest.fixture
def method_result_failed():
    """失敗したMethodSearchResult"""
    return MethodSearchResult(
        method=SearchMethod.GLOBAL,
        hits=[],
        success=False,
        error="GlobalSearch not configured",
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
        assert SearchMethod.GLOBAL.value == "global"
        assert SearchMethod.LOCAL.value == "local"


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
    
    def test_to_dict(self):
        """辞書変換"""
        config = HybridSearchConfig()
        data = config.to_dict()
        
        assert "methods" in data
        assert "fusion" in data
        assert data["fusion"] == "rrf"
    
    def test_from_dict(self):
        """辞書から復元"""
        data = {
            "methods": ["vector", "global"],
            "fusion": "weighted",
            "top_k": 15,
        }
        config = HybridSearchConfig.from_dict(data)
        
        assert SearchMethod.VECTOR in config.methods
        assert SearchMethod.GLOBAL in config.methods
        assert config.fusion == FusionMethod.WEIGHTED
        assert config.top_k == 15


# ========== Test: HybridSearchHit ==========


class TestHybridSearchHit:
    """HybridSearchHitのテスト"""
    
    def test_to_dict(self):
        """辞書変換"""
        hit = HybridSearchHit(
            chunk_id="chunk_001",
            score=0.95,
            content="Test content",
            sources=["vector", "lazy"],
            paper_id="paper_001",
        )
        
        data = hit.to_dict()
        
        assert data["chunk_id"] == "chunk_001"
        assert data["score"] == 0.95
        assert data["sources"] == ["vector", "lazy"]
        assert data["paper_id"] == "paper_001"
    
    def test_from_dict(self):
        """辞書から復元"""
        data = {
            "chunk_id": "chunk_001",
            "score": 0.85,
            "content": "Test",
            "sources": ["vector"],
        }
        hit = HybridSearchHit.from_dict(data)
        
        assert hit.chunk_id == "chunk_001"
        assert hit.score == 0.85


# ========== Test: MethodSearchResult ==========


class TestMethodSearchResult:
    """MethodSearchResultのテスト"""
    
    def test_success_result(self, sample_vector_hits):
        """成功結果"""
        result = MethodSearchResult(
            method=SearchMethod.VECTOR,
            hits=sample_vector_hits,
            success=True,
            search_time_ms=50.0,
        )
        
        assert result.method == SearchMethod.VECTOR
        assert len(result.hits) == 3
        assert result.success is True
        assert result.error is None
    
    def test_failed_result(self):
        """失敗結果"""
        result = MethodSearchResult(
            method=SearchMethod.LAZY,
            hits=[],
            success=False,
            error="Connection timeout",
        )
        
        assert result.success is False
        assert result.error == "Connection timeout"


# ========== Test: HybridSearchResult ==========


class TestHybridSearchResult:
    """HybridSearchResultのテスト"""
    
    def test_to_dict(self, sample_vector_hits, method_result_vector):
        """辞書変換"""
        result = HybridSearchResult(
            query="test query",
            hits=sample_vector_hits,
            method_results=[method_result_vector],
            fusion_method=FusionMethod.RRF,
            total_time_ms=100.0,
        )
        
        data = result.to_dict()
        
        assert data["query"] == "test query"
        assert len(data["hits"]) == 3
        assert data["fusion_method"] == "rrf"
    
    def test_success_count(self, method_result_vector, method_result_lazy, method_result_failed):
        """成功カウント"""
        result = HybridSearchResult(
            query="test",
            hits=[],
            method_results=[method_result_vector, method_result_lazy, method_result_failed],
        )
        
        assert result.success_count == 2
        assert result.failed_methods == [SearchMethod.GLOBAL]


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
        
        # chunk_002は両方に出現するので上位に
        assert len(merged) == 5
        assert "chunk_002" in [h.chunk_id for h in merged[:3]]
        
        # ソース確認
        assert "vector" in sources["chunk_002"]
        assert "lazy" in sources["chunk_002"]
    
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
    
    def test_max_fusion(self, method_result_vector, method_result_lazy):
        """最大スコア融合"""
        config = HybridSearchConfig(fusion=FusionMethod.MAX)
        merger = ResultMerger(config)
        
        merged, _ = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        assert len(merged) == 5
        # chunk_001はvectorのみで0.95
        chunk_001 = next(h for h in merged if h.chunk_id == "chunk_001")
        assert chunk_001.score == 0.95
    
    def test_combsum_fusion(self, method_result_vector, method_result_lazy):
        """CombSUM融合"""
        config = HybridSearchConfig(fusion=FusionMethod.COMBSUM)
        merger = ResultMerger(config)
        
        merged, _ = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        # chunk_002: 0.85 + 0.90 = 1.75
        chunk_002 = next(h for h in merged if h.chunk_id == "chunk_002")
        assert abs(chunk_002.score - 1.75) < 0.01
    
    def test_combmnz_fusion(self, method_result_vector, method_result_lazy):
        """CombMNZ融合"""
        config = HybridSearchConfig(fusion=FusionMethod.COMBMNZ)
        merger = ResultMerger(config)
        
        merged, _ = merger.merge(
            [method_result_vector, method_result_lazy],
            top_k=5,
        )
        
        # chunk_002: (0.85 + 0.90) * 2 = 3.5
        chunk_002 = next(h for h in merged if h.chunk_id == "chunk_002")
        assert abs(chunk_002.score - 3.5) < 0.01
    
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
        for chunk_id, src_list in sources.items():
            assert "global" not in src_list
    
    def test_empty_results(self):
        """空の結果"""
        config = HybridSearchConfig()
        merger = ResultMerger(config)
        
        merged, sources = merger.merge([], top_k=10)
        
        assert merged == []
        assert sources == {}


# ========== Test: HybridSearch ==========


class TestHybridSearch:
    """HybridSearchのテスト"""
    
    def test_init_default(self):
        """デフォルト初期化"""
        search = HybridSearch()
        
        assert search.vector_search is None
        assert search.lazy_search is None
        assert search.config is not None
    
    def test_init_with_config(self):
        """設定付き初期化"""
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            fusion=FusionMethod.WEIGHTED,
        )
        search = HybridSearch(config=config)
        
        assert search.config.methods == [SearchMethod.VECTOR]
        assert search.config.fusion == FusionMethod.WEIGHTED
    
    @pytest.mark.asyncio
    async def test_search_no_engines(self):
        """エンジンなしで検索（エラー）"""
        search = HybridSearch()
        
        result = await search.search("test query")
        
        # 全て失敗
        assert all(not r.success for r in result.method_results)
        assert len(result.hits) == 0
    
    @pytest.mark.asyncio
    async def test_search_with_vector_engine(self):
        """ベクトルエンジンのみで検索"""
        # モックエンジン
        @dataclass
        class MockHit:
            chunk_id: str
            score: float
            content: str
            metadata: dict = None
            paper_id: str = None
            paper_title: str = None
            section_type: str = None
        
        @dataclass
        class MockResult:
            hits: list
        
        mock_vector = AsyncMock()
        mock_vector.search.return_value = MockResult(
            hits=[
                MockHit(chunk_id="c1", score=0.9, content="Test 1"),
                MockHit(chunk_id="c2", score=0.8, content="Test 2"),
            ]
        )
        
        config = HybridSearchConfig(methods=[SearchMethod.VECTOR])
        search = HybridSearch(config=config, vector_search=mock_vector)
        
        result = await search.search("test query")
        
        assert len(result.method_results) == 1
        assert result.method_results[0].method == SearchMethod.VECTOR
        assert result.method_results[0].success is True
        assert len(result.hits) == 2
    
    @pytest.mark.asyncio
    async def test_search_parallel_mode(self):
        """並列検索モード"""
        @dataclass
        class MockHit:
            chunk_id: str
            score: float
            content: str
            metadata: dict = None
            paper_id: str = None
            paper_title: str = None
            section_type: str = None
        
        @dataclass
        class MockResult:
            hits: list
        
        mock_vector = AsyncMock()
        mock_vector.search.return_value = MockResult(
            hits=[MockHit(chunk_id="c1", score=0.9, content="Test")]
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            parallel=True,
        )
        search = HybridSearch(config=config, vector_search=mock_vector)
        
        result = await search.search("test")
        
        assert result.method_results[0].success is True
    
    @pytest.mark.asyncio
    async def test_search_sequential_mode(self):
        """逐次検索モード"""
        @dataclass
        class MockHit:
            chunk_id: str
            score: float
            content: str
            metadata: dict = None
            paper_id: str = None
            paper_title: str = None
            section_type: str = None
        
        @dataclass
        class MockResult:
            hits: list
        
        mock_vector = AsyncMock()
        mock_vector.search.return_value = MockResult(
            hits=[MockHit(chunk_id="c1", score=0.9, content="Test")]
        )
        
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR],
            parallel=False,
        )
        search = HybridSearch(config=config, vector_search=mock_vector)
        
        result = await search.search("test")
        
        assert result.method_results[0].success is True
    
    @pytest.mark.asyncio
    async def test_search_sync(self):
        """同期版検索（asyncioで実行）"""
        @dataclass
        class MockHit:
            chunk_id: str
            score: float
            content: str
            metadata: dict = None
            paper_id: str = None
            paper_title: str = None
            section_type: str = None
        
        @dataclass
        class MockResult:
            hits: list
        
        mock_vector = AsyncMock()
        mock_vector.search.return_value = MockResult(
            hits=[MockHit(chunk_id="c1", score=0.9, content="Test")]
        )
        
        config = HybridSearchConfig(methods=[SearchMethod.VECTOR])
        search = HybridSearch(config=config, vector_search=mock_vector)
        
        # 非同期版をテスト
        result = await search.search("test")
        
        assert len(result.hits) == 1


# ========== Test: Factory ==========


class TestCreateHybridSearch:
    """create_hybrid_search関数のテスト"""
    
    def test_default_creation(self):
        """デフォルト作成"""
        search = create_hybrid_search()
        
        assert isinstance(search, HybridSearch)
        assert SearchMethod.VECTOR in search.config.methods
        assert SearchMethod.LAZY in search.config.methods
    
    def test_custom_methods(self):
        """カスタムメソッド"""
        search = create_hybrid_search(methods=["vector", "global"])
        
        assert SearchMethod.VECTOR in search.config.methods
        assert SearchMethod.GLOBAL in search.config.methods
        assert SearchMethod.LAZY not in search.config.methods
    
    def test_custom_fusion(self):
        """カスタム融合方式"""
        search = create_hybrid_search(fusion="weighted")
        
        assert search.config.fusion == FusionMethod.WEIGHTED
    
    def test_invalid_method_ignored(self):
        """無効なメソッドは無視"""
        search = create_hybrid_search(methods=["vector", "invalid"])
        
        assert SearchMethod.VECTOR in search.config.methods
        assert len(search.config.methods) == 1
    
    def test_invalid_fusion_fallback(self):
        """無効な融合方式はRRFにフォールバック"""
        search = create_hybrid_search(fusion="invalid")
        
        assert search.config.fusion == FusionMethod.RRF
    
    def test_kwargs_passed(self):
        """追加引数が渡される"""
        search = create_hybrid_search(
            top_k=20,
            rrf_k=100,
            parallel=False,
        )
        
        assert search.config.top_k == 20
        assert search.config.rrf_k == 100
        assert search.config.parallel is False


# ========== Test: Module Exports ==========


class TestModuleExports:
    """モジュールエクスポートのテスト"""
    
    def test_all_exports_available(self):
        """全エクスポートが利用可能"""
        from monjyu.query.hybrid_search import (
            FusionMethod,
            SearchMethod,
            HybridSearchConfig,
            HybridSearchHit,
            HybridSearchResult,
            MethodSearchResult,
            HybridSearch,
            ResultMerger,
            create_hybrid_search,
            VectorSearchProtocol,
            LazySearchProtocol,
            GlobalSearchProtocol,
            LocalSearchProtocol,
        )
        
        assert FusionMethod is not None
        assert SearchMethod is not None
        assert HybridSearch is not None
        assert create_hybrid_search is not None


# ========== Test: Integration Scenarios ==========


class TestIntegrationScenarios:
    """統合シナリオテスト"""
    
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
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """一部エンジン失敗時のグレースフルデグラデーション"""
        @dataclass
        class MockHit:
            chunk_id: str
            score: float
            content: str
            metadata: dict = None
            paper_id: str = None
            paper_title: str = None
            section_type: str = None
        
        @dataclass
        class MockResult:
            hits: list
        
        mock_vector = AsyncMock()
        mock_vector.search.return_value = MockResult(
            hits=[MockHit(chunk_id="c1", score=0.9, content="Test")]
        )
        
        # Lazyエンジンは未設定
        config = HybridSearchConfig(
            methods=[SearchMethod.VECTOR, SearchMethod.LAZY],
        )
        search = HybridSearch(
            config=config,
            vector_search=mock_vector,
            lazy_search=None,
        )
        
        result = await search.search("test query")
        
        # Vectorは成功、Lazyは失敗
        vector_result = next(r for r in result.method_results if r.method == SearchMethod.VECTOR)
        lazy_result = next(r for r in result.method_results if r.method == SearchMethod.LAZY)
        
        assert vector_result.success is True
        assert lazy_result.success is False
        
        # 結果はVectorのみ
        assert len(result.hits) == 1
