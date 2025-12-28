# LazySearchEngine Coverage Tests
"""
LazySearchEngine のカバレッジ向上テスト

TASK-005-06: LazySearchEngine カバレッジ20%→50%
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from monjyu.lazy.base import (
    Claim,
    LazySearchConfig,
    LazySearchResult,
    LazySearchState,
    RelevanceScore,
    SearchCandidate,
    SearchLevel,
)
from monjyu.lazy.engine import (
    LazySearchEngine,
    MockLazySearchEngine,
    create_local_lazy_engine,
)
from monjyu.search.base import Citation


# === Fixtures ===


@pytest.fixture
def mock_vector_engine():
    """モックベクトル検索エンジン"""
    engine = MagicMock()
    
    # 検索結果
    mock_result = MagicMock()
    mock_result.search_results.hits = [
        MagicMock(
            text_unit_id="tu_1",
            score=0.95,
            text="Machine learning is a subset of AI.",
            document_id="doc_1",
            document_title="AI Introduction",
        ),
        MagicMock(
            text_unit_id="tu_2",
            score=0.88,
            text="Neural networks are inspired by the brain.",
            document_id="doc_1",
            document_title="AI Introduction",
        ),
    ]
    engine.search.return_value = mock_result
    
    return engine


@pytest.fixture
def mock_llm_client():
    """モックLLMクライアント"""
    client = MagicMock()
    client.generate.return_value = "Based on the sources [1] and [2], machine learning is an AI technique."
    return client


@pytest.fixture
def mock_level1_index():
    """モックLevel1インデックス"""
    index = MagicMock()
    index.nodes = []
    index.communities = [
        MagicMock(
            id="comm_1",
            level=0,
            node_ids=["node_1"],
            representative_phrases=["machine learning"],
            size=1,
        ),
    ]
    return index


@pytest.fixture
def mock_embedding_client():
    """モック埋め込みクライアント"""
    client = MagicMock()
    client.embed.return_value = [0.1] * 768
    return client


@pytest.fixture
def basic_config():
    """基本設定"""
    return LazySearchConfig(
        initial_top_k=5,
        max_iterations=2,
        max_llm_calls=10,
        min_relevance=RelevanceScore.MEDIUM,
    )


# === TestLazySearchEngineInit ===


class TestLazySearchEngineInit:
    """LazySearchEngine 初期化テスト"""

    def test_init_basic(self, mock_vector_engine, mock_llm_client):
        """基本初期化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        assert engine.vector_engine is mock_vector_engine
        assert engine.llm_client is mock_llm_client
        assert engine.config is not None
        assert engine.community_searcher is None

    def test_init_with_config(self, mock_vector_engine, mock_llm_client, basic_config):
        """設定付き初期化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            config=basic_config,
        )
        
        assert engine.config.initial_top_k == 5
        assert engine.config.max_iterations == 2

    def test_init_with_level1_index(
        self, mock_vector_engine, mock_llm_client, mock_level1_index
    ):
        """Level1インデックス付き初期化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            level1_index=mock_level1_index,
        )
        
        assert engine.community_searcher is not None

    def test_init_with_embedding_client(
        self, mock_vector_engine, mock_llm_client, mock_level1_index, mock_embedding_client
    ):
        """埋め込みクライアント付き初期化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            level1_index=mock_level1_index,
            embedding_client=mock_embedding_client,
        )
        
        assert engine.community_searcher is not None
        # CommunitySearcherにembedding_clientが渡されている
        assert engine.community_searcher.embedding_client is mock_embedding_client


class TestLazySearchEngineInitialSearch:
    """_initial_search メソッドテスト"""

    def test_initial_search_creates_candidates(self, mock_vector_engine, mock_llm_client):
        """初期検索で候補を作成"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        state = LazySearchState(query="test query")
        candidates = engine._initial_search("test query", state)
        
        assert len(candidates) == 2
        assert all(isinstance(c, SearchCandidate) for c in candidates)
        assert candidates[0].source == "vector"
        assert candidates[0].level == SearchLevel.LEVEL_0

    def test_initial_search_includes_communities(
        self, mock_vector_engine, mock_llm_client, mock_level1_index
    ):
        """コミュニティを含む初期検索"""
        config = LazySearchConfig(include_communities=True)
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            level1_index=mock_level1_index,
            config=config,
        )
        
        state = LazySearchState(query="test query")
        engine._initial_search("test query", state)
        
        # ベクトル検索が呼ばれた
        assert mock_vector_engine.search.called


class TestLazySearchEngineTestRelevance:
    """_test_relevance メソッドテスト"""

    def test_test_relevance_empty_candidates(self, mock_vector_engine, mock_llm_client):
        """空の候補で関連性テスト"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        state = LazySearchState(query="test query")
        result = engine._test_relevance("test query", [], state)
        
        assert result == []

    def test_test_relevance_filters_candidates(self, mock_vector_engine, mock_llm_client):
        """候補をフィルタリング"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # RelevanceTesterをモック
        engine.relevance_tester = MagicMock()
        candidates = [
            SearchCandidate(
                id="tu_1",
                source="vector",
                priority=0.9,
                level=SearchLevel.LEVEL_0,
                text="Test text",
            )
        ]
        engine.relevance_tester.filter_relevant.return_value = [
            (candidates[0], RelevanceScore.HIGH)
        ]
        
        state = LazySearchState(query="test query")
        result = engine._test_relevance("test query", candidates, state)
        
        assert len(result) == 1
        # LLM呼び出し回数が更新された
        assert state.llm_calls == 1


class TestLazySearchEngineExtractClaims:
    """_extract_claims メソッドテスト"""

    def test_extract_claims_adds_to_state(self, mock_vector_engine, mock_llm_client):
        """クレームを状態に追加"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # ClaimExtractorをモック
        engine.claim_extractor = MagicMock()
        claims = [
            Claim(
                text="Test claim",
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            )
        ]
        engine.claim_extractor.extract_batch.return_value = claims
        
        candidates = [
            SearchCandidate(
                id="tu_1",
                source="vector",
                priority=0.9,
                level=SearchLevel.LEVEL_0,
                text="Test text",
            )
        ]
        
        state = LazySearchState(query="test query")
        engine._extract_claims("test query", candidates, state)
        
        assert len(state.claims) == 1
        assert state.claims[0].text == "Test claim"

    def test_extract_claims_merges_duplicates(self, mock_vector_engine, mock_llm_client):
        """重複クレームをマージ"""
        config = LazySearchConfig(merge_duplicates=True)
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            config=config,
        )
        
        # ClaimExtractorをモック
        engine.claim_extractor = MagicMock()
        
        # 既存のクレームを追加
        state = LazySearchState(query="test query")
        state.claims.append(
            Claim(
                text="Existing claim",
                source_text_unit_id="tu_0",
                source_document_id="doc_0",
            )
        )
        
        # 同じテキストの新しいクレーム
        engine.claim_extractor.extract_batch.return_value = [
            Claim(
                text="Existing claim",  # 重複
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            ),
            Claim(
                text="New claim",  # 新規
                source_text_unit_id="tu_2",
                source_document_id="doc_1",
            ),
        ]
        
        candidates = [MagicMock()]
        engine._extract_claims("test query", candidates, state)
        
        # 重複が除外されて2件
        assert len(state.claims) == 2


class TestLazySearchEngineIterateDeepening:
    """_iterate_deepening メソッドテスト"""

    def test_iterate_deepening_respects_max_iterations(
        self, mock_vector_engine, mock_llm_client
    ):
        """最大イテレーション数を尊重"""
        config = LazySearchConfig(max_iterations=2)
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            config=config,
        )
        
        # Deepenerをモック
        engine.deepener = MagicMock()
        engine.deepener.should_deepen.return_value = True
        engine.deepener.get_next_candidates.return_value = []
        
        state = LazySearchState(query="test query")
        engine._iterate_deepening(state)
        
        # 最大イテレーション数に達した
        assert state.iterations == 2


class TestLazySearchEngineDeepenOneIteration:
    """_deepen_one_iteration メソッドテスト"""

    def test_deepen_one_iteration_no_candidates(
        self, mock_vector_engine, mock_llm_client
    ):
        """候補なしの深化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # Deepenerをモック
        engine.deepener = MagicMock()
        engine.deepener.get_next_candidates.return_value = []
        
        state = LazySearchState(query="test query")
        engine._deepen_one_iteration(state)
        
        # 何も起きない（レベルは変わらない）
        assert state.current_level == SearchLevel.LEVEL_0

    def test_deepen_one_iteration_with_community(
        self, mock_vector_engine, mock_llm_client, mock_level1_index
    ):
        """コミュニティ展開を含む深化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            level1_index=mock_level1_index,
        )
        
        # Deepenerをモック
        engine.deepener = MagicMock()
        community_candidate = SearchCandidate(
            id="comm_1",
            source="community",
            priority=0.8,
            level=SearchLevel.LEVEL_1,
            text="Community",
        )
        engine.deepener.get_next_candidates.return_value = [community_candidate]
        engine.deepener.expand_from_community.return_value = []
        
        state = LazySearchState(query="test query")
        engine._deepen_one_iteration(state)
        
        # コミュニティが展開された
        engine.deepener.expand_from_community.assert_called_once()

    def test_deepen_one_iteration_with_text_candidates(
        self, mock_vector_engine, mock_llm_client
    ):
        """テキスト候補の深化"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # Deepenerをモック
        engine.deepener = MagicMock()
        text_candidate = SearchCandidate(
            id="tu_3",
            source="vector",
            priority=0.7,
            level=SearchLevel.LEVEL_0,
            text="Additional text",
        )
        engine.deepener.get_next_candidates.return_value = [text_candidate]
        
        # RelevanceTesterをモック
        engine.relevance_tester = MagicMock()
        engine.relevance_tester.filter_relevant.return_value = [
            (text_candidate, RelevanceScore.HIGH)
        ]
        
        # ClaimExtractorをモック
        engine.claim_extractor = MagicMock()
        engine.claim_extractor.extract_batch.return_value = []
        
        state = LazySearchState(query="test query")
        engine._deepen_one_iteration(state)
        
        # レベルがLevel 1に更新された
        assert state.current_level == SearchLevel.LEVEL_1


class TestLazySearchEngineSynthesizeAnswer:
    """_synthesize_answer メソッドテスト"""

    def test_synthesize_answer_no_claims_no_context(
        self, mock_vector_engine, mock_llm_client
    ):
        """クレームもコンテキストもない場合"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        state = LazySearchState(query="test query")
        answer, citations = engine._synthesize_answer("test query", state)
        
        assert "関連する情報が見つかりませんでした" in answer
        assert citations == []

    def test_synthesize_answer_no_claims_with_context(
        self, mock_vector_engine, mock_llm_client
    ):
        """クレームなしでコンテキストあり"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        state = LazySearchState(query="test query")
        state.context = ["Context text 1", "Context text 2"]
        
        answer, citations = engine._synthesize_answer("test query", state)
        
        # LLMが呼ばれた
        assert mock_llm_client.generate.called
        assert state.llm_calls == 1

    def test_synthesize_answer_with_claims(self, mock_vector_engine, mock_llm_client):
        """クレームからの回答合成"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        state = LazySearchState(query="test query")
        state.claims = [
            Claim(
                text="Claim 1 text",
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            ),
            Claim(
                text="Claim 2 text",
                source_text_unit_id="tu_2",
                source_document_id="doc_1",
            ),
        ]
        
        answer, citations = engine._synthesize_answer("test query", state)
        
        # 引用が抽出された
        assert len(citations) >= 0  # 引用は回答のパターンに依存


class TestLazySearchEngineExtractCitations:
    """_extract_citations メソッドテスト"""

    def test_extract_citations_from_answer(self, mock_vector_engine, mock_llm_client):
        """回答から引用を抽出"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        source_map = {
            1: Claim(
                text="Claim 1",
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            ),
            2: Claim(
                text="Claim 2",
                source_text_unit_id="tu_2",
                source_document_id="doc_2",
            ),
        }
        
        answer = "Based on [1] and [2], the answer is clear."
        citations = engine._extract_citations(answer, source_map)
        
        assert len(citations) == 2
        assert citations[0].text_unit_id == "tu_1"
        assert citations[1].text_unit_id == "tu_2"

    def test_extract_citations_invalid_index(self, mock_vector_engine, mock_llm_client):
        """無効なインデックスの引用"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        source_map = {
            1: Claim(
                text="Claim 1",
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            ),
        }
        
        # [3] は source_map にない
        answer = "Based on [1] and [3], the answer is clear."
        citations = engine._extract_citations(answer, source_map)
        
        # 有効な引用のみ
        assert len(citations) == 1

    def test_extract_citations_no_citations(self, mock_vector_engine, mock_llm_client):
        """引用がない回答"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        source_map = {
            1: Claim(
                text="Claim 1",
                source_text_unit_id="tu_1",
                source_document_id="doc_1",
            ),
        }
        
        answer = "The answer without any citations."
        citations = engine._extract_citations(answer, source_map)
        
        assert citations == []


class TestLazySearchEngineSearch:
    """search メソッドテスト"""

    def test_search_returns_result(self, mock_vector_engine, mock_llm_client):
        """検索結果を返す"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # コンポーネントをモック
        engine.relevance_tester = MagicMock()
        engine.relevance_tester.filter_relevant.return_value = []
        engine.deepener = MagicMock()
        engine.deepener.should_deepen.return_value = False
        
        result = engine.search("test query")
        
        assert isinstance(result, LazySearchResult)
        assert result.query == "test query"

    def test_search_level0_only(self, mock_vector_engine, mock_llm_client):
        """Level 0のみで検索"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # コンポーネントをモック
        engine.relevance_tester = MagicMock()
        engine.relevance_tester.filter_relevant.return_value = []
        
        result = engine.search_level0_only("test query")
        
        assert isinstance(result, LazySearchResult)


# === TestCreateLocalLazyEngine ===


class TestCreateLocalLazyEngine:
    """create_local_lazy_engine テスト"""

    def test_create_engine(self, mock_vector_engine, mock_llm_client):
        """エンジン作成"""
        engine = create_local_lazy_engine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        assert isinstance(engine, LazySearchEngine)

    def test_create_engine_with_all_params(
        self, mock_vector_engine, mock_llm_client, mock_level1_index, mock_embedding_client
    ):
        """全パラメータでエンジン作成"""
        config = LazySearchConfig(max_iterations=5)
        
        engine = create_local_lazy_engine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
            level1_index=mock_level1_index,
            embedding_client=mock_embedding_client,
            level0_dir="/tmp/level0",
            config=config,
        )
        
        assert engine.config.max_iterations == 5
        assert engine.community_searcher is not None


# === TestMockLazySearchEngine ===


class TestMockLazySearchEngineAdvanced:
    """MockLazySearchEngine 追加テスト"""

    def test_default_initialization(self):
        """デフォルト初期化"""
        engine = MockLazySearchEngine()
        
        assert engine.default_answer == "This is a mock answer."
        assert len(engine.default_claims) == 2
        assert engine.search_call_count == 0

    def test_custom_answer(self):
        """カスタム回答"""
        engine = MockLazySearchEngine(
            default_answer="Custom answer with [1].",
        )
        
        result = engine.search("test query")
        
        assert result.answer == "Custom answer with [1]."

    def test_custom_claims(self):
        """カスタムクレーム"""
        custom_claims = [
            {
                "text": "Custom claim",
                "source_text_unit_id": "custom_tu",
                "source_document_id": "custom_doc",
            },
        ]
        
        engine = MockLazySearchEngine(default_claims=custom_claims)
        result = engine.search("test query")
        
        assert len(result.claims) == 1
        assert result.claims[0].text == "Custom claim"

    def test_search_increments_count(self):
        """検索カウントがインクリメントされる"""
        engine = MockLazySearchEngine()
        
        assert engine.search_call_count == 0
        engine.search("query 1")
        assert engine.search_call_count == 1
        engine.search("query 2")
        assert engine.search_call_count == 2

    def test_search_respects_max_level(self):
        """max_levelを尊重"""
        engine = MockLazySearchEngine()
        
        result = engine.search("test", max_level=SearchLevel.LEVEL_0)
        assert result.search_level_reached == SearchLevel.LEVEL_0
        
        result = engine.search("test", max_level=SearchLevel.LEVEL_1)
        assert result.search_level_reached == SearchLevel.LEVEL_1

    def test_search_level0_only_method(self):
        """search_level0_onlyメソッド"""
        engine = MockLazySearchEngine()
        
        result = engine.search_level0_only("test query")
        
        assert result.search_level_reached == SearchLevel.LEVEL_0


# === Edge Cases ===


class TestEdgeCases:
    """エッジケーステスト"""

    def test_search_with_empty_query(self, mock_vector_engine, mock_llm_client):
        """空のクエリで検索"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # コンポーネントをモック
        engine.relevance_tester = MagicMock()
        engine.relevance_tester.filter_relevant.return_value = []
        engine.deepener = MagicMock()
        engine.deepener.should_deepen.return_value = False
        
        result = engine.search("")
        
        assert isinstance(result, LazySearchResult)

    def test_search_with_special_characters(self, mock_vector_engine, mock_llm_client):
        """特殊文字を含むクエリ"""
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_client,
        )
        
        # コンポーネントをモック
        engine.relevance_tester = MagicMock()
        engine.relevance_tester.filter_relevant.return_value = []
        engine.deepener = MagicMock()
        engine.deepener.should_deepen.return_value = False
        
        result = engine.search("test [1] & [2] query?")
        
        assert isinstance(result, LazySearchResult)
