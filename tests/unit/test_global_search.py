"""Unit tests for GlobalSearch module."""

import pytest
from dataclasses import asdict

from monjyu.query.global_search import (
    GlobalSearch,
    GlobalSearchConfig,
    GlobalSearchResult,
    MapResult,
    CommunityInfo,
    InMemoryCommunityStore,
    MockLLMClient,
    get_map_prompt,
    get_reduce_prompt,
    MAP_PROMPT_EN,
    MAP_PROMPT_JA,
    REDUCE_PROMPT_EN,
    REDUCE_PROMPT_JA,
)


# =============================================================================
# GlobalSearchConfig Tests
# =============================================================================


class TestGlobalSearchConfig:
    """GlobalSearchConfigのテスト"""

    def test_default_values(self):
        """デフォルト値のテスト"""
        config = GlobalSearchConfig()
        assert config.community_level == 1
        assert config.top_k_communities == 10
        assert config.map_reduce_enabled is True
        assert config.max_context_tokens == 8000
        assert config.temperature == 0.0
        assert config.response_language == "auto"

    def test_custom_values(self):
        """カスタム値のテスト"""
        config = GlobalSearchConfig(
            community_level=2,
            top_k_communities=5,
            map_reduce_enabled=False,
            max_context_tokens=4000,
            temperature=0.5,
            response_language="ja",
        )
        assert config.community_level == 2
        assert config.top_k_communities == 5
        assert config.map_reduce_enabled is False
        assert config.max_context_tokens == 4000
        assert config.temperature == 0.5
        assert config.response_language == "ja"

    def test_to_dict(self):
        """to_dictのテスト"""
        config = GlobalSearchConfig(community_level=2, top_k_communities=5)
        d = config.to_dict()
        assert d["community_level"] == 2
        assert d["top_k_communities"] == 5
        assert "map_reduce_enabled" in d

    def test_from_dict(self):
        """from_dictのテスト"""
        data = {
            "community_level": 3,
            "top_k_communities": 15,
            "map_reduce_enabled": False,
        }
        config = GlobalSearchConfig.from_dict(data)
        assert config.community_level == 3
        assert config.top_k_communities == 15
        assert config.map_reduce_enabled is False

    def test_from_dict_with_defaults(self):
        """from_dictでデフォルト値が適用されるテスト"""
        data = {"community_level": 2}
        config = GlobalSearchConfig.from_dict(data)
        assert config.community_level == 2
        assert config.top_k_communities == 10  # default


# =============================================================================
# MapResult Tests
# =============================================================================


class TestMapResult:
    """MapResultのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        result = MapResult(
            community_id="c1",
            community_title="ML Community",
            partial_answer="Machine learning is...",
            relevance_score=0.8,
            tokens_used=150,
        )
        assert result.community_id == "c1"
        assert result.community_title == "ML Community"
        assert result.partial_answer == "Machine learning is..."
        assert result.relevance_score == 0.8
        assert result.tokens_used == 150

    def test_to_dict(self):
        """to_dictのテスト"""
        result = MapResult(
            community_id="c1",
            community_title="Test",
            partial_answer="Answer",
            relevance_score=0.5,
            tokens_used=100,
        )
        d = result.to_dict()
        assert d["community_id"] == "c1"
        assert d["relevance_score"] == 0.5


# =============================================================================
# CommunityInfo Tests
# =============================================================================


class TestCommunityInfo:
    """CommunityInfoのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        info = CommunityInfo(
            community_id="c1",
            title="ML Community",
            summary="This community focuses on machine learning.",
            level=1,
            size=50,
            key_entities=["neural network", "deep learning"],
            findings=["Finding 1", "Finding 2"],
        )
        assert info.community_id == "c1"
        assert info.title == "ML Community"
        assert info.level == 1
        assert info.size == 50
        assert len(info.key_entities) == 2
        assert len(info.findings) == 2

    def test_default_values(self):
        """デフォルト値のテスト"""
        info = CommunityInfo(
            community_id="c1",
            title="Test",
            summary="Test summary",
        )
        assert info.level == 0
        assert info.size == 0
        assert info.key_entities == []
        assert info.findings == []

    def test_to_dict(self):
        """to_dictのテスト"""
        info = CommunityInfo(
            community_id="c1",
            title="Test",
            summary="Summary",
            level=2,
            size=30,
        )
        d = info.to_dict()
        assert d["community_id"] == "c1"
        assert d["level"] == 2

    def test_from_dict(self):
        """from_dictのテスト"""
        data = {
            "community_id": "c2",
            "title": "NLP Community",
            "summary": "Natural language processing",
            "level": 1,
            "size": 25,
            "key_entities": ["BERT", "GPT"],
            "findings": ["Finding A"],
        }
        info = CommunityInfo.from_dict(data)
        assert info.community_id == "c2"
        assert info.title == "NLP Community"
        assert len(info.key_entities) == 2


# =============================================================================
# GlobalSearchResult Tests
# =============================================================================


class TestGlobalSearchResult:
    """GlobalSearchResultのテスト"""

    def test_creation(self):
        """基本的な作成テスト"""
        result = GlobalSearchResult(
            query="What is machine learning?",
            answer="Machine learning is...",
            communities_used=[],
            map_results=[],
            processing_time_ms=500,
            tokens_used=1000,
            community_level=1,
        )
        assert result.query == "What is machine learning?"
        assert result.answer == "Machine learning is..."
        assert result.processing_time_ms == 500
        assert result.tokens_used == 1000
        assert result.community_level == 1

    def test_to_dict(self):
        """to_dictのテスト"""
        community = CommunityInfo(
            community_id="c1",
            title="Test",
            summary="Summary",
        )
        map_result = MapResult(
            community_id="c1",
            community_title="Test",
            partial_answer="Partial",
            relevance_score=0.7,
            tokens_used=50,
        )
        result = GlobalSearchResult(
            query="Query",
            answer="Answer",
            communities_used=[community],
            map_results=[map_result],
            processing_time_ms=100,
            tokens_used=200,
            community_level=1,
        )
        d = result.to_dict()
        assert d["query"] == "Query"
        assert len(d["communities_used"]) == 1
        assert len(d["map_results"]) == 1


# =============================================================================
# Prompts Tests
# =============================================================================


class TestPrompts:
    """プロンプト関数のテスト"""

    def test_get_map_prompt_english(self):
        """英語Mapプロンプトのテスト"""
        prompt = get_map_prompt("en")
        assert "helpful assistant" in prompt
        assert "{query}" in prompt

    def test_get_map_prompt_japanese(self):
        """日本語Mapプロンプトのテスト"""
        prompt = get_map_prompt("ja")
        assert "アシスタント" in prompt
        assert "{query}" in prompt

    def test_get_reduce_prompt_english(self):
        """英語Reduceプロンプトのテスト"""
        prompt = get_reduce_prompt("en")
        assert "synthesizing" in prompt
        assert "{partial_answers}" in prompt

    def test_get_reduce_prompt_japanese(self):
        """日本語Reduceプロンプトのテスト"""
        prompt = get_reduce_prompt("ja")
        assert "統合" in prompt
        assert "{partial_answers}" in prompt


# =============================================================================
# InMemoryCommunityStore Tests
# =============================================================================


class TestInMemoryCommunityStore:
    """InMemoryCommunityStoreのテスト"""

    def test_add_and_get(self):
        """追加と取得のテスト"""
        store = InMemoryCommunityStore()
        community = CommunityInfo(
            community_id="c1",
            title="ML",
            summary="ML community",
            level=1,
            size=50,
        )
        store.add_community(community)
        
        result = store.get_communities_by_level(1)
        assert len(result) == 1
        assert result[0].community_id == "c1"

    def test_get_by_level_filtering(self):
        """レベルフィルタリングのテスト"""
        store = InMemoryCommunityStore()
        store.add_community(CommunityInfo("c1", "ML", "ML summary", level=1, size=50))
        store.add_community(CommunityInfo("c2", "NLP", "NLP summary", level=2, size=30))
        store.add_community(CommunityInfo("c3", "CV", "CV summary", level=1, size=40))

        level1 = store.get_communities_by_level(1)
        assert len(level1) == 2
        
        level2 = store.get_communities_by_level(2)
        assert len(level2) == 1

    def test_get_top_communities(self):
        """上位コミュニティ取得のテスト"""
        store = InMemoryCommunityStore()
        store.add_community(CommunityInfo("c1", "ML", "ML", level=1, size=50))
        store.add_community(CommunityInfo("c2", "NLP", "NLP", level=1, size=30))
        store.add_community(CommunityInfo("c3", "CV", "CV", level=1, size=40))

        top2 = store.get_top_communities(level=1, top_k=2)
        assert len(top2) == 2
        assert top2[0].size == 50  # largest first
        assert top2[1].size == 40

    def test_clear(self):
        """クリアのテスト"""
        store = InMemoryCommunityStore()
        store.add_community(CommunityInfo("c1", "Test", "Test", level=1, size=10))
        assert store.count() == 1
        
        store.clear()
        assert store.count() == 0

    def test_count(self):
        """カウントのテスト"""
        store = InMemoryCommunityStore()
        assert store.count() == 0
        
        store.add_community(CommunityInfo("c1", "A", "A", level=1, size=10))
        store.add_community(CommunityInfo("c2", "B", "B", level=1, size=20))
        assert store.count() == 2


# =============================================================================
# MockLLMClient Tests
# =============================================================================


class TestMockLLMClient:
    """MockLLMClientのテスト"""

    def test_default_response(self):
        """デフォルト応答のテスト"""
        client = MockLLMClient()
        response = client.generate("random prompt")
        assert response == "This is a mock response."

    def test_custom_response(self):
        """カスタム応答のテスト"""
        client = MockLLMClient(
            responses={"machine learning": "ML is a subset of AI."}
        )
        response = client.generate("What is machine learning?")
        assert response == "ML is a subset of AI."

    def test_count_tokens(self):
        """トークンカウントのテスト"""
        client = MockLLMClient()
        # 100文字 * 0.25 = 25トークン
        tokens = client.count_tokens("a" * 100)
        assert tokens == 25


# =============================================================================
# GlobalSearch Tests
# =============================================================================


class TestGlobalSearch:
    """GlobalSearchのテスト"""

    @pytest.fixture
    def setup_search(self):
        """テスト用のGlobalSearchセットアップ"""
        store = InMemoryCommunityStore()
        store.add_community(
            CommunityInfo(
                community_id="c1",
                title="Machine Learning Community",
                summary="This community focuses on machine learning algorithms and applications.",
                level=1,
                size=100,
                key_entities=["neural network", "deep learning", "CNN", "RNN"],
                findings=[
                    "Deep learning has revolutionized image recognition.",
                    "Transfer learning reduces training time significantly.",
                ],
            )
        )
        store.add_community(
            CommunityInfo(
                community_id="c2",
                title="Natural Language Processing Community",
                summary="NLP research focuses on text understanding and generation.",
                level=1,
                size=80,
                key_entities=["BERT", "GPT", "transformer", "attention"],
                findings=[
                    "Transformer architecture has become the standard for NLP.",
                    "Large language models show emergent abilities.",
                ],
            )
        )
        store.add_community(
            CommunityInfo(
                community_id="c3",
                title="Computer Vision Community",
                summary="CV research on image and video analysis.",
                level=2,
                size=60,
                key_entities=["ResNet", "YOLO", "segmentation"],
                findings=[
                    "Object detection accuracy has improved significantly.",
                ],
            )
        )

        llm = MockLLMClient(
            responses={
                "machine learning": "Machine learning involves training models on data to make predictions.",
                "nlp": "NLP enables computers to understand human language.",
                "neural network": "Neural networks are computational models inspired by biological neurons.",
            },
            default_response="Based on the community data, here is the relevant information.",
        )

        search = GlobalSearch(
            llm_client=llm,
            community_store=store,
            config=GlobalSearchConfig(
                community_level=1,
                top_k_communities=10,
            ),
        )

        return search, store, llm

    def test_search_basic(self, setup_search):
        """基本検索のテスト"""
        search, _, _ = setup_search
        result = search.search("What is machine learning?")
        
        assert isinstance(result, GlobalSearchResult)
        assert result.query == "What is machine learning?"
        assert result.answer != ""
        assert result.processing_time_ms >= 0

    def test_search_with_level(self, setup_search):
        """レベル指定検索のテスト"""
        search, _, _ = setup_search
        result = search.search("What is computer vision?", level=2)
        
        assert result.community_level == 2
        # level=2のコミュニティが1つある
        assert len(result.communities_used) == 1

    def test_search_empty_communities(self):
        """コミュニティなしの検索テスト"""
        store = InMemoryCommunityStore()
        llm = MockLLMClient()
        search = GlobalSearch(llm_client=llm, community_store=store)
        
        result = search.search("Any question")
        
        assert "No community data" in result.answer
        assert len(result.communities_used) == 0

    def test_search_with_config_override(self, setup_search):
        """設定オーバーライドのテスト"""
        search, _, _ = setup_search
        config = GlobalSearchConfig(
            community_level=1,
            top_k_communities=1,
        )
        
        result = search.search("What is NLP?", config=config)
        
        # top_k=1なので1コミュニティのみ
        assert len(result.communities_used) == 1

    def test_search_map_reduce_disabled(self, setup_search):
        """Map-Reduce無効化のテスト"""
        search, _, _ = setup_search
        config = GlobalSearchConfig(
            community_level=1,
            map_reduce_enabled=False,
        )
        
        result = search.search("What is deep learning?", config=config)
        
        # Map-Reduce無効なのでmap_resultsは空
        assert len(result.map_results) == 0
        assert result.answer != ""

    def test_search_returns_tokens_used(self, setup_search):
        """トークン使用量が返されるテスト"""
        search, _, _ = setup_search
        result = search.search("What is AI?")
        
        assert result.tokens_used >= 0

    def test_map_phase_filters_by_relevance(self, setup_search):
        """Mapフェーズで関連性フィルタリングされるテスト"""
        search, store, llm = setup_search
        
        # 「関連情報なし」を返すレスポンスを設定
        llm.responses["specific topic"] = "No relevant information found."
        
        result = search.search("specific topic not in data")
        
        # 関連性スコアでフィルタリングされる
        assert isinstance(result.map_results, list)

    def test_relevance_calculation(self, setup_search):
        """関連性スコア計算のテスト"""
        search, _, _ = setup_search
        
        # 短い回答 = 低スコア
        score_short = search._calculate_relevance("OK")
        assert score_short == 0.2
        
        # 「関連情報なし」= 0スコア
        score_no_info = search._calculate_relevance("No relevant information found")
        assert score_no_info == 0.0
        
        # 長い回答 = 高スコア
        score_long = search._calculate_relevance("a" * 400)
        assert score_long == 0.9


# =============================================================================
# Integration Tests
# =============================================================================


class TestGlobalSearchIntegration:
    """GlobalSearch統合テスト"""

    def test_full_search_workflow(self):
        """完全な検索ワークフローのテスト"""
        # セットアップ
        store = InMemoryCommunityStore()
        store.add_community(
            CommunityInfo(
                community_id="research-ml",
                title="ML Research",
                summary="Machine learning research community",
                level=1,
                size=150,
                key_entities=["supervised learning", "unsupervised learning"],
                findings=[
                    "Supervised learning is most common in industry.",
                    "Unsupervised learning is growing in popularity.",
                ],
            )
        )

        llm = MockLLMClient(
            responses={
                "learning": "Learning methods include supervised and unsupervised approaches. "
                           "Supervised learning uses labeled data while unsupervised learning "
                           "finds patterns in unlabeled data."
            }
        )

        search = GlobalSearch(
            llm_client=llm,
            community_store=store,
            config=GlobalSearchConfig(
                community_level=1,
                top_k_communities=5,
                map_reduce_enabled=True,
            ),
        )

        # 検索実行
        result = search.search("What types of learning are there?")

        # 検証
        assert result.query == "What types of learning are there?"
        assert len(result.answer) > 0
        assert len(result.communities_used) == 1
        assert result.communities_used[0].community_id == "research-ml"
        assert result.processing_time_ms >= 0
        assert result.tokens_used > 0
        assert result.community_level == 1

    def test_multi_community_search(self):
        """複数コミュニティ検索のテスト"""
        store = InMemoryCommunityStore()
        
        for i in range(5):
            store.add_community(
                CommunityInfo(
                    community_id=f"community-{i}",
                    title=f"Research Area {i}",
                    summary=f"Summary for area {i}",
                    level=1,
                    size=100 - i * 10,
                    key_entities=[f"entity-{i}-a", f"entity-{i}-b"],
                    findings=[f"Finding {i}"],
                )
            )

        llm = MockLLMClient(default_response="Combined research findings show interesting patterns.")
        
        search = GlobalSearch(
            llm_client=llm,
            community_store=store,
            config=GlobalSearchConfig(
                community_level=1,
                top_k_communities=3,
            ),
        )

        result = search.search("What are the main research findings?")

        # top_k=3なので3コミュニティ
        assert len(result.communities_used) == 3
        # サイズ順（大きい順）
        assert result.communities_used[0].size == 100
        assert result.communities_used[1].size == 90
        assert result.communities_used[2].size == 80

    def test_language_specific_search(self):
        """言語指定検索のテスト"""
        store = InMemoryCommunityStore()
        store.add_community(
            CommunityInfo(
                community_id="jp-research",
                title="日本語研究",
                summary="日本語に関する研究コミュニティ",
                level=1,
                size=50,
                key_entities=["形態素解析", "係り受け"],
                findings=["日本語処理は形態素解析が重要"],
            )
        )

        llm = MockLLMClient(
            responses={"日本語": "日本語処理には形態素解析が不可欠です。"}
        )

        search = GlobalSearch(
            llm_client=llm,
            community_store=store,
            config=GlobalSearchConfig(
                response_language="ja",
            ),
        )

        result = search.search("日本語処理について教えて")

        assert result.answer != ""
        assert "日本語" in result.communities_used[0].title
