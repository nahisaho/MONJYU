# Lazy Search Unit Tests
"""
FEAT-005: Lazy Search 単体テスト

TASK-005-07: 単体テスト作成
"""

import pytest
from unittest.mock import MagicMock, patch
from monjyu.lazy.base import (
    SearchLevel,
    RelevanceScore,
    Claim,
    SearchCandidate,
    LazySearchState,
    LazySearchResult,
    LazySearchConfig,
)
from monjyu.lazy.relevance_tester import RelevanceTester, MockRelevanceTester
from monjyu.lazy.claim_extractor import ClaimExtractor, MockClaimExtractor
from monjyu.lazy.iterative_deepener import IterativeDeepener, MockIterativeDeepener
from monjyu.lazy.community_searcher import CommunitySearcher, MockCommunitySearcher
from monjyu.lazy.engine import LazySearchEngine, MockLazySearchEngine


# === Base Types Tests ===


class TestSearchLevel:
    """SearchLevel のテスト"""

    def test_level_values(self):
        """レベル値の確認"""
        assert SearchLevel.LEVEL_0.value == 0
        assert SearchLevel.LEVEL_1.value == 1
        assert SearchLevel.LEVEL_2.value == 2

    def test_level_comparison(self):
        """レベル比較"""
        assert SearchLevel.LEVEL_0.value < SearchLevel.LEVEL_1.value
        assert SearchLevel.LEVEL_1.value < SearchLevel.LEVEL_2.value


class TestRelevanceScore:
    """RelevanceScore のテスト"""

    def test_score_values(self):
        """スコア値の確認"""
        assert RelevanceScore.LOW.value == 0
        assert RelevanceScore.MEDIUM.value == 1
        assert RelevanceScore.HIGH.value == 2

    def test_score_comparison(self):
        """スコア比較"""
        assert RelevanceScore.LOW.value < RelevanceScore.MEDIUM.value
        assert RelevanceScore.MEDIUM.value < RelevanceScore.HIGH.value


class TestClaim:
    """Claim のテスト"""

    def test_claim_creation(self):
        """クレーム作成"""
        claim = Claim(
            text="Test claim",
            source_text_unit_id="tu_1",
            source_document_id="doc_1",
        )
        assert claim.text == "Test claim"
        assert claim.source_text_unit_id == "tu_1"
        assert claim.confidence == 1.0
        assert claim.extracted_at != ""

    def test_claim_to_dict(self):
        """辞書変換"""
        claim = Claim(
            text="Test claim",
            source_text_unit_id="tu_1",
            source_document_id="doc_1",
        )
        data = claim.to_dict()
        assert data["text"] == "Test claim"
        assert data["source_text_unit_id"] == "tu_1"
        assert "extracted_at" in data

    def test_claim_from_dict(self):
        """辞書から復元"""
        data = {
            "text": "Test claim",
            "source_text_unit_id": "tu_1",
            "source_document_id": "doc_1",
            "confidence": 0.9,
            "relevance_score": "HIGH",
        }
        claim = Claim.from_dict(data)
        assert claim.text == "Test claim"
        assert claim.confidence == 0.9
        assert claim.relevance_score == RelevanceScore.HIGH


class TestSearchCandidate:
    """SearchCandidate のテスト"""

    def test_candidate_creation(self):
        """候補作成"""
        candidate = SearchCandidate(
            id="tu_1",
            source="vector",
            priority=0.9,
            level=SearchLevel.LEVEL_0,
            text="Test text",
        )
        assert candidate.id == "tu_1"
        assert candidate.source == "vector"
        assert candidate.priority == 0.9

    def test_candidate_comparison(self):
        """優先度比較（heapq用）"""
        c1 = SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        c2 = SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0)
        # 優先度が高いほうが「小さい」（heapqで先に出る）
        assert c1 < c2

    def test_candidate_equality(self):
        """等価性"""
        c1 = SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        c2 = SearchCandidate(id="1", source="vector", priority=0.5, level=SearchLevel.LEVEL_0)
        assert c1 == c2  # IDとソースが同じなら等価


class TestLazySearchState:
    """LazySearchState のテスト"""

    def test_state_creation(self):
        """状態作成"""
        state = LazySearchState(query="Test query")
        assert state.query == "Test query"
        assert state.llm_calls == 0
        assert state.current_level == SearchLevel.LEVEL_0

    def test_add_candidate(self):
        """候補追加"""
        state = LazySearchState(query="Test")
        candidate = SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        state.add_candidate(candidate)
        assert state.queue_size == 1

    def test_pop_candidate(self):
        """候補取得（優先度順）"""
        state = LazySearchState(query="Test")
        c1 = SearchCandidate(id="1", source="vector", priority=0.7, level=SearchLevel.LEVEL_0)
        c2 = SearchCandidate(id="2", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        state.add_candidate(c1)
        state.add_candidate(c2)
        
        # 優先度が高い順に取得
        popped = state.pop_candidate()
        assert popped.id == "2"
        assert popped.priority == 0.9

    def test_mark_visited(self):
        """訪問済みマーク"""
        state = LazySearchState(query="Test")
        candidate = SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        state.mark_visited(candidate)
        assert "1" in state.visited_text_units

    def test_is_visited(self):
        """訪問済み確認"""
        state = LazySearchState(query="Test")
        candidate = SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0)
        assert not state.is_visited(candidate)
        state.mark_visited(candidate)
        assert state.is_visited(candidate)

    def test_state_properties(self):
        """状態プロパティ"""
        state = LazySearchState(query="Test")
        state.claims.append(Claim(text="Claim 1", source_text_unit_id="tu_1", source_document_id="doc_1"))
        state.visited_text_units.add("tu_1")
        
        assert state.claim_count == 1
        assert state.visited_count == 1


class TestLazySearchConfig:
    """LazySearchConfig のテスト"""

    def test_default_config(self):
        """デフォルト設定"""
        config = LazySearchConfig()
        assert config.initial_top_k == 20
        assert config.max_iterations == 5
        assert config.min_relevance == RelevanceScore.MEDIUM

    def test_custom_config(self):
        """カスタム設定"""
        config = LazySearchConfig(
            initial_top_k=10,
            max_llm_calls=10,
            include_communities=False,
        )
        assert config.initial_top_k == 10
        assert config.max_llm_calls == 10
        assert config.include_communities is False


# === Component Tests ===


class TestMockRelevanceTester:
    """MockRelevanceTester のテスト"""

    def test_default_score(self):
        """デフォルトスコア"""
        tester = MockRelevanceTester(default_score=RelevanceScore.HIGH)
        score = tester.test("query", "text")
        assert score == RelevanceScore.HIGH

    def test_keyword_score(self):
        """キーワードスコア"""
        tester = MockRelevanceTester(default_score=RelevanceScore.LOW)
        tester.set_keyword_score("important", RelevanceScore.HIGH)
        
        score1 = tester.test("query", "This is important text")
        score2 = tester.test("query", "This is regular text")
        
        assert score1 == RelevanceScore.HIGH
        assert score2 == RelevanceScore.LOW

    def test_batch_test(self):
        """バッチテスト"""
        tester = MockRelevanceTester(default_score=RelevanceScore.MEDIUM)
        scores = tester.test_batch("query", ["text1", "text2", "text3"])
        assert len(scores) == 3
        assert all(s == RelevanceScore.MEDIUM for s in scores)

    def test_filter_relevant(self):
        """関連性フィルタリング"""
        tester = MockRelevanceTester(default_score=RelevanceScore.MEDIUM)
        candidates = [
            SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0, text="text1"),
            SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0, text="text2"),
        ]
        
        results = tester.filter_relevant("query", candidates, min_relevance=RelevanceScore.MEDIUM)
        assert len(results) == 2


class TestMockClaimExtractor:
    """MockClaimExtractor のテスト"""

    def test_default_claims(self):
        """デフォルトクレーム"""
        extractor = MockClaimExtractor()
        claims = extractor.extract("query", "text")
        assert len(claims) == 2

    def test_custom_claims(self):
        """カスタムクレーム"""
        extractor = MockClaimExtractor(default_claims=["Claim A", "Claim B", "Claim C"])
        claims = extractor.extract("query", "text")
        assert len(claims) == 3
        assert claims[0].text == "Claim A"

    def test_batch_extract(self):
        """バッチ抽出"""
        extractor = MockClaimExtractor(default_claims=["Claim"])
        candidates = [
            SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0, text="text1"),
            SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0, text="text2"),
        ]
        
        claims = extractor.extract_batch("query", candidates)
        assert len(claims) == 2


class TestMockIterativeDeepener:
    """MockIterativeDeepener のテスト"""

    def test_should_deepen_true(self):
        """深化すべき（True）"""
        deepener = MockIterativeDeepener(should_deepen_result=True, max_deepening_count=3)
        state = LazySearchState(query="Test")
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        
        assert deepener.should_deepen(state) is True

    def test_should_deepen_false_empty_queue(self):
        """深化すべきでない（キュー空）"""
        deepener = MockIterativeDeepener(should_deepen_result=True)
        state = LazySearchState(query="Test")
        
        assert deepener.should_deepen(state) is False

    def test_should_deepen_false_max_iterations(self):
        """深化すべきでない（最大イテレーション）"""
        deepener = MockIterativeDeepener(max_deepening_count=2)
        state = LazySearchState(query="Test")
        state.iterations = 2
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        
        assert deepener.should_deepen(state) is False

    def test_get_next_candidates(self):
        """次の候補取得"""
        deepener = MockIterativeDeepener()
        state = LazySearchState(query="Test")
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        state.add_candidate(SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0))
        
        candidates = deepener.get_next_candidates(state, batch_size=2)
        assert len(candidates) == 2


class TestMockCommunitySearcher:
    """MockCommunitySearcher のテスト"""

    def test_search(self):
        """コミュニティ検索"""
        searcher = MockCommunitySearcher()
        candidates = searcher.search("machine learning", top_k=3)
        
        assert len(candidates) == 2
        assert candidates[0].source == "community"

    def test_get_text_units(self):
        """TextUnit取得"""
        searcher = MockCommunitySearcher()
        text_units = searcher.get_text_units("comm_1")
        
        assert len(text_units) == 3
        assert text_units[0][0] == "tu_1"  # text_unit_id


class TestMockLazySearchEngine:
    """MockLazySearchEngine のテスト"""

    def test_search(self):
        """検索"""
        engine = MockLazySearchEngine(
            default_answer="Mock answer with [1] and [2] citations."
        )
        result = engine.search("test query")
        
        assert result.query == "test query"
        assert result.answer == "Mock answer with [1] and [2] citations."
        assert len(result.claims) == 2
        assert len(result.citations) == 2

    def test_search_level0_only(self):
        """Level 0のみ検索"""
        engine = MockLazySearchEngine()
        result = engine.search_level0_only("test query")
        
        assert result.search_level_reached == SearchLevel.LEVEL_0

    def test_call_count(self):
        """呼び出し回数"""
        engine = MockLazySearchEngine()
        engine.search("query1")
        engine.search("query2")
        
        assert engine.search_call_count == 2


# === LazySearchResult Tests ===


class TestLazySearchResult:
    """LazySearchResult のテスト"""

    def test_result_to_dict(self):
        """辞書変換"""
        result = LazySearchResult(
            query="test query",
            answer="test answer",
            claims=[],
            citations=[],
            search_level_reached=SearchLevel.LEVEL_1,
            llm_calls=5,
            tokens_used=500,
            total_time_ms=100.0,
        )
        
        data = result.to_dict()
        assert data["query"] == "test query"
        assert data["search_level_reached"] == "LEVEL_1"
        assert data["llm_calls"] == 5


# === Integration with Mock Tests ===


class TestRelevanceTesterWithMockLLM:
    """RelevanceTester with Mock LLM"""

    def test_test_high_relevance(self):
        """HIGH関連性テスト"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "HIGH"
        
        tester = RelevanceTester(mock_llm)
        score = tester.test("What is machine learning?", "Machine learning is a type of AI.")
        
        assert score == RelevanceScore.HIGH
        mock_llm.generate.assert_called_once()

    def test_test_low_relevance(self):
        """LOW関連性テスト"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "LOW"
        
        tester = RelevanceTester(mock_llm)
        score = tester.test("What is machine learning?", "The weather is nice today.")
        
        assert score == RelevanceScore.LOW

    def test_test_medium_relevance(self):
        """MEDIUM関連性テスト"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "MEDIUM"
        
        tester = RelevanceTester(mock_llm)
        score = tester.test("What is machine learning?", "AI has many applications.")
        
        assert score == RelevanceScore.MEDIUM

    def test_test_error_handling(self):
        """エラーハンドリング"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = Exception("API Error")
        
        tester = RelevanceTester(mock_llm)
        score = tester.test("query", "text")
        
        # エラー時はLOW
        assert score == RelevanceScore.LOW


class TestClaimExtractorWithMockLLM:
    """ClaimExtractor with Mock LLM"""

    def test_extract_claims(self):
        """クレーム抽出"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
- Machine learning uses data to learn patterns
- Neural networks are inspired by the brain
- Deep learning uses multiple layers
"""
        
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("What is ML?", "Sample text", "tu_1", "doc_1")
        
        assert len(claims) == 3
        assert claims[0].text == "Machine learning uses data to learn patterns"
        assert claims[0].source_text_unit_id == "tu_1"

    def test_extract_with_bullet_variants(self):
        """異なる箇条書き形式"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
- Claim 1
* Claim 2
・Claim 3
"""
        
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("query", "text")
        
        assert len(claims) == 3

    def test_merge_duplicates(self):
        """重複マージ"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = """
- Machine learning is important
- machine learning is important
- Different claim
"""
        
        extractor = ClaimExtractor(mock_llm)
        claims = extractor.extract("query", "text")
        
        # _merge_duplicates で重複除去
        # extract内部では呼ばれないので3つ
        assert len(claims) == 3


class TestIterativeDeepenerWithMockLLM:
    """IterativeDeepener with Mock LLM"""

    def test_should_deepen_insufficient(self):
        """INSUFFICIENT判定"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "INSUFFICIENT"
        
        deepener = IterativeDeepener(mock_llm, max_llm_calls=20)
        state = LazySearchState(query="Test")
        state.claims = [Claim(text="Claim", source_text_unit_id="tu_1", source_document_id="doc_1")]
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        
        # min_claims_for_check以下なので深化続行
        assert deepener.should_deepen(state) is True

    def test_should_deepen_sufficient(self):
        """SUFFICIENT判定"""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "SUFFICIENT"
        
        deepener = IterativeDeepener(mock_llm, min_claims_for_check=2)
        state = LazySearchState(query="Test")
        # 十分なクレーム数
        for i in range(5):
            state.claims.append(Claim(text=f"Claim {i}", source_text_unit_id=f"tu_{i}", source_document_id="doc_1"))
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        
        assert deepener.should_deepen(state) is False

    def test_should_deepen_max_llm_calls(self):
        """LLMコール上限"""
        mock_llm = MagicMock()
        deepener = IterativeDeepener(mock_llm, max_llm_calls=5)
        state = LazySearchState(query="Test")
        state.llm_calls = 5
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        
        assert deepener.should_deepen(state) is False
        mock_llm.generate.assert_not_called()  # LLMは呼ばれない


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
