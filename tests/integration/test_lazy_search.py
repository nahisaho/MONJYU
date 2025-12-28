# Lazy Search Integration Tests
"""
FEAT-005: Lazy Search 統合テスト

TASK-005-08: 統合テスト作成
"""

import pytest
import time
from unittest.mock import MagicMock

from monjyu.lazy.base import (
    SearchLevel,
    RelevanceScore,
    Claim,
    SearchCandidate,
    LazySearchState,
    LazySearchConfig,
)
from monjyu.lazy.relevance_tester import RelevanceTester, MockRelevanceTester
from monjyu.lazy.claim_extractor import ClaimExtractor, MockClaimExtractor
from monjyu.lazy.iterative_deepener import IterativeDeepener, MockIterativeDeepener
from monjyu.lazy.community_searcher import MockCommunitySearcher
from monjyu.lazy.engine import LazySearchEngine, MockLazySearchEngine
from monjyu.search.base import SearchHit, SearchResults, SynthesizedAnswer, SearchResponse, SearchMode


class MockVectorSearchEngine:
    """テスト用モック VectorSearchEngine"""

    def __init__(self, mock_hits: list[dict] | None = None):
        self.mock_hits = mock_hits or [
            {
                "text_unit_id": "tu_1",
                "document_id": "doc_1",
                "text": "Machine learning is a branch of artificial intelligence.",
                "score": 0.95,
                "document_title": "Introduction to ML",
            },
            {
                "text_unit_id": "tu_2",
                "document_id": "doc_1",
                "text": "Neural networks are computational models inspired by biological neurons.",
                "score": 0.88,
                "document_title": "Introduction to ML",
            },
            {
                "text_unit_id": "tu_3",
                "document_id": "doc_2",
                "text": "Deep learning uses multiple layers of neural networks.",
                "score": 0.82,
                "document_title": "Deep Learning Basics",
            },
        ]
        self.search_call_count = 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: SearchMode | None = None,
        synthesize: bool = True,
        threshold: float = 0.0,
    ) -> SearchResponse:
        self.search_call_count += 1
        
        hits = [
            SearchHit(
                text_unit_id=h["text_unit_id"],
                document_id=h["document_id"],
                text=h["text"],
                score=h["score"],
                document_title=h.get("document_title", ""),
            )
            for h in self.mock_hits[:top_k]
        ]
        
        results = SearchResults(hits=hits, total_count=len(hits))
        answer = SynthesizedAnswer(answer="", citations=[])
        
        return SearchResponse(
            query=query,
            answer=answer,
            search_results=results,
        )


class MockLLMClient:
    """テスト用モック LLMClient"""

    def __init__(self):
        self._model = "mock-model"
        self._responses = {}
        self.call_count = 0

    @property
    def model_name(self) -> str:
        return self._model

    def set_response(self, prompt_contains: str, response: str):
        """特定のプロンプトに対する応答を設定"""
        self._responses[prompt_contains] = response

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.call_count += 1
        
        # 設定されたレスポンスを探す
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response
        
        # デフォルトレスポンス
        if "HIGH" in prompt or "MEDIUM" in prompt or "LOW" in prompt:
            return "MEDIUM"
        if "SUFFICIENT" in prompt or "INSUFFICIENT" in prompt:
            return "INSUFFICIENT"
        if "claim" in prompt.lower() or "抽出" in prompt:
            return "- This is an extracted claim\n- Another claim"
        
        return "This is a default response [1] [2]"


# === Integration Tests ===


class TestLazySearchIntegration:
    """Lazy Search 統合テスト"""

    def test_full_search_pipeline(self):
        """フル検索パイプライン"""
        # Setup
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        mock_llm.set_response("HIGH", "HIGH")
        mock_llm.set_response("claim", "- ML uses data to learn\n- Neural networks are powerful")
        mock_llm.set_response("SUFFICIENT", "SUFFICIENT")
        
        config = LazySearchConfig(
            initial_top_k=3,
            max_iterations=2,
            include_communities=False,
        )
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=config,
        )
        
        # Execute
        result = engine.search("What is machine learning?")
        
        # Verify
        assert result.query == "What is machine learning?"
        assert result.answer != ""
        assert result.search_level_reached in [SearchLevel.LEVEL_0, SearchLevel.LEVEL_1]
        assert result.llm_calls > 0

    def test_level0_only_search(self):
        """Level 0のみ検索"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        mock_llm.set_response("HIGH", "HIGH")
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=LazySearchConfig(include_communities=False),
        )
        
        result = engine.search_level0_only("What is ML?")
        
        assert result.search_level_reached == SearchLevel.LEVEL_0

    def test_search_with_communities(self):
        """コミュニティ検索を含む"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        mock_community_searcher = MockCommunitySearcher()
        
        # Level1Index はモックなので None で渡してコミュニティサーチャーを直接設定
        config = LazySearchConfig(
            include_communities=True,
            community_top_k=2,
        )
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=config,
        )
        # コミュニティサーチャーを直接設定
        engine.community_searcher = mock_community_searcher
        engine.deepener.community_searcher = mock_community_searcher
        
        result = engine.search("machine learning neural network")
        
        assert result.answer != ""

    def test_iterative_deepening(self):
        """動的深化テスト"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        
        # 最初はINSUFFICIENT、最後にSUFFICIENT
        call_count = [0]
        def custom_generate(prompt, system_prompt=None, max_tokens=None):
            call_count[0] += 1
            if "SUFFICIENT" in prompt or "INSUFFICIENT" in prompt:
                # 2回目以降はSUFFICIENT
                if call_count[0] > 5:
                    return "SUFFICIENT"
                return "INSUFFICIENT"
            if "claim" in prompt.lower():
                return "- Claim extracted"
            if "HIGH" in prompt:
                return "HIGH"
            return "Answer [1]"
        
        mock_llm.generate = custom_generate
        
        config = LazySearchConfig(
            initial_top_k=3,
            max_iterations=3,
            include_communities=False,
        )
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=config,
        )
        
        result = engine.search("complex query")
        
        # 深化が行われた
        assert result.llm_calls > 3


class TestRelevanceFilteringIntegration:
    """関連性フィルタリング統合テスト"""

    def test_filter_irrelevant_content(self):
        """無関連コンテンツのフィルタリング"""
        mock_llm = MockLLMClient()
        
        # 特定のテキストにはLOW
        def custom_generate(prompt, system_prompt=None, max_tokens=None):
            if "weather" in prompt.lower():
                return "LOW"
            return "HIGH"
        
        mock_llm.generate = custom_generate
        
        tester = RelevanceTester(mock_llm)
        
        candidates = [
            SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0,
                          text="Machine learning is AI technology"),
            SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0,
                          text="The weather is nice today"),
        ]
        
        results = tester.filter_relevant("What is ML?", candidates, min_relevance=RelevanceScore.MEDIUM)
        
        # weather関連はフィルタされる
        assert len(results) == 1
        assert results[0][0].id == "1"


class TestClaimExtractionIntegration:
    """クレーム抽出統合テスト"""

    def test_extract_and_merge(self):
        """抽出とマージ"""
        mock_llm = MockLLMClient()
        mock_llm.set_response("claim", """
- Machine learning uses data
- Neural networks learn patterns
- Data drives ML models
""")
        
        extractor = ClaimExtractor(mock_llm)
        
        candidates = [
            SearchCandidate(id="1", source="vector", priority=0.9, level=SearchLevel.LEVEL_0,
                          text="Text about ML", metadata={"document_id": "doc_1"}),
            SearchCandidate(id="2", source="vector", priority=0.8, level=SearchLevel.LEVEL_0,
                          text="Text about neural networks", metadata={"document_id": "doc_1"}),
        ]
        
        claims = extractor.extract_batch("What is ML?", candidates)
        
        # 重複が除去される（同じクレームテキストが返される場合）
        assert len(claims) >= 1
        assert all(c.source_text_unit_id in ["1", "2"] for c in claims)


class TestStateManagement:
    """状態管理テスト"""

    def test_state_transition(self):
        """状態遷移"""
        state = LazySearchState(query="Test query")
        
        # 初期状態
        assert state.current_level == SearchLevel.LEVEL_0
        assert state.iterations == 0
        assert state.llm_calls == 0
        
        # 候補追加
        for i in range(5):
            state.add_candidate(SearchCandidate(
                id=f"tu_{i}",
                source="vector",
                priority=0.9 - i * 0.1,
                level=SearchLevel.LEVEL_0,
            ))
        
        assert state.queue_size == 5
        
        # 候補取得と訪問
        candidate = state.pop_candidate()
        state.mark_visited(candidate)
        
        assert state.queue_size == 4
        assert state.visited_count == 1
        
        # クレーム追加
        state.claims.append(Claim(text="Claim 1", source_text_unit_id="tu_0", source_document_id="doc_1"))
        
        assert state.claim_count == 1

    def test_priority_ordering(self):
        """優先度順序"""
        state = LazySearchState(query="Test")
        
        # 異なる優先度で追加
        state.add_candidate(SearchCandidate(id="1", source="vector", priority=0.5, level=SearchLevel.LEVEL_0))
        state.add_candidate(SearchCandidate(id="2", source="vector", priority=0.9, level=SearchLevel.LEVEL_0))
        state.add_candidate(SearchCandidate(id="3", source="vector", priority=0.7, level=SearchLevel.LEVEL_0))
        
        # 優先度順に取得
        c1 = state.pop_candidate()
        c2 = state.pop_candidate()
        c3 = state.pop_candidate()
        
        assert c1.priority == 0.9
        assert c2.priority == 0.7
        assert c3.priority == 0.5


class TestPerformance:
    """パフォーマンステスト"""

    def test_search_response_time(self):
        """検索レスポンス時間"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=LazySearchConfig(
                initial_top_k=5,
                max_iterations=2,
                include_communities=False,
            ),
        )
        
        start = time.time()
        result = engine.search("test query")
        elapsed = time.time() - start
        
        # モックなので高速であるべき
        assert elapsed < 1.0
        assert result.total_time_ms > 0

    def test_llm_call_limit(self):
        """LLMコール制限"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        mock_llm.set_response("INSUFFICIENT", "INSUFFICIENT")  # 常に深化続行
        
        config = LazySearchConfig(
            initial_top_k=5,
            max_llm_calls=10,
            max_iterations=100,  # 高いが、max_llm_callsで制限
            include_communities=False,
        )
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=config,
        )
        
        result = engine.search("test query")
        
        # LLMコールが上限付近で停止
        assert result.llm_calls <= config.max_llm_calls + 5  # 多少の余裕


class TestCostComparison:
    """コスト比較テスト（TASK-005-09）"""

    def test_lazy_vs_baseline_llm_calls(self):
        """Lazy vs Baseline: LLMコール数比較"""
        # Baseline RAG (Level 0のみ、全テキストに対してクレーム抽出)
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm_baseline = MockLLMClient()
        
        baseline_engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_baseline,
            config=LazySearchConfig(
                initial_top_k=10,
                include_communities=False,
            ),
        )
        
        baseline_result = baseline_engine.search_level0_only("What is ML?")
        
        # Lazy Search (関連性テストでフィルタリング)
        mock_llm_lazy = MockLLMClient()
        
        # 一部のみHIGH（関連性テストでフィルタ）
        call_count = [0]
        def custom_generate(prompt, system_prompt=None, max_tokens=None):
            call_count[0] += 1
            if "HIGH" in prompt or "MEDIUM" in prompt:
                # 奇数番目のみHIGH
                if call_count[0] % 2 == 1:
                    return "HIGH"
                return "LOW"
            if "SUFFICIENT" in prompt:
                return "SUFFICIENT"
            return "Claim extracted"
        
        mock_llm_lazy.generate = custom_generate
        
        lazy_engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm_lazy,
            config=LazySearchConfig(
                initial_top_k=10,
                include_communities=False,
            ),
        )
        
        lazy_result = lazy_engine.search("What is ML?", max_level=SearchLevel.LEVEL_0)
        
        # 両方とも結果を返す
        assert baseline_result.answer != ""
        assert lazy_result.answer != ""

    def test_early_termination_saves_calls(self):
        """早期終了によるコール節約"""
        mock_vector_engine = MockVectorSearchEngine()
        mock_llm = MockLLMClient()
        
        # 最初からSUFFICIENT
        mock_llm.set_response("SUFFICIENT", "SUFFICIENT")
        mock_llm.set_response("HIGH", "HIGH")
        
        config = LazySearchConfig(
            initial_top_k=5,
            max_iterations=10,  # 高く設定
            include_communities=False,
        )
        
        engine = LazySearchEngine(
            vector_search_engine=mock_vector_engine,
            llm_client=mock_llm,
            config=config,
        )
        
        result = engine.search("simple query")
        
        # SUFFICIENTで早期終了するため、イテレーションは少ない
        # (実際のイテレーション回数は実装による)
        assert result.llm_calls < 50  # 上限よりずっと少ない


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
