# Search Unit Tests
"""
検索コンポーネントの単体テスト

TASK-004-07: 単体テスト作成
"""

from __future__ import annotations

import pytest

from monjyu.search.base import (
    Citation,
    SearchHit,
    SearchMode,
    SearchResults,
    SynthesizedAnswer,
    SearchResponse,
)


class TestSearchHit:
    """SearchHitのテスト"""

    def test_create_hit(self):
        """SearchHit作成"""
        hit = SearchHit(
            text_unit_id="tu_001",
            document_id="doc_001",
            text="This is a test text.",
            score=0.85,
            chunk_index=0,
            document_title="Test Document",
        )

        assert hit.text_unit_id == "tu_001"
        assert hit.document_id == "doc_001"
        assert hit.score == 0.85
        assert hit.document_title == "Test Document"

    def test_to_dict(self):
        """辞書変換"""
        hit = SearchHit(
            text_unit_id="tu_001",
            document_id="doc_001",
            text="Test",
            score=0.9,
        )

        d = hit.to_dict()
        assert d["text_unit_id"] == "tu_001"
        assert d["score"] == 0.9

    def test_from_dict(self):
        """辞書から復元"""
        data = {
            "text_unit_id": "tu_002",
            "document_id": "doc_002",
            "text": "Restored text",
            "score": 0.75,
        }

        hit = SearchHit.from_dict(data)
        assert hit.text_unit_id == "tu_002"
        assert hit.score == 0.75


class TestSearchResults:
    """SearchResultsのテスト"""

    def test_create_results(self):
        """SearchResults作成"""
        hits = [
            SearchHit("tu_001", "doc_001", "Text 1", 0.9),
            SearchHit("tu_002", "doc_001", "Text 2", 0.8),
        ]

        results = SearchResults(
            hits=hits,
            total_count=2,
            search_time_ms=15.5,
        )

        assert len(results.hits) == 2
        assert results.total_count == 2

    def test_texts_property(self):
        """textsプロパティ"""
        hits = [
            SearchHit("tu_001", "doc_001", "Text 1", 0.9),
            SearchHit("tu_002", "doc_001", "Text 2", 0.8),
        ]

        results = SearchResults(hits=hits, total_count=2)

        assert results.texts == ["Text 1", "Text 2"]

    def test_top_score_property(self):
        """top_scoreプロパティ"""
        hits = [
            SearchHit("tu_001", "doc_001", "Text 1", 0.9),
            SearchHit("tu_002", "doc_001", "Text 2", 0.8),
        ]

        results = SearchResults(hits=hits, total_count=2)

        assert results.top_score == 0.9

    def test_top_score_empty(self):
        """空の結果でtop_score"""
        results = SearchResults(hits=[], total_count=0)
        assert results.top_score == 0.0

    def test_text_unit_ids_property(self):
        """text_unit_idsプロパティ"""
        hits = [
            SearchHit("tu_001", "doc_001", "Text 1", 0.9),
            SearchHit("tu_002", "doc_001", "Text 2", 0.8),
        ]

        results = SearchResults(hits=hits, total_count=2)

        assert results.text_unit_ids == ["tu_001", "tu_002"]


class TestCitation:
    """Citationのテスト"""

    def test_create_citation(self):
        """Citation作成"""
        citation = Citation(
            text_unit_id="tu_001",
            document_id="doc_001",
            document_title="Test Paper",
            text_snippet="This is the relevant snippet...",
            relevance_score=0.85,
        )

        assert citation.text_unit_id == "tu_001"
        assert citation.document_title == "Test Paper"
        assert citation.relevance_score == 0.85

    def test_to_dict(self):
        """辞書変換"""
        citation = Citation(
            text_unit_id="tu_001",
            document_id="doc_001",
            document_title="Test",
            text_snippet="Snippet",
            relevance_score=0.9,
        )

        d = citation.to_dict()
        assert d["document_title"] == "Test"
        assert d["relevance_score"] == 0.9


class TestSynthesizedAnswer:
    """SynthesizedAnswerのテスト"""

    def test_create_answer(self):
        """SynthesizedAnswer作成"""
        citations = [
            Citation("tu_001", "doc_001", "Paper 1", "snippet", 0.9),
        ]

        answer = SynthesizedAnswer(
            answer="This is the answer based on [1].",
            citations=citations,
            confidence=0.85,
            model="gpt-4o-mini",
        )

        assert "answer" in answer.answer
        assert len(answer.citations) == 1
        assert answer.confidence == 0.85

    def test_to_dict(self):
        """辞書変換"""
        answer = SynthesizedAnswer(
            answer="Answer text",
            citations=[],
            confidence=0.7,
        )

        d = answer.to_dict()
        assert d["answer"] == "Answer text"
        assert d["confidence"] == 0.7


class TestSearchResponse:
    """SearchResponseのテスト"""

    def test_create_response(self):
        """SearchResponse作成"""
        hits = [SearchHit("tu_001", "doc_001", "Text", 0.9)]
        results = SearchResults(hits=hits, total_count=1)
        answer = SynthesizedAnswer(answer="Answer", citations=[])

        response = SearchResponse(
            query="What is this?",
            answer=answer,
            search_results=results,
            total_time_ms=100.0,
            mode=SearchMode.VECTOR,
        )

        assert response.query == "What is this?"
        assert response.mode == SearchMode.VECTOR

    def test_to_dict(self):
        """辞書変換"""
        hits = [SearchHit("tu_001", "doc_001", "Text", 0.9)]
        results = SearchResults(hits=hits, total_count=1)
        answer = SynthesizedAnswer(answer="Answer", citations=[])

        response = SearchResponse(
            query="Test query",
            answer=answer,
            search_results=results,
        )

        d = response.to_dict()
        assert d["query"] == "Test query"
        assert d["mode"] == "vector"


class TestSearchMode:
    """SearchModeのテスト"""

    def test_modes(self):
        """モード値"""
        assert SearchMode.VECTOR.value == "vector"
        assert SearchMode.KEYWORD.value == "keyword"
        assert SearchMode.HYBRID.value == "hybrid"


# === Query Encoder Tests ===


class TestQueryEncoder:
    """QueryEncoderのテスト"""

    def test_encode(self):
        """エンコード"""
        from monjyu.search.query_encoder import QueryEncoder

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        encoder = QueryEncoder(MockEmbeddingClient())
        vector = encoder.encode("test query")

        assert len(vector) == 3
        assert vector == [0.1, 0.2, 0.3]

    def test_encode_caching(self):
        """キャッシュ機能"""
        from monjyu.search.query_encoder import QueryEncoder

        call_count = 0

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                nonlocal call_count
                call_count += 1
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        encoder = QueryEncoder(MockEmbeddingClient())

        # 同じクエリを2回呼び出し
        encoder.encode("test query")
        encoder.encode("test query")

        # 1回だけ埋め込みが実行される
        assert call_count == 1

    def test_encode_batch(self):
        """バッチエンコード"""
        from monjyu.search.query_encoder import QueryEncoder

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(1, len(texts) + 1)]

        encoder = QueryEncoder(MockEmbeddingClient())
        vectors = encoder.encode_batch(["query1", "query2"])

        assert len(vectors) == 2

    def test_clear_cache(self):
        """キャッシュクリア"""
        from monjyu.search.query_encoder import QueryEncoder

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [0.1, 0.2, 0.3]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        encoder = QueryEncoder(MockEmbeddingClient())
        encoder.encode("test")
        assert len(encoder._cache) == 1

        encoder.clear_cache()
        assert len(encoder._cache) == 0


# === Vector Searcher Tests ===


class TestInMemoryVectorSearcher:
    """InMemoryVectorSearcherのテスト"""

    def test_add_and_search(self):
        """追加と検索"""
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        searcher = InMemoryVectorSearcher()

        searcher.add(
            text_unit_id="tu_001",
            vector=[1.0, 0.0, 0.0],
            text="First document",
            document_id="doc_001",
        )
        searcher.add(
            text_unit_id="tu_002",
            vector=[0.0, 1.0, 0.0],
            text="Second document",
            document_id="doc_002",
        )

        results = searcher.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

        assert len(results.hits) == 2
        assert results.hits[0].text_unit_id == "tu_001"
        assert results.hits[0].score > results.hits[1].score

    def test_search_with_threshold(self):
        """閾値付き検索"""
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        searcher = InMemoryVectorSearcher()

        searcher.add("tu_001", [1.0, 0.0, 0.0], "High similarity")
        searcher.add("tu_002", [0.0, 1.0, 0.0], "Low similarity")

        results = searcher.search(
            query_vector=[1.0, 0.0, 0.0],
            top_k=10,
            threshold=0.9,
        )

        # 高い類似度のものだけ返る
        assert len(results.hits) == 1
        assert results.hits[0].text_unit_id == "tu_001"

    def test_clear(self):
        """クリア"""
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        searcher = InMemoryVectorSearcher()
        searcher.add("tu_001", [1.0, 0.0, 0.0], "Test")

        assert len(searcher) == 1

        searcher.clear()
        assert len(searcher) == 0


# === Answer Synthesizer Tests ===


class TestAnswerSynthesizer:
    """AnswerSynthesizerのテスト"""

    def test_synthesize(self):
        """回答合成"""
        from monjyu.search.answer_synthesizer import AnswerSynthesizer, MockLLMClient

        llm = MockLLMClient()
        llm.set_response("質問", "The answer is based on [1]. This is important.")

        synthesizer = AnswerSynthesizer(llm)

        context = [
            SearchHit(
                text_unit_id="tu_001",
                document_id="doc_001",
                text="Context text here",
                score=0.9,
                document_title="Test Paper",
            ),
        ]

        answer = synthesizer.synthesize("質問: What is this?", context)

        assert answer.answer != ""
        assert answer.model == "mock-llm"

    def test_synthesize_empty_context(self):
        """空のコンテキスト"""
        from monjyu.search.answer_synthesizer import AnswerSynthesizer, MockLLMClient

        synthesizer = AnswerSynthesizer(MockLLMClient())

        answer = synthesizer.synthesize("Test query", [])

        assert "見つかりませんでした" in answer.answer
        assert len(answer.citations) == 0

    def test_extract_citations(self):
        """引用抽出"""
        from monjyu.search.answer_synthesizer import AnswerSynthesizer, MockLLMClient

        synthesizer = AnswerSynthesizer(MockLLMClient())

        context = [
            SearchHit("tu_001", "doc_001", "Text 1", 0.9, document_title="Paper 1"),
            SearchHit("tu_002", "doc_002", "Text 2", 0.8, document_title="Paper 2"),
        ]

        response = "The answer is based on [1] and [2]."
        _, citations = synthesizer._extract_citations(response, context)

        assert len(citations) == 2
        assert citations[0].document_title == "Paper 1"
        assert citations[1].document_title == "Paper 2"


# === VectorSearchEngine Tests ===


class TestVectorSearchConfig:
    """VectorSearchConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定"""
        from monjyu.search.engine import VectorSearchConfig

        config = VectorSearchConfig()

        assert config.top_k == 10
        assert config.mode == SearchMode.VECTOR
        assert config.synthesize is True

    def test_custom_config(self):
        """カスタム設定"""
        from monjyu.search.engine import VectorSearchConfig

        config = VectorSearchConfig(
            top_k=5,
            mode=SearchMode.HYBRID,
            hybrid_alpha=0.7,
        )

        assert config.top_k == 5
        assert config.mode == SearchMode.HYBRID
        assert config.hybrid_alpha == 0.7

    def test_to_dict(self):
        """辞書変換"""
        from monjyu.search.engine import VectorSearchConfig

        config = VectorSearchConfig(top_k=20)
        d = config.to_dict()

        assert d["top_k"] == 20
        assert d["mode"] == "vector"


class TestVectorSearchEngine:
    """VectorSearchEngineのテスト"""

    def test_search(self):
        """基本検索"""
        from monjyu.search.answer_synthesizer import MockLLMClient
        from monjyu.search.engine import VectorSearchConfig, VectorSearchEngine
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0] for _ in texts]

        searcher = InMemoryVectorSearcher()
        searcher.add("tu_001", [1.0, 0.0, 0.0], "Relevant text", document_title="Paper")

        engine = VectorSearchEngine(
            embedding_client=MockEmbeddingClient(),
            vector_searcher=searcher,
            llm_client=MockLLMClient(),
            config=VectorSearchConfig(synthesize=False),
        )

        response = engine.search("test query")

        assert response.query == "test query"
        assert len(response.search_results.hits) == 1

    def test_retrieve_only(self):
        """検索のみ（合成なし）"""
        from monjyu.search.answer_synthesizer import MockLLMClient
        from monjyu.search.engine import VectorSearchEngine
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0] for _ in texts]

        searcher = InMemoryVectorSearcher()
        searcher.add("tu_001", [1.0, 0.0, 0.0], "Test")

        engine = VectorSearchEngine(
            embedding_client=MockEmbeddingClient(),
            vector_searcher=searcher,
            llm_client=MockLLMClient(),
        )

        results = engine.retrieve_only("query")

        assert isinstance(results, SearchResults)
        assert len(results.hits) == 1

    def test_get_stats(self):
        """統計情報取得"""
        from monjyu.search.answer_synthesizer import MockLLMClient
        from monjyu.search.engine import VectorSearchEngine
        from monjyu.search.vector_searcher import InMemoryVectorSearcher

        class MockEmbeddingClient:
            def embed(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0]

            def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [[1.0, 0.0, 0.0] for _ in texts]

        engine = VectorSearchEngine(
            embedding_client=MockEmbeddingClient(),
            vector_searcher=InMemoryVectorSearcher(),
            llm_client=MockLLMClient(),
        )

        stats = engine.get_stats()

        assert "config" in stats
        assert "encoder_cache_size" in stats
