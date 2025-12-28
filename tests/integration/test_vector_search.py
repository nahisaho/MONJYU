# Vector Search Integration Tests
"""
ベクトル検索の統合テスト

TASK-004-08: 統合テスト作成
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from monjyu.search.base import SearchMode, SearchResults
from monjyu.search.answer_synthesizer import MockLLMClient
from monjyu.search.engine import VectorSearchConfig, VectorSearchEngine
from monjyu.search.vector_searcher import InMemoryVectorSearcher


class MockEmbeddingClient:
    """テスト用埋め込みクライアント"""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._embeddings: dict[str, list[float]] = {}

    def set_embedding(self, text: str, vector: list[float]) -> None:
        """特定テキストの埋め込みを設定"""
        self._embeddings[text] = vector

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        if text in self._embeddings:
            return self._embeddings[text]
        # デフォルト: テキストのハッシュを基にベクトル生成
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        vector = [b / 255.0 for b in hash_bytes[: self.dimension]]
        # 次元数を調整
        while len(vector) < self.dimension:
            vector.extend(vector)
        return vector[: self.dimension]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括変換"""
        return [self.embed(text) for text in texts]


class TestVectorSearchIntegration:
    """ベクトル検索統合テスト"""

    @pytest.fixture
    def sample_documents(self):
        """サンプルドキュメント"""
        return [
            {
                "text_unit_id": "tu_001",
                "document_id": "doc_001",
                "document_title": "Deep Learning Fundamentals",
                "text": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn representations of data.",
            },
            {
                "text_unit_id": "tu_002",
                "document_id": "doc_001",
                "document_title": "Deep Learning Fundamentals",
                "text": "Convolutional neural networks (CNNs) are particularly effective for image recognition tasks.",
            },
            {
                "text_unit_id": "tu_003",
                "document_id": "doc_002",
                "document_title": "Natural Language Processing",
                "text": "Transformers have revolutionized natural language processing by enabling parallel processing of sequences.",
            },
            {
                "text_unit_id": "tu_004",
                "document_id": "doc_002",
                "document_title": "Natural Language Processing",
                "text": "BERT and GPT are transformer-based models that achieve state-of-the-art results on many NLP benchmarks.",
            },
            {
                "text_unit_id": "tu_005",
                "document_id": "doc_003",
                "document_title": "Reinforcement Learning",
                "text": "Reinforcement learning trains agents to make decisions by rewarding desired behaviors and punishing undesired ones.",
            },
        ]

    @pytest.fixture
    def embedding_client(self):
        """埋め込みクライアント"""
        client = MockEmbeddingClient(dimension=16)
        # 類似クエリに類似ベクトルを設定
        client.set_embedding("deep learning", [1.0, 0.9, 0.1, 0.0] + [0.0] * 12)
        client.set_embedding("neural network", [0.9, 1.0, 0.1, 0.0] + [0.0] * 12)
        client.set_embedding("transformer NLP", [0.1, 0.1, 1.0, 0.9] + [0.0] * 12)
        return client

    @pytest.fixture
    def vector_searcher(self, sample_documents, embedding_client):
        """ベクトル検索器（データ投入済み）"""
        searcher = InMemoryVectorSearcher()

        for doc in sample_documents:
            vector = embedding_client.embed(doc["text"])
            searcher.add(
                text_unit_id=doc["text_unit_id"],
                vector=vector,
                text=doc["text"],
                document_id=doc["document_id"],
                document_title=doc["document_title"],
            )

        return searcher

    @pytest.fixture
    def search_engine(self, embedding_client, vector_searcher):
        """検索エンジン"""
        llm_client = MockLLMClient()
        llm_client.set_response(
            "deep learning",
            "Deep learning uses neural networks with multiple layers [1]. "
            "CNNs are effective for image recognition [2].",
        )
        llm_client.set_response(
            "NLP",
            "Transformers have revolutionized NLP [1]. "
            "BERT and GPT are transformer-based models [2].",
        )

        return VectorSearchEngine(
            embedding_client=embedding_client,
            vector_searcher=vector_searcher,
            llm_client=llm_client,
            config=VectorSearchConfig(top_k=3),
        )

    def test_basic_search(self, search_engine):
        """基本検索テスト"""
        response = search_engine.search("deep learning")

        assert response.query == "deep learning"
        assert len(response.search_results.hits) <= 3
        assert response.total_time_ms > 0

    def test_search_relevance(self, search_engine, embedding_client):
        """検索関連性テスト"""
        # Deep learningクエリ用のベクトルを設定
        embedding_client.set_embedding(
            "What is deep learning?", [1.0, 0.9, 0.1, 0.0] + [0.0] * 12
        )

        response = search_engine.search("What is deep learning?", synthesize=False)

        # Deep learning関連のドキュメントが上位に来る
        hits = response.search_results.hits
        assert len(hits) > 0

        # 最上位の結果がdeep learning関連であることを確認
        top_hit = hits[0]
        assert (
            "deep learning" in top_hit.text.lower()
            or "neural" in top_hit.text.lower()
        )

    def test_search_with_synthesis(self, search_engine):
        """回答合成付き検索"""
        response = search_engine.search("deep learning", synthesize=True)

        assert response.answer.answer != ""
        # 引用が含まれている
        assert "[1]" in response.answer.answer or "[2]" in response.answer.answer

    def test_search_without_synthesis(self, search_engine):
        """回答合成なし検索"""
        response = search_engine.search("deep learning", synthesize=False)

        assert response.answer.answer == ""
        assert len(response.answer.citations) == 0
        # 検索結果は返る
        assert len(response.search_results.hits) > 0

    def test_retrieve_only(self, search_engine):
        """retrieve_only メソッド"""
        results = search_engine.retrieve_only("transformer NLP")

        assert isinstance(results, SearchResults)
        assert len(results.hits) > 0

    def test_top_k_parameter(self, search_engine):
        """top_kパラメータ"""
        response_3 = search_engine.search("neural network", top_k=3, synthesize=False)
        response_1 = search_engine.search("neural network", top_k=1, synthesize=False)

        assert len(response_1.search_results.hits) <= 1
        assert len(response_3.search_results.hits) <= 3

    def test_threshold_filtering(self, search_engine):
        """閾値フィルタリング"""
        # 高い閾値
        response = search_engine.search(
            "completely unrelated query xyz123",
            threshold=0.99,
            synthesize=False,
        )

        # 閾値が高いので結果は少ないか0
        assert len(response.search_results.hits) <= 1

    def test_search_mode_vector(self, search_engine):
        """ベクトル検索モード"""
        response = search_engine.search(
            "deep learning",
            mode=SearchMode.VECTOR,
            synthesize=False,
        )

        assert response.mode == SearchMode.VECTOR
        assert len(response.search_results.hits) > 0


class TestSearchEngineE2E:
    """検索エンジンE2Eテスト"""

    def test_full_pipeline(self):
        """フルパイプラインテスト"""
        # 1. コンポーネント準備
        embedding_client = MockEmbeddingClient(dimension=8)
        searcher = InMemoryVectorSearcher()
        llm_client = MockLLMClient()

        # 2. ドキュメント追加
        documents = [
            ("tu_001", "Graph neural networks learn representations.", "GNN Paper"),
            ("tu_002", "Knowledge graphs store structured information.", "KG Paper"),
            ("tu_003", "Attention mechanisms improve model performance.", "Attention"),
        ]

        for tu_id, text, title in documents:
            vector = embedding_client.embed(text)
            searcher.add(
                text_unit_id=tu_id,
                vector=vector,
                text=text,
                document_title=title,
            )

        # 3. 検索エンジン作成
        engine = VectorSearchEngine(
            embedding_client=embedding_client,
            vector_searcher=searcher,
            llm_client=llm_client,
            config=VectorSearchConfig(
                top_k=3,
                synthesize=True,
            ),
        )

        # 4. 検索実行
        response = engine.search("graph neural network")

        # 5. 検証
        assert response.query == "graph neural network"
        assert len(response.search_results.hits) > 0
        assert response.total_time_ms > 0
        assert response.search_time_ms > 0

    def test_multiple_searches(self):
        """複数検索の連続実行"""
        embedding_client = MockEmbeddingClient(dimension=8)
        searcher = InMemoryVectorSearcher()

        # データ追加
        for i in range(10):
            vector = embedding_client.embed(f"document {i}")
            searcher.add(f"tu_{i:03d}", vector, f"Document {i} content")

        engine = VectorSearchEngine(
            embedding_client=embedding_client,
            vector_searcher=searcher,
            llm_client=MockLLMClient(),
            config=VectorSearchConfig(synthesize=False),
        )

        # 複数検索
        queries = ["document 1", "document 5", "document 9"]
        for query in queries:
            response = engine.search(query)
            assert len(response.search_results.hits) > 0

        # キャッシュが蓄積される
        stats = engine.get_stats()
        assert stats["encoder_cache_size"] == 3


class TestPerformance:
    """パフォーマンステスト"""

    def test_search_latency(self):
        """検索レイテンシ"""
        embedding_client = MockEmbeddingClient(dimension=128)
        searcher = InMemoryVectorSearcher()

        # 1000ドキュメント追加
        for i in range(1000):
            vector = embedding_client.embed(f"document content {i}")
            searcher.add(f"tu_{i:04d}", vector, f"Document {i}")

        engine = VectorSearchEngine(
            embedding_client=embedding_client,
            vector_searcher=searcher,
            llm_client=MockLLMClient(),
            config=VectorSearchConfig(synthesize=False),
        )

        # 検索実行
        response = engine.search("test query")

        # 検索時間は妥当な範囲内
        assert response.search_time_ms < 1000  # 1秒未満
