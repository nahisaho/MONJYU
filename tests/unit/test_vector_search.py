"""VectorSearch unit tests."""

from typing import List
from unittest.mock import AsyncMock

import numpy as np
import pytest
from numpy.typing import NDArray


# ==== モックエンベッダー ====


class MockEmbedder:
    """テスト用モックエンベッダー"""
    
    def __init__(self, dim: int = 384):
        self._dim = dim
    
    async def embed(self, text: str) -> NDArray[np.float32]:
        """テキストを埋め込み"""
        # テキストの長さに基づいて決定的なベクトルを生成
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self._dim).astype(np.float32)
    
    async def embed_batch(self, texts: List[str]) -> NDArray[np.float32]:
        """バッチ埋め込み"""
        return np.array([await self.embed(t) for t in texts])
    
    @property
    def dimension(self) -> int:
        """埋め込み次元"""
        return self._dim


# ==== SearchHit テスト ====


class TestSearchHit:
    """SearchHit のテスト"""
    
    def test_creation(self):
        """SearchHit作成テスト"""
        from monjyu.query.vector_search import SearchHit
        
        hit = SearchHit(
            chunk_id="chunk_1",
            score=0.95,
            content="Test content",
        )
        
        assert hit.chunk_id == "chunk_1"
        assert hit.score == 0.95
        assert hit.content == "Test content"
        assert hit.metadata == {}
    
    def test_with_metadata(self):
        """メタデータ付きSearchHitテスト"""
        from monjyu.query.vector_search import SearchHit
        
        hit = SearchHit(
            chunk_id="chunk_1",
            score=0.9,
            content="Content",
            metadata={"key": "value"},
            paper_id="paper_1",
            paper_title="Test Paper",
            section_type="introduction",
        )
        
        assert hit.paper_id == "paper_1"
        assert hit.paper_title == "Test Paper"
        assert hit.section_type == "introduction"
        assert hit.metadata["key"] == "value"
    
    def test_to_dict(self):
        """辞書変換テスト"""
        from monjyu.query.vector_search import SearchHit
        
        hit = SearchHit(
            chunk_id="chunk_1",
            score=0.85,
            content="Test",
            paper_id="paper_1",
        )
        
        data = hit.to_dict()
        
        assert data["chunk_id"] == "chunk_1"
        assert data["score"] == 0.85
        assert data["paper_id"] == "paper_1"
    
    def test_from_dict(self):
        """辞書から作成テスト"""
        from monjyu.query.vector_search import SearchHit
        
        data = {
            "chunk_id": "chunk_2",
            "score": 0.75,
            "content": "Content",
            "paper_id": "paper_2",
        }
        
        hit = SearchHit.from_dict(data)
        
        assert hit.chunk_id == "chunk_2"
        assert hit.score == 0.75
        assert hit.paper_id == "paper_2"


# ==== VectorSearchConfig テスト ====


class TestVectorSearchConfig:
    """VectorSearchConfig のテスト"""
    
    def test_default_config(self):
        """デフォルト設定テスト"""
        from monjyu.query.vector_search import VectorSearchConfig
        
        config = VectorSearchConfig()
        
        assert config.top_k == 10
        assert config.min_score == 0.0
        assert config.include_metadata is True
        assert config.rerank is False
    
    def test_custom_config(self):
        """カスタム設定テスト"""
        from monjyu.query.vector_search import VectorSearchConfig
        
        config = VectorSearchConfig(
            top_k=20,
            min_score=0.5,
            rerank=True,
            rerank_model="bge-reranker",
        )
        
        assert config.top_k == 20
        assert config.min_score == 0.5
        assert config.rerank is True
        assert config.rerank_model == "bge-reranker"
    
    def test_to_dict(self):
        """辞書変換テスト"""
        from monjyu.query.vector_search import VectorSearchConfig
        
        config = VectorSearchConfig(top_k=15)
        data = config.to_dict()
        
        assert data["top_k"] == 15


# ==== VectorSearchResult テスト ====


class TestVectorSearchResult:
    """VectorSearchResult のテスト"""
    
    def test_creation(self):
        """結果作成テスト"""
        from monjyu.query.vector_search import VectorSearchResult, SearchHit
        
        result = VectorSearchResult(
            hits=[SearchHit(chunk_id="1", score=0.9, content="Test")],
            total_count=1,
            processing_time_ms=50.0,
            query="test query",
        )
        
        assert len(result.hits) == 1
        assert result.total_count == 1
        assert result.processing_time_ms == 50.0
        assert result.query == "test query"
    
    def test_empty_result(self):
        """空結果テスト"""
        from monjyu.query.vector_search import VectorSearchResult
        
        result = VectorSearchResult()
        
        assert len(result.hits) == 0
        assert result.total_count == 0
    
    def test_to_dict(self):
        """辞書変換テスト"""
        from monjyu.query.vector_search import VectorSearchResult, SearchHit
        
        result = VectorSearchResult(
            hits=[SearchHit(chunk_id="1", score=0.9, content="Test")],
            total_count=1,
        )
        
        data = result.to_dict()
        
        assert len(data["hits"]) == 1
        assert data["hits"][0]["chunk_id"] == "1"
    
    def test_from_dict(self):
        """辞書から作成テスト"""
        from monjyu.query.vector_search import VectorSearchResult
        
        data = {
            "hits": [{"chunk_id": "1", "score": 0.9, "content": "Test"}],
            "total_count": 1,
            "processing_time_ms": 25.0,
        }
        
        result = VectorSearchResult.from_dict(data)
        
        assert len(result.hits) == 1
        assert result.processing_time_ms == 25.0


# ==== IndexedDocument テスト ====


class TestIndexedDocument:
    """IndexedDocument のテスト"""
    
    def test_creation(self):
        """作成テスト"""
        from monjyu.query.vector_search.types import IndexedDocument
        
        doc = IndexedDocument(
            chunk_id="chunk_1",
            content="Test content",
            vector=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            paper_id="paper_1",
        )
        
        assert doc.chunk_id == "chunk_1"
        assert doc.content == "Test content"
        assert len(doc.vector) == 3
        assert doc.paper_id == "paper_1"


# ==== コサイン類似度テスト ====


class TestCosineSimilarity:
    """cosine_similarity のテスト"""
    
    def test_identical_vectors(self):
        """同一ベクトルのテスト"""
        from monjyu.query.vector_search import cosine_similarity
        
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        
        scores = cosine_similarity(v, docs)
        
        assert np.isclose(scores[0], 1.0)
    
    def test_orthogonal_vectors(self):
        """直交ベクトルのテスト"""
        from monjyu.query.vector_search import cosine_similarity
        
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        
        scores = cosine_similarity(v, docs)
        
        assert np.isclose(scores[0], 0.0, atol=1e-6)
    
    def test_multiple_documents(self):
        """複数ドキュメントのテスト"""
        from monjyu.query.vector_search import cosine_similarity
        
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        docs = np.array([
            [1.0, 0.0, 0.0],  # 同一
            [0.0, 1.0, 0.0],  # 直交
            [0.5, 0.5, 0.0],  # 部分一致
        ], dtype=np.float32)
        
        scores = cosine_similarity(v, docs)
        
        assert len(scores) == 3
        assert scores[0] > scores[2] > scores[1]


# ==== キーワードマッチテスト ====


class TestKeywordMatchScore:
    """keyword_match_score のテスト"""
    
    def test_exact_match(self):
        """完全一致テスト"""
        from monjyu.query.vector_search import keyword_match_score
        
        score = keyword_match_score("test", "test")
        
        assert score == 1.0
    
    def test_partial_match(self):
        """部分一致テスト"""
        from monjyu.query.vector_search import keyword_match_score
        
        score = keyword_match_score("test query", "this is a test")
        
        assert 0.0 < score < 1.0
    
    def test_no_match(self):
        """マッチなしテスト"""
        from monjyu.query.vector_search import keyword_match_score
        
        score = keyword_match_score("apple", "banana orange")
        
        assert score == 0.0
    
    def test_case_insensitive(self):
        """大文字小文字無視テスト"""
        from monjyu.query.vector_search import keyword_match_score
        
        score = keyword_match_score("TEST", "test")
        
        assert score == 1.0


# ==== InMemoryVectorSearch テスト ====


class TestInMemoryVectorSearch:
    """InMemoryVectorSearch のテスト"""
    
    @pytest.fixture
    def embedder(self):
        """エンベッダー作成"""
        return MockEmbedder(dim=128)
    
    @pytest.fixture
    def search(self, embedder):
        """検索インスタンス作成"""
        from monjyu.query.vector_search import InMemoryVectorSearch
        return InMemoryVectorSearch(embedder=embedder)
    
    @pytest.mark.asyncio
    async def test_add_documents(self, search):
        """ドキュメント追加テスト"""
        from monjyu.query.vector_search.types import IndexedDocument
        
        docs = [
            IndexedDocument(
                chunk_id="1",
                content="Deep learning is a subset of machine learning",
                vector=np.random.randn(128).astype(np.float32),
            ),
            IndexedDocument(
                chunk_id="2",
                content="Transformers revolutionized NLP",
                vector=np.random.randn(128).astype(np.float32),
            ),
        ]
        
        count = await search.add_documents(docs)
        
        assert count == 2
        assert search.count() == 2
    
    @pytest.mark.asyncio
    async def test_add_texts(self, search):
        """テキスト追加テスト"""
        texts = [
            "Deep learning is powerful",
            "Neural networks learn from data",
        ]
        
        count = await search.add_texts(texts)
        
        assert count == 2
        assert search.count() == 2
    
    @pytest.mark.asyncio
    async def test_search(self, search):
        """テキスト検索テスト"""
        # データ追加
        texts = [
            "BERT is a transformer-based model for NLP",
            "GPT uses transformer architecture for generation",
            "CNN is used for image classification",
        ]
        await search.add_texts(texts)
        
        # 検索
        result = await search.search("transformer NLP", top_k=2)
        
        assert len(result.hits) <= 2
        assert result.processing_time_ms > 0
        assert result.query == "transformer NLP"
    
    @pytest.mark.asyncio
    async def test_search_by_vector(self, search, embedder):
        """ベクトル検索テスト"""
        # データ追加
        texts = ["Deep learning", "Machine learning", "Reinforcement learning"]
        await search.add_texts(texts)
        
        # ベクトルで検索
        query_vector = await embedder.embed("learning")
        result = await search.search_by_vector(query_vector, top_k=2)
        
        assert len(result.hits) <= 2
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, search):
        """ハイブリッド検索テスト"""
        # データ追加
        texts = [
            "BERT is a powerful NLP model",
            "GPT generates coherent text",
            "Image classification uses CNN",
        ]
        await search.add_texts(texts)
        
        # ハイブリッド検索
        result = await search.hybrid_search("NLP model", top_k=2, alpha=0.5)
        
        assert len(result.hits) <= 2
        assert result.query == "NLP model"
    
    @pytest.mark.asyncio
    async def test_search_with_filter(self, search):
        """フィルタ付き検索テスト"""
        # データ追加
        texts = ["BERT model", "GPT model", "CNN model"]
        metadatas = [
            {"paper_id": "paper_1"},
            {"paper_id": "paper_2"},
            {"paper_id": "paper_1"},
        ]
        await search.add_texts(texts, metadatas=metadatas)
        
        # フィルタ付き検索
        result = await search.search("model", top_k=10, filter={"paper_id": "paper_1"})
        
        # paper_1 のみ取得
        for hit in result.hits:
            assert hit.metadata.get("paper_id") == "paper_1"
    
    @pytest.mark.asyncio
    async def test_search_empty_index(self, search):
        """空インデックス検索テスト"""
        result = await search.search("test query")
        
        assert len(result.hits) == 0
        assert result.total_count == 0
    
    @pytest.mark.asyncio
    async def test_clear(self, search):
        """クリアテスト"""
        await search.add_texts(["test"])
        assert search.count() == 1
        
        search.clear()
        
        assert search.count() == 0
    
    @pytest.mark.asyncio
    async def test_min_score_filter(self, search):
        """最小スコアフィルタテスト"""
        from monjyu.query.vector_search import (
            InMemoryVectorSearch,
            VectorSearchConfig,
        )
        
        config = VectorSearchConfig(min_score=0.9)
        search_with_min = InMemoryVectorSearch(
            embedder=search.embedder,
            config=config,
        )
        
        await search_with_min.add_texts(["apple", "banana", "orange"])
        result = await search_with_min.search("completely unrelated query xyz")
        
        # 低スコアの結果は除外される
        for hit in result.hits:
            assert hit.score >= 0.9


# ==== ファクトリ関数テスト ====


class TestCreateInMemorySearch:
    """create_in_memory_search のテスト"""
    
    def test_create_default(self):
        """デフォルト作成テスト"""
        from monjyu.query.vector_search import create_in_memory_search
        
        embedder = MockEmbedder()
        search = create_in_memory_search(embedder)
        
        assert search is not None
        assert search.count() == 0
    
    def test_create_with_config(self):
        """設定付き作成テスト"""
        from monjyu.query.vector_search import (
            create_in_memory_search,
            VectorSearchConfig,
        )
        
        embedder = MockEmbedder()
        config = VectorSearchConfig(top_k=5, min_score=0.5)
        search = create_in_memory_search(embedder, config)
        
        assert search.config.top_k == 5
        assert search.config.min_score == 0.5


# ==== 統合テスト ====


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_search_workflow(self):
        """完全な検索ワークフローテスト"""
        from monjyu.query.vector_search import (
            InMemoryVectorSearch,
            VectorSearchConfig,
        )
        
        # セットアップ
        embedder = MockEmbedder(dim=256)
        config = VectorSearchConfig(
            top_k=3,
            include_metadata=True,
        )
        search = InMemoryVectorSearch(embedder=embedder, config=config)
        
        # データ追加
        texts = [
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "GPT-3: Language Models are Few-Shot Learners",
            "ResNet: Deep Residual Learning for Image Recognition",
            "Attention Is All You Need - introducing the Transformer architecture",
            "Word2Vec: Efficient Estimation of Word Representations",
        ]
        metadatas = [
            {"paper_id": "bert", "section_type": "title"},
            {"paper_id": "gpt3", "section_type": "title"},
            {"paper_id": "resnet", "section_type": "title"},
            {"paper_id": "transformer", "section_type": "title"},
            {"paper_id": "word2vec", "section_type": "title"},
        ]
        await search.add_texts(texts, metadatas=metadatas)
        
        # 検索実行
        result = await search.search("transformer architecture NLP")
        
        # 検証
        assert result.total_count > 0
        assert result.processing_time_ms > 0
        assert all(hit.score > 0 for hit in result.hits)
    
    @pytest.mark.asyncio
    async def test_hybrid_search_combines_scores(self):
        """ハイブリッド検索スコア結合テスト"""
        from monjyu.query.vector_search import InMemoryVectorSearch
        
        embedder = MockEmbedder(dim=128)
        search = InMemoryVectorSearch(embedder=embedder)
        
        # キーワードマッチが高いドキュメントを追加
        texts = [
            "deep learning neural network",
            "machine learning algorithm",
            "transformer attention mechanism",
        ]
        await search.add_texts(texts)
        
        # ベクトルのみ
        vector_result = await search.hybrid_search("deep learning", alpha=1.0)
        
        # キーワードのみ
        keyword_result = await search.hybrid_search("deep learning", alpha=0.0)
        
        # ハイブリッド
        hybrid_result = await search.hybrid_search("deep learning", alpha=0.5)
        
        # 全て結果を返す
        assert vector_result.total_count > 0
        assert keyword_result.total_count > 0
        assert hybrid_result.total_count > 0
