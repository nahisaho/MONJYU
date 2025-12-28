"""Unit tests for Azure AI Search VectorStore.

Azure SDKがインストールされていない環境でもテスト可能。
"""

import asyncio
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from dataclasses import dataclass

import pytest


# ========== Azure SDK モック設定 ==========

# Azureモジュールをモック
azure_mock = MagicMock()
azure_mock.core = MagicMock()
azure_mock.core.credentials = MagicMock()
azure_mock.core.credentials.AzureKeyCredential = MagicMock

azure_mock.identity = MagicMock()
azure_mock.identity.DefaultAzureCredential = MagicMock

azure_mock.search = MagicMock()
azure_mock.search.documents = MagicMock()
azure_mock.search.documents.SearchClient = MagicMock
azure_mock.search.documents.aio = MagicMock()
azure_mock.search.documents.aio.SearchClient = MagicMock

azure_mock.search.documents.indexes = MagicMock()
azure_mock.search.documents.indexes.SearchIndexClient = MagicMock
azure_mock.search.documents.indexes.aio = MagicMock()
azure_mock.search.documents.indexes.aio.SearchIndexClient = MagicMock

azure_mock.search.documents.indexes.models = MagicMock()
azure_mock.search.documents.indexes.models.SearchIndex = MagicMock
azure_mock.search.documents.indexes.models.VectorSearch = MagicMock
azure_mock.search.documents.indexes.models.VectorSearchAlgorithmConfiguration = MagicMock
azure_mock.search.documents.indexes.models.HnswAlgorithmConfiguration = MagicMock
azure_mock.search.documents.indexes.models.VectorSearchProfile = MagicMock
azure_mock.search.documents.indexes.models.SearchField = MagicMock
azure_mock.search.documents.indexes.models.SearchFieldDataType = MagicMock
azure_mock.search.documents.indexes.models.SearchableField = MagicMock
azure_mock.search.documents.indexes.models.SimpleField = MagicMock
azure_mock.search.documents.indexes.models.SemanticConfiguration = MagicMock
azure_mock.search.documents.indexes.models.SemanticField = MagicMock
azure_mock.search.documents.indexes.models.SemanticPrioritizedFields = MagicMock
azure_mock.search.documents.indexes.models.SemanticSearch = MagicMock

azure_mock.search.documents.models = MagicMock()
azure_mock.search.documents.models.VectorizedQuery = MagicMock

# モジュールパッチ
sys.modules["azure"] = azure_mock
sys.modules["azure.core"] = azure_mock.core
sys.modules["azure.core.credentials"] = azure_mock.core.credentials
sys.modules["azure.identity"] = azure_mock.identity
sys.modules["azure.search"] = azure_mock.search
sys.modules["azure.search.documents"] = azure_mock.search.documents
sys.modules["azure.search.documents.aio"] = azure_mock.search.documents.aio
sys.modules["azure.search.documents.indexes"] = azure_mock.search.documents.indexes
sys.modules["azure.search.documents.indexes.aio"] = azure_mock.search.documents.indexes.aio
sys.modules["azure.search.documents.indexes.models"] = azure_mock.search.documents.indexes.models
sys.modules["azure.search.documents.models"] = azure_mock.search.documents.models


# ========== Test Fixtures ==========


@pytest.fixture
def mock_search_client():
    """SearchClientのモック"""
    mock = AsyncMock()
    mock.search = AsyncMock()
    mock.upload_documents = AsyncMock()
    mock.delete_documents = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_index_client():
    """SearchIndexClientのモック"""
    mock = AsyncMock()
    mock.get_index = AsyncMock()
    mock.create_or_update_index = AsyncMock()
    mock.delete_index = AsyncMock()
    mock.get_index_statistics = AsyncMock(
        return_value=MagicMock(document_count=100, storage_size=1024)
    )
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def sample_search_results():
    """サンプル検索結果"""
    results = [
        {
            "id": "doc_001",
            "text_unit_id": "tu_001",
            "document_id": "paper_001",
            "content": "GraphRAGは知識グラフを活用した検索手法です。",
            "document_title": "GraphRAG入門",
            "chunk_index": 0,
            "@search.score": 0.95,
        },
        {
            "id": "doc_002",
            "text_unit_id": "tu_002",
            "document_id": "paper_001",
            "content": "ベクトル検索はセマンティック類似度を計算します。",
            "document_title": "GraphRAG入門",
            "chunk_index": 1,
            "@search.score": 0.85,
        },
        {
            "id": "doc_003",
            "text_unit_id": "tu_003",
            "document_id": "paper_002",
            "content": "LLMとRAGの組み合わせが効果的です。",
            "document_title": "RAG論文",
            "chunk_index": 0,
            "@search.score": 0.75,
        },
    ]
    return results


@pytest.fixture
def sample_documents():
    """サンプルドキュメント"""
    return [
        {
            "text_unit_id": "tu_001",
            "document_id": "doc_001",
            "content": "GraphRAGの概要説明",
            "document_title": "GraphRAG入門",
            "chunk_index": 0,
        },
        {
            "text_unit_id": "tu_002",
            "document_id": "doc_001",
            "content": "ベクトル検索の説明",
            "document_title": "GraphRAG入門",
            "chunk_index": 1,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """サンプル埋め込み"""
    return [
        [0.1] * 1536,
        [0.2] * 1536,
    ]


# ========== AzureSearchVectorStoreConfig Tests ==========


class TestAzureSearchVectorStoreConfig:
    """AzureSearchVectorStoreConfig テスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict("os.environ", {}, clear=True):
            config = AzureSearchVectorStoreConfig(
                endpoint="https://test.search.windows.net",
                api_key="test-key",
            )
        
        assert config.index_name == "monjyu-vectors"
        assert config.vector_dimensions == 1536
        assert config.enable_semantic_search is True
        assert config.batch_size == 1000
    
    def test_config_from_env(self):
        """環境変数から設定"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict("os.environ", {
            "AZURE_SEARCH_ENDPOINT": "https://env-test.search.windows.net",
            "AZURE_SEARCH_KEY": "env-test-key",
        }):
            config = AzureSearchVectorStoreConfig()
        
        assert config.endpoint == "https://env-test.search.windows.net"
        assert config.api_key == "env-test-key"
    
    def test_config_validation_missing_endpoint(self):
        """エンドポイント欠落時のバリデーション"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict("os.environ", {}, clear=True):
            config = AzureSearchVectorStoreConfig()
        
        with pytest.raises(ValueError, match="endpoint is required"):
            config.validate()
    
    def test_config_validation_missing_credentials(self):
        """クレデンシャル欠落時のバリデーション"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict("os.environ", {}, clear=True):
            config = AzureSearchVectorStoreConfig(
                endpoint="https://test.search.windows.net",
            )
        
        with pytest.raises(ValueError, match="API key or Managed Identity"):
            config.validate()
    
    def test_config_validation_managed_identity(self):
        """Managed Identity使用時のバリデーション"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        with patch.dict("os.environ", {}, clear=True):
            config = AzureSearchVectorStoreConfig(
                endpoint="https://test.search.windows.net",
                use_managed_identity=True,
            )
        
        # エラーなし
        config.validate()
    
    def test_config_to_dict(self):
        """辞書変換"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        config = AzureSearchVectorStoreConfig(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        data = config.to_dict()
        
        assert data["endpoint"] == "https://test.search.windows.net"
        assert data["index_name"] == "test-index"
        assert "api_key" not in data  # APIキーは含めない
    
    def test_hnsw_parameters(self):
        """HNSWパラメータ設定"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStoreConfig
        
        config = AzureSearchVectorStoreConfig(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            hnsw_m=8,
            hnsw_ef_construction=800,
            hnsw_ef_search=1000,
        )
        
        assert config.hnsw_m == 8
        assert config.hnsw_ef_construction == 800
        assert config.hnsw_ef_search == 1000


# ========== AzureSearchVectorStore Tests ==========


class TestAzureSearchVectorStore:
    """AzureSearchVectorStore テスト"""
    
    def test_init_with_params(self):
        """パラメータ指定での初期化"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="custom-index",
        )
        
        assert store.config.endpoint == "https://test.search.windows.net"
        assert store.config.index_name == "custom-index"
    
    def test_init_with_config(self):
        """Config指定での初期化"""
        from monjyu.search.azure_vector_store import (
            AzureSearchVectorStore,
            AzureSearchVectorStoreConfig,
        )
        
        config = AzureSearchVectorStoreConfig(
            endpoint="https://config-test.search.windows.net",
            api_key="config-key",
            vector_dimensions=768,
        )
        store = AzureSearchVectorStore(config=config)
        
        assert store.config.vector_dimensions == 768
    
    def test_init_from_env(self):
        """環境変数からの初期化"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        with patch.dict("os.environ", {
            "AZURE_SEARCH_ENDPOINT": "https://env.search.windows.net",
            "AZURE_SEARCH_KEY": "env-key",
        }):
            store = AzureSearchVectorStore()
        
        assert store.config.endpoint == "https://env.search.windows.net"
    
    def test_credential_property_initialized_once(self):
        """クレデンシャルは一度だけ初期化される"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        # 初回取得
        cred1 = store._create_credential()
        # キャッシュされているので同じオブジェクト
        cred2 = store._create_credential()
        
        assert cred1 is cred2


# ========== Index Management Tests ==========


class TestIndexManagement:
    """インデックス管理テスト"""
    
    @pytest.fixture
    def store_with_mocks(self, mock_index_client, mock_search_client):
        """モック付きVectorStore"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        store._async_index_client = mock_index_client
        store._async_search_client = mock_search_client
        
        return store
    
    @pytest.mark.asyncio
    async def test_get_index_stats(self, store_with_mocks):
        """インデックス統計取得"""
        stats = await store_with_mocks.get_index_stats_async()
        
        assert stats["document_count"] == 100
        assert stats["storage_size_bytes"] == 1024


# ========== Document Operations Tests ==========


class TestDocumentOperations:
    """ドキュメント操作テスト"""
    
    @pytest.fixture
    def store_with_mocks(self, mock_search_client):
        """モック付きVectorStore"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        store._async_search_client = mock_search_client
        
        return store
    
    @pytest.mark.asyncio
    async def test_add_documents(
        self,
        store_with_mocks,
        mock_search_client,
        sample_documents,
        sample_embeddings,
    ):
        """ドキュメント追加"""
        count = await store_with_mocks.add_documents_async(
            documents=sample_documents,
            embeddings=sample_embeddings,
        )
        
        assert count == 2
        mock_search_client.upload_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_documents_empty(self, store_with_mocks, mock_search_client):
        """空のドキュメントリスト"""
        count = await store_with_mocks.add_documents_async(documents=[])
        
        assert count == 0
        mock_search_client.upload_documents.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_add_documents_batching(self, store_with_mocks, mock_search_client):
        """バッチ処理"""
        store_with_mocks.config.batch_size = 2
        
        documents = [
            {"text_unit_id": f"tu_{i}", "content": f"Content {i}"}
            for i in range(5)
        ]
        
        count = await store_with_mocks.add_documents_async(documents=documents)
        
        assert count == 5
        # 3回呼び出し（2, 2, 1）
        assert mock_search_client.upload_documents.call_count == 3
    
    @pytest.mark.asyncio
    async def test_delete_documents(self, store_with_mocks, mock_search_client):
        """ドキュメント削除"""
        count = await store_with_mocks.delete_documents_async(["id1", "id2", "id3"])
        
        assert count == 3
        mock_search_client.delete_documents.assert_called_once()


# ========== Search Operations Tests ==========


class TestSearchOperations:
    """検索操作テスト"""
    
    @pytest.fixture
    def store_with_mocks(self, mock_search_client, sample_search_results):
        """モック付きVectorStore"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        # 非同期イテレータクラスを作成
        class AsyncSearchResults:
            def __init__(self, results):
                self.results = results
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.results):
                    raise StopAsyncIteration
                result = self.results[self.index]
                self.index += 1
                return result
        
        # AsyncMockで非同期イテレータを返す
        async def mock_search(*args, **kwargs):
            return AsyncSearchResults(sample_search_results)
        
        mock_search_client.search = mock_search
        store._async_search_client = mock_search_client
        store.config.retry_delay = 0.01  # テスト高速化
        
        return store
    
    @pytest.mark.asyncio
    async def test_vector_search(self, store_with_mocks):
        """ベクトル検索"""
        from monjyu.search.base import SearchResults
        
        query_embedding = [0.1] * 1536
        
        results = await store_with_mocks.search_async(
            query_embedding=query_embedding,
            top_k=10,
        )
        
        assert isinstance(results, SearchResults)
        assert len(results.hits) == 3
        assert results.hits[0].score == 0.95
    
    @pytest.mark.asyncio
    async def test_search_result_fields(self, store_with_mocks):
        """検索結果フィールドの確認"""
        query_embedding = [0.1] * 1536
        
        results = await store_with_mocks.search_async(
            query_embedding=query_embedding,
            top_k=10,
        )
        
        hit = results.hits[0]
        assert hit.text_unit_id == "tu_001"
        assert hit.document_id == "paper_001"
        assert hit.text == "GraphRAGは知識グラフを活用した検索手法です。"
        assert hit.document_title == "GraphRAG入門"
    
    @pytest.mark.asyncio
    async def test_hybrid_search(self, store_with_mocks, sample_search_results):
        """ハイブリッド検索"""
        from monjyu.search.base import SearchResults
        
        query_embedding = [0.1] * 1536
        
        results = await store_with_mocks.hybrid_search_async(
            query_text="GraphRAGとは",
            query_embedding=query_embedding,
            top_k=10,
        )
        
        assert isinstance(results, SearchResults)
    
    def test_sync_search(self, store_with_mocks, sample_search_results):
        """同期検索"""
        from monjyu.search.base import SearchResults
        
        query_embedding = [0.1] * 1536
        
        results = store_with_mocks.search(
            query_embedding=query_embedding,
            top_k=10,
        )
        
        assert isinstance(results, SearchResults)


# ========== AzureSearchEngine Tests ==========


class TestAzureSearchEngine:
    """AzureSearchEngine テスト"""
    
    @pytest.fixture
    def mock_embedding_client(self):
        """モック埋め込みクライアント"""
        client = MagicMock()
        client.embed = AsyncMock(return_value=[0.1] * 1536)
        return client
    
    @pytest.fixture
    def search_engine(self, mock_embedding_client, mock_search_client, sample_search_results):
        """テスト用検索エンジン"""
        from monjyu.search.azure_vector_store import (
            AzureSearchVectorStore,
            AzureSearchEngine,
        )
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        # 非同期イテレータクラスを作成
        class AsyncSearchResults:
            def __init__(self, results):
                self.results = results
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.results):
                    raise StopAsyncIteration
                result = self.results[self.index]
                self.index += 1
                return result
        
        # AsyncMockで非同期イテレータを返す
        async def mock_search(*args, **kwargs):
            return AsyncSearchResults(sample_search_results)
        
        mock_search_client.search = mock_search
        store._async_search_client = mock_search_client
        store.config.retry_delay = 0.01
        
        return AzureSearchEngine(
            vector_store=store,
            embedding_client=mock_embedding_client,
        )
    
    @pytest.mark.asyncio
    async def test_search_async_hybrid(self, search_engine):
        """ハイブリッド検索（非同期）"""
        from monjyu.search.base import SearchResults
        
        results = await search_engine.search_async(
            query="GraphRAGとは何ですか？",
            top_k=10,
            use_hybrid=True,
        )
        
        assert isinstance(results, SearchResults)
        assert len(results.hits) == 3
    
    @pytest.mark.asyncio
    async def test_search_async_vector_only(self, search_engine):
        """ベクトル検索のみ（非同期）"""
        from monjyu.search.base import SearchResults
        
        results = await search_engine.search_async(
            query="GraphRAGとは何ですか？",
            top_k=10,
            use_hybrid=False,
        )
        
        assert isinstance(results, SearchResults)
    
    def test_search_sync(self, search_engine):
        """同期検索"""
        from monjyu.search.base import SearchResults
        
        results = search_engine.search(
            query="GraphRAGとは何ですか？",
            top_k=10,
        )
        
        assert isinstance(results, SearchResults)


# ========== Factory Function Tests ==========


class TestFactoryFunctions:
    """ファクトリー関数テスト"""
    
    def test_create_azure_vector_store(self):
        """create_azure_vector_store"""
        from monjyu.search.azure_vector_store import create_azure_vector_store
        
        store = create_azure_vector_store(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="test-index",
        )
        
        assert store.config.index_name == "test-index"
    
    def test_create_azure_search_engine(self):
        """create_azure_search_engine"""
        from monjyu.search.azure_vector_store import (
            AzureSearchVectorStore,
            create_azure_search_engine,
        )
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        mock_embedding = MagicMock()
        engine = create_azure_search_engine(store, mock_embedding)
        
        assert engine.vector_store == store
        assert engine.embedding_client == mock_embedding


# ========== Context Manager Tests ==========


class TestContextManager:
    """コンテキストマネージャーテスト"""
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_search_client, mock_index_client):
        """非同期コンテキストマネージャー"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        store._async_search_client = mock_search_client
        store._async_index_client = mock_index_client
        
        async with store as s:
            assert s is store
        
        mock_search_client.close.assert_called_once()
        mock_index_client.close.assert_called_once()


# ========== Retry Logic Tests ==========


class TestRetryLogic:
    """リトライロジックテスト"""
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """失敗時のリトライ"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        store.config.max_retries = 3
        store.config.retry_delay = 0.01  # テスト高速化
        
        call_count = 0
        
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await store._retry_async(failing_operation)
        
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """リトライ回数超過"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        store.config.max_retries = 2
        store.config.retry_delay = 0.01
        
        async def always_failing():
            raise Exception("Permanent failure")
        
        with pytest.raises(Exception, match="Permanent failure"):
            await store._retry_async(always_failing)


# ========== Integration Tests ==========


class TestIntegration:
    """統合テスト（モック使用）"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(
        self,
        mock_search_client,
        mock_index_client,
        sample_documents,
        sample_embeddings,
    ):
        """完全なワークフロー"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        store._async_index_client = mock_index_client
        store._async_search_client = mock_search_client
        
        # ドキュメント追加
        count = await store.add_documents_async(
            documents=sample_documents,
            embeddings=sample_embeddings,
        )
        
        assert count == 2
        
        # クローズ
        await store.close()
        
        mock_search_client.close.assert_called_once()
        mock_index_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_and_transform(
        self,
        mock_search_client,
        sample_search_results,
    ):
        """検索と結果変換"""
        from monjyu.search.azure_vector_store import AzureSearchVectorStore
        from monjyu.search.base import SearchResults
        
        store = AzureSearchVectorStore(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        # 非同期イテレータクラスを作成
        class AsyncSearchResults:
            def __init__(self, results):
                self.results = results
                self.index = 0
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.results):
                    raise StopAsyncIteration
                result = self.results[self.index]
                self.index += 1
                return result
        
        # AsyncMockで非同期イテレータを返す
        async def mock_search(*args, **kwargs):
            return AsyncSearchResults(sample_search_results)
        
        mock_search_client.search = mock_search
        store._async_search_client = mock_search_client
        store.config.retry_delay = 0.01
        
        query_embedding = [0.1] * 1536
        results = await store.search_async(
            query_embedding=query_embedding,
            top_k=10,
        )
        
        # SearchResultsに変換されていること
        assert isinstance(results, SearchResults)
        
        # スコア順でソートされていること
        scores = [hit.score for hit in results.hits]
        assert scores == sorted(scores, reverse=True)
