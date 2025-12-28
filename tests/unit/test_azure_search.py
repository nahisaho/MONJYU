# Azure AI Search ユニットテスト
"""
Azure AI Search Vector Indexer のユニットテスト

FEAT-014: Azure AI Search 統合テスト
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Azure SDK をモック
azure_mock = MagicMock()
azure_core_mock = MagicMock()
azure_identity_mock = MagicMock()
azure_search_mock = MagicMock()

sys.modules["azure"] = azure_mock
sys.modules["azure.core"] = azure_core_mock
sys.modules["azure.core.credentials"] = azure_core_mock
sys.modules["azure.identity"] = azure_identity_mock
sys.modules["azure.search"] = azure_search_mock
sys.modules["azure.search.documents"] = azure_search_mock
sys.modules["azure.search.documents.indexes"] = azure_search_mock
sys.modules["azure.search.documents.indexes.models"] = azure_search_mock
sys.modules["azure.search.documents.models"] = azure_search_mock

# モッククラスを設定
azure_core_mock.AzureKeyCredential = MagicMock
azure_identity_mock.DefaultAzureCredential = MagicMock
azure_search_mock.SearchClient = MagicMock
azure_search_mock.SearchIndexClient = MagicMock
azure_search_mock.VectorizedQuery = MagicMock

from monjyu.index.azure_search import (
    AzureAISearchIndexer,
    AzureSearchConfig,
    create_azure_search_indexer,
)
from monjyu.index.base import SearchResult


class TestAzureSearchConfig:
    """AzureSearchConfig のテスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        config = AzureSearchConfig(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        assert config.endpoint == "https://test.search.windows.net"
        assert config.api_key == "test-key"
        assert config.index_name == "monjyu-text-units"
        assert config.use_managed_identity is False
        
        # ベクトル設定
        assert config.vector_field == "vector"
        assert config.vector_dimensions == 1536
        assert config.text_field == "content"
        
        # HNSW パラメータ（Azure推奨値）
        assert config.hnsw_m == 4
        assert config.hnsw_ef_construction == 400
        assert config.hnsw_ef_search == 500
        
        # リトライ設定
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        
        # セマンティック検索
        assert config.enable_semantic_search is True
    
    def test_env_var_fallback(self):
        """環境変数からの取得テスト"""
        with patch.dict(os.environ, {
            "AZURE_SEARCH_ENDPOINT": "https://env.search.windows.net",
            "AZURE_SEARCH_KEY": "env-api-key",
        }):
            config = AzureSearchConfig()
            
            assert config.endpoint == "https://env.search.windows.net"
            assert config.api_key == "env-api-key"
    
    def test_explicit_values_override_env(self):
        """明示的な値が環境変数より優先されるテスト"""
        with patch.dict(os.environ, {
            "AZURE_SEARCH_ENDPOINT": "https://env.search.windows.net",
        }):
            config = AzureSearchConfig(
                endpoint="https://explicit.search.windows.net",
                api_key="explicit-key",
            )
            
            assert config.endpoint == "https://explicit.search.windows.net"
    
    def test_custom_hnsw_parameters(self):
        """カスタムHNSWパラメータのテスト"""
        config = AzureSearchConfig(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            hnsw_m=8,
            hnsw_ef_construction=800,
            hnsw_ef_search=1000,
        )
        
        assert config.hnsw_m == 8
        assert config.hnsw_ef_construction == 800
        assert config.hnsw_ef_search == 1000
    
    def test_managed_identity_mode(self):
        """Managed Identity モードのテスト"""
        config = AzureSearchConfig(
            endpoint="https://test.search.windows.net",
            use_managed_identity=True,
        )
        
        assert config.use_managed_identity is True
        assert config.api_key is None  # API キー不要


class TestAzureAISearchIndexer:
    """AzureAISearchIndexer のテスト"""
    
    def test_init_with_api_key(self):
        """APIキー認証での初期化テスト"""
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        
        assert indexer.config.endpoint == "https://test.search.windows.net"
        assert indexer.config.api_key == "test-key"
        assert indexer.config.use_managed_identity is False
    
    def test_init_with_config(self):
        """設定オブジェクトでの初期化テスト"""
        config = AzureSearchConfig(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="custom-index",
            vector_dimensions=768,
        )
        
        indexer = AzureAISearchIndexer(config=config)
        
        assert indexer.config.index_name == "custom-index"
        assert indexer.config.vector_dimensions == 768
    
    def test_init_missing_endpoint(self):
        """エンドポイント未設定エラーのテスト"""
        with pytest.raises(ValueError, match="エンドポイントが設定されていません"):
            AzureAISearchIndexer(api_key="test-key")
    
    def test_init_missing_credentials(self):
        """認証情報未設定エラーのテスト"""
        with pytest.raises(ValueError, match="APIキーまたはManaged Identity"):
            AzureAISearchIndexer(endpoint="https://test.search.windows.net")
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    def test_search(self, mock_search_client):
        """ベクトル検索のテスト"""
        # モックの設定
        mock_results = [
            {
                "id": "doc1",
                "@search.score": 0.95,
                "content": "Test content 1",
            },
            {
                "id": "doc2",
                "@search.score": 0.85,
                "content": "Test content 2",
            },
        ]
        mock_search_client.search.return_value = iter(mock_results)
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        # 検索実行
        query_embedding = [0.1] * 1536
        results = indexer.search(query_embedding, top_k=5)
        
        # 結果の検証
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        assert results[0].metadata["content"] == "Test content 1"
        
        # search が正しく呼び出されたか
        mock_search_client.search.assert_called_once()
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    def test_search_hybrid(self, mock_search_client):
        """ハイブリッド検索のテスト"""
        mock_results = [
            {
                "id": "doc1",
                "@search.score": 0.95,
                "@search.reranker_score": 3.5,
                "content": "Transformer architecture",
            },
        ]
        mock_search_client.search.return_value = iter(mock_results)
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        # ハイブリッド検索実行
        results = indexer.search_hybrid(
            query_text="transformer",
            query_embedding=[0.1] * 1536,
            top_k=5,
            use_semantic_reranker=True,
        )
        
        # 結果の検証
        assert len(results) == 1
        assert results[0].metadata.get("reranker_score") == 3.5
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    def test_add_documents(self, mock_search_client):
        """ドキュメント追加のテスト"""
        mock_search_client.upload_documents.return_value = None
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        # ドキュメント追加
        embeddings = [[0.1] * 1536, [0.2] * 1536]
        ids = ["doc1", "doc2"]
        metadata = [
            {"content": "Content 1"},
            {"content": "Content 2"},
        ]
        
        indexer.add(embeddings, ids, metadata)
        
        # upload_documents が呼ばれたか
        mock_search_client.upload_documents.assert_called_once()
        call_args = mock_search_client.upload_documents.call_args
        documents = call_args.kwargs["documents"]
        
        assert len(documents) == 2
        assert documents[0]["id"] == "doc1"
        assert documents[0]["content"] == "Content 1"
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    def test_count(self, mock_search_client):
        """ドキュメント数取得のテスト"""
        mock_search_client.get_document_count.return_value = 100
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        count = indexer.count()
        
        assert count == 100
        mock_search_client.get_document_count.assert_called_once()
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    def test_delete(self, mock_search_client):
        """ドキュメント削除のテスト"""
        mock_search_client.delete_documents.return_value = None
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        indexer.delete(["doc1", "doc2"])
        
        mock_search_client.delete_documents.assert_called_once()
        call_args = mock_search_client.delete_documents.call_args
        documents = call_args.kwargs["documents"]
        
        assert len(documents) == 2
        assert documents[0]["id"] == "doc1"


class TestRetryLogic:
    """リトライロジックのテスト"""
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    @patch("monjyu.index.azure_search.time.sleep")
    def test_retry_on_failure(self, mock_sleep, mock_search_client):
        """失敗時のリトライテスト"""
        # 最初の2回は失敗、3回目で成功
        mock_search_client.upload_documents.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            None,  # 成功
        ]
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        # エラーが発生しないことを確認
        indexer.add([[0.1] * 1536], ["doc1"])
        
        # 3回呼ばれたことを確認
        assert mock_search_client.upload_documents.call_count == 3
        
        # sleep が2回呼ばれたことを確認（指数バックオフ）
        assert mock_sleep.call_count == 2
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.search_client")
    @patch("monjyu.index.azure_search.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep, mock_search_client):
        """最大リトライ回数超過のテスト"""
        mock_search_client.upload_documents.side_effect = Exception("Persistent error")
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._search_client = mock_search_client
        
        # 最大リトライ後にエラーが発生
        with pytest.raises(Exception, match="Persistent error"):
            indexer.add([[0.1] * 1536], ["doc1"])
        
        # max_retries 回呼ばれた
        assert mock_search_client.upload_documents.call_count == 3


class TestFactoryFunction:
    """ファクトリ関数のテスト"""
    
    def test_create_azure_search_indexer(self):
        """ファクトリ関数のテスト"""
        indexer = create_azure_search_indexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
            index_name="custom-index",
        )
        
        assert isinstance(indexer, AzureAISearchIndexer)
        assert indexer.config.index_name == "custom-index"
    
    def test_create_with_managed_identity(self):
        """Managed Identity でのファクトリ関数テスト"""
        indexer = create_azure_search_indexer(
            endpoint="https://test.search.windows.net",
            use_managed_identity=True,
        )
        
        assert indexer.config.use_managed_identity is True


class TestIndexCreation:
    """インデックス作成のテスト"""
    
    @pytest.mark.skip(reason="Azure SDK types require actual module imports - tested in integration")
    def test_create_index_if_not_exists_new(self):
        """新規インデックス作成のテスト（統合テストで実施）"""
        pass
    
    @patch("monjyu.index.azure_search.AzureAISearchIndexer.index_client")
    def test_create_index_if_not_exists_existing(self, mock_index_client):
        """既存インデックスのテスト"""
        # get_index で成功（存在する）
        mock_index_client.get_index.return_value = MagicMock()
        
        indexer = AzureAISearchIndexer(
            endpoint="https://test.search.windows.net",
            api_key="test-key",
        )
        indexer._index_client = mock_index_client
        
        result = indexer.create_index_if_not_exists()
        
        assert result is False  # 既存
        mock_index_client.create_or_update_index.assert_not_called()
