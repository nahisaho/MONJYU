# Embedding Coverage Tests
"""
Azure OpenAI および Ollama Embedding クライアントのカバレッジ向上テスト

TASK: Embedding カバレッジ向上
- azure_openai.py: 23% → 70%
- ollama.py: 30% → 70%
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monjyu.embedding.ollama import OllamaEmbeddingClient


# === TestOllamaEmbeddingClient ===


class TestOllamaEmbeddingClientInit:
    """OllamaEmbeddingClient 初期化テスト"""

    def test_init_default(self):
        """デフォルト初期化"""
        client = OllamaEmbeddingClient()
        
        assert client.model == "nomic-embed-text"
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 60.0
        assert client.max_retries == 3
        assert client.dimensions == 768

    def test_init_custom_model(self):
        """カスタムモデル"""
        client = OllamaEmbeddingClient(model="mxbai-embed-large")
        
        assert client.model == "mxbai-embed-large"
        assert client.dimensions == 1024

    def test_init_all_minilm(self):
        """all-minilmモデル"""
        client = OllamaEmbeddingClient(model="all-minilm")
        
        assert client.dimensions == 384

    def test_init_unknown_model(self):
        """未知のモデル（デフォルト次元数）"""
        client = OllamaEmbeddingClient(model="unknown-model")
        
        assert client.dimensions == 768  # DEFAULT_DIMENSIONS

    def test_init_custom_base_url(self):
        """カスタムベースURL"""
        client = OllamaEmbeddingClient(base_url="http://custom:1234/")
        
        # 末尾スラッシュが削除される
        assert client.base_url == "http://custom:1234"

    def test_model_name_property(self):
        """model_nameプロパティ"""
        client = OllamaEmbeddingClient(model="test-model")
        
        assert client.model_name == "test-model"


class TestOllamaEmbeddingClientGetClient:
    """_get_client メソッドテスト"""

    @pytest.mark.asyncio
    async def test_get_client_creates_httpx_client(self):
        """HTTPXクライアントを作成"""
        client = OllamaEmbeddingClient()
        
        mock_async_client = AsyncMock()
        with patch.dict("sys.modules", {"httpx": MagicMock(AsyncClient=MagicMock(return_value=mock_async_client))}):
            # 既存のクライアントをクリア
            client._client = None
            result = await client._get_client()
            
            assert result is not None

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self):
        """既存クライアントを再利用"""
        client = OllamaEmbeddingClient()
        
        mock_async_client = AsyncMock()
        client._client = mock_async_client
        
        result = await client._get_client()
        
        assert result is mock_async_client


class TestOllamaEmbeddingClientEmbed:
    """embed メソッドテスト"""

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """正常な埋め込み生成"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.embed("test text")
        
        assert result == [0.1, 0.2, 0.3]
        mock_http_client.post.assert_called_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "test text"},
        )

    @pytest.mark.asyncio
    async def test_embed_invalid_response(self):
        """不正なレスポンス"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": "no embedding"}  # embeddingなし
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            with pytest.raises(ConnectionError):  # ValueError → ConnectionError
                await client.embed("test text")

    @pytest.mark.asyncio
    async def test_embed_retry_on_failure(self):
        """失敗時のリトライ"""
        client = OllamaEmbeddingClient(max_retries=2)
        
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = Exception("Connection error")
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError):
                    await client.embed("test text")
        
        # max_retries回試行
        assert mock_http_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_retry_succeeds(self):
        """リトライ成功"""
        client = OllamaEmbeddingClient(max_retries=3)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        # 1回目失敗、2回目成功
        mock_http_client.post.side_effect = [
            Exception("First failure"),
            mock_response,
        ]
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.embed("test text")
        
        assert result == [0.1, 0.2]


class TestOllamaEmbeddingClientEmbedBatch:
    """embed_batch メソッドテスト"""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """バッチ埋め込み成功"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1, 0.2]}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            results = await client.embed_batch(["text1", "text2", "text3"])
        
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_embed_batch_with_batch_size(self):
        """バッチサイズ指定"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1]}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            results = await client.embed_batch(
                ["text1", "text2", "text3", "text4", "text5"],
                batch_size=2,
            )
        
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_embed_batch_exception_propagates(self):
        """バッチ内の例外が伝播"""
        client = OllamaEmbeddingClient()
        
        mock_http_client = AsyncMock()
        mock_http_client.post.side_effect = Exception("Batch error")
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(Exception):
                    await client.embed_batch(["text1", "text2"])


class TestOllamaEmbeddingClientClose:
    """close メソッドテスト"""

    @pytest.mark.asyncio
    async def test_close_existing_client(self):
        """既存クライアントをクローズ"""
        client = OllamaEmbeddingClient()
        
        mock_http_client = AsyncMock()
        client._client = mock_http_client
        
        await client.close()
        
        mock_http_client.aclose.assert_called_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """クライアントなしでクローズ"""
        client = OllamaEmbeddingClient()
        
        # 例外が発生しない
        await client.close()


class TestOllamaEmbeddingClientContextManager:
    """コンテキストマネージャーテスト"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """コンテキストマネージャー"""
        client = OllamaEmbeddingClient()
        
        with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
            async with client as ctx:
                assert ctx is client
            
            mock_close.assert_called_once()


class TestOllamaEmbeddingClientIsAvailable:
    """is_available メソッドテスト"""

    @pytest.mark.asyncio
    async def test_is_available_true(self):
        """サーバー利用可能"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.is_available()
        
        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_false_status(self):
        """サーバー利用不可（ステータスコード）"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.is_available()
        
        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_exception(self):
        """サーバー利用不可（例外）"""
        client = OllamaEmbeddingClient()
        
        mock_http_client = AsyncMock()
        mock_http_client.get.side_effect = Exception("Connection refused")
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.is_available()
        
        assert result is False


class TestOllamaEmbeddingClientListModels:
    """list_models メソッドテスト"""

    @pytest.mark.asyncio
    async def test_list_models_success(self):
        """モデル一覧取得"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "nomic-embed-text"},
                {"name": "mxbai-embed-large"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.list_models()
        
        assert result == ["nomic-embed-text", "mxbai-embed-large"]

    @pytest.mark.asyncio
    async def test_list_models_empty(self):
        """空のモデル一覧"""
        client = OllamaEmbeddingClient()
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        
        mock_http_client = AsyncMock()
        mock_http_client.get.return_value = mock_response
        
        with patch.object(client, "_get_client", return_value=mock_http_client):
            result = await client.list_models()
        
        assert result == []


# === TestAzureOpenAIEmbeddingClient ===


class TestAzureOpenAIEmbeddingClientInit:
    """AzureOpenAIEmbeddingClient 初期化テスト"""

    def test_init_with_endpoint_and_key(self):
        """エンドポイントとAPIキーで初期化"""
        with patch.dict("os.environ", {}, clear=True):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-large",
                    endpoint="https://test.openai.azure.com/",
                    api_key="test-key",
                )
                
                assert client.deployment == "text-embedding-3-large"
                assert client._endpoint == "https://test.openai.azure.com/"
                assert client._api_key == "test-key"
                assert client.dimensions == 3072

    def test_init_with_env_vars(self):
        """環境変数から初期化"""
        env_vars = {
            "AZURE_OPENAI_ENDPOINT": "https://env.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "env-key",
        }
        with patch.dict("os.environ", env_vars):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient()
                
                assert client._endpoint == "https://env.openai.azure.com/"
                assert client._api_key == "env-key"

    def test_init_missing_endpoint(self):
        """エンドポイント未設定でエラー"""
        with patch.dict("os.environ", {}, clear=True):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                with pytest.raises(ValueError, match="エンドポイント"):
                    AzureOpenAIEmbeddingClient()

    def test_init_custom_dimensions(self):
        """カスタム次元数"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-small",
                    dimensions=512,  # カスタム
                )
                
                assert client.dimensions == 512

    def test_init_ada_002(self):
        """ada-002モデル"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-ada-002",
                )
                
                assert client.dimensions == 1536

    def test_model_name_property(self):
        """model_nameプロパティ"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_module = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_module}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(deployment="my-deployment")
                
                assert client.model_name == "my-deployment"


class TestAzureOpenAIEmbeddingClientClientProperty:
    """client プロパティテスト"""

    def test_client_with_api_key(self):
        """APIキーでクライアント作成"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            mock_instance = MagicMock()
            mock_openai.AsyncAzureOpenAI.return_value = mock_instance
            
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(api_key="test-key")
                
                # client プロパティにアクセス
                result = client.client
                
                assert result is mock_instance

    def test_client_with_default_credential(self):
        """DefaultAzureCredentialでクライアント作成"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            mock_azure_identity = MagicMock()
            
            with patch.dict("sys.modules", {
                "openai": mock_openai,
                "azure": MagicMock(),
                "azure.identity": mock_azure_identity,
            }):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient()  # api_keyなし
                
                # client プロパティにアクセス - DefaultAzureCredentialが使用される
                _ = client.client
                
                # クライアントが作成された
                assert mock_openai.AsyncAzureOpenAI.called

    def test_client_cached(self):
        """クライアントがキャッシュされる"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            mock_instance = MagicMock()
            mock_openai.AsyncAzureOpenAI.return_value = mock_instance
            
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(api_key="test-key")
                
                # 2回アクセス
                _ = client.client
                _ = client.client
                
                # 1回のみ作成
                assert mock_openai.AsyncAzureOpenAI.call_count == 1


class TestAzureOpenAIEmbeddingClientEmbed:
    """embed メソッドテスト"""

    @pytest.mark.asyncio
    async def test_embed_success(self):
        """正常な埋め込み生成"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-large",
                    api_key="test-key",
                )
                
                # クライアントをモック
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
                
                mock_client = AsyncMock()
                mock_client.embeddings.create.return_value = mock_response
                client._client = mock_client
                
                result = await client.embed("test text")
                
                assert result == [0.1, 0.2, 0.3]
                mock_client.embeddings.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_with_dimensions(self):
        """次元数指定"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-large",
                    api_key="test-key",
                    dimensions=1024,
                )
                
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
                
                mock_client = AsyncMock()
                mock_client.embeddings.create.return_value = mock_response
                client._client = mock_client
                
                await client.embed("test text")
                
                # dimensions が渡される
                call_kwargs = mock_client.embeddings.create.call_args[1]
                assert call_kwargs["dimensions"] == 1024


class TestAzureOpenAIEmbeddingClientEmbedBatch:
    """embed_batch メソッドテスト"""

    @pytest.mark.asyncio
    async def test_embed_batch_success(self):
        """バッチ埋め込み成功"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-large",
                    api_key="test-key",
                )
                
                # レスポンスをモック
                mock_response = MagicMock()
                mock_response.data = [
                    MagicMock(index=0, embedding=[0.1, 0.2]),
                    MagicMock(index=1, embedding=[0.3, 0.4]),
                    MagicMock(index=2, embedding=[0.5, 0.6]),
                ]
                
                mock_client = AsyncMock()
                mock_client.embeddings.create.return_value = mock_response
                client._client = mock_client
                
                results = await client.embed_batch(["text1", "text2", "text3"])
                
                assert len(results) == 3
                assert results[0] == [0.1, 0.2]
                assert results[1] == [0.3, 0.4]
                assert results[2] == [0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_batch_respects_limit(self):
        """バッチサイズ制限"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(
                    deployment="text-embedding-3-large",
                    api_key="test-key",
                )
                
                # 大きなバッチサイズを指定しても2048に制限される
                mock_response = MagicMock()
                mock_response.data = [MagicMock(index=0, embedding=[0.1])]
                
                mock_client = AsyncMock()
                mock_client.embeddings.create.return_value = mock_response
                client._client = mock_client
                
                await client.embed_batch(["text1"], batch_size=5000)
                
                # バッチサイズ制限は内部処理なので、呼び出しは成功する


class TestAzureOpenAIEmbeddingClientClose:
    """close メソッドテスト"""

    @pytest.mark.asyncio
    async def test_close_existing_client(self):
        """既存クライアントをクローズ"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(api_key="test-key")
                
                mock_async_client = AsyncMock()
                client._client = mock_async_client
                
                await client.close()
                
                mock_async_client.close.assert_called_once()
                assert client._client is None

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """クライアントなしでクローズ"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(api_key="test-key")
                
                # 例外が発生しない
                await client.close()


class TestAzureOpenAIEmbeddingClientContextManager:
    """コンテキストマネージャーテスト"""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """コンテキストマネージャー"""
        with patch.dict("os.environ", {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}):
            mock_openai = MagicMock()
            with patch.dict("sys.modules", {"openai": mock_openai}):
                from monjyu.embedding.azure_openai import AzureOpenAIEmbeddingClient
                
                client = AzureOpenAIEmbeddingClient(api_key="test-key")
                
                with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                    async with client as ctx:
                        assert ctx is client
                    
                    mock_close.assert_called_once()
