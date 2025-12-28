# Azure OpenAI Embedding Client
"""
Embedding client using Azure OpenAI for production use.

Supports models:
- text-embedding-3-large (3072 dimensions)
- text-embedding-3-small (1536 dimensions)
- text-embedding-ada-002 (1536 dimensions)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from monjyu.embedding.base import EmbeddingClient

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI


class AzureOpenAIEmbeddingClient(EmbeddingClient):
    """Azure OpenAI 埋め込みクライアント（本番用）
    
    Azure OpenAIを使用してテキストの埋め込みを生成する。
    高精度で本番環境に最適。
    
    Example:
        >>> client = AzureOpenAIEmbeddingClient(
        ...     deployment="text-embedding-3-large",
        ...     endpoint="https://xxx.openai.azure.com/",
        ...     api_key="your-api-key",
        ... )
        >>> embedding = await client.embed("Hello, world!")
        >>> print(len(embedding))
        3072
    
    Environment Variables:
        AZURE_OPENAI_ENDPOINT: Azure OpenAI エンドポイント
        AZURE_OPENAI_API_KEY: API キー
    """
    
    # モデルごとの次元数マッピング
    DIMENSIONS: dict[str, int] = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    # デフォルト次元数
    DEFAULT_DIMENSIONS = 1536
    
    def __init__(
        self,
        deployment: str = "text-embedding-3-large",
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview",
        dimensions: int | None = None,
        max_retries: int = 3,
    ) -> None:
        """初期化
        
        Args:
            deployment: Azure OpenAI デプロイメント名
            endpoint: Azure OpenAI エンドポイント（省略時は環境変数から取得）
            api_key: APIキー（省略時は環境変数から取得）
            api_version: API バージョン
            dimensions: 埋め込み次元数（text-embedding-3で変更可能）
            max_retries: 最大リトライ回数
        """
        from openai import AsyncAzureOpenAI
        
        self.deployment = deployment
        self.api_version = api_version
        self.max_retries = max_retries
        
        # 次元数の決定
        if dimensions is not None:
            self._dimensions = dimensions
        else:
            self._dimensions = self.DIMENSIONS.get(deployment, self.DEFAULT_DIMENSIONS)
        
        # クライアント初期化
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        
        if not self._endpoint:
            msg = "Azure OpenAI エンドポイントが設定されていません (AZURE_OPENAI_ENDPOINT)"
            raise ValueError(msg)
        
        self._client: AsyncAzureOpenAI | None = None
    
    @property
    def client(self) -> "AsyncAzureOpenAI":
        """クライアントを取得（遅延初期化）"""
        if self._client is None:
            from openai import AsyncAzureOpenAI
            
            if self._api_key:
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self._endpoint,
                    api_key=self._api_key,
                    api_version=self.api_version,
                    max_retries=self.max_retries,
                )
            else:
                # DefaultAzureCredential を使用
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
                
                credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    credential, "https://cognitiveservices.azure.com/.default"
                )
                
                self._client = AsyncAzureOpenAI(
                    azure_endpoint=self._endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=self.api_version,
                    max_retries=self.max_retries,
                )
        
        return self._client
    
    @property
    def dimensions(self) -> int:
        """埋め込みの次元数"""
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        """モデル名（デプロイメント名）"""
        return self.deployment
    
    async def embed(self, text: str) -> list[float]:
        """単一テキストの埋め込みを生成
        
        Args:
            text: 埋め込みを生成するテキスト
            
        Returns:
            埋め込みベクトル
        """
        kwargs: dict[str, Any] = {
            "input": text,
            "model": self.deployment,
        }
        
        # text-embedding-3 系は次元数を指定可能
        if self.deployment.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions
        
        response = await self.client.embeddings.create(**kwargs)
        return response.data[0].embedding
    
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """バッチでテキストの埋め込みを生成
        
        Azure OpenAIはネイティブバッチをサポート。
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ（最大2048）
            
        Returns:
            埋め込みベクトルのリスト
        """
        results: list[list[float]] = []
        
        # Azure OpenAIのバッチ制限
        batch_size = min(batch_size, 2048)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            kwargs: dict[str, Any] = {
                "input": batch,
                "model": self.deployment,
            }
            
            if self.deployment.startswith("text-embedding-3"):
                kwargs["dimensions"] = self._dimensions
            
            response = await self.client.embeddings.create(**kwargs)
            
            # レスポンスの順序を保持
            batch_embeddings = [None] * len(batch)
            for item in response.data:
                batch_embeddings[item.index] = item.embedding
            
            results.extend(batch_embeddings)
        
        return results
    
    async def close(self) -> None:
        """クライアントをクローズ"""
        if self._client is not None:
            await self._client.close()
            self._client = None
    
    async def __aenter__(self) -> "AzureOpenAIEmbeddingClient":
        """コンテキストマネージャー開始"""
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """コンテキストマネージャー終了"""
        await self.close()
