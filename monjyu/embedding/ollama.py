# Ollama Embedding Client
"""
Embedding client using Ollama for local development.

Supports models:
- nomic-embed-text (768 dimensions)
- mxbai-embed-large (1024 dimensions)
- all-minilm (384 dimensions)
"""

from __future__ import annotations

import asyncio
from typing import Any

from monjyu.embedding.base import EmbeddingClient


class OllamaEmbeddingClient(EmbeddingClient):
    """Ollama 埋め込みクライアント（ローカル開発用）
    
    Ollamaを使用してテキストの埋め込みを生成する。
    ローカル環境での開発・テストに最適。
    
    Example:
        >>> client = OllamaEmbeddingClient(model="nomic-embed-text")
        >>> embedding = await client.embed("Hello, world!")
        >>> print(len(embedding))
        768
        
        >>> # バッチ処理
        >>> embeddings = await client.embed_batch(["text1", "text2"])
        >>> print(len(embeddings))
        2
    
    Attributes:
        model: 使用するOllamaモデル名
        base_url: Ollama APIのベースURL
    """
    
    # モデルごとの次元数マッピング
    DIMENSIONS: dict[str, int] = {
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "all-minilm": 384,
        "snowflake-arctic-embed": 1024,
        "bge-m3": 1024,
    }
    
    # デフォルト次元数（未知のモデル用）
    DEFAULT_DIMENSIONS = 768
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        """初期化
        
        Args:
            model: Ollamaモデル名
            base_url: Ollama APIのベースURL
            timeout: リクエストタイムアウト（秒）
            max_retries: 最大リトライ回数
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._dimensions = self.DIMENSIONS.get(model, self.DEFAULT_DIMENSIONS)
        self._client: Any = None
    
    @property
    def dimensions(self) -> int:
        """埋め込みの次元数"""
        return self._dimensions
    
    @property
    def model_name(self) -> str:
        """モデル名"""
        return self.model
    
    async def _get_client(self) -> Any:
        """HTTPクライアントを取得（遅延初期化）"""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client
    
    async def embed(self, text: str) -> list[float]:
        """単一テキストの埋め込みを生成
        
        Args:
            text: 埋め込みを生成するテキスト
            
        Returns:
            埋め込みベクトル
            
        Raises:
            ConnectionError: Ollamaサーバーに接続できない場合
            ValueError: レスポンスが不正な場合
        """
        client = await self._get_client()
        
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                
                data = response.json()
                if "embedding" not in data:
                    msg = f"レスポンスに 'embedding' が含まれていません: {data}"
                    raise ValueError(msg)
                
                return data["embedding"]
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数バックオフ
        
        msg = f"Ollama API呼び出しに失敗しました: {last_error}"
        raise ConnectionError(msg) from last_error
    
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 10,
    ) -> list[list[float]]:
        """バッチでテキストの埋め込みを生成
        
        Ollamaはネイティブバッチをサポートしていないため、
        並行リクエストで処理する。
        
        Args:
            texts: テキストのリスト
            batch_size: 同時処理数
            
        Returns:
            埋め込みベクトルのリスト
        """
        results: list[list[float]] = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.embed(text) for text in batch],
                return_exceptions=True,
            )
            
            for result in batch_results:
                if isinstance(result, Exception):
                    raise result
                results.append(result)
        
        return results
    
    async def close(self) -> None:
        """クライアントをクローズ"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def __aenter__(self) -> "OllamaEmbeddingClient":
        """コンテキストマネージャー開始"""
        return self
    
    async def __aexit__(self, *args: Any) -> None:
        """コンテキストマネージャー終了"""
        await self.close()
    
    async def is_available(self) -> bool:
        """Ollamaサーバーが利用可能かチェック
        
        Returns:
            利用可能な場合True
        """
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> list[str]:
        """利用可能なモデル一覧を取得
        
        Returns:
            モデル名のリスト
        """
        client = await self._get_client()
        response = await client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        
        data = response.json()
        return [model["name"] for model in data.get("models", [])]
