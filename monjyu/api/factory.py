# MONJYU Component Factory
"""
monjyu.api.factory - コンポーネントファクトリー

FEAT-007: Python API (MONJYU Facade)
- 各コンポーネントの遅延初期化
- 依存性注入のサポート
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol

from monjyu.api.base import MONJYUConfig

if TYPE_CHECKING:
    from monjyu.citation import CitationNetworkManager


# ========== Protocols ==========


class EmbeddingClientProtocol(Protocol):
    """埋め込みクライアントプロトコル"""

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class LLMClientProtocol(Protocol):
    """LLMクライアントプロトコル"""

    def generate(self, prompt: str, **kwargs: Any) -> str: ...


class VectorSearcherProtocol(Protocol):
    """ベクトル検索プロトコル"""

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]: ...


# ========== Component Factory ==========


class ComponentFactory:
    """コンポーネントファクトリー"""

    def __init__(self, config: MONJYUConfig):
        self.config = config
        self._cached_components: dict[str, Any] = {}

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cached_components.clear()

    def get_embedding_client(self) -> EmbeddingClientProtocol:
        """埋め込みクライアントを取得"""
        if "embedding" not in self._cached_components:
            self._cached_components["embedding"] = self._create_embedding_client()
        return self._cached_components["embedding"]

    def get_llm_client(self) -> LLMClientProtocol:
        """LLMクライアントを取得"""
        if "llm" not in self._cached_components:
            self._cached_components["llm"] = self._create_llm_client()
        return self._cached_components["llm"]

    def get_citation_network_manager(self) -> CitationNetworkManager:
        """引用ネットワークマネージャーを取得"""
        if "citation" not in self._cached_components:
            self._cached_components["citation"] = self._create_citation_manager()
        return self._cached_components["citation"]

    # ========== Internal Creation Methods ==========

    def _create_embedding_client(self) -> EmbeddingClientProtocol:
        """埋め込みクライアントを作成"""
        if self.config.environment == "azure":
            return self._create_azure_embedding_client()
        else:
            return self._create_local_embedding_client()

    def _create_local_embedding_client(self) -> EmbeddingClientProtocol:
        """ローカル埋め込みクライアントを作成"""
        # MockまたはOllamaクライアント
        return MockEmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )

    def _create_azure_embedding_client(self) -> EmbeddingClientProtocol:
        """Azure埋め込みクライアントを作成"""
        # 将来的にはAzure OpenAI Embeddingを実装
        return MockEmbeddingClient(
            model=self.config.embedding_model,
            base_url=self.config.azure_openai_endpoint or "",
        )

    def _create_llm_client(self) -> LLMClientProtocol:
        """LLMクライアントを作成"""
        if self.config.environment == "azure":
            return self._create_azure_llm_client()
        else:
            return self._create_local_llm_client()

    def _create_local_llm_client(self) -> LLMClientProtocol:
        """ローカルLLMクライアントを作成"""
        return MockLLMClient(
            model=self.config.llm_model,
            base_url=self.config.ollama_base_url,
        )

    def _create_azure_llm_client(self) -> LLMClientProtocol:
        """Azure LLMクライアントを作成"""
        return MockLLMClient(
            model=self.config.llm_model,
            base_url=self.config.azure_openai_endpoint or "",
        )

    def _create_citation_manager(self) -> CitationNetworkManager:
        """引用ネットワークマネージャーを作成"""
        from monjyu.citation import CitationNetworkManager

        return CitationNetworkManager()


# ========== Mock Implementations ==========


class MockEmbeddingClient:
    """モック埋め込みクライアント（テスト・開発用）"""

    def __init__(self, model: str = "", base_url: str = ""):
        self.model = model
        self.base_url = base_url
        self._dimension = 384

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込み"""
        # 簡易的なハッシュベース埋め込み
        import hashlib

        hash_val = hashlib.md5(text.encode()).hexdigest()
        # ハッシュを数値ベクトルに変換
        vector = []
        for i in range(0, len(hash_val), 2):
            val = int(hash_val[i : i + 2], 16) / 255.0 - 0.5
            vector.append(val)
        # 次元を調整
        while len(vector) < self._dimension:
            vector.extend(vector[: self._dimension - len(vector)])
        return vector[: self._dimension]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを埋め込み"""
        return [self.embed(text) for text in texts]


class MockLLMClient:
    """モックLLMクライアント（テスト・開発用）"""

    def __init__(self, model: str = "", base_url: str = ""):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """テキスト生成"""
        # モックレスポンス
        return f"[Mock Response for: {prompt[:50]}...]"


class MockVectorSearcher:
    """モックベクトル検索（テスト・開発用）"""

    def __init__(self, data: list[dict[str, Any]] | None = None):
        self._data = data or []

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """ベクトル検索"""
        # モック結果
        return self._data[:top_k]
