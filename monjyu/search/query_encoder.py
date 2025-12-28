# Query Encoder
"""
クエリエンコーダー - クエリを埋め込みベクトルに変換

TASK-004-01: QueryEncoder実装
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    pass


class EmbeddingClientProtocol(Protocol):
    """埋め込みクライアントプロトコル"""

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括変換"""
        ...


class QueryEncoder:
    """クエリエンコーダー"""

    def __init__(self, embedding_client: EmbeddingClientProtocol):
        """
        Args:
            embedding_client: 埋め込みクライアント
        """
        self.embedding_client = embedding_client
        self._cache: dict[str, list[float]] = {}

    def encode(self, query: str) -> list[float]:
        """
        クエリを埋め込みベクトルに変換

        Args:
            query: クエリ文字列

        Returns:
            埋め込みベクトル
        """
        # キャッシュチェック
        if query in self._cache:
            return self._cache[query]

        # 埋め込み生成
        vector = self.embedding_client.embed(query)

        # キャッシュに保存
        self._cache[query] = vector

        return vector

    def encode_batch(self, queries: list[str]) -> list[list[float]]:
        """
        複数クエリを一括変換

        Args:
            queries: クエリ文字列のリスト

        Returns:
            埋め込みベクトルのリスト
        """
        # キャッシュされていないクエリを抽出
        uncached_queries = [q for q in queries if q not in self._cache]

        if uncached_queries:
            # 一括で埋め込み生成
            vectors = self.embedding_client.embed_batch(uncached_queries)

            # キャッシュに保存
            for query, vector in zip(uncached_queries, vectors):
                self._cache[query] = vector

        # 結果を返す
        return [self._cache[q] for q in queries]

    def clear_cache(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()


class LLMClientProtocol(Protocol):
    """LLMクライアントプロトコル"""

    @property
    def model_name(self) -> str:
        """モデル名"""
        ...

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """テキスト生成"""
        ...


class QueryExpander:
    """クエリ拡張"""

    EXPANSION_PROMPT = """
以下のクエリに対して、{num_expansions}個の異なる言い回しや関連クエリを生成してください。
各クエリは改行で区切ってください。

クエリ: {query}

異なる言い回し:
"""

    def __init__(self, llm_client: LLMClientProtocol):
        """
        Args:
            llm_client: LLMクライアント
        """
        self.llm_client = llm_client

    def expand(self, query: str, num_expansions: int = 3) -> list[str]:
        """
        クエリを複数の関連クエリに拡張

        Args:
            query: 元のクエリ
            num_expansions: 拡張数

        Returns:
            元のクエリ + 拡張クエリのリスト
        """
        prompt = self.EXPANSION_PROMPT.format(
            query=query, num_expansions=num_expansions
        )

        response = self.llm_client.generate(prompt)
        expansions = [line.strip() for line in response.strip().split("\n") if line.strip()]

        # 元のクエリを先頭に追加
        return [query] + expansions[:num_expansions]


# === Ollama Embedding Client ===


class OllamaEmbeddingClient:
    """Ollama埋め込みクライアント"""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
    ):
        """
        Args:
            model: Ollamaモデル名
            host: Ollamaホスト
        """
        self.model = model
        self.host = host
        self._client = None

    @property
    def client(self):
        """Ollamaクライアント（遅延初期化）"""
        if self._client is None:
            try:
                import ollama

                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._client

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        response = self.client.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括変換"""
        # Ollamaはバッチ埋め込みをサポートしていないので逐次処理
        return [self.embed(text) for text in texts]


# === OpenAI Embedding Client ===


class OpenAIEmbeddingClient:
    """OpenAI埋め込みクライアント"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        """
        Args:
            model: OpenAIモデル名
            api_key: APIキー（省略時は環境変数から取得）
        """
        self.model = model
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        """OpenAIクライアント（遅延初期化）"""
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括変換"""
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


# === Azure OpenAI Embedding Client ===


class AzureOpenAIEmbeddingClient:
    """Azure OpenAI埋め込みクライアント"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-02-01",
    ):
        """
        Args:
            endpoint: Azure OpenAIエンドポイント
            api_key: APIキー
            deployment_name: デプロイメント名
            api_version: APIバージョン
        """
        self.endpoint = endpoint
        self.deployment_name = deployment_name
        self.api_version = api_version
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        """AzureOpenAIクライアント（遅延初期化）"""
        if self._client is None:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_key=self._api_key,
                    api_version=self.api_version,
                )
            except ImportError:
                raise RuntimeError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        response = self.client.embeddings.create(
            model=self.deployment_name, input=text
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストを一括変換"""
        response = self.client.embeddings.create(
            model=self.deployment_name, input=texts
        )
        return [item.embedding for item in response.data]
