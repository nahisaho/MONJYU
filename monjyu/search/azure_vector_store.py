"""Azure AI Search VectorStore - Production Vector Storage.

Azure AI Searchをベクトルストアとして使用するための統合モジュール。
VectorSearcherProtocolを実装し、既存の検索パイプラインと統合。

References:
- https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-create-index
- https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol

from monjyu.search.base import SearchHit, SearchResults

logger = logging.getLogger(__name__)


# ========== Configuration ==========


@dataclass
class AzureSearchVectorStoreConfig:
    """Azure AI Search VectorStore 設定
    
    Attributes:
        endpoint: Azure AI Search エンドポイント
        index_name: インデックス名
        api_key: APIキー（Managed Identity使用時は不要）
        use_managed_identity: Managed Identity を使用
        
        vector_field: ベクトルフィールド名
        vector_dimensions: ベクトル次元数
        content_field: コンテンツフィールド名
        title_field: タイトルフィールド名
        id_field: IDフィールド名
        
        enable_semantic_search: セマンティック検索を有効化
        semantic_config_name: セマンティック設定名
        
        hnsw_m: HNSWグラフのリンク数
        hnsw_ef_construction: インデックス構築時のef値
        hnsw_ef_search: 検索時のef値
        
        batch_size: バッチアップロードサイズ
        max_retries: 最大リトライ回数
        retry_delay: リトライ遅延（秒）
    """
    endpoint: str | None = None
    index_name: str = "monjyu-vectors"
    api_key: str | None = None
    use_managed_identity: bool = False
    
    # フィールド設定
    vector_field: str = "vector"
    vector_dimensions: int = 1536
    content_field: str = "content"
    title_field: str = "document_title"
    id_field: str = "text_unit_id"
    
    # セマンティック検索
    enable_semantic_search: bool = True
    semantic_config_name: str = "monjyu-semantic-config"
    
    # HNSW パラメータ
    hnsw_m: int = 4
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 500
    
    # バッチ設定
    batch_size: int = 1000
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 追加フィールド
    filterable_fields: List[str] = field(
        default_factory=lambda: ["document_id", "chunk_index"]
    )
    searchable_fields: List[str] = field(
        default_factory=lambda: ["content", "document_title"]
    )
    
    def __post_init__(self):
        """環境変数からの取得"""
        if self.endpoint is None:
            self.endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        if self.api_key is None and not self.use_managed_identity:
            self.api_key = os.environ.get("AZURE_SEARCH_KEY")
    
    def validate(self) -> None:
        """設定の検証"""
        if not self.endpoint:
            raise ValueError(
                "Azure AI Search endpoint is required. "
                "Set AZURE_SEARCH_ENDPOINT environment variable or provide endpoint parameter."
            )
        if not self.api_key and not self.use_managed_identity:
            raise ValueError(
                "Either API key or Managed Identity is required. "
                "Set AZURE_SEARCH_KEY environment variable or use_managed_identity=True."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "endpoint": self.endpoint,
            "index_name": self.index_name,
            "use_managed_identity": self.use_managed_identity,
            "vector_field": self.vector_field,
            "vector_dimensions": self.vector_dimensions,
            "content_field": self.content_field,
            "title_field": self.title_field,
            "enable_semantic_search": self.enable_semantic_search,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef_search": self.hnsw_ef_search,
            "batch_size": self.batch_size,
        }


# ========== Protocols ==========


class EmbeddingProtocol(Protocol):
    """埋め込みプロトコル"""
    
    async def embed(self, text: str) -> List[float]:
        """テキストを埋め込む"""
        ...
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """テキストをバッチで埋め込む"""
        ...


# ========== Azure Search VectorStore ==========


class AzureSearchVectorStore:
    """Azure AI Search VectorStore
    
    Azure AI Searchをベクトルストアとして使用。
    VectorSearcherProtocolに準拠。
    
    Features:
        - ベクトル検索
        - ハイブリッド検索（ベクトル + キーワード）
        - セマンティックリランキング
        - バッチアップロード
        - フィルタリング
        - 非同期サポート
    
    Example:
        >>> # 初期化
        >>> store = AzureSearchVectorStore(
        ...     endpoint="https://xxx.search.windows.net",
        ...     api_key="your-api-key",
        ... )
        
        >>> # インデックス作成
        >>> await store.create_index_async()
        
        >>> # ドキュメント追加
        >>> await store.add_documents_async(documents)
        
        >>> # 検索
        >>> results = await store.search_async(query_embedding, top_k=10)
    """
    
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str = "monjyu-vectors",
        use_managed_identity: bool = False,
        config: AzureSearchVectorStoreConfig | None = None,
    ):
        """初期化
        
        Args:
            endpoint: Azure AI Search エンドポイント
            api_key: APIキー
            index_name: インデックス名
            use_managed_identity: Managed Identity を使用
            config: 詳細設定
        """
        if config is not None:
            self.config = config
        else:
            self.config = AzureSearchVectorStoreConfig(
                endpoint=endpoint,
                api_key=api_key,
                index_name=index_name,
                use_managed_identity=use_managed_identity,
            )
        
        self.config.validate()
        
        self._credential = None
        self._search_client = None
        self._index_client = None
        self._async_search_client = None
        self._async_index_client = None
    
    def _create_credential(self):
        """クレデンシャルを作成"""
        if self._credential is not None:
            return self._credential
        
        if self.config.use_managed_identity:
            from azure.identity import DefaultAzureCredential
            self._credential = DefaultAzureCredential()
        else:
            from azure.core.credentials import AzureKeyCredential
            self._credential = AzureKeyCredential(self.config.api_key)
        
        return self._credential
    
    @property
    def search_client(self):
        """SearchClient を取得（同期版）"""
        if self._search_client is None:
            from azure.search.documents import SearchClient
            self._search_client = SearchClient(
                endpoint=self.config.endpoint,
                index_name=self.config.index_name,
                credential=self._create_credential(),
            )
        return self._search_client
    
    @property
    def index_client(self):
        """SearchIndexClient を取得（同期版）"""
        if self._index_client is None:
            from azure.search.documents.indexes import SearchIndexClient
            self._index_client = SearchIndexClient(
                endpoint=self.config.endpoint,
                credential=self._create_credential(),
            )
        return self._index_client
    
    @property
    def async_search_client(self):
        """SearchClient を取得（非同期版）"""
        if self._async_search_client is None:
            from azure.search.documents.aio import SearchClient
            self._async_search_client = SearchClient(
                endpoint=self.config.endpoint,
                index_name=self.config.index_name,
                credential=self._create_credential(),
            )
        return self._async_search_client
    
    @property
    def async_index_client(self):
        """SearchIndexClient を取得（非同期版）"""
        if self._async_index_client is None:
            from azure.search.documents.indexes.aio import SearchIndexClient
            self._async_index_client = SearchIndexClient(
                endpoint=self.config.endpoint,
                credential=self._create_credential(),
            )
        return self._async_index_client
    
    # ========== Index Management ==========
    
    def create_index(self, force: bool = False) -> bool:
        """インデックスを作成（同期版）
        
        Args:
            force: 既存インデックスを削除して再作成
            
        Returns:
            True: 新規作成, False: 既存
        """
        return asyncio.get_event_loop().run_until_complete(
            self.create_index_async(force)
        )
    
    async def create_index_async(self, force: bool = False) -> bool:
        """インデックスを作成（非同期版）
        
        Args:
            force: 既存インデックスを削除して再作成
            
        Returns:
            True: 新規作成, False: 既存
        """
        from azure.search.documents.indexes.models import (
            HnswAlgorithmConfiguration,
            HnswParameters,
            SearchableField,
            SearchField,
            SearchFieldDataType,
            SearchIndex,
            SemanticConfiguration,
            SemanticField,
            SemanticPrioritizedFields,
            SemanticSearch,
            SimpleField,
            VectorSearch,
            VectorSearchAlgorithmMetric,
            VectorSearchProfile,
        )
        
        index_name = self.config.index_name
        
        # 既存インデックスチェック
        try:
            existing_index = await self.async_index_client.get_index(index_name)
            if existing_index and not force:
                logger.info(f"Index '{index_name}' already exists")
                return False
            if existing_index and force:
                await self.async_index_client.delete_index(index_name)
                logger.info(f"Deleted existing index '{index_name}'")
        except Exception:
            pass  # インデックスが存在しない
        
        logger.info(f"Creating index '{index_name}'...")
        
        # フィールド定義
        fields = [
            # キーフィールド
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            # テキストユニットID
            SimpleField(
                name=self.config.id_field,
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # ベクトルフィールド
            SearchField(
                name=self.config.vector_field,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.config.vector_dimensions,
                vector_search_profile_name="monjyu-vector-profile",
            ),
            # コンテンツフィールド
            SearchableField(
                name=self.config.content_field,
                type=SearchFieldDataType.String,
                analyzer_name="ja.lucene",
            ),
            # タイトルフィールド
            SearchableField(
                name=self.config.title_field,
                type=SearchFieldDataType.String,
            ),
        ]
        
        # 追加のフィルタ可能フィールド
        for field_name in self.config.filterable_fields:
            if field_name not in [f.name for f in fields]:
                fields.append(SimpleField(
                    name=field_name,
                    type=SearchFieldDataType.String,
                    filterable=True,
                ))
        
        # スコアフィールド
        fields.append(SimpleField(
            name="chunk_index",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ))
        
        # HNSW ベクトル検索設定
        vector_search = VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="monjyu-hnsw-algorithm",
                    parameters=HnswParameters(
                        m=self.config.hnsw_m,
                        ef_construction=self.config.hnsw_ef_construction,
                        ef_search=self.config.hnsw_ef_search,
                        metric=VectorSearchAlgorithmMetric.COSINE,
                    ),
                ),
            ],
            profiles=[
                VectorSearchProfile(
                    name="monjyu-vector-profile",
                    algorithm_configuration_name="monjyu-hnsw-algorithm",
                ),
            ],
        )
        
        # セマンティック検索設定
        semantic_search = None
        if self.config.enable_semantic_search:
            semantic_config = SemanticConfiguration(
                name=self.config.semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name=self.config.title_field),
                    content_fields=[
                        SemanticField(field_name=self.config.content_field),
                    ],
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # インデックス作成
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        
        await self.async_index_client.create_or_update_index(index)
        logger.info(f"Index '{index_name}' created successfully")
        return True
    
    async def delete_index_async(self) -> None:
        """インデックスを削除"""
        await self.async_index_client.delete_index(self.config.index_name)
        logger.info(f"Index '{self.config.index_name}' deleted")
    
    async def get_index_stats_async(self) -> Dict[str, Any]:
        """インデックス統計を取得"""
        stats = await self.async_index_client.get_index_statistics(
            self.config.index_name
        )
        return {
            "document_count": stats.document_count,
            "storage_size_bytes": stats.storage_size,
        }
    
    # ========== Document Operations ==========
    
    async def add_documents_async(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """ドキュメントを追加（非同期版）
        
        Args:
            documents: ドキュメントリスト
            embeddings: 埋め込みベクトル（省略時はdocumentsから取得）
            
        Returns:
            追加されたドキュメント数
        """
        if not documents:
            return 0
        
        # 埋め込みを設定
        if embeddings:
            for i, doc in enumerate(documents):
                if i < len(embeddings):
                    doc[self.config.vector_field] = embeddings[i]
        
        # バッチアップロード
        total_uploaded = 0
        batch_size = self.config.batch_size
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # IDがなければ生成
            for j, doc in enumerate(batch):
                if "id" not in doc:
                    doc["id"] = doc.get(self.config.id_field, f"doc_{i+j}")
            
            await self._retry_async(
                self.async_search_client.upload_documents,
                documents=batch,
            )
            total_uploaded += len(batch)
            logger.debug(f"Uploaded {total_uploaded}/{len(documents)} documents")
        
        logger.info(f"Successfully uploaded {total_uploaded} documents")
        return total_uploaded
    
    async def delete_documents_async(self, ids: List[str]) -> int:
        """ドキュメントを削除
        
        Args:
            ids: 削除するIDリスト
            
        Returns:
            削除されたドキュメント数
        """
        if not ids:
            return 0
        
        documents = [{"id": id_} for id_ in ids]
        await self._retry_async(
            self.async_search_client.delete_documents,
            documents=documents,
        )
        logger.info(f"Deleted {len(ids)} documents")
        return len(ids)
    
    # ========== Search Operations ==========
    
    async def search_async(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        select_fields: Optional[List[str]] = None,
    ) -> SearchResults:
        """ベクトル検索（非同期版）
        
        Args:
            query_embedding: クエリベクトル
            top_k: 取得件数
            filter_expr: ODataフィルター式
            select_fields: 取得フィールド
            
        Returns:
            SearchResults
        """
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields=self.config.vector_field,
            exhaustive=True,
        )
        
        start_time = time.time()
        
        results = await self._retry_async(
            self.async_search_client.search,
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            select=select_fields,
            top=top_k,
            include_total_count=True,
        )
        
        hits = await self._convert_results_async(results)
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            search_time_ms=search_time_ms,
        )
    
    async def hybrid_search_async(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        use_semantic_reranker: bool = False,
    ) -> SearchResults:
        """ハイブリッド検索（ベクトル + キーワード）
        
        Args:
            query_text: テキストクエリ
            query_embedding: クエリベクトル
            top_k: 取得件数
            filter_expr: ODataフィルター式
            use_semantic_reranker: セマンティックリランカーを使用
            
        Returns:
            SearchResults
        """
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields=self.config.vector_field,
            exhaustive=True,
        )
        
        search_kwargs = {
            "search_text": query_text,
            "vector_queries": [vector_query],
            "filter": filter_expr,
            "top": top_k,
            "include_total_count": True,
        }
        
        if use_semantic_reranker and self.config.enable_semantic_search:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self.config.semantic_config_name
        
        start_time = time.time()
        
        results = await self._retry_async(
            self.async_search_client.search,
            **search_kwargs,
        )
        
        hits = await self._convert_results_async(
            results,
            include_reranker_score=use_semantic_reranker,
        )
        search_time_ms = (time.time() - start_time) * 1000
        
        return SearchResults(
            hits=hits,
            total_count=len(hits),
            search_time_ms=search_time_ms,
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> SearchResults:
        """ベクトル検索（同期版）"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.search_async(query_embedding, top_k, filter_expr)
            )
        finally:
            loop.close()
    
    # ========== Helper Methods ==========
    
    async def _convert_results_async(
        self,
        results,
        include_reranker_score: bool = False,
    ) -> List[SearchHit]:
        """Azure Search結果をSearchHitに変換"""
        hits = []
        
        async for result in results:
            result_dict = dict(result)
            
            # スコア取得
            score = result_dict.get("@search.score", 0.0)
            if include_reranker_score:
                reranker_score = result_dict.get("@search.reranker_score")
                if reranker_score is not None:
                    score = reranker_score
            
            hits.append(SearchHit(
                text_unit_id=result_dict.get(self.config.id_field, result_dict.get("id", "")),
                document_id=result_dict.get("document_id", ""),
                text=result_dict.get(self.config.content_field, ""),
                score=score,
                chunk_index=result_dict.get("chunk_index", 0),
                document_title=result_dict.get(self.config.title_field, ""),
            ))
        
        return hits
    
    async def _retry_async(self, operation, *args, **kwargs):
        """リトライ付きで非同期操作を実行"""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
        
        logger.error(f"Operation failed after {self.config.max_retries} attempts")
        raise last_error
    
    async def close(self):
        """クライアントをクローズ"""
        if self._async_search_client:
            await self._async_search_client.close()
        if self._async_index_client:
            await self._async_index_client.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ========== Search Engine Integration ==========


class AzureSearchEngine:
    """Azure AI Search統合検索エンジン
    
    VectorSearchEngineと同じインターフェースで
    Azure AI Searchを使用した検索を提供。
    """
    
    def __init__(
        self,
        vector_store: AzureSearchVectorStore,
        embedding_client: EmbeddingProtocol,
    ):
        """初期化
        
        Args:
            vector_store: Azure Search VectorStore
            embedding_client: 埋め込みクライアント
        """
        self.vector_store = vector_store
        self.embedding_client = embedding_client
    
    async def search_async(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        use_hybrid: bool = True,
        use_semantic_reranker: bool = False,
    ) -> SearchResults:
        """検索を実行
        
        Args:
            query: クエリテキスト
            top_k: 取得件数
            filter_expr: フィルター式
            use_hybrid: ハイブリッド検索を使用
            use_semantic_reranker: セマンティックリランカーを使用
            
        Returns:
            SearchResults
        """
        # クエリを埋め込み
        query_embedding = await self.embedding_client.embed(query)
        
        if use_hybrid:
            return await self.vector_store.hybrid_search_async(
                query_text=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_expr=filter_expr,
                use_semantic_reranker=use_semantic_reranker,
            )
        else:
            return await self.vector_store.search_async(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_expr=filter_expr,
            )
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> SearchResults:
        """検索を実行（同期版）"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.search_async(query, top_k, **kwargs)
            )
        finally:
            loop.close()


# ========== Factory Functions ==========


def create_azure_vector_store(
    endpoint: str | None = None,
    api_key: str | None = None,
    index_name: str = "monjyu-vectors",
    use_managed_identity: bool = False,
    **kwargs,
) -> AzureSearchVectorStore:
    """Azure Search VectorStoreを作成
    
    Args:
        endpoint: Azure AI Search エンドポイント
        api_key: APIキー
        index_name: インデックス名
        use_managed_identity: Managed Identity を使用
        **kwargs: 追加設定
        
    Returns:
        AzureSearchVectorStore
    """
    config = AzureSearchVectorStoreConfig(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        use_managed_identity=use_managed_identity,
        **kwargs,
    )
    return AzureSearchVectorStore(config=config)


def create_azure_search_engine(
    vector_store: AzureSearchVectorStore,
    embedding_client: EmbeddingProtocol,
) -> AzureSearchEngine:
    """Azure Search Engineを作成
    
    Args:
        vector_store: Azure Search VectorStore
        embedding_client: 埋め込みクライアント
        
    Returns:
        AzureSearchEngine
    """
    return AzureSearchEngine(
        vector_store=vector_store,
        embedding_client=embedding_client,
    )
