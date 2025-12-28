# Azure AI Search Vector Indexer
"""
FEAT-014: Azure AI Search ベクトルインデクサー

本番環境用の Azure AI Search 統合。
Microsoft Learn ベストプラクティスに準拠。

References:
- https://learn.microsoft.com/en-us/azure/search/search-get-started-vector
- https://learn.microsoft.com/en-us/azure/search/vector-search-how-to-query
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from monjyu.index.base import VectorIndexer, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class AzureSearchConfig:
    """Azure AI Search 設定
    
    Attributes:
        endpoint: Azure AI Search エンドポイント
        index_name: インデックス名
        api_key: APIキー（Managed Identity使用時は不要）
        use_managed_identity: Managed Identity を使用するか
        vector_field: ベクトルフィールド名
        vector_dimensions: ベクトル次元数
        text_field: テキストフィールド名（ハイブリッド検索用）
        
        # HNSW アルゴリズム設定
        hnsw_m: グラフの各ノードの双方向リンク数
        hnsw_ef_construction: インデックス構築時の動的リストサイズ
        hnsw_ef_search: 検索時の動的リストサイズ
        
        # リトライ設定
        max_retries: 最大リトライ回数
        retry_delay: 初期リトライ遅延（秒）
        
        # セマンティック検索設定
        semantic_config_name: セマンティック設定名
        enable_semantic_search: セマンティック検索を有効化
    """
    endpoint: str | None = None
    index_name: str = "monjyu-text-units"
    api_key: str | None = None
    use_managed_identity: bool = False
    
    # ベクトル設定
    vector_field: str = "vector"
    vector_dimensions: int = 1536
    text_field: str = "content"
    
    # HNSW パラメータ（Azure推奨値）
    hnsw_m: int = 4
    hnsw_ef_construction: int = 400
    hnsw_ef_search: int = 500
    
    # リトライ設定
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # セマンティック検索
    semantic_config_name: str = "monjyu-semantic-config"
    enable_semantic_search: bool = True
    
    # 追加フィールド
    filterable_fields: list[str] = field(default_factory=lambda: ["document_id", "chunk_index"])
    searchable_fields: list[str] = field(default_factory=lambda: ["content", "title"])
    
    def __post_init__(self):
        """環境変数からの取得"""
        if self.endpoint is None:
            self.endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        if self.api_key is None and not self.use_managed_identity:
            self.api_key = os.environ.get("AZURE_SEARCH_KEY")


class AzureAISearchIndexer(VectorIndexer):
    """Azure AI Search ベクトルインデクサー（本番用）
    
    Azure AI Searchを使用してベクトルインデックスを構築・検索する。
    Managed Identity とセマンティック検索をサポート。
    
    Example:
        >>> # API Key 認証
        >>> indexer = AzureAISearchIndexer(
        ...     endpoint="https://xxx.search.windows.net",
        ...     api_key="your-api-key",
        ...     index_name="monjyu-text-units",
        ... )
        
        >>> # Managed Identity 認証（推奨）
        >>> indexer = AzureAISearchIndexer(
        ...     endpoint="https://xxx.search.windows.net",
        ...     use_managed_identity=True,
        ... )
        
        >>> # ベクトル検索
        >>> results = indexer.search([0.1, 0.2, ...], top_k=5)
        
        >>> # ハイブリッド検索（ベクトル + キーワード）
        >>> results = indexer.search_hybrid(
        ...     query_text="Transformer architecture",
        ...     query_embedding=[0.1, 0.2, ...],
        ...     top_k=5,
        ... )
    
    Environment Variables:
        AZURE_SEARCH_ENDPOINT: Azure AI Search エンドポイント
        AZURE_SEARCH_KEY: API キー（Managed Identity使用時は不要）
    """
    
    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str = "monjyu-text-units",
        use_managed_identity: bool = False,
        config: AzureSearchConfig | None = None,
        **kwargs,
    ) -> None:
        """初期化
        
        Args:
            endpoint: Azure AI Search エンドポイント
            api_key: APIキー（Managed Identity使用時は不要）
            index_name: インデックス名
            use_managed_identity: Managed Identity を使用するか
            config: 詳細設定（指定時は他のパラメータより優先）
            **kwargs: 追加の設定パラメータ
        """
        # 設定の構築
        if config is not None:
            self.config = config
        else:
            config_kwargs = {k: v for k, v in kwargs.items() if hasattr(AzureSearchConfig, k)}
            self.config = AzureSearchConfig(
                endpoint=endpoint,
                api_key=api_key,
                index_name=index_name,
                use_managed_identity=use_managed_identity,
                **config_kwargs,
            )
        
        # バリデーション
        if not self.config.endpoint:
            msg = "Azure AI Search エンドポイントが設定されていません (AZURE_SEARCH_ENDPOINT)"
            raise ValueError(msg)
        
        if not self.config.api_key and not self.config.use_managed_identity:
            msg = "APIキーまたはManaged Identityのいずれかを設定してください"
            raise ValueError(msg)
        
        # クレデンシャルの作成
        self._credential = self._create_credential()
        
        # クライアントの初期化（遅延）
        self._search_client = None
        self._index_client = None
    
    def _create_credential(self):
        """クレデンシャルを作成"""
        if self.config.use_managed_identity:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        else:
            from azure.core.credentials import AzureKeyCredential
            return AzureKeyCredential(self.config.api_key)
    
    @property
    def search_client(self):
        """SearchClient を取得（遅延初期化）"""
        if self._search_client is None:
            from azure.search.documents import SearchClient
            self._search_client = SearchClient(
                endpoint=self.config.endpoint,
                index_name=self.config.index_name,
                credential=self._credential,
            )
        return self._search_client
    
    @property
    def index_client(self):
        """SearchIndexClient を取得（遅延初期化）"""
        if self._index_client is None:
            from azure.search.documents.indexes import SearchIndexClient
            self._index_client = SearchIndexClient(
                endpoint=self.config.endpoint,
                credential=self._credential,
            )
        return self._index_client
    
    def _retry_operation(self, operation, *args, **kwargs):
        """リトライ付きで操作を実行
        
        指数バックオフでリトライ。
        """
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.config.max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        
        logger.error(f"Operation failed after {self.config.max_retries} attempts: {last_error}")
        raise last_error
    
    def create_index_if_not_exists(self) -> bool:
        """インデックスを作成（存在しない場合）
        
        Returns:
            True: 新規作成、False: 既存
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
        
        # 既存インデックスをチェック
        try:
            self.index_client.get_index(self.config.index_name)
            logger.info(f"Index '{self.config.index_name}' already exists")
            return False
        except Exception:
            pass  # 存在しない
        
        logger.info(f"Creating index '{self.config.index_name}'...")
        
        # フィールド定義
        fields = [
            # キーフィールド
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
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
        ]
        
        # 検索可能フィールド
        for field_name in self.config.searchable_fields:
            fields.append(SearchableField(
                name=field_name,
                type=SearchFieldDataType.String,
            ))
        
        # フィルタ可能フィールド
        for field_name in self.config.filterable_fields:
            if field_name not in [f.name for f in fields]:
                fields.append(SimpleField(
                    name=field_name,
                    type=SearchFieldDataType.String,
                    filterable=True,
                ))
        
        # HNSW ベクトル検索設定（Azure推奨パラメータ）
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
        if self.config.enable_semantic_search and self.config.searchable_fields:
            content_fields = [
                SemanticField(field_name=f) 
                for f in self.config.searchable_fields[:3]  # 最大3フィールド
            ]
            semantic_config = SemanticConfiguration(
                name=self.config.semantic_config_name,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=content_fields,
                ),
            )
            semantic_search = SemanticSearch(configurations=[semantic_config])
        
        # インデックスを作成
        index = SearchIndex(
            name=self.config.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )
        
        self._retry_operation(self.index_client.create_or_update_index, index)
        logger.info(f"Index '{self.config.index_name}' created successfully")
        return True
    
    def build(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """インデックスを構築
        
        Args:
            embeddings: 埋め込みベクトルのリスト
            ids: ID のリスト
            metadata: メタデータのリスト
        """
        # インデックスを作成
        self.create_index_if_not_exists()
        
        # ドキュメントをアップロード
        self.add(embeddings, ids, metadata)
    
    def add(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """インデックスにデータを追加
        
        Args:
            embeddings: 埋め込みベクトルのリスト
            ids: ID のリスト
            metadata: メタデータのリスト
        """
        if len(embeddings) != len(ids):
            msg = "embeddings と ids の長さが一致しません"
            raise ValueError(msg)
        
        # ドキュメントを準備
        documents = []
        for i, (emb, id_) in enumerate(zip(embeddings, ids, strict=True)):
            doc: dict[str, Any] = {
                "id": id_,
                self.config.vector_field: emb,
            }
            if metadata and i < len(metadata):
                for key, value in metadata[i].items():
                    if key not in ["id", self.config.vector_field]:
                        doc[key] = value
            documents.append(doc)
        
        # バッチでアップロード（1000件ずつ）
        batch_size = 1000
        total_uploaded = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._retry_operation(
                self.search_client.upload_documents,
                documents=batch,
            )
            total_uploaded += len(batch)
            logger.debug(f"Uploaded {total_uploaded}/{len(documents)} documents")
        
        logger.info(f"Successfully uploaded {total_uploaded} documents")
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """ベクトル検索を実行
        
        Args:
            query_embedding: クエリベクトル
            top_k: 取得する上位件数
            filter_expr: ODataフィルター式
            
        Returns:
            検索結果のリスト
        """
        from azure.search.documents.models import VectorizedQuery
        
        vector_query = VectorizedQuery(
            vector=query_embedding,
            k_nearest_neighbors=top_k,
            fields=self.config.vector_field,
            exhaustive=True,  # 正確な結果を得る
        )
        
        results = self._retry_operation(
            self.search_client.search,
            search_text=None,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=top_k,
            include_total_count=True,
        )
        
        return self._convert_results(results)
    
    def search_hybrid(
        self,
        query_text: str,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
        use_semantic_reranker: bool = False,
    ) -> list[SearchResult]:
        """ハイブリッド検索（ベクトル + キーワード）を実行
        
        Args:
            query_text: テキストクエリ
            query_embedding: クエリベクトル
            top_k: 取得する上位件数
            filter_expr: ODataフィルター式
            use_semantic_reranker: セマンティックリランカーを使用
            
        Returns:
            検索結果のリスト
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
        
        # セマンティックリランカーを追加
        if use_semantic_reranker and self.config.enable_semantic_search:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self.config.semantic_config_name
        
        results = self._retry_operation(
            self.search_client.search,
            **search_kwargs,
        )
        
        return self._convert_results(results, include_reranker_score=use_semantic_reranker)
    
    def _convert_results(
        self,
        results,
        include_reranker_score: bool = False,
    ) -> list[SearchResult]:
        """Azure Search 結果を SearchResult に変換"""
        search_results = []
        
        for result in results:
            result_dict = dict(result)
            
            # スコア取得
            score = result_dict.get("@search.score", 0.0)
            
            # メタデータを抽出
            metadata = {
                k: v for k, v in result_dict.items()
                if not k.startswith("@") and k not in ["id", self.config.vector_field]
            }
            
            # リランカースコアがあれば追加
            if include_reranker_score:
                reranker_score = result_dict.get("@search.reranker_score")
                if reranker_score is not None:
                    metadata["reranker_score"] = reranker_score
            
            search_results.append(SearchResult(
                id=result_dict.get("id", ""),
                score=score,
                metadata=metadata,
            ))
        
        return search_results
    
    def save(self, path: Path | str) -> None:
        """インデックスを保存（Azure AI Searchは自動永続化）"""
        pass
    
    def load(self, path: Path | str) -> None:
        """インデックスを読み込み（Azure AI Searchは自動永続化）"""
        pass
    
    def count(self) -> int:
        """インデックス内のドキュメント数を取得"""
        return self._retry_operation(self.search_client.get_document_count)
    
    def delete(self, ids: list[str]) -> None:
        """指定IDのドキュメントを削除
        
        Args:
            ids: 削除するID のリスト
        """
        documents = [{"id": id_} for id_ in ids]
        self._retry_operation(
            self.search_client.delete_documents,
            documents=documents,
        )
        logger.info(f"Deleted {len(ids)} documents")
    
    def clear(self) -> None:
        """インデックス内の全ドキュメントを削除"""
        # 全ドキュメントのIDを取得
        results = self.search_client.search(
            search_text="*",
            select=["id"],
            top=10000,
        )
        ids_to_delete = [r["id"] for r in results]
        
        if ids_to_delete:
            # バッチで削除
            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i:i + batch_size]
                self.delete(batch)
        
        logger.info(f"Cleared all {len(ids_to_delete)} documents from index")
    
    def get_index_stats(self) -> dict[str, Any]:
        """インデックスの統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        index = self.index_client.get_index(self.config.index_name)
        stats = self.index_client.get_index_statistics(self.config.index_name)
        
        return {
            "name": index.name,
            "document_count": stats.document_count,
            "storage_size_bytes": stats.storage_size,
            "vector_index_size_bytes": getattr(stats, 'vector_index_size', 0),
            "field_count": len(index.fields),
            "vector_dimensions": self.config.vector_dimensions,
        }


def create_azure_search_indexer(
    endpoint: str | None = None,
    api_key: str | None = None,
    index_name: str = "monjyu-text-units",
    use_managed_identity: bool = False,
    **kwargs,
) -> AzureAISearchIndexer:
    """AzureAISearchIndexer を作成するファクトリ関数
    
    Args:
        endpoint: Azure AI Search エンドポイント
        api_key: APIキー
        index_name: インデックス名
        use_managed_identity: Managed Identity を使用
        **kwargs: 追加設定
        
    Returns:
        AzureAISearchIndexer インスタンス
    """
    return AzureAISearchIndexer(
        endpoint=endpoint,
        api_key=api_key,
        index_name=index_name,
        use_managed_identity=use_managed_identity,
        **kwargs,
    )
