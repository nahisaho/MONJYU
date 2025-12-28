# Vector Searcher
"""
ベクトル検索実装

TASK-004-02: LanceDBVectorSearcher実装
TASK-004-03: AzureAISearchVectorSearcher実装
TASK-004-04: HybridSearcher実装
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from monjyu.search.base import SearchHit, SearchResults

if TYPE_CHECKING:
    pass


@dataclass
class VectorSearcherConfig:
    """ベクトル検索設定"""

    # LanceDB設定
    db_path: str = "./output/index/level_0/vector_index"
    table_name: str = "embeddings"

    # デフォルト検索パラメータ
    default_top_k: int = 10
    default_threshold: float = 0.0


class LanceDBVectorSearcher:
    """LanceDB ベクトル検索"""

    def __init__(
        self,
        db_path: str | Path = "./output/index/level_0/vector_index",
        table_name: str = "embeddings",
    ):
        """
        Args:
            db_path: LanceDBデータベースパス
            table_name: テーブル名
        """
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = None
        self._table = None

    @property
    def db(self):
        """LanceDBデータベース（遅延初期化）"""
        if self._db is None:
            try:
                import lancedb

                self._db = lancedb.connect(str(self.db_path))
            except ImportError:
                raise RuntimeError(
                    "lancedb package not installed. Run: pip install lancedb"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to connect to LanceDB: {e}")
        return self._db

    @property
    def table(self):
        """LanceDBテーブル（遅延初期化）"""
        if self._table is None:
            try:
                self._table = self.db.open_table(self.table_name)
            except Exception as e:
                raise RuntimeError(f"Failed to open table '{self.table_name}': {e}")
        return self._table

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> SearchResults:
        """
        ベクトル検索を実行

        Args:
            query_vector: クエリベクトル
            top_k: 返す結果数
            threshold: 類似度閾値

        Returns:
            検索結果
        """
        start = time.time()

        # LanceDB検索
        results = self.table.search(query_vector).limit(top_k).to_list()

        # SearchHitに変換
        hits = []
        for r in results:
            # cosine distance → similarity
            score = 1.0 - r.get("_distance", 0.0)

            if score >= threshold:
                hits.append(
                    SearchHit(
                        text_unit_id=r["text_unit_id"],
                        document_id=r.get("document_id", ""),
                        text=r.get("text", ""),
                        score=score,
                        chunk_index=r.get("chunk_index", 0),
                        document_title=r.get("document_title", ""),
                        vector_score=score,
                    )
                )

        elapsed = (time.time() - start) * 1000

        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed,
        )

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> SearchResults:
        """
        ハイブリッド検索（ベクトル + BM25/FTS）

        Args:
            query_text: クエリテキスト
            query_vector: クエリベクトル
            top_k: 返す結果数
            alpha: ベクトルスコアの重み (0-1)

        Returns:
            検索結果
        """
        start = time.time()

        # ベクトル検索
        vector_results = self.table.search(query_vector).limit(top_k * 2).to_list()

        # FTS検索（LanceDBがFTSをサポートしている場合）
        fts_results = []
        try:
            fts_results = (
                self.table.search(query_text, query_type="fts").limit(top_k * 2).to_list()
            )
        except Exception:
            # FTSが利用不可の場合はベクトル検索のみ
            pass

        # スコア統合
        hit_map: dict[str, dict[str, Any]] = {}

        for r in vector_results:
            text_unit_id = r["text_unit_id"]
            vector_score = 1.0 - r.get("_distance", 0.0)

            if text_unit_id not in hit_map:
                hit_map[text_unit_id] = {
                    "data": r,
                    "vector_score": vector_score,
                    "keyword_score": 0.0,
                }
            else:
                hit_map[text_unit_id]["vector_score"] = vector_score

        for r in fts_results:
            text_unit_id = r["text_unit_id"]
            keyword_score = r.get("_score", 0.0)

            if text_unit_id not in hit_map:
                hit_map[text_unit_id] = {
                    "data": r,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                }
            else:
                hit_map[text_unit_id]["keyword_score"] = keyword_score

        # 統合スコア計算
        hits = []
        for text_unit_id, data in hit_map.items():
            # Reciprocal Rank Fusion (RRF) 風のスコア計算
            combined_score = (
                alpha * data["vector_score"] + (1 - alpha) * data["keyword_score"]
            )

            r = data["data"]
            hits.append(
                SearchHit(
                    text_unit_id=text_unit_id,
                    document_id=r.get("document_id", ""),
                    text=r.get("text", ""),
                    score=combined_score,
                    chunk_index=r.get("chunk_index", 0),
                    document_title=r.get("document_title", ""),
                    vector_score=data["vector_score"],
                    keyword_score=data["keyword_score"],
                )
            )

        # スコアでソートしてTop-K
        hits.sort(key=lambda x: x.score, reverse=True)
        hits = hits[:top_k]

        elapsed = (time.time() - start) * 1000

        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed,
        )

    def get_stats(self) -> dict:
        """統計情報を取得"""
        try:
            count = len(self.table.to_pandas())
            return {
                "db_path": str(self.db_path),
                "table_name": self.table_name,
                "row_count": count,
            }
        except Exception:
            return {
                "db_path": str(self.db_path),
                "table_name": self.table_name,
                "row_count": 0,
            }


class AzureAISearchVectorSearcher:
    """Azure AI Search ベクトル検索"""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
    ):
        """
        Args:
            endpoint: Azure AI Searchエンドポイント
            api_key: APIキー
            index_name: インデックス名
        """
        self.endpoint = endpoint
        self.index_name = index_name
        self._api_key = api_key
        self._client = None

    @property
    def client(self):
        """SearchClient（遅延初期化）"""
        if self._client is None:
            try:
                from azure.core.credentials import AzureKeyCredential
                from azure.search.documents import SearchClient

                self._client = SearchClient(
                    endpoint=self.endpoint,
                    index_name=self.index_name,
                    credential=AzureKeyCredential(self._api_key),
                )
            except ImportError:
                raise RuntimeError(
                    "azure-search-documents not installed. "
                    "Run: pip install azure-search-documents"
                )
        return self._client

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> SearchResults:
        """
        ベクトル検索を実行

        Args:
            query_vector: クエリベクトル
            top_k: 返す結果数
            threshold: 類似度閾値

        Returns:
            検索結果
        """
        from azure.search.documents.models import VectorizedQuery

        start = time.time()

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k,
            fields="embedding",
        )

        results = self.client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["text_unit_id", "document_id", "text", "document_title", "chunk_index"],
        )

        hits = []
        for r in results:
            score = r["@search.score"]
            if score >= threshold:
                hits.append(
                    SearchHit(
                        text_unit_id=r["text_unit_id"],
                        document_id=r.get("document_id", ""),
                        text=r.get("text", ""),
                        score=score,
                        chunk_index=r.get("chunk_index", 0),
                        document_title=r.get("document_title", ""),
                        vector_score=score,
                    )
                )

        elapsed = (time.time() - start) * 1000

        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed,
        )

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> SearchResults:
        """
        ハイブリッド検索（Semantic Ranker使用）

        Args:
            query_text: クエリテキスト
            query_vector: クエリベクトル
            top_k: 返す結果数
            alpha: ベクトルスコアの重み（Azure AI Searchでは内部で処理）

        Returns:
            検索結果
        """
        from azure.search.documents.models import VectorizedQuery

        start = time.time()

        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top_k * 2,
            fields="embedding",
        )

        results = self.client.search(
            search_text=query_text,
            vector_queries=[vector_query],
            select=["text_unit_id", "document_id", "text", "document_title", "chunk_index"],
            query_type="semantic",
            semantic_configuration_name="monjyu-semantic-config",
        )

        hits = []
        for r in results:
            # セマンティックランカースコアを優先
            score = r.get("@search.reranker_score", r["@search.score"])
            hits.append(
                SearchHit(
                    text_unit_id=r["text_unit_id"],
                    document_id=r.get("document_id", ""),
                    text=r.get("text", ""),
                    score=score,
                    chunk_index=r.get("chunk_index", 0),
                    document_title=r.get("document_title", ""),
                )
            )

        hits = hits[:top_k]
        elapsed = (time.time() - start) * 1000

        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed,
        )


class InMemoryVectorSearcher:
    """インメモリベクトル検索（テスト・小規模データ用）"""

    def __init__(self):
        """初期化"""
        self._vectors: list[dict] = []

    def add(
        self,
        text_unit_id: str,
        vector: list[float],
        text: str = "",
        document_id: str = "",
        document_title: str = "",
        chunk_index: int = 0,
    ) -> None:
        """ベクトルを追加"""
        self._vectors.append(
            {
                "text_unit_id": text_unit_id,
                "vector": vector,
                "text": text,
                "document_id": document_id,
                "document_title": document_title,
                "chunk_index": chunk_index,
            }
        )

    def add_batch(self, items: list[dict]) -> None:
        """複数ベクトルを一括追加"""
        for item in items:
            self.add(**item)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> SearchResults:
        """
        ベクトル検索を実行

        Args:
            query_vector: クエリベクトル
            top_k: 返す結果数
            threshold: 類似度閾値

        Returns:
            検索結果
        """
        import math

        start = time.time()

        # コサイン類似度計算
        def cosine_similarity(a: list[float], b: list[float]) -> float:
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)

        # 全ベクトルとの類似度計算
        scored = []
        for item in self._vectors:
            score = cosine_similarity(query_vector, item["vector"])
            if score >= threshold:
                scored.append((item, score))

        # スコアでソート
        scored.sort(key=lambda x: x[1], reverse=True)

        # SearchHitに変換
        hits = []
        for item, score in scored[:top_k]:
            hits.append(
                SearchHit(
                    text_unit_id=item["text_unit_id"],
                    document_id=item.get("document_id", ""),
                    text=item.get("text", ""),
                    score=score,
                    chunk_index=item.get("chunk_index", 0),
                    document_title=item.get("document_title", ""),
                    vector_score=score,
                )
            )

        elapsed = (time.time() - start) * 1000

        return SearchResults(
            hits=hits,
            total_count=len(hits),
            query_vector=query_vector,
            search_time_ms=elapsed,
        )

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float],
        top_k: int = 10,
        alpha: float = 0.5,
    ) -> SearchResults:
        """
        ハイブリッド検索（簡易実装）

        InMemorySearcherではBM25は未実装のため、ベクトル検索のみ
        """
        return self.search(query_vector, top_k)

    def clear(self) -> None:
        """全ベクトルをクリア"""
        self._vectors.clear()

    def __len__(self) -> int:
        """ベクトル数を返す"""
        return len(self._vectors)
