# LanceDB Vector Indexer
"""
Vector indexer using LanceDB for local development.

LanceDB is an embedded vector database that requires no server setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

from monjyu.index.base import VectorIndexer, SearchResult

if TYPE_CHECKING:
    import lancedb


class LanceDBIndexer(VectorIndexer):
    """LanceDB ベクトルインデクサー（ローカル開発用）
    
    LanceDBを使用してベクトルインデックスを構築・検索する。
    サーバー不要で、ローカル環境での開発に最適。
    
    Example:
        >>> indexer = LanceDBIndexer(db_path="./storage/lancedb")
        >>> indexer.build(
        ...     embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ...     ids=["id1", "id2"],
        ...     metadata=[{"text": "hello"}, {"text": "world"}],
        ... )
        >>> results = indexer.search([0.1, 0.2, 0.3], top_k=5)
    
    Attributes:
        db_path: LanceDBデータベースのパス
        table_name: テーブル名
    """
    
    def __init__(
        self,
        db_path: str | Path = "./storage/lancedb",
        table_name: str = "text_units",
        metric: str = "L2",
    ) -> None:
        """初期化
        
        Args:
            db_path: データベースパス
            table_name: テーブル名
            metric: 距離メトリック ("L2", "cosine", "dot")
        """
        import lancedb as ldb
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.table_name = table_name
        self.metric = metric
        
        self._db: lancedb.DBConnection = ldb.connect(str(self.db_path))
        self._table: lancedb.Table | None = None
        
        # 既存テーブルがあれば開く
        if self.table_name in self._list_table_names():
            self._table = self._db.open_table(self.table_name)
    
    def _list_table_names(self) -> list[str]:
        """テーブル名のリストを取得
        
        LanceDB の list_tables() は ListTablesResponse を返すため、
        .tables 属性でリストにアクセスする。
        """
        result = self._db.list_tables()
        if hasattr(result, 'tables'):
            return result.tables
        return list(result)
    
    def build(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """インデックスを構築（既存があれば上書き）
        
        Args:
            embeddings: 埋め込みベクトルのリスト
            ids: ID のリスト
            metadata: メタデータのリスト
        """
        if len(embeddings) != len(ids):
            msg = "embeddings と ids の長さが一致しません"
            raise ValueError(msg)
        
        # データを準備
        data = self._prepare_data(embeddings, ids, metadata)
        
        # テーブルを作成（上書き）
        self._table = self._db.create_table(
            self.table_name,
            data=data,
            mode="overwrite",
        )
    
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
        if self._table is None:
            self.build(embeddings, ids, metadata)
            return
        
        data = self._prepare_data(embeddings, ids, metadata)
        self._table.add(data)
    
    def _prepare_data(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """データを準備"""
        data = []
        for i, (emb, id_) in enumerate(zip(embeddings, ids, strict=True)):
            record: dict[str, Any] = {
                "id": id_,
                "vector": emb,
            }
            if metadata and i < len(metadata):
                # メタデータをフラット化（LanceDBはネストをサポートしない場合がある）
                for key, value in metadata[i].items():
                    if isinstance(value, (str, int, float, bool)):
                        record[key] = value
                    else:
                        record[key] = str(value)
            data.append(record)
        return data
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[SearchResult]:
        """類似検索を実行
        
        Args:
            query_embedding: クエリベクトル
            top_k: 取得する上位件数
            filter_expr: フィルター式（LanceDBのWHERE句形式）
            
        Returns:
            検索結果のリスト
        """
        if self._table is None:
            msg = "インデックスが構築されていません"
            raise ValueError(msg)
        
        # 検索クエリを構築
        query = self._table.search(query_embedding).limit(top_k)
        
        # フィルターを適用
        if filter_expr:
            query = query.where(filter_expr)
        
        # 検索を実行
        results = query.to_list()
        
        # SearchResultに変換
        return [
            SearchResult(
                id=r["id"],
                score=r.get("_distance", 0.0),
                metadata={
                    k: v for k, v in r.items()
                    if k not in ["id", "vector", "_distance"]
                },
            )
            for r in results
        ]
    
    def search_batch(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[list[SearchResult]]:
        """バッチで類似検索を実行
        
        Args:
            query_embeddings: クエリベクトルのリスト
            top_k: 取得する上位件数
            filter_expr: フィルター式
            
        Returns:
            検索結果のリストのリスト
        """
        return [
            self.search(q, top_k, filter_expr)
            for q in query_embeddings
        ]
    
    def save(self, path: Path | str) -> None:
        """インデックスを保存
        
        LanceDBは自動永続化されるため、特別な処理は不要。
        パスを変更したい場合は新しいDBに書き込む。
        
        Args:
            path: 保存先パス
        """
        path = Path(path)
        if path != self.db_path and self._table is not None:
            import lancedb as ldb
            
            new_db = ldb.connect(str(path))
            data = self._table.to_pandas().to_dict("records")
            new_db.create_table(self.table_name, data=data, mode="overwrite")
    
    def load(self, path: Path | str) -> None:
        """インデックスを読み込み
        
        Args:
            path: 読み込み元パス
        """
        import lancedb as ldb
        
        path = Path(path)
        self.db_path = path
        self._db = ldb.connect(str(path))
        
        if self.table_name in self._list_table_names():
            self._table = self._db.open_table(self.table_name)
        else:
            self._table = None
    
    def count(self) -> int:
        """インデックス内のアイテム数を取得
        
        Returns:
            アイテム数
        """
        if self._table is None:
            return 0
        return self._table.count_rows()
    
    def delete(self, ids: list[str]) -> None:
        """指定IDのアイテムを削除
        
        Args:
            ids: 削除するID のリスト
        """
        if self._table is None:
            return
        
        # LanceDBはSQLライクなフィルターで削除
        id_list = ", ".join(f"'{id_}'" for id_ in ids)
        self._table.delete(f"id IN ({id_list})")
    
    def clear(self) -> None:
        """インデックスをクリア"""
        if self.table_name in self._list_table_names():
            self._db.drop_table(self.table_name)
        self._table = None
    
    def get_by_id(self, id_: str) -> SearchResult | None:
        """IDでアイテムを取得
        
        Args:
            id_: アイテムID
            
        Returns:
            SearchResult（存在しない場合はNone）
        """
        if self._table is None:
            return None
        
        results = self._table.search().where(f"id = '{id_}'").limit(1).to_list()
        if not results:
            return None
        
        r = results[0]
        return SearchResult(
            id=r["id"],
            score=r.get("_distance", 0.0),
            metadata={k: v for k, v in r.items() if k not in ["id", "vector", "_distance"]},
        )
    
    def get_by_ids(self, ids: list[str]) -> list[SearchResult]:
        """複数IDでアイテムを取得
        
        Args:
            ids: アイテムIDのリスト
            
        Returns:
            SearchResultのリスト
        """
        if self._table is None:
            return []
        
        id_list = ", ".join(f"'{id_}'" for id_ in ids)
        results = self._table.search().where(f"id IN ({id_list})").to_list()
        
        return [
            SearchResult(
                id=r["id"],
                score=r.get("_distance", 0.0),
                metadata={k: v for k, v in r.items() if k not in ["id", "vector", "_distance"]},
            )
            for r in results
        ]
