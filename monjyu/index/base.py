# Vector Indexer Base
"""
Base classes and protocols for vector indexers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class SearchResult:
    """検索結果
    
    Attributes:
        id: ドキュメントID
        score: 類似度スコア（距離）
        metadata: メタデータ
    """
    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "score": self.score,
            "metadata": self.metadata,
        }


@runtime_checkable
class VectorIndexerProtocol(Protocol):
    """ベクトルインデクサープロトコル
    
    すべてのベクトルインデクサーが実装すべきインターフェース。
    """
    
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
        ...
    
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
        ...
    
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
            filter_expr: フィルター式
            
        Returns:
            検索結果のリスト
        """
        ...
    
    def save(self, path: Path | str) -> None:
        """インデックスを保存
        
        Args:
            path: 保存先パス
        """
        ...
    
    def load(self, path: Path | str) -> None:
        """インデックスを読み込み
        
        Args:
            path: 読み込み元パス
        """
        ...


class VectorIndexer(ABC):
    """ベクトルインデクサー抽象基底クラス
    
    すべてのベクトルインデクサーの基底クラス。
    サブクラスは各メソッドを実装する必要がある。
    """
    
    @abstractmethod
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
        ...
    
    def add(
        self,
        embeddings: list[list[float]],
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """インデックスにデータを追加
        
        デフォルト実装はbuildを呼び出す。
        サブクラスでオーバーライドして増分追加を実装可能。
        
        Args:
            embeddings: 埋め込みベクトルのリスト
            ids: ID のリスト
            metadata: メタデータのリスト
        """
        self.build(embeddings, ids, metadata)
    
    @abstractmethod
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
            filter_expr: フィルター式
            
        Returns:
            検索結果のリスト
        """
        ...
    
    def search_batch(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[list[SearchResult]]:
        """バッチで類似検索を実行
        
        デフォルト実装は単純ループ。
        サブクラスでオーバーライドしてバッチ最適化可能。
        
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
    
    @abstractmethod
    def save(self, path: Path | str) -> None:
        """インデックスを保存
        
        Args:
            path: 保存先パス
        """
        ...
    
    @abstractmethod
    def load(self, path: Path | str) -> None:
        """インデックスを読み込み
        
        Args:
            path: 読み込み元パス
        """
        ...
    
    @abstractmethod
    def count(self) -> int:
        """インデックス内のアイテム数を取得
        
        Returns:
            アイテム数
        """
        ...
    
    def delete(self, ids: list[str]) -> None:
        """指定IDのアイテムを削除
        
        Args:
            ids: 削除するID のリスト
        """
        msg = "delete は実装されていません"
        raise NotImplementedError(msg)
    
    def clear(self) -> None:
        """インデックスをクリア"""
        msg = "clear は実装されていません"
        raise NotImplementedError(msg)
