# Embedding Client Base
"""
Base classes and protocols for embedding clients.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClientProtocol(Protocol):
    """埋め込みクライアントプロトコル
    
    すべての埋め込みクライアントが実装すべきインターフェース。
    """
    
    async def embed(self, text: str) -> list[float]:
        """単一テキストの埋め込みを生成
        
        Args:
            text: 埋め込みを生成するテキスト
            
        Returns:
            埋め込みベクトル
        """
        ...
    
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """バッチでテキストの埋め込みを生成
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
            
        Returns:
            埋め込みベクトルのリスト
        """
        ...
    
    @property
    def dimensions(self) -> int:
        """埋め込みの次元数"""
        ...
    
    @property
    def model_name(self) -> str:
        """モデル名"""
        ...


@dataclass
class EmbeddingRecord:
    """埋め込みレコード
    
    Attributes:
        id: レコードID
        text_unit_id: 元のTextUnitのID
        vector: 埋め込みベクトル
        model: 使用したモデル名
        dimensions: 次元数
    """
    id: str
    text_unit_id: str
    vector: list[float]
    model: str
    dimensions: int


class EmbeddingClient(ABC):
    """埋め込みクライアント抽象基底クラス
    
    すべての埋め込みクライアントの基底クラス。
    サブクラスは embed() と embed_batch() を実装する必要がある。
    
    Example:
        >>> class MyEmbeddingClient(EmbeddingClient):
        ...     async def embed(self, text: str) -> list[float]:
        ...         # 実装
        ...         pass
    """
    
    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """単一テキストの埋め込みを生成
        
        Args:
            text: 埋め込みを生成するテキスト
            
        Returns:
            埋め込みベクトル
        """
        ...
    
    async def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """バッチでテキストの埋め込みを生成
        
        デフォルト実装は単純な並行処理。
        サブクラスでオーバーライドしてバッチAPIを使用可能。
        
        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ
            
        Returns:
            埋め込みベクトルのリスト
        """
        import asyncio
        
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[self.embed(text) for text in batch])
            results.extend(batch_results)
        return results
    
    @property
    @abstractmethod
    def dimensions(self) -> int:
        """埋め込みの次元数"""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """モデル名"""
        ...
    
    def create_record(
        self,
        text_unit_id: str,
        vector: list[float],
        record_id: str | None = None,
    ) -> EmbeddingRecord:
        """埋め込みレコードを作成
        
        Args:
            text_unit_id: TextUnitのID
            vector: 埋め込みベクトル
            record_id: レコードID（省略時は自動生成）
            
        Returns:
            EmbeddingRecord
        """
        import uuid
        
        return EmbeddingRecord(
            id=record_id or str(uuid.uuid4()),
            text_unit_id=text_unit_id,
            vector=vector,
            model=self.model_name,
            dimensions=self.dimensions,
        )
