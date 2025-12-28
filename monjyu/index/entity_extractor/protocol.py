# Entity Extractor Protocol
"""
FEAT-010: EntityExtractor プロトコル定義

エンティティ抽出の抽象インターフェース
"""

from abc import ABC, abstractmethod
from typing import List, AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from monjyu.index.entity_extractor.types import Entity, ExtractionResult


class EntityExtractorProtocol(ABC):
    """エンティティ抽出プロトコル
    
    テキストチャンクから学術エンティティを抽出するための
    抽象インターフェース。
    
    Examples:
        >>> class MyExtractor(EntityExtractorProtocol):
        ...     async def extract(self, chunk):
        ...         # 実装
        ...         pass
    """
    
    @abstractmethod
    async def extract(
        self,
        chunk: "TextChunk"
    ) -> "ExtractionResult":
        """単一チャンクからエンティティ抽出
        
        Args:
            chunk: 抽出対象のテキストチャンク
            
        Returns:
            抽出結果（エンティティリストと統計情報）
        """
        ...
    
    @abstractmethod
    async def extract_batch(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> List["ExtractionResult"]:
        """複数チャンクから一括抽出
        
        Args:
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            抽出結果のリスト
        """
        ...
    
    @abstractmethod
    async def extract_stream(
        self,
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> AsyncIterator["ExtractionResult"]:
        """ストリーミング抽出
        
        チャンクごとに結果をyieldする非同期イテレータ。
        大量のチャンクを処理する際にメモリ効率が良い。
        
        Args:
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Yields:
            各チャンクの抽出結果
        """
        ...
    
    @abstractmethod
    def merge_entities(
        self,
        entities: List["Entity"]
    ) -> List["Entity"]:
        """重複エンティティをマージ
        
        同一エンティティの異なる表現（別名、typo等）を
        統合して一つのエンティティにまとめる。
        
        Args:
            entities: マージ対象のエンティティリスト
            
        Returns:
            マージ後のエンティティリスト
        """
        ...


# Type stub for TextChunk (imported from index module)
class TextChunk:
    """テキストチャンク（型ヒント用）
    
    実際の実装は monjyu.index.text_chunker にある。
    """
    id: str
    content: str
    metadata: dict
