# Relationship Extractor Protocol
"""
FEAT-011: RelationshipExtractor プロトコル定義

関係抽出の抽象インターフェース
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from monjyu.index.relationship_extractor.types import Relationship, RelationshipExtractionResult
    from monjyu.index.entity_extractor.types import Entity


class RelationshipExtractorProtocol(ABC):
    """関係抽出プロトコル
    
    エンティティ間の関係を抽出するための抽象インターフェース。
    """
    
    @abstractmethod
    async def extract(
        self,
        entities: List["Entity"],
        chunk: "TextChunk"
    ) -> "RelationshipExtractionResult":
        """単一チャンクから関係抽出
        
        Args:
            entities: 対象エンティティリスト
            chunk: 抽出対象のテキストチャンク
            
        Returns:
            抽出結果
        """
        ...
    
    @abstractmethod
    async def extract_batch(
        self,
        entities: List["Entity"],
        chunks: List["TextChunk"],
        max_concurrent: int = 5
    ) -> List["RelationshipExtractionResult"]:
        """複数チャンクから一括抽出
        
        Args:
            entities: 対象エンティティリスト
            chunks: 抽出対象のテキストチャンクリスト
            max_concurrent: 最大同時実行数
            
        Returns:
            抽出結果のリスト
        """
        ...
    
    @abstractmethod
    def merge_relationships(
        self,
        relationships: List["Relationship"]
    ) -> List["Relationship"]:
        """重複関係をマージ
        
        Args:
            relationships: マージ対象の関係リスト
            
        Returns:
            マージ後の関係リスト
        """
        ...


# Type stub for TextChunk
class TextChunk:
    """テキストチャンク（型ヒント用）"""
    id: str
    content: str
    metadata: dict
