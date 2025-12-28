# Entity Extractor Types
"""
FEAT-010: EntityExtractor データモデル定義

学術エンティティのタイプ定義とデータ構造
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AcademicEntityType(Enum):
    """学術エンティティタイプ
    
    学術論文から抽出される主要なエンティティの分類。
    
    Examples:
        >>> entity_type = AcademicEntityType.MODEL
        >>> entity_type.value
        'model'
    """
    RESEARCHER = "researcher"      # 研究者、著者
    ORGANIZATION = "organization"  # 機関、企業、研究室
    METHOD = "method"              # アルゴリズム、手法
    MODEL = "model"                # MLモデル（GPT-4, BERT等）
    DATASET = "dataset"            # データセット（ImageNet等）
    METRIC = "metric"              # 評価指標（accuracy, F1等）
    TASK = "task"                  # タスク（分類、翻訳等）
    CONCEPT = "concept"            # 概念、理論
    TOOL = "tool"                  # ツール、フレームワーク
    PAPER = "paper"                # 参照論文


@dataclass
class Entity:
    """エンティティ
    
    学術論文から抽出されたエンティティを表現。
    
    Attributes:
        id: 一意識別子
        name: エンティティ名
        type: エンティティタイプ
        description: 説明文
        aliases: 別名リスト
        source_chunk_ids: 抽出元チャンクID
        first_mentioned_year: 初出年（わかる場合）
        external_ids: 外部ID（DOI, arXiv等）
        confidence: 信頼度スコア (0.0-1.0)
    
    Examples:
        >>> entity = Entity(
        ...     id="ent-001",
        ...     name="BERT",
        ...     type=AcademicEntityType.MODEL,
        ...     description="Bidirectional Encoder Representations from Transformers",
        ...     aliases=["Bidirectional Encoder"],
        ... )
        >>> entity.type.value
        'model'
    """
    id: str
    name: str
    type: AcademicEntityType
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    source_chunk_ids: List[str] = field(default_factory=list)
    
    # 学術固有
    first_mentioned_year: Optional[int] = None
    external_ids: Dict[str, str] = field(default_factory=dict)
    
    # 信頼度
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換
        
        Returns:
            エンティティの辞書表現
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "aliases": self.aliases,
            "source_chunk_ids": self.source_chunk_ids,
            "first_mentioned_year": self.first_mentioned_year,
            "external_ids": self.external_ids,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """辞書から生成
        
        Args:
            data: エンティティデータの辞書
            
        Returns:
            Entityインスタンス
        """
        return cls(
            id=data["id"],
            name=data["name"],
            type=AcademicEntityType(data["type"]),
            description=data.get("description", ""),
            aliases=data.get("aliases", []),
            source_chunk_ids=data.get("source_chunk_ids", []),
            first_mentioned_year=data.get("first_mentioned_year"),
            external_ids=data.get("external_ids", {}),
            confidence=data.get("confidence", 1.0),
        )
    
    def merge_with(self, other: "Entity") -> "Entity":
        """他のエンティティとマージ
        
        Args:
            other: マージ対象のエンティティ
            
        Returns:
            マージされた新しいエンティティ
        """
        merged_aliases = list(set(self.aliases + other.aliases))
        if other.name not in merged_aliases and other.name != self.name:
            merged_aliases.append(other.name)
        
        merged_chunks = list(set(self.source_chunk_ids + other.source_chunk_ids))
        
        # より長い説明を採用
        description = self.description if len(self.description) >= len(other.description) else other.description
        
        # 外部IDをマージ
        external_ids = {**self.external_ids, **other.external_ids}
        
        return Entity(
            id=self.id,
            name=self.name,
            type=self.type,
            description=description,
            aliases=merged_aliases,
            source_chunk_ids=merged_chunks,
            first_mentioned_year=self.first_mentioned_year or other.first_mentioned_year,
            external_ids=external_ids,
            confidence=max(self.confidence, other.confidence),
        )


@dataclass
class ExtractionResult:
    """抽出結果
    
    単一チャンクからのエンティティ抽出結果。
    
    Attributes:
        chunk_id: 抽出元チャンクID
        entities: 抽出されたエンティティリスト
        raw_response: LLMの生レスポンス
        tokens_used: 使用トークン数
        extraction_time_ms: 抽出時間（ミリ秒）
        error: エラーメッセージ（あれば）
    """
    chunk_id: str
    entities: List[Entity] = field(default_factory=list)
    raw_response: str = ""
    tokens_used: int = 0
    extraction_time_ms: float = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """抽出が成功したかどうか"""
        return self.error is None
    
    @property
    def entity_count(self) -> int:
        """抽出されたエンティティ数"""
        return len(self.entities)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "chunk_id": self.chunk_id,
            "entities": [e.to_dict() for e in self.entities],
            "raw_response": self.raw_response,
            "tokens_used": self.tokens_used,
            "extraction_time_ms": self.extraction_time_ms,
            "error": self.error,
        }


@dataclass
class BatchExtractionResult:
    """バッチ抽出結果
    
    複数チャンクからの抽出結果を集約。
    """
    results: List[ExtractionResult] = field(default_factory=list)
    total_entities: int = 0
    merged_entities: List[Entity] = field(default_factory=list)
    total_time_ms: float = 0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if not self.results:
            return 0.0
        return (len(self.results) - self.error_count) / len(self.results)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "result_count": len(self.results),
            "total_entities": self.total_entities,
            "merged_entity_count": len(self.merged_entities),
            "total_time_ms": self.total_time_ms,
            "error_count": self.error_count,
            "success_rate": self.success_rate,
        }
