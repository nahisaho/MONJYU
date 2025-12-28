# Relationship Extractor Types
"""
FEAT-011: RelationshipExtractor データモデル定義

エンティティ間関係のタイプ定義とデータ構造
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class AcademicRelationType(Enum):
    """学術関係タイプ
    
    学術論文におけるエンティティ間の関係分類。
    
    Examples:
        >>> rel_type = AcademicRelationType.USES
        >>> rel_type.value
        'uses'
    """
    # 技術的関係
    USES = "uses"                      # AがBを使用
    EXTENDS = "extends"                # AがBを拡張
    IMPROVES = "improves"              # AがBを改善
    IMPLEMENTS = "implements"          # AがBを実装
    BASED_ON = "based_on"              # AがBに基づく
    
    # 評価・比較
    COMPARES = "compares"              # AとBを比較
    OUTPERFORMS = "outperforms"        # AがBより優れる
    EVALUATES_ON = "evaluates_on"      # AをBで評価
    
    # 学習・データ
    TRAINED_ON = "trained_on"          # AをBで訓練
    FINE_TUNED_ON = "fine_tuned_on"    # AをBでファインチューン
    APPLIED_TO = "applied_to"          # AをBに適用
    
    # 人・組織
    PROPOSED_BY = "proposed_by"        # AをBが提案
    DEVELOPED_BY = "developed_by"      # AをBが開発
    AFFILIATED_WITH = "affiliated_with"  # AがBに所属
    COLLABORATED_WITH = "collaborated_with"  # AがBと協力
    
    # 参照・引用
    CITES = "cites"                    # AがBを引用
    REFERENCES = "references"          # AがBを参照
    
    # 一般
    RELATED_TO = "related_to"          # 一般的な関連
    SIMILAR_TO = "similar_to"          # 類似関係
    PART_OF = "part_of"                # AがBの一部


@dataclass
class Relationship:
    """関係
    
    2つのエンティティ間の関係を表現。
    
    Attributes:
        id: 一意識別子
        source_entity_id: ソースエンティティID
        target_entity_id: ターゲットエンティティID
        source_entity_name: ソースエンティティ名
        target_entity_name: ターゲットエンティティ名
        type: 関係タイプ
        description: 関係の説明
        weight: 関係の重み（0.0-1.0）
        source_chunk_ids: 抽出元チャンクID
        evidence: 関係の根拠となるテキスト
        confidence: 信頼度スコア
    
    Examples:
        >>> rel = Relationship(
        ...     id="rel-001",
        ...     source_entity_id="ent-001",
        ...     target_entity_id="ent-002",
        ...     source_entity_name="BERT",
        ...     target_entity_name="Transformer",
        ...     type=AcademicRelationType.BASED_ON,
        ...     description="BERT is based on Transformer architecture",
        ... )
    """
    id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    type: AcademicRelationType
    description: str = ""
    weight: float = 1.0
    source_chunk_ids: List[str] = field(default_factory=list)
    evidence: str = ""
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "source_entity_id": self.source_entity_id,
            "target_entity_id": self.target_entity_id,
            "source_entity_name": self.source_entity_name,
            "target_entity_name": self.target_entity_name,
            "type": self.type.value,
            "description": self.description,
            "weight": self.weight,
            "source_chunk_ids": self.source_chunk_ids,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relationship":
        """辞書から生成"""
        return cls(
            id=data["id"],
            source_entity_id=data["source_entity_id"],
            target_entity_id=data["target_entity_id"],
            source_entity_name=data.get("source_entity_name", ""),
            target_entity_name=data.get("target_entity_name", ""),
            type=AcademicRelationType(data["type"]),
            description=data.get("description", ""),
            weight=data.get("weight", 1.0),
            source_chunk_ids=data.get("source_chunk_ids", []),
            evidence=data.get("evidence", ""),
            confidence=data.get("confidence", 1.0),
        )
    
    def reverse(self) -> "Relationship":
        """関係を逆転（双方向関係用）"""
        return Relationship(
            id=f"{self.id}_rev",
            source_entity_id=self.target_entity_id,
            target_entity_id=self.source_entity_id,
            source_entity_name=self.target_entity_name,
            target_entity_name=self.source_entity_name,
            type=self.type,
            description=self.description,
            weight=self.weight,
            source_chunk_ids=self.source_chunk_ids,
            evidence=self.evidence,
            confidence=self.confidence,
        )
    
    @property
    def key(self) -> str:
        """関係のユニークキー（重複検出用）"""
        return f"{self.source_entity_name}|{self.type.value}|{self.target_entity_name}"


@dataclass
class RelationshipExtractionResult:
    """関係抽出結果
    
    Attributes:
        chunk_id: 抽出元チャンクID
        relationships: 抽出された関係リスト
        raw_response: LLMの生レスポンス
        extraction_time_ms: 抽出時間（ミリ秒）
        error: エラーメッセージ
    """
    chunk_id: str
    relationships: List[Relationship] = field(default_factory=list)
    raw_response: str = ""
    extraction_time_ms: float = 0
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """抽出が成功したかどうか"""
        return self.error is None
    
    @property
    def relationship_count(self) -> int:
        """抽出された関係数"""
        return len(self.relationships)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "chunk_id": self.chunk_id,
            "relationships": [r.to_dict() for r in self.relationships],
            "raw_response": self.raw_response,
            "extraction_time_ms": self.extraction_time_ms,
            "error": self.error,
        }


@dataclass
class BatchRelationshipResult:
    """バッチ抽出結果"""
    results: List[RelationshipExtractionResult] = field(default_factory=list)
    total_relationships: int = 0
    merged_relationships: List[Relationship] = field(default_factory=list)
    total_time_ms: float = 0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if not self.results:
            return 0.0
        return (len(self.results) - self.error_count) / len(self.results)
