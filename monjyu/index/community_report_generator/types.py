# Community Report Generator Types
"""
FEAT-013: CommunityReportGenerator データモデル定義

コミュニティレポートの型定義
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Finding:
    """発見事項
    
    コミュニティから抽出された重要な知見。
    
    Attributes:
        id: 一意識別子
        summary: 発見の要約
        explanation: 詳細説明
        evidence: 根拠となる情報
    """
    id: str
    summary: str
    explanation: str = ""
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "id": self.id,
            "summary": self.summary,
            "explanation": self.explanation,
            "evidence": self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Finding":
        """辞書から生成"""
        return cls(
            id=data.get("id", ""),
            summary=data.get("summary", ""),
            explanation=data.get("explanation", ""),
            evidence=data.get("evidence", []),
        )


@dataclass
class CommunityReport:
    """コミュニティレポート
    
    コミュニティのエグゼクティブサマリー。
    
    Attributes:
        community_id: 対応コミュニティのID
        title: コミュニティを表すタイトル
        summary: 概要（1-2文）
        full_content: 詳細な説明
        findings: 発見事項リスト
        rating: 重要度スコア (0-10)
        rating_explanation: 評価の理由
        entity_count: 含まれるエンティティ数
        relationship_count: 含まれる関係数
        level: コミュニティの階層レベル
        created_at: 生成日時
        metadata: 追加メタデータ
    
    Examples:
        >>> report = CommunityReport(
        ...     community_id="comm-001",
        ...     title="Transformer Architecture Research",
        ...     summary="This community focuses on...",
        ... )
    """
    community_id: str
    title: str
    summary: str
    full_content: str = ""
    findings: List[Finding] = field(default_factory=list)
    
    # 重要度評価
    rating: float = 0.0
    rating_explanation: str = ""
    
    # 統計情報
    entity_count: int = 0
    relationship_count: int = 0
    level: int = 0
    
    # メタデータ
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    @property
    def finding_count(self) -> int:
        """発見事項の数"""
        return len(self.findings)
    
    def add_finding(self, finding: Finding) -> None:
        """発見事項を追加"""
        self.findings.append(finding)
    
    def get_findings_summary(self) -> List[str]:
        """発見事項の要約リスト"""
        return [f.summary for f in self.findings]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "full_content": self.full_content,
            "findings": [f.to_dict() for f in self.findings],
            "rating": self.rating,
            "rating_explanation": self.rating_explanation,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
            "level": self.level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunityReport":
        """辞書から生成"""
        findings = [
            Finding.from_dict(f) if isinstance(f, dict) else f
            for f in data.get("findings", [])
        ]
        
        created_at = None
        if data.get("created_at"):
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"])
            elif isinstance(data["created_at"], datetime):
                created_at = data["created_at"]
        
        return cls(
            community_id=data.get("community_id", ""),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            full_content=data.get("full_content", ""),
            findings=findings,
            rating=data.get("rating", 0.0),
            rating_explanation=data.get("rating_explanation", ""),
            entity_count=data.get("entity_count", 0),
            relationship_count=data.get("relationship_count", 0),
            level=data.get("level", 0),
            created_at=created_at,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ReportGenerationResult:
    """レポート生成結果
    
    Attributes:
        reports: 生成されたレポートリスト
        total_communities: 処理したコミュニティ数
        successful: 成功数
        failed: 失敗数
        generation_time_ms: 生成時間
        errors: エラーメッセージ
    """
    reports: List[CommunityReport] = field(default_factory=list)
    total_communities: int = 0
    successful: int = 0
    failed: int = 0
    generation_time_ms: float = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_communities == 0:
            return 0.0
        return self.successful / self.total_communities
    
    def add_report(self, report: CommunityReport) -> None:
        """レポートを追加"""
        self.reports.append(report)
        self.successful += 1
    
    def add_error(self, error: str) -> None:
        """エラーを追加"""
        self.errors.append(error)
        self.failed += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書に変換"""
        return {
            "reports": [r.to_dict() for r in self.reports],
            "total_communities": self.total_communities,
            "successful": self.successful,
            "failed": self.failed,
            "success_rate": self.success_rate,
            "generation_time_ms": self.generation_time_ms,
            "errors": self.errors,
        }
