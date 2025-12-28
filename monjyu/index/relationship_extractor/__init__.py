# Relationship Extractor - FEAT-011
"""
MONJYU Relationship Extractor Module

LLMベースのエンティティ間関係抽出（GraphRAG Level 2）
"""

from monjyu.index.relationship_extractor.types import (
    AcademicRelationType,
    Relationship,
    RelationshipExtractionResult,
)
from monjyu.index.relationship_extractor.protocol import RelationshipExtractorProtocol
from monjyu.index.relationship_extractor.llm_extractor import LLMRelationshipExtractor
from monjyu.index.relationship_extractor.prompts import RELATIONSHIP_EXTRACTION_PROMPT

__all__ = [
    "AcademicRelationType",
    "Relationship",
    "RelationshipExtractionResult",
    "RelationshipExtractorProtocol",
    "LLMRelationshipExtractor",
    "RELATIONSHIP_EXTRACTION_PROMPT",
]
