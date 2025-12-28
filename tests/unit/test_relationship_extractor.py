# Relationship Extractor Unit Tests
"""
FEAT-011: RelationshipExtractor 単体テスト
"""

import pytest
import json
from unittest.mock import Mock
from dataclasses import dataclass

from monjyu.index.relationship_extractor.types import (
    AcademicRelationType,
    Relationship,
    RelationshipExtractionResult,
    BatchRelationshipResult,
)
from monjyu.index.relationship_extractor.prompts import (
    RELATIONSHIP_EXTRACTION_PROMPT,
    get_relationship_prompt,
    format_entities_for_prompt,
)
from monjyu.index.relationship_extractor.llm_extractor import (
    LLMRelationshipExtractor,
)
from monjyu.index.entity_extractor.types import AcademicEntityType, Entity


# ========== Mock Classes ==========


@dataclass
class MockTextChunk:
    """テストチャンク"""
    id: str
    content: str
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MockLLMClient:
    """モックLLMクライアント"""
    
    def __init__(self, response: str = None):
        self._response = response or '{"relationships": []}'
        self._call_count = 0
    
    def set_response(self, response: str):
        self._response = response
    
    async def chat(self, prompt: str) -> str:
        self._call_count += 1
        return self._response


def create_test_entities() -> list:
    """テスト用エンティティを作成"""
    return [
        Entity(
            id="ent-001",
            name="BERT",
            type=AcademicEntityType.MODEL,
            description="Language model",
            source_chunk_ids=["chunk-1"],
        ),
        Entity(
            id="ent-002",
            name="Transformer",
            type=AcademicEntityType.METHOD,
            description="Attention architecture",
            source_chunk_ids=["chunk-1"],
        ),
        Entity(
            id="ent-003",
            name="Google",
            type=AcademicEntityType.ORGANIZATION,
            description="Tech company",
            source_chunk_ids=["chunk-1"],
        ),
        Entity(
            id="ent-004",
            name="ImageNet",
            type=AcademicEntityType.DATASET,
            description="Image dataset",
            source_chunk_ids=["chunk-1"],
        ),
    ]


# ========== Type Tests ==========


class TestAcademicRelationType:
    """関係タイプのテスト"""
    
    def test_all_types_defined(self):
        """全タイプが定義されている"""
        assert len(AcademicRelationType) >= 20
    
    def test_type_values(self):
        """タイプ値が正しい"""
        assert AcademicRelationType.USES.value == "uses"
        assert AcademicRelationType.EXTENDS.value == "extends"
        assert AcademicRelationType.BASED_ON.value == "based_on"
        assert AcademicRelationType.PROPOSED_BY.value == "proposed_by"
    
    def test_type_from_string(self):
        """文字列からの変換"""
        assert AcademicRelationType("uses") == AcademicRelationType.USES
        assert AcademicRelationType("trained_on") == AcademicRelationType.TRAINED_ON


class TestRelationship:
    """関係のテスト"""
    
    def test_relationship_creation(self):
        """関係作成"""
        rel = Relationship(
            id="rel-001",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            source_entity_name="BERT",
            target_entity_name="Transformer",
            type=AcademicRelationType.BASED_ON,
            description="BERT is based on Transformer",
        )
        assert rel.id == "rel-001"
        assert rel.source_entity_name == "BERT"
        assert rel.target_entity_name == "Transformer"
        assert rel.type == AcademicRelationType.BASED_ON
    
    def test_to_dict(self):
        """辞書変換"""
        rel = Relationship(
            id="rel-002",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            source_entity_name="GPT",
            target_entity_name="OpenAI",
            type=AcademicRelationType.DEVELOPED_BY,
            description="GPT developed by OpenAI",
        )
        data = rel.to_dict()
        
        assert data["id"] == "rel-002"
        assert data["source_entity_name"] == "GPT"
        assert data["type"] == "developed_by"
    
    def test_from_dict(self):
        """辞書から生成"""
        data = {
            "id": "rel-003",
            "source_entity_id": "ent-a",
            "target_entity_id": "ent-b",
            "source_entity_name": "ResNet",
            "target_entity_name": "ImageNet",
            "type": "trained_on",
            "description": "ResNet trained on ImageNet",
        }
        rel = Relationship.from_dict(data)
        
        assert rel.id == "rel-003"
        assert rel.type == AcademicRelationType.TRAINED_ON
    
    def test_key_property(self):
        """キー生成"""
        rel = Relationship(
            id="rel-004",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            source_entity_name="BERT",
            target_entity_name="Transformer",
            type=AcademicRelationType.BASED_ON,
        )
        assert rel.key == "BERT|based_on|Transformer"
    
    def test_reverse(self):
        """関係の逆転"""
        rel = Relationship(
            id="rel-005",
            source_entity_id="ent-001",
            target_entity_id="ent-002",
            source_entity_name="BERT",
            target_entity_name="Transformer",
            type=AcademicRelationType.USES,
        )
        reversed_rel = rel.reverse()
        
        assert reversed_rel.source_entity_name == "Transformer"
        assert reversed_rel.target_entity_name == "BERT"
        assert reversed_rel.id == "rel-005_rev"


class TestRelationshipExtractionResult:
    """抽出結果のテスト"""
    
    def test_success_result(self):
        """成功結果"""
        result = RelationshipExtractionResult(
            chunk_id="chunk-1",
            relationships=[
                Relationship(
                    id="r1",
                    source_entity_id="e1",
                    target_entity_id="e2",
                    source_entity_name="A",
                    target_entity_name="B",
                    type=AcademicRelationType.USES,
                ),
            ],
        )
        
        assert result.success is True
        assert result.relationship_count == 1
    
    def test_error_result(self):
        """エラー結果"""
        result = RelationshipExtractionResult(
            chunk_id="chunk-2",
            relationships=[],
            error="Parse error",
        )
        
        assert result.success is False
        assert result.relationship_count == 0


class TestBatchRelationshipResult:
    """バッチ結果のテスト"""
    
    def test_success_rate(self):
        """成功率計算"""
        result = BatchRelationshipResult(
            results=[
                RelationshipExtractionResult(chunk_id="1", relationships=[]),
                RelationshipExtractionResult(chunk_id="2", relationships=[], error="Err"),
                RelationshipExtractionResult(chunk_id="3", relationships=[]),
            ],
            error_count=1,
        )
        
        assert result.success_rate == pytest.approx(2/3, rel=0.01)


# ========== Prompt Tests ==========


class TestPrompts:
    """プロンプトのテスト"""
    
    def test_format_entities(self):
        """エンティティフォーマット"""
        entities = create_test_entities()
        formatted = format_entities_for_prompt(entities)
        
        assert "BERT" in formatted
        assert "model" in formatted
        assert "Transformer" in formatted
    
    def test_get_relationship_prompt_english(self):
        """英語プロンプト"""
        entities = create_test_entities()
        text = "BERT is based on Transformer architecture."
        prompt = get_relationship_prompt(entities, text, "en")
        
        assert "BERT" in prompt
        assert "Transformer" in prompt
        assert "Relationship Types" in prompt
    
    def test_get_relationship_prompt_japanese(self):
        """日本語プロンプト"""
        entities = create_test_entities()
        text = "BERTはTransformerアーキテクチャに基づく。"
        prompt = get_relationship_prompt(entities, text, "ja")
        
        assert "関係タイプ" in prompt


# ========== LLMRelationshipExtractor Tests ==========


class TestLLMRelationshipExtractor:
    """LLM関係抽出のテスト"""
    
    @pytest.fixture
    def mock_llm(self):
        return MockLLMClient()
    
    @pytest.fixture
    def extractor(self, mock_llm):
        return LLMRelationshipExtractor(mock_llm)
    
    @pytest.fixture
    def entities(self):
        return create_test_entities()
    
    @pytest.mark.asyncio
    async def test_extract_single_relationship(self, mock_llm, extractor, entities):
        """単一関係抽出"""
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Transformer",
                    "type": "BASED_ON",
                    "description": "BERT is based on Transformer",
                    "evidence": "BERT uses the Transformer architecture"
                }
            ]
        }))
        
        chunk = MockTextChunk(id="chunk-1", content="BERT is based on Transformer.")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert len(result.relationships) == 1
        assert result.relationships[0].source_entity_name == "BERT"
        assert result.relationships[0].target_entity_name == "Transformer"
        assert result.relationships[0].type == AcademicRelationType.BASED_ON
    
    @pytest.mark.asyncio
    async def test_extract_multiple_relationships(self, mock_llm, extractor, entities):
        """複数関係抽出"""
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Transformer",
                    "type": "BASED_ON",
                    "description": "Based on",
                    "evidence": "..."
                },
                {
                    "source": "BERT",
                    "target": "Google",
                    "type": "PROPOSED_BY",
                    "description": "Proposed by Google",
                    "evidence": "..."
                }
            ]
        }))
        
        chunk = MockTextChunk(id="chunk-1", content="BERT by Google uses Transformer.")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert len(result.relationships) == 2
    
    @pytest.mark.asyncio
    async def test_extract_no_relationships(self, mock_llm, extractor, entities):
        """関係なしの抽出"""
        mock_llm.set_response('{"relationships": []}')
        
        chunk = MockTextChunk(id="chunk-1", content="No relationships here.")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert len(result.relationships) == 0
    
    @pytest.mark.asyncio
    async def test_extract_with_insufficient_entities(self, mock_llm, extractor):
        """エンティティ不足"""
        entities = [
            Entity(id="e1", name="BERT", type=AcademicEntityType.MODEL, description=""),
        ]
        
        chunk = MockTextChunk(id="chunk-1", content="Only BERT mentioned.")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert len(result.relationships) == 0
    
    @pytest.mark.asyncio
    async def test_extract_invalid_json(self, mock_llm, extractor, entities):
        """不正なJSON応答"""
        mock_llm.set_response("This is not JSON")
        
        chunk = MockTextChunk(id="chunk-1", content="Some text")
        result = await extractor.extract(entities, chunk)
        
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_extract_unknown_entity_filtered(self, mock_llm, extractor, entities):
        """未知のエンティティは除外"""
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Unknown Entity",
                    "type": "USES",
                    "description": "...",
                    "evidence": "..."
                }
            ]
        }))
        
        chunk = MockTextChunk(id="chunk-1", content="BERT uses something.")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert len(result.relationships) == 0  # Unknown entity filtered
    
    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_llm, extractor, entities):
        """バッチ抽出"""
        mock_llm.set_response('{"relationships": []}')
        
        chunks = [
            MockTextChunk(id=f"chunk-{i}", content=f"Content {i}")
            for i in range(5)
        ]
        
        results = await extractor.extract_batch(entities, chunks, max_concurrent=2)
        
        assert len(results) == 5
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_extract_all_with_merge(self, mock_llm, extractor, entities):
        """全抽出+マージ"""
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Transformer",
                    "type": "BASED_ON",
                    "description": "Based on",
                    "evidence": "..."
                }
            ]
        }))
        
        chunks = [
            MockTextChunk(id=f"chunk-{i}", content="BERT uses Transformer")
            for i in range(3)
        ]
        
        result = await extractor.extract_all(entities, chunks)
        
        assert result.total_relationships == 3
        assert len(result.merged_relationships) == 1  # All merged
        assert len(result.merged_relationships[0].source_chunk_ids) == 3
    
    def test_merge_relationships(self, extractor):
        """関係マージ"""
        relationships = [
            Relationship(
                id="r1",
                source_entity_id="e1",
                target_entity_id="e2",
                source_entity_name="BERT",
                target_entity_name="Transformer",
                type=AcademicRelationType.BASED_ON,
                source_chunk_ids=["c1"],
            ),
            Relationship(
                id="r2",
                source_entity_id="e1",
                target_entity_id="e2",
                source_entity_name="BERT",
                target_entity_name="Transformer",
                type=AcademicRelationType.BASED_ON,
                source_chunk_ids=["c2"],
            ),
            Relationship(
                id="r3",
                source_entity_id="e1",
                target_entity_id="e3",
                source_entity_name="BERT",
                target_entity_name="Google",
                type=AcademicRelationType.PROPOSED_BY,
                source_chunk_ids=["c1"],
            ),
        ]
        
        merged = extractor.merge_relationships(relationships)
        
        assert len(merged) == 2
        
        bert_transformer = next(r for r in merged if r.target_entity_name == "Transformer")
        assert len(bert_transformer.source_chunk_ids) == 2


# ========== Relation Type Parsing Tests ==========


class TestRelationTypeParsing:
    """関係タイプ解析のテスト"""
    
    @pytest.mark.asyncio
    async def test_unknown_type_falls_back(self):
        """未知のタイプはRELATED_TOにフォールバック"""
        mock_llm = MockLLMClient()
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Transformer",
                    "type": "UNKNOWN_TYPE",
                    "description": "...",
                    "evidence": "..."
                }
            ]
        }))
        
        extractor = LLMRelationshipExtractor(mock_llm)
        entities = create_test_entities()
        chunk = MockTextChunk(id="c1", content="Test")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert result.relationships[0].type == AcademicRelationType.RELATED_TO
    
    @pytest.mark.asyncio
    async def test_lowercase_type(self):
        """小文字タイプも受け付ける"""
        mock_llm = MockLLMClient()
        mock_llm.set_response(json.dumps({
            "relationships": [
                {
                    "source": "BERT",
                    "target": "Transformer",
                    "type": "based_on",
                    "description": "...",
                    "evidence": "..."
                }
            ]
        }))
        
        extractor = LLMRelationshipExtractor(mock_llm)
        entities = create_test_entities()
        chunk = MockTextChunk(id="c1", content="Test")
        result = await extractor.extract(entities, chunk)
        
        assert result.success
        assert result.relationships[0].type == AcademicRelationType.BASED_ON
