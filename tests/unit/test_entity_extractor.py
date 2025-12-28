# Entity Extractor Unit Tests
"""
FEAT-010: EntityExtractor 単体テスト
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock
from dataclasses import dataclass
from typing import Optional

from monjyu.index.entity_extractor.types import (
    AcademicEntityType,
    Entity,
    ExtractionResult,
    BatchExtractionResult,
)
from monjyu.index.entity_extractor.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    get_extraction_prompt,
)
from monjyu.index.entity_extractor.llm_extractor import (
    LLMEntityExtractor,
    CachedEntityExtractor,
)


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
        self._response = response or '{"entities": []}'
        self._fail_count = 0
        self._call_count = 0
    
    def set_response(self, response: str):
        self._response = response
    
    def fail_first_n(self, n: int):
        self._fail_count = n
    
    async def chat(self, prompt: str) -> str:
        self._call_count += 1
        if self._call_count <= self._fail_count:
            raise ValueError("Simulated failure")
        return self._response


# ========== Type Tests ==========


class TestAcademicEntityType:
    """エンティティタイプのテスト"""
    
    def test_all_types_defined(self):
        """全タイプが定義されている"""
        assert len(AcademicEntityType) == 10
    
    def test_type_values(self):
        """タイプ値が正しい"""
        assert AcademicEntityType.RESEARCHER.value == "researcher"
        assert AcademicEntityType.ORGANIZATION.value == "organization"
        assert AcademicEntityType.METHOD.value == "method"
        assert AcademicEntityType.MODEL.value == "model"
        assert AcademicEntityType.DATASET.value == "dataset"
        assert AcademicEntityType.METRIC.value == "metric"
        assert AcademicEntityType.TASK.value == "task"
        assert AcademicEntityType.CONCEPT.value == "concept"
        assert AcademicEntityType.TOOL.value == "tool"
        assert AcademicEntityType.PAPER.value == "paper"
    
    def test_type_from_string(self):
        """文字列からの変換"""
        assert AcademicEntityType("model") == AcademicEntityType.MODEL
        assert AcademicEntityType("dataset") == AcademicEntityType.DATASET


class TestEntity:
    """エンティティのテスト"""
    
    def test_entity_creation(self):
        """エンティティ作成"""
        entity = Entity(
            id="ent-001",
            name="BERT",
            type=AcademicEntityType.MODEL,
            description="A language model",
        )
        assert entity.id == "ent-001"
        assert entity.name == "BERT"
        assert entity.type == AcademicEntityType.MODEL
        assert entity.description == "A language model"
        assert entity.aliases == []
        assert entity.confidence == 1.0
    
    def test_entity_with_aliases(self):
        """エイリアス付きエンティティ"""
        entity = Entity(
            id="ent-002",
            name="GPT-4",
            type=AcademicEntityType.MODEL,
            description="Large language model",
            aliases=["GPT4", "GPT 4"],
        )
        assert len(entity.aliases) == 2
        assert "GPT4" in entity.aliases
    
    def test_to_dict(self):
        """辞書変換"""
        entity = Entity(
            id="ent-003",
            name="ImageNet",
            type=AcademicEntityType.DATASET,
            description="Large image dataset",
        )
        data = entity.to_dict()
        
        assert data["id"] == "ent-003"
        assert data["name"] == "ImageNet"
        assert data["type"] == "dataset"
        assert data["description"] == "Large image dataset"
    
    def test_from_dict(self):
        """辞書から生成"""
        data = {
            "id": "ent-004",
            "name": "Transformer",
            "type": "method",
            "description": "Attention-based architecture",
            "aliases": ["Attention is All You Need"],
        }
        entity = Entity.from_dict(data)
        
        assert entity.id == "ent-004"
        assert entity.name == "Transformer"
        assert entity.type == AcademicEntityType.METHOD
        assert len(entity.aliases) == 1
    
    def test_merge_with(self):
        """エンティティマージ"""
        entity1 = Entity(
            id="ent-a",
            name="BERT",
            type=AcademicEntityType.MODEL,
            description="Language model",
            source_chunk_ids=["chunk-1"],
        )
        entity2 = Entity(
            id="ent-b",
            name="bert",
            type=AcademicEntityType.MODEL,
            description="Bidirectional Encoder Representations from Transformers",
            source_chunk_ids=["chunk-2"],
            aliases=["Bidirectional Encoder"],
        )
        
        merged = entity1.merge_with(entity2)
        
        assert merged.id == "ent-a"  # Keep first ID
        assert merged.name == "BERT"  # Keep first name
        assert "chunk-1" in merged.source_chunk_ids
        assert "chunk-2" in merged.source_chunk_ids
        assert "bert" in merged.aliases or "Bidirectional Encoder" in merged.aliases
        # Longer description should be kept
        assert "Bidirectional" in merged.description


class TestExtractionResult:
    """抽出結果のテスト"""
    
    def test_success_result(self):
        """成功結果"""
        result = ExtractionResult(
            chunk_id="chunk-1",
            entities=[
                Entity(id="e1", name="BERT", type=AcademicEntityType.MODEL, description=""),
            ],
            raw_response='{"entities": [...]}',
            extraction_time_ms=100.5,
        )
        
        assert result.success is True
        assert result.entity_count == 1
        assert result.error is None
    
    def test_error_result(self):
        """エラー結果"""
        result = ExtractionResult(
            chunk_id="chunk-2",
            entities=[],
            raw_response="Invalid response",
            error="JSON parse error",
        )
        
        assert result.success is False
        assert result.entity_count == 0
        assert result.error == "JSON parse error"
    
    def test_to_dict(self):
        """辞書変換"""
        result = ExtractionResult(
            chunk_id="chunk-3",
            entities=[],
            extraction_time_ms=50.0,
        )
        data = result.to_dict()
        
        assert data["chunk_id"] == "chunk-3"
        assert data["extraction_time_ms"] == 50.0


class TestBatchExtractionResult:
    """バッチ抽出結果のテスト"""
    
    def test_success_rate(self):
        """成功率計算"""
        result = BatchExtractionResult(
            results=[
                ExtractionResult(chunk_id="1", entities=[]),
                ExtractionResult(chunk_id="2", entities=[], error="Error"),
                ExtractionResult(chunk_id="3", entities=[]),
            ],
            error_count=1,
        )
        
        assert result.success_rate == pytest.approx(2/3, rel=0.01)
    
    def test_empty_results(self):
        """空の結果"""
        result = BatchExtractionResult()
        assert result.success_rate == 0.0


# ========== Prompt Tests ==========


class TestPrompts:
    """プロンプトのテスト"""
    
    def test_extraction_prompt_format(self):
        """抽出プロンプトのフォーマット"""
        text = "BERT is a language model."
        prompt = get_extraction_prompt(text, "en")
        
        assert "BERT is a language model." in prompt
        assert "Entity Types" in prompt
        assert "RESEARCHER" in prompt
        assert "MODEL" in prompt
    
    def test_japanese_prompt(self):
        """日本語プロンプト"""
        text = "BERTは言語モデルです。"
        prompt = get_extraction_prompt(text, "ja")
        
        assert text in prompt
        assert "エンティティタイプ" in prompt


# ========== LLMEntityExtractor Tests ==========


class TestLLMEntityExtractor:
    """LLMエンティティ抽出のテスト"""
    
    @pytest.fixture
    def mock_llm(self):
        """モックLLMを作成"""
        return MockLLMClient()
    
    @pytest.fixture
    def extractor(self, mock_llm):
        """抽出器を作成"""
        return LLMEntityExtractor(mock_llm)
    
    @pytest.mark.asyncio
    async def test_extract_single_chunk(self, mock_llm, extractor):
        """単一チャンク抽出"""
        mock_llm.set_response(json.dumps({
            "entities": [
                {
                    "name": "BERT",
                    "type": "MODEL",
                    "description": "Language model by Google",
                    "aliases": ["Bidirectional Encoder"],
                }
            ]
        }))
        
        chunk = MockTextChunk(id="chunk-1", content="BERT is a language model.")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert len(result.entities) == 1
        assert result.entities[0].name == "BERT"
        assert result.entities[0].type == AcademicEntityType.MODEL
    
    @pytest.mark.asyncio
    async def test_extract_multiple_entities(self, mock_llm, extractor):
        """複数エンティティ抽出"""
        mock_llm.set_response(json.dumps({
            "entities": [
                {"name": "GPT-4", "type": "MODEL", "description": "LLM", "aliases": []},
                {"name": "OpenAI", "type": "ORGANIZATION", "description": "AI company", "aliases": []},
                {"name": "ImageNet", "type": "DATASET", "description": "Image dataset", "aliases": []},
            ]
        }))
        
        chunk = MockTextChunk(id="chunk-2", content="GPT-4 by OpenAI was trained on ImageNet.")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert len(result.entities) == 3
        
        names = [e.name for e in result.entities]
        assert "GPT-4" in names
        assert "OpenAI" in names
        assert "ImageNet" in names
    
    @pytest.mark.asyncio
    async def test_extract_no_entities(self, mock_llm, extractor):
        """エンティティなしの抽出"""
        mock_llm.set_response('{"entities": []}')
        
        chunk = MockTextChunk(id="chunk-3", content="The weather is nice today.")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert len(result.entities) == 0
    
    @pytest.mark.asyncio
    async def test_extract_invalid_json(self, mock_llm, extractor):
        """不正なJSON応答"""
        mock_llm.set_response("This is not JSON")
        
        chunk = MockTextChunk(id="chunk-4", content="Some text")
        result = await extractor.extract(chunk)
        
        assert result.success is False
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_extract_with_retry(self, mock_llm):
        """リトライテスト"""
        mock_llm.fail_first_n(2)
        mock_llm.set_response('{"entities": [{"name": "Test", "type": "CONCEPT", "description": "", "aliases": []}]}')
        
        extractor = LLMEntityExtractor(mock_llm, max_retries=3, retry_delay=0.01)
        chunk = MockTextChunk(id="chunk-5", content="Test content")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert len(result.entities) == 1
    
    @pytest.mark.asyncio
    async def test_extract_batch(self, mock_llm, extractor):
        """バッチ抽出"""
        mock_llm.set_response('{"entities": [{"name": "Entity", "type": "CONCEPT", "description": "", "aliases": []}]}')
        
        chunks = [
            MockTextChunk(id=f"chunk-{i}", content=f"Content {i}")
            for i in range(5)
        ]
        
        results = await extractor.extract_batch(chunks, max_concurrent=2)
        
        assert len(results) == 5
        assert all(r.success for r in results)
    
    @pytest.mark.asyncio
    async def test_extract_all_with_merge(self, mock_llm, extractor):
        """全抽出+マージ"""
        mock_llm.set_response('{"entities": [{"name": "BERT", "type": "MODEL", "description": "", "aliases": []}]}')
        
        chunks = [
            MockTextChunk(id=f"chunk-{i}", content=f"BERT content {i}")
            for i in range(3)
        ]
        
        result = await extractor.extract_all(chunks)
        
        assert result.total_entities == 3  # 3 chunks, each has 1 entity
        assert len(result.merged_entities) == 1  # All merged into 1
        assert result.merged_entities[0].name == "BERT"
        assert len(result.merged_entities[0].source_chunk_ids) == 3
    
    def test_merge_entities(self, extractor):
        """エンティティマージ"""
        entities = [
            Entity(id="1", name="BERT", type=AcademicEntityType.MODEL, description="Model 1", source_chunk_ids=["c1"]),
            Entity(id="2", name="bert", type=AcademicEntityType.MODEL, description="Model 2", source_chunk_ids=["c2"]),
            Entity(id="3", name="GPT", type=AcademicEntityType.MODEL, description="GPT model", source_chunk_ids=["c3"]),
        ]
        
        merged = extractor.merge_entities(entities)
        
        assert len(merged) == 2  # BERT (merged) and GPT
        
        bert_entity = next(e for e in merged if "bert" in e.name.lower())
        assert len(bert_entity.source_chunk_ids) == 2
    
    def test_normalize_name(self, extractor):
        """名前の正規化"""
        assert extractor._normalize_name("BERT") == "bert"
        assert extractor._normalize_name("  GPT-4  ") == "gpt-4"
        assert extractor._normalize_name("Image Net") == "image net"


# ========== CachedEntityExtractor Tests ==========


class TestCachedEntityExtractor:
    """キャッシュ付き抽出のテスト"""
    
    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """キャッシュヒット"""
        mock_llm = MockLLMClient()
        mock_llm.set_response('{"entities": [{"name": "Test", "type": "CONCEPT", "description": "", "aliases": []}]}')
        
        extractor = CachedEntityExtractor(mock_llm)
        chunk = MockTextChunk(id="chunk-1", content="Test content")
        
        # First call
        result1 = await extractor.extract(chunk)
        assert result1.success
        
        # Second call (should hit cache)
        result2 = await extractor.extract(chunk)
        assert result2.success
        
        # Only 1 LLM call should be made
        assert mock_llm._call_count == 1
        assert extractor.cache_size == 1
    
    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """キャッシュクリア"""
        mock_llm = MockLLMClient()
        mock_llm.set_response('{"entities": []}')
        
        extractor = CachedEntityExtractor(mock_llm)
        chunk = MockTextChunk(id="chunk-1", content="Test")
        
        await extractor.extract(chunk)
        assert extractor.cache_size == 1
        
        extractor.clear_cache()
        assert extractor.cache_size == 0


# ========== Entity Type Parsing Tests ==========


class TestEntityTypeParsing:
    """エンティティタイプ解析のテスト"""
    
    @pytest.mark.asyncio
    async def test_unknown_type_falls_back_to_concept(self):
        """未知のタイプはCONCEPTにフォールバック"""
        mock_llm = MockLLMClient()
        mock_llm.set_response(json.dumps({
            "entities": [
                {"name": "Unknown", "type": "UNKNOWN_TYPE", "description": "", "aliases": []},
            ]
        }))
        
        extractor = LLMEntityExtractor(mock_llm)
        chunk = MockTextChunk(id="chunk-1", content="Unknown entity")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert result.entities[0].type == AcademicEntityType.CONCEPT
    
    @pytest.mark.asyncio
    async def test_lowercase_type(self):
        """小文字タイプも受け付ける"""
        mock_llm = MockLLMClient()
        mock_llm.set_response(json.dumps({
            "entities": [
                {"name": "Test", "type": "model", "description": "", "aliases": []},
            ]
        }))
        
        extractor = LLMEntityExtractor(mock_llm)
        chunk = MockTextChunk(id="chunk-1", content="Test content")
        result = await extractor.extract(chunk)
        
        assert result.success
        assert result.entities[0].type == AcademicEntityType.MODEL
