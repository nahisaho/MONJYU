# Test Community Report Generator - FEAT-013
"""
CommunityReportGenerator の単体テスト
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime


class TestFinding:
    """Finding データモデルのテスト"""
    
    def test_finding_creation(self):
        """Finding作成テスト"""
        from monjyu.index.community_report_generator.types import Finding
        
        finding = Finding(
            id="finding-1",
            summary="Important discovery",
            explanation="Detailed explanation",
            evidence=["Evidence 1", "Evidence 2"],
        )
        
        assert finding.id == "finding-1"
        assert finding.summary == "Important discovery"
        assert len(finding.evidence) == 2
    
    def test_finding_defaults(self):
        """Findingデフォルト値テスト"""
        from monjyu.index.community_report_generator.types import Finding
        
        finding = Finding(id="f1", summary="Test")
        
        assert finding.explanation == ""
        assert finding.evidence == []
    
    def test_finding_to_dict(self):
        """to_dictテスト"""
        from monjyu.index.community_report_generator.types import Finding
        
        finding = Finding(
            id="f1",
            summary="Test",
            explanation="Details",
            evidence=["E1"],
        )
        
        data = finding.to_dict()
        
        assert data["id"] == "f1"
        assert data["summary"] == "Test"
        assert data["evidence"] == ["E1"]
    
    def test_finding_from_dict(self):
        """from_dictテスト"""
        from monjyu.index.community_report_generator.types import Finding
        
        data = {
            "id": "f1",
            "summary": "Restored finding",
            "evidence": ["E1", "E2"],
        }
        
        finding = Finding.from_dict(data)
        
        assert finding.id == "f1"
        assert finding.summary == "Restored finding"


class TestCommunityReport:
    """CommunityReport データモデルのテスト"""
    
    def test_report_creation(self):
        """Report作成テスト"""
        from monjyu.index.community_report_generator.types import (
            CommunityReport,
            Finding,
        )
        
        report = CommunityReport(
            community_id="comm-001",
            title="Transformer Research",
            summary="This community focuses on transformer architectures.",
            rating=8.5,
        )
        
        assert report.community_id == "comm-001"
        assert report.title == "Transformer Research"
        assert report.rating == 8.5
    
    def test_report_defaults(self):
        """Reportデフォルト値テスト"""
        from monjyu.index.community_report_generator.types import CommunityReport
        
        report = CommunityReport(
            community_id="test",
            title="Test",
            summary="Summary",
        )
        
        assert report.full_content == ""
        assert report.findings == []
        assert report.rating == 0.0
        assert report.entity_count == 0
        assert report.level == 0
        assert report.created_at is not None
    
    def test_report_finding_count(self):
        """finding_countプロパティテスト"""
        from monjyu.index.community_report_generator.types import (
            CommunityReport,
            Finding,
        )
        
        report = CommunityReport(
            community_id="test",
            title="Test",
            summary="Summary",
            findings=[
                Finding(id="f1", summary="Finding 1"),
                Finding(id="f2", summary="Finding 2"),
            ],
        )
        
        assert report.finding_count == 2
    
    def test_report_add_finding(self):
        """add_findingテスト"""
        from monjyu.index.community_report_generator.types import (
            CommunityReport,
            Finding,
        )
        
        report = CommunityReport(
            community_id="test",
            title="Test",
            summary="Summary",
        )
        
        report.add_finding(Finding(id="f1", summary="New finding"))
        
        assert report.finding_count == 1
    
    def test_report_get_findings_summary(self):
        """get_findings_summaryテスト"""
        from monjyu.index.community_report_generator.types import (
            CommunityReport,
            Finding,
        )
        
        report = CommunityReport(
            community_id="test",
            title="Test",
            summary="Summary",
            findings=[
                Finding(id="f1", summary="Finding A"),
                Finding(id="f2", summary="Finding B"),
            ],
        )
        
        summaries = report.get_findings_summary()
        
        assert summaries == ["Finding A", "Finding B"]
    
    def test_report_to_dict(self):
        """to_dictテスト"""
        from monjyu.index.community_report_generator.types import (
            CommunityReport,
            Finding,
        )
        
        report = CommunityReport(
            community_id="comm-001",
            title="Test Report",
            summary="Summary",
            rating=7.5,
            findings=[Finding(id="f1", summary="F1")],
        )
        
        data = report.to_dict()
        
        assert data["community_id"] == "comm-001"
        assert data["title"] == "Test Report"
        assert data["rating"] == 7.5
        assert len(data["findings"]) == 1
        assert "created_at" in data
    
    def test_report_from_dict(self):
        """from_dictテスト"""
        from monjyu.index.community_report_generator.types import CommunityReport
        
        data = {
            "community_id": "comm-001",
            "title": "Restored Report",
            "summary": "Restored summary",
            "rating": 8.0,
            "findings": [{"id": "f1", "summary": "Finding"}],
            "created_at": "2024-01-01T12:00:00",
        }
        
        report = CommunityReport.from_dict(data)
        
        assert report.community_id == "comm-001"
        assert report.rating == 8.0
        assert len(report.findings) == 1


class TestReportGenerationResult:
    """ReportGenerationResult のテスト"""
    
    def test_result_creation(self):
        """Result作成テスト"""
        from monjyu.index.community_report_generator.types import ReportGenerationResult
        
        result = ReportGenerationResult(total_communities=10)
        
        assert result.total_communities == 10
        assert result.successful == 0
        assert result.failed == 0
    
    def test_result_success_rate(self):
        """success_rateテスト"""
        from monjyu.index.community_report_generator.types import (
            ReportGenerationResult,
            CommunityReport,
        )
        
        result = ReportGenerationResult(total_communities=10)
        
        for i in range(7):
            result.add_report(CommunityReport(
                community_id=f"comm-{i}",
                title="Test",
                summary="Summary",
            ))
        
        for i in range(3):
            result.add_error(f"Error {i}")
        
        assert result.success_rate == 0.7
    
    def test_result_to_dict(self):
        """to_dictテスト"""
        from monjyu.index.community_report_generator.types import (
            ReportGenerationResult,
            CommunityReport,
        )
        
        result = ReportGenerationResult(total_communities=2)
        result.add_report(CommunityReport(
            community_id="c1",
            title="Test",
            summary="Summary",
        ))
        result.add_error("Error 1")
        
        data = result.to_dict()
        
        assert data["total_communities"] == 2
        assert data["successful"] == 1
        assert data["failed"] == 1
        assert len(data["reports"]) == 1


class TestPrompts:
    """プロンプトのテスト"""
    
    def test_get_prompts_en(self):
        """英語プロンプト取得テスト"""
        from monjyu.index.community_report_generator.prompts import get_prompts
        
        prompts = get_prompts("en")
        
        assert "system" in prompts
        assert "user" in prompts
        assert "research communities" in prompts["system"]
    
    def test_get_prompts_ja(self):
        """日本語プロンプト取得テスト"""
        from monjyu.index.community_report_generator.prompts import get_prompts
        
        prompts = get_prompts("ja")
        
        assert "system" in prompts
        assert "user" in prompts
        assert "研究コミュニティ" in prompts["system"]
    
    def test_format_entities_section(self):
        """エンティティセクションフォーマットテスト"""
        from monjyu.index.community_report_generator.prompts import format_entities_section
        
        entities = [
            {"name": "Transformer", "type": "METHOD", "description": "Neural network architecture"},
            {"name": "BERT", "type": "MODEL"},
        ]
        
        formatted = format_entities_section(entities)
        
        assert "Transformer" in formatted
        assert "METHOD" in formatted
        assert "BERT" in formatted
    
    def test_format_entities_section_empty(self):
        """空エンティティセクションテスト"""
        from monjyu.index.community_report_generator.prompts import format_entities_section
        
        formatted = format_entities_section([])
        
        assert "(No entities)" in formatted
    
    def test_format_relationships_section(self):
        """関係性セクションフォーマットテスト"""
        from monjyu.index.community_report_generator.prompts import format_relationships_section
        
        relationships = [
            {"source": "BERT", "target": "Transformer", "type": "BASED_ON"},
        ]
        
        formatted = format_relationships_section(relationships)
        
        assert "BERT" in formatted
        assert "BASED_ON" in formatted
        assert "Transformer" in formatted
    
    def test_build_report_prompt(self):
        """プロンプト構築テスト"""
        from monjyu.index.community_report_generator.prompts import build_report_prompt
        
        entities = [{"name": "Entity1", "type": "TYPE1"}]
        relationships = [{"source": "E1", "target": "E2", "type": "REL"}]
        
        prompts = build_report_prompt(
            community_id="comm-001",
            level=1,
            entities=entities,
            relationships=relationships,
            language="en",
        )
        
        assert "system" in prompts
        assert "user" in prompts
        assert "comm-001" in prompts["user"]


class TestCommunityReportGeneratorConfig:
    """設定のテスト"""
    
    def test_config_defaults(self):
        """デフォルト設定テスト"""
        from monjyu.index.community_report_generator.generator import (
            CommunityReportGeneratorConfig,
        )
        
        config = CommunityReportGeneratorConfig()
        
        assert config.language == "en"
        assert config.max_retries == 3
        assert config.max_concurrent == 5
    
    def test_config_custom(self):
        """カスタム設定テスト"""
        from monjyu.index.community_report_generator.generator import (
            CommunityReportGeneratorConfig,
        )
        
        config = CommunityReportGeneratorConfig(
            language="ja",
            max_retries=5,
            max_concurrent=10,
        )
        
        assert config.language == "ja"
        assert config.max_retries == 5
        assert config.max_concurrent == 10


class TestCommunityReportGenerator:
    """CommunityReportGenerator のテスト"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """モックLLMクライアント"""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=json.dumps({
            "title": "Test Community Report",
            "summary": "This is a test summary",
            "full_content": "Detailed content here",
            "findings": [
                {
                    "id": "finding-1",
                    "summary": "First finding",
                    "explanation": "Explanation",
                    "evidence": ["E1"]
                }
            ],
            "rating": 7.5,
            "rating_explanation": "Good community"
        }))
        return mock
    
    @pytest.mark.asyncio
    async def test_generator_creation(self, mock_llm_client):
        """Generator作成テスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        assert generator is not None
        assert generator.llm_client is mock_llm_client
    
    @pytest.mark.asyncio
    async def test_generate_report(self, mock_llm_client):
        """レポート生成テスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        report = await generator.generate(
            community_id="comm-001",
            level=0,
            entities=[{"name": "Entity1", "type": "TYPE1"}],
            relationships=[],
        )
        
        assert report.community_id == "comm-001"
        assert report.title == "Test Community Report"
        assert report.rating == 7.5
        assert len(report.findings) == 1
    
    @pytest.mark.asyncio
    async def test_generate_batch(self, mock_llm_client):
        """バッチ生成テスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        communities = [
            {
                "community_id": "comm-001",
                "level": 0,
                "entities": [{"name": "E1", "type": "T1"}],
                "relationships": [],
            },
            {
                "community_id": "comm-002",
                "level": 0,
                "entities": [{"name": "E2", "type": "T2"}],
                "relationships": [],
            },
        ]
        
        result = await generator.generate_batch(communities)
        
        assert result.total_communities == 2
        assert result.successful == 2
        assert len(result.reports) == 2


class TestJsonParsing:
    """JSONパースのテスト"""
    
    @pytest.fixture
    def mock_llm_client(self):
        """モックLLMクライアント"""
        return AsyncMock()
    
    @pytest.mark.asyncio
    async def test_parse_json_in_code_block(self, mock_llm_client):
        """コードブロック内のJSONパーステスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm_client.generate = AsyncMock(return_value="""
Here is the report:

```json
{
    "title": "Code Block Report",
    "summary": "Summary in code block",
    "findings": [],
    "rating": 6.0,
    "rating_explanation": "Test"
}
```

Hope this helps!
""")
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        report = await generator.generate(
            community_id="comm-001",
            level=0,
            entities=[],
            relationships=[],
        )
        
        assert report.title == "Code Block Report"
    
    @pytest.mark.asyncio
    async def test_parse_raw_json(self, mock_llm_client):
        """生JSONパーステスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm_client.generate = AsyncMock(return_value="""
{"title": "Raw JSON Report", "summary": "Direct JSON", "findings": [], "rating": 5.0, "rating_explanation": ""}
""")
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        report = await generator.generate(
            community_id="comm-001",
            level=0,
            entities=[],
            relationships=[],
        )
        
        assert report.title == "Raw JSON Report"
    
    @pytest.mark.asyncio
    async def test_parse_invalid_json(self, mock_llm_client):
        """不正JSONパーステスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm_client.generate = AsyncMock(return_value="""
This is not valid JSON at all.
No JSON here.
""")
        
        generator = CommunityReportGenerator(mock_llm_client)
        
        report = await generator.generate(
            community_id="comm-001",
            level=0,
            entities=[],
            relationships=[],
        )
        
        # デフォルト値が使われる
        assert report.title == "Untitled Community"


class TestCreateReportGenerator:
    """create_report_generatorファクトリ関数のテスト"""
    
    def test_create_generator(self):
        """ファクトリ関数テスト"""
        from monjyu.index.community_report_generator import (
            create_report_generator,
        )
        
        mock_llm = AsyncMock()
        
        generator = create_report_generator(
            llm_client=mock_llm,
            language="ja",
            max_concurrent=10,
        )
        
        assert generator.config.language == "ja"
        assert generator.config.max_concurrent == 10


class TestIntegration:
    """統合テスト"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """完全ワークフローテスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
            CommunityReport,
            Finding,
        )
        
        # モックLLM
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "title": "Academic Research Community",
            "summary": "This community focuses on NLP research.",
            "full_content": "Detailed analysis of the community...",
            "findings": [
                {
                    "id": "f1",
                    "summary": "Transformer is widely used",
                    "explanation": "The transformer architecture...",
                    "evidence": ["Paper A", "Paper B"]
                },
                {
                    "id": "f2",
                    "summary": "BERT dominates benchmarks",
                    "explanation": "BERT and its variants...",
                    "evidence": ["Benchmark results"]
                }
            ],
            "rating": 8.5,
            "rating_explanation": "High impact research area"
        }))
        
        # 1. 生成器作成
        generator = CommunityReportGenerator(mock_llm)
        
        # 2. 入力データ
        entities = [
            {"name": "Transformer", "type": "METHOD", "description": "Attention-based architecture"},
            {"name": "BERT", "type": "MODEL", "description": "Pre-trained language model"},
            {"name": "Google", "type": "ORGANIZATION"},
        ]
        relationships = [
            {"source": "BERT", "target": "Transformer", "type": "BASED_ON"},
            {"source": "Google", "target": "Transformer", "type": "PROPOSED"},
        ]
        
        # 3. レポート生成
        report = await generator.generate(
            community_id="nlp-community-001",
            level=1,
            entities=entities,
            relationships=relationships,
        )
        
        # 4. 検証
        assert report.community_id == "nlp-community-001"
        assert report.title == "Academic Research Community"
        assert report.rating == 8.5
        assert report.finding_count == 2
        assert report.entity_count == 3
        assert report.relationship_count == 2
        
        # 5. シリアライズ
        data = report.to_dict()
        assert "findings" in data
        assert len(data["findings"]) == 2
    
    @pytest.mark.asyncio
    async def test_batch_workflow(self):
        """バッチワークフローテスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm = AsyncMock()
        call_count = 0
        
        async def mock_generate(messages):
            nonlocal call_count
            call_count += 1
            return json.dumps({
                "title": f"Community {call_count}",
                "summary": f"Summary {call_count}",
                "findings": [],
                "rating": 5.0 + call_count,
                "rating_explanation": "Test"
            })
        
        mock_llm.generate = mock_generate
        
        generator = CommunityReportGenerator(mock_llm)
        
        communities = [
            {"community_id": f"comm-{i}", "entities": [], "relationships": []}
            for i in range(5)
        ]
        
        result = await generator.generate_batch(communities)
        
        assert result.total_communities == 5
        assert result.successful == 5
        assert result.failed == 0
        assert result.success_rate == 1.0


class TestGenerateFromEntities:
    """generate_from_entities統合テスト"""
    
    @pytest.mark.asyncio
    async def test_generate_from_entities_basic(self):
        """基本的なgenerate_from_entitiesテスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "title": "Research Community",
            "summary": "A group of related research concepts",
            "full_content": "Detailed content",
            "findings": [{"id": "f1", "summary": "Finding 1"}],
            "rating": 7.5,
            "rating_explanation": "Good community"
        }))
        
        generator = CommunityReportGenerator(mock_llm)
        
        entities = [
            {"id": "e1", "name": "Transformer", "type": "METHOD"},
            {"id": "e2", "name": "BERT", "type": "METHOD"},
            {"id": "e3", "name": "Attention", "type": "CONCEPT"},
        ]
        relationships = [
            {"source": "e1", "target": "e3", "type": "USES"},
            {"source": "e2", "target": "e1", "type": "BASED_ON"},
        ]
        
        reports = await generator.generate_from_entities(entities, relationships)
        
        assert len(reports) >= 1
        assert all(isinstance(r.community_id, str) for r in reports)
    
    @pytest.mark.asyncio
    async def test_generate_from_entities_empty(self):
        """空のエンティティリストでgenerate_from_entitiesテスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm = AsyncMock()
        generator = CommunityReportGenerator(mock_llm)
        
        reports = await generator.generate_from_entities([], [])
        
        assert reports == []
    
    @pytest.mark.asyncio
    async def test_generate_from_entities_disconnected(self):
        """非連結グラフでgenerate_from_entitiesテスト"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
        )
        
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(return_value=json.dumps({
            "title": "Community",
            "summary": "Summary",
            "findings": [],
            "rating": 5.0,
        }))
        
        generator = CommunityReportGenerator(mock_llm)
        
        # 非連結のエンティティ
        entities = [
            {"id": "e1", "name": "Node1", "type": "CONCEPT"},
            {"id": "e2", "name": "Node2", "type": "CONCEPT"},
            {"id": "e3", "name": "Node3", "type": "CONCEPT"},
        ]
        # e1-e2は接続、e3は孤立
        relationships = [
            {"source": "e1", "target": "e2", "type": "RELATED"},
        ]
        
        reports = await generator.generate_from_entities(entities, relationships)
        
        # 連結成分ごとにコミュニティが生成されるはず
        assert len(reports) >= 1


class TestSyncLLMAdapter:
    """SyncLLMAdapterテスト"""
    
    @pytest.mark.asyncio
    async def test_sync_llm_adapter_basic(self):
        """基本的なSyncLLMAdapterテスト"""
        from monjyu.index.community_report_generator import SyncLLMAdapter
        
        class MockSyncClient:
            def generate(self, prompt, **kwargs):
                return f"Response to: {prompt[:20]}"
        
        adapter = SyncLLMAdapter(MockSyncClient())
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        
        response = await adapter.generate(messages)
        
        assert "Response to:" in response
    
    @pytest.mark.asyncio
    async def test_sync_llm_adapter_message_format(self):
        """メッセージフォーマット変換テスト"""
        from monjyu.index.community_report_generator import SyncLLMAdapter
        
        received_prompt = None
        
        class MockSyncClient:
            def generate(self, prompt, **kwargs):
                nonlocal received_prompt
                received_prompt = prompt
                return "OK"
        
        adapter = SyncLLMAdapter(MockSyncClient())
        
        messages = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
        ]
        
        await adapter.generate(messages)
        
        assert "System: System message" in received_prompt
        assert "User: User message" in received_prompt


class TestDefaultLLMClient:
    """デフォルトLLMクライアント生成テスト"""
    
    def test_generator_without_llm_client(self):
        """LLMクライアントなしでジェネレータ作成"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
            CommunityReportGeneratorConfig,
        )
        
        # llm_client=Noneで作成可能
        generator = CommunityReportGenerator()
        
        assert generator.llm_client is not None
    
    def test_generator_with_custom_config(self):
        """カスタム設定でジェネレータ作成"""
        from monjyu.index.community_report_generator import (
            CommunityReportGenerator,
            CommunityReportGeneratorConfig,
        )
        
        config = CommunityReportGeneratorConfig(
            language="ja",
            llm_model="custom-model",
        )
        
        generator = CommunityReportGenerator(config=config)
        
        assert generator.config.language == "ja"
        assert generator.config.llm_model == "custom-model"
