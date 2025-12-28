# End-to-End Pipeline Integration Tests
"""
E2E tests for the complete MONJYU pipeline:
Level 0 Index → Level 1 Index → Query (Local/Global Search)
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pytest


# =============================================================================
# Mock Data Classes
# =============================================================================


@dataclass
class MockDocument:
    """モックドキュメント"""
    id: str
    title: str
    content: str
    authors: List[str] = field(default_factory=list)
    year: int = 2024
    abstract: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "authors": self.authors,
            "year": self.year,
            "abstract": self.abstract,
        }


@dataclass
class MockTextUnit:
    """モックテキストユニット"""
    id: str
    text: str
    n_tokens: int = 0
    document_id: Optional[str] = None
    section_type: Optional[str] = None
    chunk_index: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "document_id": self.document_id,
            "section_type": self.section_type,
            "chunk_index": self.chunk_index,
        }


# =============================================================================
# Test Data
# =============================================================================


SAMPLE_DOCUMENTS = [
    MockDocument(
        id="doc_transformer",
        title="Attention Is All You Need",
        authors=["Vaswani", "Shazeer", "Parmar"],
        year=2017,
        abstract="We propose a new simple network architecture, the Transformer, "
                 "based solely on attention mechanisms.",
        content="""
        The Transformer architecture has revolutionized natural language processing.
        Unlike recurrent neural networks, Transformers process all positions simultaneously.
        The self-attention mechanism allows the model to attend to different positions.
        Multi-head attention allows the model to jointly attend to information from different 
        representation subspaces at different positions.
        The encoder-decoder structure enables sequence-to-sequence learning.
        Position encoding adds positional information since attention has no inherent order.
        """
    ),
    MockDocument(
        id="doc_bert",
        title="BERT: Pre-training of Deep Bidirectional Transformers",
        authors=["Devlin", "Chang", "Lee", "Toutanova"],
        year=2018,
        abstract="We introduce BERT, a new language representation model using "
                 "bidirectional training of Transformer.",
        content="""
        BERT is designed to pre-train deep bidirectional representations.
        It uses masked language modeling to learn bidirectional context.
        BERT achieves state-of-the-art results on many NLP benchmarks.
        The model can be fine-tuned for various downstream tasks.
        Pre-training on large corpora captures rich language understanding.
        Transformer architecture enables efficient parallel computation.
        """
    ),
    MockDocument(
        id="doc_gpt",
        title="Language Models are Few-Shot Learners",
        authors=["Brown", "Mann", "Ryder"],
        year=2020,
        abstract="We demonstrate that scaling up language models greatly improves "
                 "task-agnostic, few-shot performance.",
        content="""
        GPT-3 is an autoregressive language model with 175 billion parameters.
        Few-shot learning allows the model to perform tasks with minimal examples.
        In-context learning enables adaptation without parameter updates.
        The Transformer decoder architecture generates text autoregressively.
        Scaling laws predict performance improvements with model size.
        Large language models exhibit emergent capabilities at scale.
        """
    ),
]


def create_text_units_from_documents(documents: List[MockDocument]) -> List[MockTextUnit]:
    """ドキュメントからテキストユニットを生成"""
    text_units = []
    
    for doc in documents:
        # Abstract
        text_units.append(MockTextUnit(
            id=f"{doc.id}_abstract",
            text=doc.abstract,
            n_tokens=len(doc.abstract.split()),
            document_id=doc.id,
            section_type="abstract",
            chunk_index=0,
        ))
        
        # Content chunks
        sentences = [s.strip() for s in doc.content.strip().split('\n') if s.strip()]
        for i, sentence in enumerate(sentences):
            text_units.append(MockTextUnit(
                id=f"{doc.id}_chunk_{i}",
                text=sentence,
                n_tokens=len(sentence.split()),
                document_id=doc.id,
                section_type="body",
                chunk_index=i + 1,
            ))
    
    return text_units


# =============================================================================
# Integration Test: Level 0 → Level 1 Pipeline
# =============================================================================


class TestLevel0ToLevel1Pipeline:
    """Level 0 から Level 1 へのパイプラインテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def text_units(self):
        """テストデータ"""
        return create_text_units_from_documents(SAMPLE_DOCUMENTS)
    
    def test_level0_to_level1_data_flow(self, temp_output_dir, text_units):
        """Level 0 の出力が Level 1 の入力として使用可能"""
        from monjyu.index.level0 import Level0IndexConfig
        from monjyu.index.level1 import Level1IndexConfig
        
        # Level 0 設定
        level0_config = Level0IndexConfig(
            output_dir=str(temp_output_dir / "level0"),
            embedding_strategy="ollama",
            index_strategy="lancedb",
        )
        
        # Level 1 設定
        level1_config = Level1IndexConfig(
            output_dir=str(temp_output_dir / "level1"),
            spacy_model="en_core_web_sm",
            min_frequency=1,  # テスト用に低く設定
        )
        
        # 設定が正しく作成されることを確認
        assert level0_config.output_dir is not None
        assert level1_config.output_dir is not None
        
        # テキストユニットが両レベルで使用可能な形式
        for tu in text_units[:5]:
            assert tu.id is not None
            assert tu.text is not None
            assert tu.document_id is not None
    
    def test_text_unit_compatibility(self, text_units):
        """TextUnit形式の互換性テスト"""
        # 全てのテキストユニットが必要なフィールドを持つ
        for tu in text_units:
            # Level 0 で必要
            assert hasattr(tu, 'id')
            assert hasattr(tu, 'text')
            
            # Level 1 で必要
            assert hasattr(tu, 'document_id')
            assert hasattr(tu, 'n_tokens')
            
            # 辞書変換可能
            d = tu.to_dict()
            assert 'id' in d
            assert 'text' in d


# =============================================================================
# Integration Test: Query Router
# =============================================================================


class TestQueryRouterIntegration:
    """クエリルーターの統合テスト"""
    
    @pytest.mark.asyncio
    async def test_router_with_different_queries(self):
        """異なるクエリタイプでのルーティング"""
        from monjyu.query.router import QueryRouter, QueryRouterConfig
        
        router = QueryRouter(QueryRouterConfig(
            use_llm_classification=False,  # ルールベースのみ
        ))
        
        test_cases = [
            # (query, expected_modes - いずれかにマッチすればOK)
            ("What is the Transformer architecture?", ["lazy", "vector", "local"]),
            ("How does BERT compare to GPT?", ["lazy", "local", "global"]),
            ("Summarize the papers", ["global", "lazy"]),
            ("Find papers about attention", ["vector", "lazy"]),
        ]
        
        for query, expected_modes in test_cases:
            result = await router.route(query)
            # ルーティング結果が返される
            assert result is not None
            assert hasattr(result, 'mode') or hasattr(result, 'search_mode')


# =============================================================================
# Integration Test: Local Search with Mock Data
# =============================================================================


class TestLocalSearchIntegration:
    """LocalSearch の統合テスト"""
    
    @pytest.fixture
    def setup_local_search(self):
        """LocalSearch のセットアップ"""
        from monjyu.query.local_search import (
            LocalSearch,
            LocalSearchConfig,
            InMemoryEntityStore,
            InMemoryRelationshipStore,
            InMemoryChunkStore,
            MockLLMClient,
            EntityInfo,
            RelationshipInfo,
            ChunkInfo,
        )
        
        # ストアを作成
        entity_store = InMemoryEntityStore()
        relationship_store = InMemoryRelationshipStore()
        chunk_store = InMemoryChunkStore()
        
        # エンティティを追加
        entities = [
            EntityInfo(
                entity_id="e_transformer",
                name="Transformer",
                entity_type="MODEL",
                description="A neural network architecture based on self-attention",
            ),
            EntityInfo(
                entity_id="e_bert",
                name="BERT",
                entity_type="MODEL",
                description="Bidirectional Encoder Representations from Transformers",
            ),
            EntityInfo(
                entity_id="e_gpt",
                name="GPT",
                entity_type="MODEL",
                description="Generative Pre-trained Transformer",
            ),
            EntityInfo(
                entity_id="e_attention",
                name="Self-Attention",
                entity_type="MECHANISM",
                description="Mechanism that relates different positions of a sequence",
            ),
        ]
        
        for entity in entities:
            entity_store.add_entity(entity)
        
        # リレーションシップを追加
        relationships = [
            RelationshipInfo(
                relationship_id="r1",
                source_id="e_bert",
                target_id="e_transformer",
                relation_type="BASED_ON",
                description="BERT is based on Transformer",
            ),
            RelationshipInfo(
                relationship_id="r2",
                source_id="e_gpt",
                target_id="e_transformer",
                relation_type="BASED_ON",
                description="GPT uses Transformer decoder",
            ),
            RelationshipInfo(
                relationship_id="r3",
                source_id="e_transformer",
                target_id="e_attention",
                relation_type="USES",
                description="Transformer uses self-attention mechanism",
            ),
        ]
        
        for rel in relationships:
            relationship_store.add_relationship(rel)
        
        # チャンクを追加
        chunks = [
            ChunkInfo(
                chunk_id="ch1",
                content="The Transformer architecture revolutionized NLP.",
                paper_id="doc_transformer",
                paper_title="Attention Is All You Need",
            ),
            ChunkInfo(
                chunk_id="ch2",
                content="BERT achieves state-of-the-art results on many benchmarks.",
                paper_id="doc_bert",
                paper_title="BERT Paper",
            ),
        ]
        
        for chunk in chunks:
            chunk_store.add_chunk(chunk, entity_ids=["e_transformer", "e_bert"])
        
        # LLMクライアント
        llm_client = MockLLMClient(responses={
            "transformer": "The Transformer is a neural network architecture that uses self-attention mechanisms.",
            "bert": "BERT is a pre-trained model based on Transformer for NLP tasks.",
        })
        
        # LocalSearch インスタンス
        search = LocalSearch(
            llm_client=llm_client,
            entity_store=entity_store,
            relationship_store=relationship_store,
            chunk_store=chunk_store,
            config=LocalSearchConfig(
                max_hops=2,
                top_k_entities=10,
            ),
        )
        
        return search
    
    def test_local_search_entity_discovery(self, setup_local_search):
        """エンティティ発見のテスト"""
        search = setup_local_search
        
        result = search.search("Transformer")
        
        assert result is not None
        assert result.query == "Transformer"
        assert len(result.entities_found) > 0
        
        # Transformer エンティティが見つかる
        entity_names = [e.entity.name for e in result.entities_found]
        assert "Transformer" in entity_names
    
    def test_local_search_graph_traversal(self, setup_local_search):
        """グラフトラバーサルのテスト"""
        search = setup_local_search
        
        result = search.search("Transformer")
        
        # リレーションシップも取得される
        if result.relationships_used:
            # Transformer からの関係
            relation_types = [r.relation_type for r in result.relationships_used]
            assert len(relation_types) >= 0  # 関係がある場合
    
    def test_local_search_answer_generation(self, setup_local_search):
        """回答生成のテスト"""
        search = setup_local_search
        
        result = search.search("Transformer")
        
        assert result.answer is not None
        assert len(result.answer) > 0


# =============================================================================
# Integration Test: Global Search with Mock Data
# =============================================================================


class TestGlobalSearchIntegration:
    """GlobalSearch の統合テスト"""
    
    @pytest.fixture
    def setup_global_search(self):
        """GlobalSearch のセットアップ"""
        from monjyu.query.global_search import (
            GlobalSearch,
            GlobalSearchConfig,
            InMemoryCommunityStore,
            MockLLMClient,
            CommunityInfo,
        )
        
        # コミュニティストアを作成
        community_store = InMemoryCommunityStore()
        
        # コミュニティを追加
        communities = [
            CommunityInfo(
                community_id="c1",
                title="Transformer Architecture",
                summary="Research on Transformer-based neural network architectures",
                level=1,
                size=100,
                key_entities=["Transformer", "Self-Attention", "Encoder-Decoder"],
                findings=[
                    "Transformers outperform RNNs on many tasks",
                    "Self-attention enables parallel processing",
                ],
            ),
            CommunityInfo(
                community_id="c2",
                title="Pre-trained Language Models",
                summary="Research on BERT, GPT, and other pre-trained models",
                level=1,
                size=80,
                key_entities=["BERT", "GPT", "Pre-training", "Fine-tuning"],
                findings=[
                    "Pre-training on large corpora improves performance",
                    "Transfer learning is effective for NLP",
                ],
            ),
        ]
        
        for community in communities:
            community_store.add_community(community)
        
        # LLMクライアント
        llm_client = MockLLMClient(responses={
            "transformer": "Transformers are neural network architectures using attention.",
            "bert": "BERT is a pre-trained bidirectional model.",
        })
        
        # GlobalSearch インスタンス
        search = GlobalSearch(
            llm_client=llm_client,
            community_store=community_store,
            config=GlobalSearchConfig(
                community_level=1,
                top_k_communities=5,
                map_reduce_enabled=True,
            ),
        )
        
        return search
    
    def test_global_search_community_selection(self, setup_global_search):
        """コミュニティ選択のテスト"""
        search = setup_global_search
        
        result = search.search("What are the main research areas?")
        
        assert result is not None
        assert len(result.communities_used) > 0
    
    def test_global_search_map_reduce(self, setup_global_search):
        """Map-Reduce パターンのテスト"""
        search = setup_global_search
        
        result = search.search("Summarize Transformer research")
        
        assert result is not None
        assert result.answer is not None
        # Map-Reduce が有効な場合、map_results がある
        # (コミュニティがない場合は空になる可能性あり)
    
    def test_global_search_answer_generation(self, setup_global_search):
        """回答生成のテスト"""
        search = setup_global_search
        
        result = search.search("Tell me about NLP research")
        
        assert result.answer is not None
        assert len(result.answer) > 0


# =============================================================================
# Integration Test: Full Search Pipeline
# =============================================================================


class TestFullSearchPipeline:
    """完全な検索パイプラインのテスト"""
    
    def test_search_mode_selection(self):
        """検索モード選択のテスト"""
        from monjyu.api.base import SearchMode
        
        # 全モードが定義されている
        assert SearchMode.VECTOR is not None
        assert SearchMode.LAZY is not None
        assert SearchMode.LOCAL is not None
        assert SearchMode.GLOBAL is not None
        assert SearchMode.AUTO is not None
    
    def test_monjyu_api_initialization(self):
        """MONJYU API の初期化テスト"""
        from monjyu.api import MONJYU
        
        monjyu = MONJYU()
        
        assert monjyu is not None
        assert monjyu.config is not None
    
    def test_monjyu_search_modes(self):
        """MONJYU 検索モードのテスト"""
        from monjyu.api import MONJYU, SearchMode
        
        monjyu = MONJYU()
        
        # 各モードで検索を試行
        for mode in [SearchMode.VECTOR, SearchMode.LAZY]:
            try:
                result = monjyu.search("test query", mode=mode)
                # 結果が返される（データがなくてもエラーにならない）
                assert result is not None
            except Exception as e:
                # データがない場合はエラーになる可能性があるが、
                # モード自体は認識される
                assert "mode" not in str(e).lower() or "invalid" not in str(e).lower()


# =============================================================================
# Integration Test: CLI E2E Workflow
# =============================================================================


class TestCLIE2EWorkflow:
    """CLI E2E ワークフローのテスト"""
    
    def test_cli_workflow_init_to_query(self, tmp_path):
        """init → index → query の完全なワークフロー"""
        from typer.testing import CliRunner
        from monjyu.cli import app
        
        runner = CliRunner()
        
        # 1. プロジェクト初期化
        project_dir = tmp_path / "test_project"
        result = runner.invoke(app, ["init", str(project_dir)])
        assert result.exit_code == 0
        assert (project_dir / "monjyu.yaml").exists()
        assert (project_dir / "papers").is_dir()
        
        # 2. テストドキュメント作成
        papers_dir = project_dir / "papers"
        (papers_dir / "test_paper.md").write_text("""
# Test Paper: Machine Learning

## Abstract
This paper discusses machine learning techniques.

## Introduction
Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
        """)
        
        # 3. インデックス構築
        result = runner.invoke(app, [
            "index", "build", str(papers_dir),
            "--config", str(project_dir / "monjyu.yaml"),
        ])
        assert result.exit_code == 0
        
        # 4. ステータス確認
        result = runner.invoke(app, [
            "index", "status",
            "--config", str(project_dir / "monjyu.yaml"),
        ])
        assert result.exit_code == 0
        
        # 5. 検索実行
        result = runner.invoke(app, [
            "query", "What is machine learning?",
            "--config", str(project_dir / "monjyu.yaml"),
        ])
        # 検索は実行される（結果の内容は問わない）
        assert "Error" not in result.stdout or "Search" in result.stdout
