# Index Level 1 Integration Tests
"""
Integration tests for Level 1 index building.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from monjyu.index.level1 import Level1IndexBuilder, Level1IndexConfig, Level1Index
from monjyu.nlp.base import NLPFeatures
from monjyu.graph.base import NounPhraseNode, Community


@dataclass
class MockTextUnit:
    """モックTextUnit"""
    id: str
    text: str
    n_tokens: int = 0
    document_id: str | None = None
    section_type: str | None = None


class TestLevel1IndexConfig:
    """Level1IndexConfigのテスト"""
    
    def test_default_config(self):
        """デフォルト設定"""
        config = Level1IndexConfig()
        
        assert config.spacy_model == "en_core_web_sm"
        assert config.min_frequency == 2
        assert config.resolution == 1.0
    
    def test_custom_config(self):
        """カスタム設定"""
        config = Level1IndexConfig(
            output_dir="/custom/path",
            min_frequency=3,
            resolution=0.5,
        )
        
        assert config.output_dir == "/custom/path"
        assert config.min_frequency == 3
        assert config.resolution == 0.5


class TestLevel1IndexBuilder:
    """Level1IndexBuilderの統合テスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "output"
    
    @pytest.fixture
    def sample_text_units(self):
        """サンプルTextUnits"""
        return [
            MockTextUnit(
                id="tu_001",
                text="Deep learning has revolutionized natural language processing. "
                     "Neural networks can learn complex patterns from data.",
                document_id="doc_001",
                section_type="abstract",
            ),
            MockTextUnit(
                id="tu_002",
                text="The transformer architecture uses attention mechanisms. "
                     "BERT and GPT are transformer-based models.",
                document_id="doc_001",
                section_type="body",
            ),
            MockTextUnit(
                id="tu_003",
                text="Natural language processing enables computers to understand text. "
                     "Deep learning models achieve state-of-the-art results.",
                document_id="doc_002",
                section_type="abstract",
            ),
            MockTextUnit(
                id="tu_004",
                text="Attention mechanisms allow models to focus on relevant parts. "
                     "The neural network learns representations automatically.",
                document_id="doc_002",
                section_type="body",
            ),
        ]
    
    def test_builder_init(self, temp_output_dir):
        """ビルダー初期化"""
        config = Level1IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        assert builder.config == config
        assert builder.output_dir == temp_output_dir
    
    def test_build_index(self, temp_output_dir, sample_text_units):
        """インデックス構築"""
        try:
            config = Level1IndexConfig(
                output_dir=temp_output_dir,
                show_progress=False,
                min_frequency=1,  # テスト用に低く設定
            )
            builder = Level1IndexBuilder(config)
            
            index = builder.build(sample_text_units)
            
            assert isinstance(index, Level1Index)
            assert index.feature_count == 4
            assert index.node_count >= 1
            # コミュニティは検出されるかもしれない
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise
    
    def test_build_creates_files(self, temp_output_dir, sample_text_units):
        """ファイルが作成される"""
        try:
            config = Level1IndexConfig(
                output_dir=temp_output_dir,
                show_progress=False,
                min_frequency=1,
            )
            builder = Level1IndexBuilder(config)
            
            builder.build(sample_text_units)
            
            # Parquetファイルが作成されていることを確認
            assert (temp_output_dir / "nlp_features.parquet").exists()
            # ノードがある場合のみ
            # assert (temp_output_dir / "noun_phrase_nodes.parquet").exists()
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise
    
    def test_get_stats(self, temp_output_dir, sample_text_units):
        """統計情報取得"""
        try:
            config = Level1IndexConfig(
                output_dir=temp_output_dir,
                show_progress=False,
                min_frequency=1,
            )
            builder = Level1IndexBuilder(config)
            
            # 構築前
            stats = builder.get_stats()
            assert stats["nlp_features_count"] == 0
            
            # 構築後
            builder.build(sample_text_units)
            stats = builder.get_stats()
            assert stats["nlp_features_count"] == 4
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise
    
    def test_load_existing_index(self, temp_output_dir, sample_text_units):
        """既存インデックスの読み込み"""
        try:
            config = Level1IndexConfig(
                output_dir=temp_output_dir,
                show_progress=False,
                min_frequency=1,
            )
            builder = Level1IndexBuilder(config)
            
            # 構築
            builder.build(sample_text_units)
            
            # 新しいビルダーで読み込み
            new_builder = Level1IndexBuilder(config)
            loaded_index = new_builder.load()
            
            assert loaded_index is not None
            assert loaded_index.feature_count == 4
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise
    
    def test_load_nonexistent_index(self, temp_output_dir):
        """存在しないインデックスの読み込み"""
        config = Level1IndexConfig(
            output_dir=temp_output_dir,
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        loaded_index = builder.load()
        
        assert loaded_index is None


class TestLevel1IndexE2E:
    """Level 1インデックスのE2Eテスト"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """一時出力ディレクトリ"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "output"
    
    def test_full_pipeline(self, temp_output_dir):
        """フルパイプラインテスト"""
        try:
            # 1. テキストユニット準備
            text_units = [
                MockTextUnit(
                    id="tu_001",
                    text="Graph neural networks learn representations of graph-structured data. "
                         "They aggregate information from neighboring nodes.",
                    document_id="paper_001",
                ),
                MockTextUnit(
                    id="tu_002",
                    text="Knowledge graphs represent entities and their relationships. "
                         "Graph neural networks can reason over knowledge graphs.",
                    document_id="paper_001",
                ),
                MockTextUnit(
                    id="tu_003",
                    text="The attention mechanism allows models to focus on relevant information. "
                         "Graph attention networks use attention for neighbor aggregation.",
                    document_id="paper_002",
                ),
            ]
            
            # 2. インデックス構築
            config = Level1IndexConfig(
                output_dir=temp_output_dir,
                show_progress=False,
                min_frequency=1,
                hierarchical_levels=2,
            )
            builder = Level1IndexBuilder(config)
            
            index = builder.build(text_units)
            
            # 3. 検証
            assert index.feature_count == 3
            
            # NLP特徴量の確認
            for features in index.nlp_features:
                assert features.text_unit_id.startswith("tu_")
                assert isinstance(features.keywords, list)
                assert isinstance(features.noun_phrases, list)
            
            # ノードの確認
            if index.node_count > 0:
                for node in index.nodes:
                    assert isinstance(node, NounPhraseNode)
                    assert node.frequency >= 1
            
            # コミュニティの確認
            for community in index.communities:
                assert isinstance(community, Community)
                assert len(community.node_ids) > 0
            
            # 4. 統計確認
            stats = builder.get_stats()
            assert stats["nlp_features_count"] == 3
            
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise


class TestNLPProcessingIntegration:
    """NLP処理の統合テスト"""
    
    def test_academic_terms_detection(self):
        """学術用語検出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            
            processor = SpacyNLPProcessor(academic_mode=True)
            
            text = "BERT uses the transformer architecture with attention mechanisms."
            features = processor.process(text, "tu_001")
            
            # 学術用語が検出される
            entity_texts = [e[0].lower() for e in features.entities]
            assert any("transformer" in t or "attention" in t for t in entity_texts)
            
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise
    
    def test_method_extraction(self):
        """手法名抽出"""
        try:
            from monjyu.nlp.spacy_processor import SpacyNLPProcessor
            
            processor = SpacyNLPProcessor()
            
            text = "We compare BERT, GPT-4, and ResNet on various benchmarks."
            methods = processor.extract_methods(text)
            
            # CamelCase と ACRONYM が検出される
            assert len(methods) >= 1
            
        except RuntimeError as e:
            if "spaCy model" in str(e):
                pytest.skip("spaCy model not installed")
            raise


class TestGraphBuildingIntegration:
    """グラフ構築の統合テスト"""
    
    def test_cooccurrence_graph(self):
        """共起グラフ構築"""
        from monjyu.graph.noun_phrase_graph import NounPhraseGraphBuilder
        from monjyu.nlp.base import NLPFeatures
        
        builder = NounPhraseGraphBuilder(min_frequency=1)
        
        features = [
            NLPFeatures(
                text_unit_id="tu_001",
                noun_phrases=["deep learning", "neural network", "attention"],
            ),
            NLPFeatures(
                text_unit_id="tu_002",
                noun_phrases=["neural network", "transformer", "attention"],
            ),
        ]
        
        text_units = [
            MockTextUnit("tu_001", "text1", document_id="doc_001"),
            MockTextUnit("tu_002", "text2", document_id="doc_001"),
        ]
        
        nodes, edges = builder.build_from_features(features, text_units)
        
        # neural network と attention が2回出現
        assert any(n.frequency >= 2 for n in nodes)
        
        # 共起エッジが存在
        assert len(edges) >= 1
    
    def test_community_detection_on_graph(self):
        """グラフからのコミュニティ検出"""
        import networkx as nx
        from monjyu.graph.community_detector import LeidenCommunityDetector
        
        # テスト用グラフ
        graph = nx.Graph()
        
        # クラスタ1: NLP関連
        for i, phrase in enumerate(["nlp", "text", "language"]):
            graph.add_node(f"np_{i}", phrase=phrase, frequency=5)
        graph.add_edge("np_0", "np_1", weight=3)
        graph.add_edge("np_1", "np_2", weight=3)
        graph.add_edge("np_0", "np_2", weight=3)
        
        # クラスタ2: Vision関連
        for i, phrase in enumerate(["vision", "image", "pixel"], start=3):
            graph.add_node(f"np_{i}", phrase=phrase, frequency=4)
        graph.add_edge("np_3", "np_4", weight=3)
        graph.add_edge("np_4", "np_5", weight=3)
        graph.add_edge("np_3", "np_5", weight=3)
        
        detector = LeidenCommunityDetector(resolution=1.0)
        communities = detector.detect(graph)
        
        assert len(communities) >= 1
        
        # 各コミュニティに代表フレーズがある
        for comm in communities:
            assert len(comm.representative_phrases) > 0
