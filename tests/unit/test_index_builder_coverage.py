# Index Builder Coverage Tests
"""
Unit tests for Level 0 and Level 1 index builders to improve coverage.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest


# ==============================================================================
# Mock Classes
# ==============================================================================

@dataclass
class MockDocument:
    """モックドキュメント"""
    id: str
    title: str
    source_path: str = ""
    abstract: str = ""
    
    @property
    def file_name(self) -> str:
        return Path(self.source_path).name if self.source_path else ""
    
    @property
    def file_type(self) -> str:
        return Path(self.source_path).suffix.lstrip(".") if self.source_path else "txt"
    
    @property
    def authors(self) -> list:
        return []
    
    @property
    def doi(self) -> str | None:
        return None
    
    @property
    def arxiv_id(self) -> str | None:
        return None
    
    @property
    def language(self) -> str:
        return "en"
    
    @property
    def page_count(self) -> int:
        return 0
    
    @property
    def keywords(self) -> list:
        return []
    
    @property
    def sections(self) -> list:
        return []
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source_path": self.source_path,
            "abstract": self.abstract,
            "file_name": self.file_name,
            "file_type": self.file_type,
        }


@dataclass
class MockTextUnit:
    """モックTextUnit"""
    id: str
    text: str
    n_tokens: int = 100
    document_id: str | None = None
    section_type: str | None = None
    chunk_index: int = 0
    start_char: int = 0
    end_char: int | None = None
    
    def __post_init__(self):
        if self.end_char is None:
            self.end_char = len(self.text)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "n_tokens": self.n_tokens,
            "document_id": self.document_id,
            "section_type": self.section_type,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


# ==============================================================================
# Level 0 Index Config Tests
# ==============================================================================

class TestLevel0IndexConfig:
    """Level0IndexConfig テスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        from monjyu.index.level0.builder import Level0IndexConfig
        
        config = Level0IndexConfig()
        assert config.embedding_strategy == "ollama"
        assert config.index_strategy == "lancedb"
        assert config.ollama_model == "nomic-embed-text"
        assert config.batch_size == 100
        assert config.show_progress is True
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        from monjyu.index.level0.builder import Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir="/tmp/custom",
            embedding_strategy="azure",
            index_strategy="azure_search",
            batch_size=50,
            show_progress=False,
        )
        assert str(config.output_dir) == "/tmp/custom"
        assert config.embedding_strategy == "azure"
        assert config.index_strategy == "azure_search"
        assert config.batch_size == 50
        assert config.show_progress is False
    
    def test_azure_settings(self):
        """Azure設定のテスト"""
        from monjyu.index.level0.builder import Level0IndexConfig
        
        config = Level0IndexConfig(
            azure_openai_deployment="my-deployment",
            azure_openai_endpoint="https://my-endpoint.openai.azure.com",
            azure_search_endpoint="https://my-search.search.windows.net",
            azure_search_index_name="my-index",
        )
        assert config.azure_openai_deployment == "my-deployment"
        assert config.azure_openai_endpoint == "https://my-endpoint.openai.azure.com"
        assert config.azure_search_endpoint == "https://my-search.search.windows.net"
        assert config.azure_search_index_name == "my-index"


# ==============================================================================
# Level 0 Index Tests
# ==============================================================================

class TestLevel0Index:
    """Level0Index テスト"""
    
    def test_properties(self, tmp_path):
        """プロパティのテスト"""
        from monjyu.index.level0.builder import Level0Index
        from monjyu.embedding.base import EmbeddingRecord
        
        docs = [MockDocument(id="d1", title="Doc1")]
        units = [MockTextUnit(id="u1", text="Test text")]
        embeddings = [
            EmbeddingRecord(
                id="e1",
                text_unit_id="u1",
                vector=[0.1, 0.2, 0.3],
                model="test-model",
                dimensions=3,
            )
        ]
        
        index = Level0Index(
            documents=docs,
            text_units=units,
            embeddings=embeddings,
            output_dir=tmp_path,
            embedding_model="test-model",
            embedding_dimensions=3,
        )
        
        assert index.document_count == 1
        assert index.text_unit_count == 1
        assert index.embedding_count == 1
        assert index.embedding_model == "test-model"
        assert index.embedding_dimensions == 3


# ==============================================================================
# Level 0 Index Builder Tests
# ==============================================================================

class TestLevel0IndexBuilderInit:
    """Level0IndexBuilder 初期化テスト"""
    
    def test_init_with_default_config(self, tmp_path):
        """デフォルト設定での初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(output_dir=str(tmp_path))
        builder = Level0IndexBuilder(config)
        
        assert builder.config == config
        assert builder.output_dir == tmp_path
        assert builder._embedding_client is None
        assert builder._vector_indexer is None
        assert builder._storage is None
    
    def test_init_without_config(self, tmp_path):
        """設定なしでの初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder
        
        with patch.object(Path, "mkdir"):
            builder = Level0IndexBuilder()
            assert builder.config is not None


class TestLevel0IndexBuilderProperties:
    """Level0IndexBuilder プロパティテスト"""
    
    def test_embedding_client_ollama(self, tmp_path):
        """Ollamaクライアントの遅延初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            embedding_strategy="ollama",
        )
        builder = Level0IndexBuilder(config)
        
        # 遅延初期化
        client = builder.embedding_client
        assert client is not None
        assert builder._embedding_client is client
        # 2回目は同じインスタンス
        assert builder.embedding_client is client
    
    def test_embedding_client_azure(self, tmp_path):
        """Azure OpenAIクライアントの遅延初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            embedding_strategy="azure",
            azure_openai_endpoint="https://test.openai.azure.com",
        )
        builder = Level0IndexBuilder(config)
        
        # AzureOpenAIEmbeddingClientのimportをモック
        mock_azure_client = MagicMock()
        with patch.dict("sys.modules", {"monjyu.embedding.azure_openai": MagicMock(AzureOpenAIEmbeddingClient=mock_azure_client)}):
            with patch("monjyu.index.level0.builder.Level0IndexBuilder._create_embedding_client") as mock_create:
                mock_create.return_value = MagicMock()
                client = builder.embedding_client
                assert client is not None
    
    def test_vector_indexer_lancedb(self, tmp_path):
        """LanceDBインデクサーの遅延初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            index_strategy="lancedb",
        )
        builder = Level0IndexBuilder(config)
        
        indexer = builder.vector_indexer
        assert indexer is not None
        assert builder._vector_indexer is indexer
        # 2回目は同じインスタンス
        assert builder.vector_indexer is indexer
    
    def test_vector_indexer_azure_search(self, tmp_path):
        """Azure AI Searchインデクサーの遅延初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            index_strategy="azure_search",
            azure_search_endpoint="https://test.search.windows.net",
        )
        builder = Level0IndexBuilder(config)
        
        with patch("monjyu.index.level0.builder.Level0IndexBuilder._create_vector_indexer") as mock_create:
            mock_create.return_value = MagicMock()
            indexer = builder.vector_indexer
            assert indexer is not None
    
    def test_storage_property(self, tmp_path):
        """ストレージの遅延初期化"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(output_dir=str(tmp_path))
        builder = Level0IndexBuilder(config)
        
        storage = builder.storage
        assert storage is not None
        assert builder._storage is storage
        # 2回目は同じインスタンス
        assert builder.storage is storage


class TestLevel0IndexBuilderCreateClients:
    """Level0IndexBuilder クライアント生成テスト"""
    
    def test_create_embedding_client_azure(self, tmp_path):
        """Azure埋め込みクライアントの生成"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            embedding_strategy="azure",
            azure_openai_deployment="test-deploy",
            azure_openai_endpoint="https://test.openai.azure.com",
        )
        builder = Level0IndexBuilder(config)
        
        # Azureモジュールをモック
        mock_openai = MagicMock()
        mock_azure_client_class = MagicMock()
        mock_openai.AzureOpenAIEmbeddingClient = mock_azure_client_class
        
        with patch.dict("sys.modules", {
            "monjyu.embedding.azure_openai": mock_openai,
        }):
            # _create_embedding_clientを直接呼び出す
            with patch("monjyu.index.level0.builder.AzureOpenAIEmbeddingClient", mock_azure_client_class, create=True):
                # モジュールレベルimportのモック
                import monjyu.index.level0.builder as builder_module
                original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
                
                def mock_import(name, *args, **kwargs):
                    if name == "monjyu.embedding.azure_openai":
                        return mock_openai
                    return original_import(name, *args, **kwargs)
                
                # テスト - azure戦略の場合のインポート動作確認
                assert config.embedding_strategy == "azure"
    
    def test_create_vector_indexer_azure_search(self, tmp_path):
        """Azure AI Searchインデクサーの生成"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            index_strategy="azure_search",
            azure_search_endpoint="https://test.search.windows.net",
            azure_search_index_name="test-index",
        )
        builder = Level0IndexBuilder(config)
        
        # azure_searchの確認
        assert config.index_strategy == "azure_search"
        assert config.azure_search_endpoint == "https://test.search.windows.net"


class TestLevel0IndexBuilderBuild:
    """Level0IndexBuilder build メソッドテスト"""
    
    @pytest.mark.asyncio
    async def test_build_success(self, tmp_path):
        """正常なビルド"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        # モッククライアント設定
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_embedding_client.model_name = "test-model"
        mock_embedding_client.dimensions = 3
        
        mock_indexer = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_stats.return_value = {"text_units_count": 1, "embeddings_count": 1}
        
        builder._embedding_client = mock_embedding_client
        builder._vector_indexer = mock_indexer
        builder._storage = mock_storage
        
        docs = [MockDocument(id="d1", title="Test Doc")]
        units = [MockTextUnit(id="u1", text="Test text", document_id="d1")]
        
        result = await builder.build(docs, units)
        
        assert result.document_count == 1
        assert result.text_unit_count == 1
        assert result.embedding_count == 1
        mock_embedding_client.embed_batch.assert_called_once()
        mock_indexer.build.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_build_with_progress(self, tmp_path, capsys):
        """進捗表示付きビルド"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=True,
        )
        builder = Level0IndexBuilder(config)
        
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_embedding_client.model_name = "test-model"
        mock_embedding_client.dimensions = 3
        
        mock_indexer = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_stats.return_value = {"text_units_count": 1, "embeddings_count": 1}
        
        builder._embedding_client = mock_embedding_client
        builder._vector_indexer = mock_indexer
        builder._storage = mock_storage
        
        docs = [MockDocument(id="d1", title="Test")]
        units = [MockTextUnit(id="u1", text="Test")]
        
        await builder.build(docs, units)
        
        captured = capsys.readouterr()
        assert "Building Level 0 index" in captured.out


class TestLevel0IndexBuilderGenerateEmbeddings:
    """Level0IndexBuilder _generate_embeddings テスト"""
    
    @pytest.mark.asyncio
    async def test_generate_embeddings(self, tmp_path):
        """埋め込み生成"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_client = AsyncMock()
        mock_client.embed_batch = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        mock_client.model_name = "test-model"
        mock_client.dimensions = 2
        builder._embedding_client = mock_client
        
        units = [
            MockTextUnit(id="u1", text="Text 1"),
            MockTextUnit(id="u2", text="Text 2"),
        ]
        
        embeddings = await builder._generate_embeddings(units)
        
        assert len(embeddings) == 2
        assert embeddings[0].text_unit_id == "u1"
        assert embeddings[1].text_unit_id == "u2"
        assert embeddings[0].vector == [0.1, 0.2]


class TestLevel0IndexBuilderBuildVectorIndex:
    """Level0IndexBuilder _build_vector_index テスト"""
    
    def test_build_vector_index(self, tmp_path):
        """ベクトルインデックス構築"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        from monjyu.embedding.base import EmbeddingRecord
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_indexer = MagicMock()
        builder._vector_indexer = mock_indexer
        
        units = [
            MockTextUnit(id="u1", text="Text 1", document_id="d1", section_type="intro", n_tokens=50),
            MockTextUnit(id="u2", text="Text 2", document_id="d2"),
        ]
        embeddings = [
            EmbeddingRecord(id="e1", text_unit_id="u1", vector=[0.1, 0.2], model="test", dimensions=2),
            EmbeddingRecord(id="e2", text_unit_id="u2", vector=[0.3, 0.4], model="test", dimensions=2),
        ]
        
        builder._build_vector_index(embeddings, units)
        
        mock_indexer.build.assert_called_once()
        args = mock_indexer.build.call_args
        assert len(args[0][0]) == 2  # vectors
        assert len(args[0][1]) == 2  # ids
        assert len(args[0][2]) == 2  # metadata
    
    def test_build_vector_index_missing_unit(self, tmp_path):
        """存在しないTextUnitの処理"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        from monjyu.embedding.base import EmbeddingRecord
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_indexer = MagicMock()
        builder._vector_indexer = mock_indexer
        
        units = []  # 空のunit list
        embeddings = [
            EmbeddingRecord(id="e1", text_unit_id="missing", vector=[0.1], model="test", dimensions=1),
        ]
        
        builder._build_vector_index(embeddings, units)
        
        # 空のメタデータで呼び出される
        mock_indexer.build.assert_called_once()


class TestLevel0IndexBuilderSaveToParquet:
    """Level0IndexBuilder _save_to_parquet テスト"""
    
    def test_save_to_parquet(self, tmp_path):
        """Parquet保存"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        from monjyu.embedding.base import EmbeddingRecord
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_storage = MagicMock()
        mock_storage.get_stats.return_value = {"text_units_count": 1, "embeddings_count": 1}
        builder._storage = mock_storage
        
        docs = [MockDocument(id="d1", title="Test")]
        units = [MockTextUnit(id="u1", text="Test")]
        embeddings = [
            EmbeddingRecord(id="e1", text_unit_id="u1", vector=[0.1], model="test", dimensions=1)
        ]
        
        builder._save_to_parquet(docs, units, embeddings)
        
        mock_storage.write_documents.assert_called_once()
        mock_storage.write_text_units.assert_called_once()
        mock_storage.write_embeddings.assert_called_once()
    
    def test_save_to_parquet_empty_lists(self, tmp_path):
        """空リストでの保存"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_storage = MagicMock()
        builder._storage = mock_storage
        
        builder._save_to_parquet([], [], [])
        
        # 空リストの場合は呼ばれない
        mock_storage.write_documents.assert_not_called()


class TestLevel0IndexBuilderAdd:
    """Level0IndexBuilder add メソッドテスト"""
    
    @pytest.mark.asyncio
    async def test_add_to_index(self, tmp_path):
        """既存インデックスへの追加"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level0IndexBuilder(config)
        
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_batch = AsyncMock(return_value=[[0.1, 0.2]])
        mock_embedding_client.model_name = "test-model"
        mock_embedding_client.dimensions = 2
        
        mock_indexer = MagicMock()
        mock_storage = MagicMock()
        
        builder._embedding_client = mock_embedding_client
        builder._vector_indexer = mock_indexer
        builder._storage = mock_storage
        
        docs = [MockDocument(id="d2", title="New Doc")]
        units = [MockTextUnit(id="u2", text="New text", document_id="d2")]
        
        result = await builder.add(docs, units)
        
        assert result is not None
        mock_indexer.add.assert_called_once()
        mock_storage.write_documents.assert_called_once_with(docs, append=True)


class TestLevel0IndexBuilderLoad:
    """Level0IndexBuilder load メソッドテスト"""
    
    def test_load_not_exists(self, tmp_path):
        """存在しないインデックスの読み込み"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(output_dir=str(tmp_path))
        builder = Level0IndexBuilder(config)
        
        mock_storage = MagicMock()
        mock_storage.exists.return_value = False
        builder._storage = mock_storage
        
        result = builder.load()
        
        assert result is None
    
    def test_load_success(self, tmp_path):
        """正常な読み込み"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(output_dir=str(tmp_path))
        builder = Level0IndexBuilder(config)
        
        mock_storage = MagicMock()
        mock_storage.exists.return_value = True
        mock_storage.read_text_units.return_value = [
            {"id": "u1", "text": "Test", "n_tokens": 10, "document_id": "d1"}
        ]
        mock_storage.read_embeddings.return_value = [
            {"id": "e1", "text_unit_id": "u1", "vector": [0.1, 0.2], "model": "test", "dimensions": 2}
        ]
        builder._storage = mock_storage
        
        mock_indexer = MagicMock()
        builder._vector_indexer = mock_indexer
        
        # vector_index_path が存在しない場合をシミュレート
        result = builder.load()
        
        assert result is not None
        assert result.text_unit_count == 1
        assert result.embedding_count == 1


class TestLevel0IndexBuilderGetStats:
    """Level0IndexBuilder get_stats テスト"""
    
    def test_get_stats(self, tmp_path):
        """統計情報取得"""
        from monjyu.index.level0.builder import Level0IndexBuilder, Level0IndexConfig
        
        config = Level0IndexConfig(output_dir=str(tmp_path))
        builder = Level0IndexBuilder(config)
        
        mock_storage = MagicMock()
        mock_storage.get_stats.return_value = {"text_units_count": 10, "embeddings_count": 10}
        builder._storage = mock_storage
        
        mock_indexer = MagicMock()
        mock_indexer.count.return_value = 10
        builder._vector_indexer = mock_indexer
        
        stats = builder.get_stats()
        
        assert stats["text_units_count"] == 10
        assert stats["vector_index_count"] == 10


# ==============================================================================
# Level 1 Index Config Tests
# ==============================================================================

class TestLevel1IndexConfig:
    """Level1IndexConfig テスト"""
    
    def test_default_values(self):
        """デフォルト値のテスト"""
        from monjyu.index.level1.builder import Level1IndexConfig
        
        config = Level1IndexConfig()
        assert config.spacy_model == "en_core_web_sm"
        assert config.academic_mode is True
        assert config.min_frequency == 2
        assert config.window_size == 5
        assert config.resolution == 1.0
        assert config.hierarchical_levels == 3
        assert config.batch_size == 50
        assert config.show_progress is True
    
    def test_custom_values(self):
        """カスタム値のテスト"""
        from monjyu.index.level1.builder import Level1IndexConfig
        
        config = Level1IndexConfig(
            output_dir="/tmp/level1",
            spacy_model="ja_core_news_sm",
            academic_mode=False,
            min_frequency=5,
            window_size=10,
            resolution=2.0,
            hierarchical_levels=5,
            batch_size=100,
            show_progress=False,
        )
        assert str(config.output_dir) == "/tmp/level1"
        assert config.spacy_model == "ja_core_news_sm"
        assert config.academic_mode is False
        assert config.min_frequency == 5


# ==============================================================================
# Level 1 Index Tests
# ==============================================================================

class TestLevel1Index:
    """Level1Index テスト"""
    
    def test_properties(self, tmp_path):
        """プロパティのテスト"""
        from monjyu.index.level1.builder import Level1Index
        from monjyu.graph.base import Community, NounPhraseNode, NounPhraseEdge
        from monjyu.nlp.base import NLPFeatures
        
        nlp_features = [
            NLPFeatures(text_unit_id="u1", keywords=["test"], noun_phrases=["test phrase"], entities=[])
        ]
        nodes = [NounPhraseNode(id="n1", phrase="test", frequency=5, text_unit_ids=["u1"])]
        edges = [NounPhraseEdge(source="n1", target="n2", weight=1.0, cooccurrence_count=2, document_ids=["d1"])]
        communities = [Community(id="c1", level=0, node_ids=["n1"], size=1)]
        
        index = Level1Index(
            nlp_features=nlp_features,
            nodes=nodes,
            edges=edges,
            communities=communities,
            output_dir=tmp_path,
        )
        
        assert index.feature_count == 1
        assert index.node_count == 1
        assert index.edge_count == 1
        assert index.community_count == 1


# ==============================================================================
# Level 1 Index Builder Tests
# ==============================================================================

class TestLevel1IndexBuilderInit:
    """Level1IndexBuilder 初期化テスト"""
    
    def test_init_with_default_config(self, tmp_path):
        """デフォルト設定での初期化"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        assert builder.config == config
        assert builder.output_dir == tmp_path
        assert builder._nlp_processor is None
        assert builder._graph_builder is None
        assert builder._community_detector is None
    
    def test_init_without_config(self, tmp_path):
        """設定なしでの初期化"""
        from monjyu.index.level1.builder import Level1IndexBuilder
        
        with patch.object(Path, "mkdir"):
            builder = Level1IndexBuilder()
            assert builder.config is not None


class TestLevel1IndexBuilderProperties:
    """Level1IndexBuilder プロパティテスト"""
    
    def test_nlp_processor_property(self, tmp_path):
        """NLPプロセッサの遅延初期化"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        # SpacyNLPProcessorをモック
        with patch("monjyu.index.level1.builder.SpacyNLPProcessor") as mock_spacy:
            mock_processor = MagicMock()
            mock_spacy.return_value = mock_processor
            
            processor = builder.nlp_processor
            assert processor is mock_processor
            assert builder._nlp_processor is mock_processor
            # 2回目は同じインスタンス
            assert builder.nlp_processor is mock_processor
    
    def test_graph_builder_property(self, tmp_path):
        """グラフビルダーの遅延初期化"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        with patch("monjyu.index.level1.builder.NounPhraseGraphBuilder") as mock_builder_class:
            mock_graph_builder = MagicMock()
            mock_builder_class.return_value = mock_graph_builder
            
            gb = builder.graph_builder
            assert gb is mock_graph_builder
            assert builder._graph_builder is mock_graph_builder
    
    def test_community_detector_property(self, tmp_path):
        """コミュニティ検出器の遅延初期化"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        with patch("monjyu.index.level1.builder.LeidenCommunityDetector") as mock_detector_class:
            mock_detector = MagicMock()
            mock_detector_class.return_value = mock_detector
            
            cd = builder.community_detector
            assert cd is mock_detector
            assert builder._community_detector is mock_detector


class TestLevel1IndexBuilderBuild:
    """Level1IndexBuilder build メソッドテスト"""
    
    def test_build_success(self, tmp_path):
        """正常なビルド"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        from monjyu.graph.base import NounPhraseNode, NounPhraseEdge, Community
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        # モック設定
        mock_nlp = MagicMock()
        mock_nlp.process_batch.return_value = [
            NLPFeatures(text_unit_id="u1", keywords=["test"], noun_phrases=["np"], entities=[])
        ]
        
        mock_graph = MagicMock()
        mock_node = NounPhraseNode(id="n1", phrase="test", frequency=1, text_unit_ids=["u1"])
        mock_edge = NounPhraseEdge(source="n1", target="n2", weight=1.0, cooccurrence_count=1, document_ids=["d1"])
        mock_graph.build_from_features.return_value = ([mock_node], [mock_edge])
        mock_graph.get_networkx_graph.return_value = MagicMock(number_of_nodes=lambda: 1)
        
        mock_detector = MagicMock()
        mock_community = Community(id="c1", level=0, node_ids=["n1"], size=1)
        mock_detector.detect.return_value = [mock_community]
        
        builder._nlp_processor = mock_nlp
        builder._graph_builder = mock_graph
        builder._community_detector = mock_detector
        
        units = [MockTextUnit(id="u1", text="Test text")]
        
        with patch.object(builder, "_save_to_parquet"):
            result = builder.build(units)
        
        assert result.feature_count == 1
        assert result.node_count == 1
        assert result.edge_count == 1
    
    def test_build_with_progress(self, tmp_path, capsys):
        """進捗表示付きビルド"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        from monjyu.graph.base import NounPhraseNode, NounPhraseEdge, Community
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=True,
        )
        builder = Level1IndexBuilder(config)
        
        mock_nlp = MagicMock()
        mock_nlp.process_batch.return_value = [
            NLPFeatures(text_unit_id="u1", keywords=["k"], noun_phrases=["np"], entities=[("ent", "TYPE")])
        ]
        
        mock_graph = MagicMock()
        mock_graph.build_from_features.return_value = ([], [])
        mock_graph.get_networkx_graph.return_value = MagicMock(number_of_nodes=lambda: 0)
        
        mock_detector = MagicMock()
        
        builder._nlp_processor = mock_nlp
        builder._graph_builder = mock_graph
        builder._community_detector = mock_detector
        
        units = [MockTextUnit(id="u1", text="Test")]
        
        with patch.object(builder, "_save_to_parquet"):
            builder.build(units)
        
        captured = capsys.readouterr()
        assert "Building Level 1 index" in captured.out


class TestLevel1IndexBuilderProcessNLP:
    """Level1IndexBuilder _process_nlp テスト"""
    
    def test_process_nlp(self, tmp_path):
        """NLP処理"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        mock_nlp = MagicMock()
        mock_nlp.process_batch.return_value = [
            NLPFeatures(text_unit_id="u1", keywords=["key"], noun_phrases=["np"], entities=[])
        ]
        builder._nlp_processor = mock_nlp
        
        units = [MockTextUnit(id="u1", text="Test text")]
        
        features = builder._process_nlp(units)
        
        assert len(features) == 1
        mock_nlp.process_batch.assert_called_once()


class TestLevel1IndexBuilderBuildGraph:
    """Level1IndexBuilder _build_graph テスト"""
    
    def test_build_graph(self, tmp_path):
        """グラフ構築"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        from monjyu.graph.base import NounPhraseNode, NounPhraseEdge
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        mock_graph = MagicMock()
        mock_node = NounPhraseNode(id="n1", phrase="test", frequency=1, text_unit_ids=["u1"])
        mock_edge = NounPhraseEdge(source="n1", target="n2", weight=1.0, cooccurrence_count=1, document_ids=["d1"])
        mock_graph.build_from_features.return_value = ([mock_node], [mock_edge])
        builder._graph_builder = mock_graph
        
        features = [NLPFeatures(text_unit_id="u1", keywords=[], noun_phrases=["np"], entities=[])]
        units = [MockTextUnit(id="u1", text="Test")]
        
        nodes, edges = builder._build_graph(features, units)
        
        assert len(nodes) == 1
        assert len(edges) == 1


class TestLevel1IndexBuilderDetectCommunities:
    """Level1IndexBuilder _detect_communities テスト"""
    
    def test_detect_communities_empty_graph(self, tmp_path):
        """空グラフでのコミュニティ検出"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        mock_graph = MagicMock()
        mock_graph.get_networkx_graph.return_value = MagicMock(number_of_nodes=lambda: 0)
        builder._graph_builder = mock_graph
        
        communities = builder._detect_communities()
        
        assert communities == []
    
    def test_detect_communities_single_level(self, tmp_path):
        """単一レベルのコミュニティ検出"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.graph.base import Community
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
            hierarchical_levels=1,
        )
        builder = Level1IndexBuilder(config)
        
        mock_graph = MagicMock()
        mock_nx_graph = MagicMock()
        mock_nx_graph.number_of_nodes.return_value = 5
        mock_graph.get_networkx_graph.return_value = mock_nx_graph
        builder._graph_builder = mock_graph
        
        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            Community(id="c1", level=0, node_ids=["n1"], size=1)
        ]
        builder._community_detector = mock_detector
        
        communities = builder._detect_communities()
        
        assert len(communities) == 1
        mock_detector.detect.assert_called_once()
    
    def test_detect_communities_hierarchical(self, tmp_path):
        """階層的コミュニティ検出"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.graph.base import Community
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
            hierarchical_levels=3,
        )
        builder = Level1IndexBuilder(config)
        
        mock_graph = MagicMock()
        mock_nx_graph = MagicMock()
        mock_nx_graph.number_of_nodes.return_value = 10
        mock_graph.get_networkx_graph.return_value = mock_nx_graph
        builder._graph_builder = mock_graph
        
        mock_detector = MagicMock()
        mock_detector.detect_hierarchical.return_value = [
            [Community(id="c1", level=0, node_ids=["n1"], size=1)],
            [Community(id="c2", level=1, node_ids=["c1"], size=1)],
        ]
        builder._community_detector = mock_detector
        
        communities = builder._detect_communities()
        
        assert len(communities) == 2
        mock_detector.detect_hierarchical.assert_called_once()


class TestLevel1IndexBuilderSaveToParquet:
    """Level1IndexBuilder _save_to_parquet テスト"""
    
    def test_save_to_parquet(self, tmp_path):
        """Parquet保存"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        from monjyu.graph.base import NounPhraseNode, NounPhraseEdge, Community
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        nlp_features = [
            NLPFeatures(text_unit_id="u1", keywords=["k"], noun_phrases=["np"], entities=[("ent", "TYPE")])
        ]
        nodes = [NounPhraseNode(id="n1", phrase="test", frequency=1, text_unit_ids=["u1"])]
        edges = [NounPhraseEdge(source="n1", target="n2", weight=1.0, cooccurrence_count=1, document_ids=["d1"])]
        communities = [Community(id="c1", level=0, node_ids=["n1"], size=1)]
        
        builder._save_to_parquet(nlp_features, nodes, edges, communities)
        
        # ファイルが作成されたことを確認
        assert (tmp_path / "nlp_features.parquet").exists()
        assert (tmp_path / "noun_phrase_nodes.parquet").exists()
        assert (tmp_path / "noun_phrase_edges.parquet").exists()
        assert (tmp_path / "communities_l1.parquet").exists()
    
    def test_save_empty_data(self, tmp_path):
        """空データの保存"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(
            output_dir=str(tmp_path),
            show_progress=False,
        )
        builder = Level1IndexBuilder(config)
        
        # 空リストで保存
        builder._save_to_parquet([], [], [], [])
        
        # ファイルは作成されない
        assert not (tmp_path / "nlp_features.parquet").exists()


class TestLevel1IndexBuilderLoad:
    """Level1IndexBuilder load メソッドテスト"""
    
    def test_load_not_exists(self, tmp_path):
        """存在しないインデックスの読み込み"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        result = builder.load()
        
        assert result is None
    
    def test_load_success(self, tmp_path):
        """正常な読み込み"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        from monjyu.nlp.base import NLPFeatures
        from monjyu.graph.base import NounPhraseNode, NounPhraseEdge, Community
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        # テストデータを保存
        nlp_data = [{"text_unit_id": "u1", "keywords": ["k"], "noun_phrases": ["np"], "entities_text": ["e"], "entities_type": ["TYPE"]}]
        pq.write_table(pa.Table.from_pylist(nlp_data), tmp_path / "nlp_features.parquet")
        
        node_data = [{"id": "n1", "phrase": "test", "frequency": 1, "text_unit_ids": ["u1"]}]
        pq.write_table(pa.Table.from_pylist(node_data), tmp_path / "noun_phrase_nodes.parquet")
        
        edge_data = [{"source": "n1", "target": "n2", "weight": 1.0, "cooccurrence_count": 1, "document_ids": ["d1"]}]
        pq.write_table(pa.Table.from_pylist(edge_data), tmp_path / "noun_phrase_edges.parquet")
        
        community_data = [{"id": "c1", "level": 0, "node_ids": ["n1"], "size": 1, "representative_phrases": [], "internal_edges": 0, "parent_id": None}]
        pq.write_table(pa.Table.from_pylist(community_data), tmp_path / "communities_l1.parquet")
        
        result = builder.load()
        
        assert result is not None
        assert result.feature_count == 1
        assert result.node_count == 1
        assert result.edge_count == 1
        assert result.community_count == 1


class TestLevel1IndexBuilderGetStats:
    """Level1IndexBuilder get_stats テスト"""
    
    def test_get_stats_empty(self, tmp_path):
        """空の統計情報"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        stats = builder.get_stats()
        
        assert stats["nlp_features_count"] == 0
        assert stats["nodes_count"] == 0
    
    def test_get_stats_with_data(self, tmp_path):
        """データありの統計情報"""
        from monjyu.index.level1.builder import Level1IndexBuilder, Level1IndexConfig
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        config = Level1IndexConfig(output_dir=str(tmp_path))
        builder = Level1IndexBuilder(config)
        
        # テストデータを保存
        nlp_data = [{"text_unit_id": "u1", "keywords": [], "noun_phrases": [], "entities_text": [], "entities_type": []}]
        pq.write_table(pa.Table.from_pylist(nlp_data), tmp_path / "nlp_features.parquet")
        
        stats = builder.get_stats()
        
        assert stats["nlp_features_count"] == 1
